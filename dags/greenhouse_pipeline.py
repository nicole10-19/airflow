from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import logging  
import pandas as pd 
import time 
from sqlalchemy import create_engine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import os


sys.path.insert(0, '/opt/airflow/dags') 

from sensor_simulator import run

log = logging.getLogger(__name__) 

DB_CONN = 'postgresql+psycopg2://airflow:airflow@postgres:5432/greenhouse_db' # Stringa di connessione a PostgreSQL 


def generate_data():
    # Chiamata a run() che genera (o aggiorna) il file sensori.csv con nuove misurazioni 
    run() 


def load_data():

    # La funzione legge il CSV generato dal simulatore e ritorna il DataFrame in formato JSON
    # così che Airflow può passarlo alla task successiva
    df = pd.read_csv("/opt/airflow/dags/sensori.csv", sep=";")
    log.info(f"Caricati {len(df)} record dal CSV")
    return df.to_json() 


    # NB: viene utilizzato '**kwargs' perchè la funzione può ricevere un numero variabile di argomenti
def clean_data(**kwargs):

    ti = kwargs['ti'] 

    # Recupera i dati letti dal CSV
    data = ti.xcom_pull(task_ids='load_data') 
    df = pd.read_json(data) 

    righe_iniziali = len(df)


    #  Scarta le righe senza valore o senza timestamps, in quanto non utilizzabili 
    mask_scartati = df["value"].isna() | (df["value"] == "null") | \
                    df["day_time"].isna() | (df["day_time"] == "null")

    df_discarded = df[mask_scartati].copy() 
    df_discarded["discard_reason"] = "missing_value_or_timestamp"

    df_clean = df[~mask_scartati].copy()

    
    # Conversionein float, se un valore non è convertibile lo trasforma in NaN  e identifica valori non numerici 
    df_clean["value"] = pd.to_numeric(df_clean["value"], errors="coerce")
    mask_non_numerici = df_clean["value"].isna()


    if mask_non_numerici.any():

        # Aggiunge gli scarti agli altri e rimuove i non numerici dai dati puliti
        extra_scartati = df_clean[mask_non_numerici].copy()
        extra_scartati["discard_reason"] = "non_numeric_value"
        df_discarded = pd.concat([df_discarded, extra_scartati], ignore_index=True)
        df_clean = df_clean[~mask_non_numerici]

    df_clean["id_sensor"] = df_clean["id_sensor"].replace("null", "UNKNOWN")
    df_clean["id_sensor"] = df_clean["id_sensor"].fillna("UNKNOWN")

    log.info(
        f"Cleaning: {righe_iniziali} totali → "
        f"{len(df_clean)} validi, {len(df_discarded)} scartati"
    )

    # Passa sia i dati puliti che quelli scartati alla task successiva 
    ti.xcom_push(key="clean_df",     value=df_clean.to_json())
    ti.xcom_push(key="discarded_df", value=df_discarded.to_json())


def anomaly_detection(**kwargs):
    start_time= time.time()
    ti = kwargs['ti']

    # Recupero i parametri puliti e analizzo ogni parametro separatamente
    data = ti.xcom_pull(task_ids='clean_data', key="clean_df")
    df = pd.read_json(data)

    df['value'] = df['value'].astype(float)

    parameters = df["parameter_name"].unique()
    result = []

    for param in parameters:
        #Prende solo i dati di quel parametro 
        df_param = df[df["parameter_name"] == param].copy()
        df_param = df_param.dropna(subset=['value'])

        # Evito di applicare DBSCAN su dataset troppo piccoli 
        if len(df_param) < 5:
            log.warning(f"Parametro '{param}': soli {len(df_param)} record, skippato")
            continue

        scaler = StandardScaler()
        normalized_values = scaler.fit_transform(df_param[['value']])

        # Se tutti i valori sono uguali, la varianza è zero e quindi DBSCAN non funziona 
        if pd.isna(normalized_values).any():
            log.warning(f"Parametro '{param}': varianza zero, skippato")
            continue

        df_param['value_normalized'] = normalized_values

        dbscan = DBSCAN(eps=0.5, min_samples=3)
        labels = dbscan.fit_predict(df_param[['value_normalized']])

        #Definito True se DBSCAN considera il punto un outlier 
        df_param['anomaly_detected_by_dbscan'] = (labels == -1)
        df_param['confidence_score'] = 0.0

        if 'anomaly' not in df_param.columns:
            df_param['anomaly'] = False

        n_anomalie = df_param['anomaly_detected_by_dbscan'].sum()
        log.info(f"Parametro '{param}': {len(df_param)} record, {n_anomalie} anomalie DBSCAN")

        result.append(df_param)

    if not result:
        log.error("Nessun parametro processato — controlla il CSV.")
        raise ValueError("anomaly_detection: nessun dato disponibile dopo il filtraggio")

    df_final = pd.concat(result, ignore_index=True)

    tp= 0 # True positive
    fp=0 # False positive
    fn=0 # False negative


    for idx, row in df_final.iterrows(): # Per ogni riga del DataFrame

        #Estrai i due valori che servono 

        is_anomaly_real = (row['anomaly'] == True) or (row['anomaly'] == True) # Controlla se il campo anomaly è True
        is_anomaly_detected = row['anomaly_detected_by_dbscan'] # Estrae il risultato di DBSCAN 


        #Confronta i due risultati e conta
        if is_anomaly_real and is_anomaly_detected:
            tp += 1
        elif not is_anomaly_real and is_anomaly_detected:
            fp +=1
        elif is_anomaly_real and not is_anomaly_detected:
            fn += 1
    
    if (tp+fp)>0: 
        precision = tp/ (tp+fp) # Maggiore è la percentuale di 'precision' , più DBSCAN è sensibile e marchia tutto come anomalia
    else:
        precision = 0.0
    
    if(tp+fn) >0:
        recall = tp/ (tp+fn) # Più il valore di 'recall' è alto, più DBSCAN non segnale quasi nessuna anomalia 
    else:
        recall= 0.0
    
    if (precision + recall) >0:
        f1= 2* (precision *recall) /(precision +recall) # Calcolo di una media armonica che penalizza i modelli squilibrati
    else:
        f1= 0.0
    
    log.info(f"Metriche: TP={tp}, FP={fp}, FN={fn}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    end_time = time.time()
    execution_time = end_time -start_time

    metrics={
        "execution_time": execution_time,
        "true_positive": tp,
        "false_positive": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall" : recall,
        "f1_score": f1
    }
    # Passa i risultati al task successivo via XCom

    ti.xcom_push(key="anomaly_df", value=df_final.to_json())

    ti.xcom_push(key = "metrics", value={
            "execution_time": execution_time,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })


def save_results(**kwargs):
    ti = kwargs['ti']

    # Recupera i dati elaborati e quelli scartati, prepara i DataFrame finali e la connessione  al DB 
    df_processed  = pd.read_json(ti.xcom_pull(task_ids='anomaly_detection', key="anomaly_df"))
    df_discarded   = pd.read_json(ti.xcom_pull(task_ids='clean_data',        key="discarded_df"))


    metrics = ti.xcom_pull(task_ids= 'anomaly_detection', key='metrics')
    engine = create_engine(DB_CONN)

    columns_base = ['id_sensor', 'day_time', 'parameter_name', 'value',
                    'anomaly', 'anomaly_detected_by_dbscan', 'confidence_score']

    # --- Tabella 1: dati sani ---
    df_clean_out = df_processed[~df_processed['anomaly_detected_by_dbscan']][columns_base].copy()

    # --- Tabella 2: anomalie ---
    df_anomalies = df_processed[
        df_processed['anomaly_detected_by_dbscan'] | df_processed['anomaly']
    ][columns_base].copy()

    # --- Tabella 3: scartati ---
    discard_cols = ['id_sensor', 'day_time', 'parameter_name', 'value', 'discard_reason']
    for col in discard_cols:
        if col not in df_discarded.columns:
            df_discarded[col] = None
    df_discarded_out = df_discarded[discard_cols].copy()

    #Scrittura dei risultati finali nel databse e gestione degli errori 
    try:
        df_clean_out.to_sql(
            'sensor_measurements_clean',
            engine, if_exists='append', index=False
        )
        log.info(f"Salvati {len(df_clean_out)} record sani")

        df_anomalies.to_sql(
            'sensor_measurements_anomalies',
            engine, if_exists='append', index=False
        )
        log.info(f"Salvate {len(df_anomalies)} anomalie")

        df_discarded_out.to_sql(
            'sensor_measurements_discarded',
            engine, if_exists='append', index=False
        )
        log.info(f"Salvati {len(df_discarded_out)} record scartati")

    except Exception as e:
        log.error(f"Errore salvataggio PostgreSQL: {e}")
        raise

    #Salvataggio metriche in PostgreSQL

    if metrics is not None:
        df_metrics = pd.DataFrame([{
            'execution_date': datetime.now(),
            'task_name': 'anomaly_detection',
            'execution_time': metrics['execution_time'],
            'true_positives': metrics['true_positives'],
            'false_positives': metrics['false_positives'],
            'false_negatives': metrics['false_negatives'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }])
    
    try:
        df_metrics.to_sql('metrics_log', engine, if_exists='append', index= False)
        log.info(f"metriche salvate in PostgreSQL: TP={metrics['true_positives']}, F1={metrics['f1_score']:.3f}")
    except Exception as e:
        log.error(f"Errore salvataggio metriche PostgreSQL: {e}")
    

    #Salvataggio metriche in CSV
    csv_path= "/opt/airflow/dags/metrics.csv"

    try:
        if os.path.exists(csv_path):
            df_existing =pd.read_csv(csv_path)
            df_metrics= pd.concat([df_existing, df_metrics], ignore_index = True)
        
        df_metrics.to_csv(csv_path, index= False, sep=";")
        log.info(f"Metriche salvate in CSV: {csv_path}")
    except Exception as e:
        log.error(f"Errore salvataggio metriche in csv: {e}")
    

with DAG(
    dag_id="greenhouse_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task_generate = PythonOperator(
        task_id="generate_data",
        python_callable=generate_data,
    )
    task_load = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
    )
    task_clean = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )
    task_anomaly = PythonOperator(
        task_id="anomaly_detection",
        python_callable=anomaly_detection,
    )
    task_save = PythonOperator(
        task_id="save_results",
        python_callable=save_results,
    )

    task_generate >> task_load >> task_clean >> task_anomaly >> task_save