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
from sklearn.ensemble import IsolationForest

sys.path.insert(0, '/opt/airflow/dags') 

from sensor_simulator import run

log = logging.getLogger(__name__) 

DB_CONN = 'postgresql+psycopg2://airflow:airflow@postgres:5432/greenhouse_db' # Stringa di connessione a PostgreSQL 


def calc_performance_metrics(tp, fp, fn):
    precision = tp / (tp +fp ) if (tp + fp ) >0 else 0.0 

    recall = tp / (tp + fn) if (tp + fp) >0 else 0.0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1 


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
    start_time = time.time()
    ti = kwargs['ti']

    # 1. Recupero i dati puliti da XCom
    data = ti.xcom_pull(task_ids='clean_data', key="clean_df")
    df = pd.read_json(data)
    df['value'] = df['value'].astype(float)

    # 2. Inizializzazione statistiche per il confronto (Ground Truth)
    stats = {
        'dbscan':  {'tp': 0, 'fp': 0, 'fn': 0},
        'iforest': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    parameters = df["parameter_name"].unique()
    result = []

    for param in parameters:
        # Prende solo i dati di quel parametro 
        df_param = df[df["parameter_name"] == param].copy()
        df_param = df_param.dropna(subset=['value'])

        # Evito di applicare algoritmi su dataset troppo piccoli 
        if len(df_param) < 5:
            log.warning(f"Parametro '{param}': soli {len(df_param)} record, skippato")
            continue

        scaler = StandardScaler()
        # Nota: usiamo [[ ]] per evitare i warning di feature names con Isolation Forest
        df_param['value_normalized'] = scaler.fit_transform(df_param[['value']])

        # --- ALGORITMO 1: DBSCAN ---
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        # Il nome della colonna deve corrispondere a quello usato nel DB e nei log
        df_param['anomaly_detected_by_dbscan'] = (dbscan.fit_predict(df_param[['value_normalized']]) == -1)

        # --- ALGORITMO 2: ISOLATION FOREST ---
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        df_param['anomaly_iforest'] = (iso_forest.fit_predict(df_param[['value_normalized']]) == -1)

        # --- 3. CONFRONTO CON GROUND TRUTH (Colonna 'anomaly' del simulatore) ---
        for idx, row in df_param.iterrows():
            real = bool(row['anomaly'])

            # Statistiche DBSCAN (corretto refuso dbascan)
            if real and row['anomaly_detected_by_dbscan']: 
                stats['dbscan']['tp'] += 1
            elif not real and row['anomaly_detected_by_dbscan']: 
                stats['dbscan']['fp'] += 1
            elif real and not row['anomaly_detected_by_dbscan']:
                stats['dbscan']['fn'] += 1
            
            # Statistiche Isolation Forest
            if real and row['anomaly_iforest']:
                stats['iforest']['tp'] += 1
            elif not real and row['anomaly_iforest']:
                stats['iforest']['fp'] += 1
            elif real and not row['anomaly_iforest']:
                stats['iforest']['fn'] += 1 

        # Aggiungiamo i metadati richiesti dal DB (anche se fissi)
        df_param['confidence_score'] = 0.0
        
        # Log dei risultati per questo parametro
        n_db = df_param['anomaly_detected_by_dbscan'].sum()
        n_if = df_param['anomaly_iforest'].sum()
        log.info(f"Parametro '{param}': record={len(df_param)}, DBSCAN={n_db}, I-Forest={n_if}")

        result.append(df_param)

    if not result:
        log.error("Nessun parametro processato — controlla il CSV.")
        raise ValueError("anomaly_detection: nessun dato disponibile dopo il filtraggio")

    # Uniamo tutti i parametri in un unico DataFrame
    df_final = pd.concat(result, ignore_index=True)

    # 4. CALCOLO METRICHE FINALI (usando la funzione esterna calc_performance_metrics)
    p_db, r_db, f1_db = calc_performance_metrics(stats['dbscan']['tp'], stats['dbscan']['fp'], stats['dbscan']['fn'])
    p_if, r_if, f1_if = calc_performance_metrics(stats['iforest']['tp'], stats['iforest']['fp'], stats['iforest']['fn'])

    execution_time = time.time() - start_time

    # Prepariamo il pacchetto metriche da passare a save_results
    metrics_summary = {
        "execution_time": execution_time,
        "dbscan": {
            "tp": stats['dbscan']['tp'], "fp": stats['dbscan']['fp'], "fn": stats['dbscan']['fn'],
            "precision": p_db, "recall": r_db, "f1_score": f1_db
        },
        "iforest": {
            "tp": stats['iforest']['tp'], "fp": stats['iforest']['fp'], "fn": stats['iforest']['fn'],
            "precision": p_if, "recall": r_if, "f1_score": f1_if
        }
    }

    # Passiamo i risultati al task successivo via XCom
    ti.xcom_push(key="anomaly_df", value=df_final.to_json())
    ti.xcom_push(key="metrics", value=metrics_summary)

    log.info(f"Analisi completata in {execution_time:.2f}s. DBSCAN F1: {f1_db:.3f}, I-Forest F1: {f1_if:.3f}")

def save_results(**kwargs):
    ti = kwargs['ti']

    # Recupero dati e metriche da XCom
    df_processed = pd.read_json(ti.xcom_pull(task_ids='anomaly_detection', key="anomaly_df"))
    df_discarded = pd.read_json(ti.xcom_pull(task_ids='clean_data', key="discarded_df"))
    metrics = ti.xcom_pull(task_ids='anomaly_detection', key='metrics')
    
    engine = create_engine(DB_CONN)

    # Colonne base per le tabelle dei sensori
    columns_base = ['id_sensor', 'day_time', 'parameter_name', 'value',
                    'anomaly', 'anomaly_detected_by_dbscan', 'confidence_score']

    # --- Salvataggio Tabelle Sensori ---
    try:
        # Tabella 1 Dati sani 
        df_clean_out = df_processed[~df_processed['anomaly_detected_by_dbscan']][columns_base].copy()
        df_clean_out.to_sql('sensor_measurements_clean', engine, if_exists='append', index=False)

        # Tabella 2 Anomalie
        df_anomalies = df_processed[
            df_processed['anomaly_detected_by_dbscan'] | df_processed['anomaly']
        ][columns_base].copy()
        df_anomalies.to_sql('sensor_measurements_anomalies', engine, if_exists='append', index=False)

        # Tabella 3 Scartati
        discard_cols = ['id_sensor', 'day_time', 'parameter_name', 'value', 'discard_reason']
        df_discarded_out = df_discarded[discard_cols].copy()
        df_discarded_out.to_sql('sensor_measurements_discarded', engine, if_exists='append', index=False)
        
        log.info("Dati dei sensori salvati correttamente nelle 3 tabelle.")
    except Exception as e:
        log.error(f"Errore salvataggio tabelle sensori: {e}")
        raise

    if metrics is not None:
        rows_metrics = []
        for algo_key in ['dbscan', 'iforest']:
            m = metrics[algo_key]
            rows_metrics.append({
                'execution_date': datetime.now(),
                'algorithm_name': 'DBSCAN' if algo_key == 'dbscan' else 'IsolationForest',
                'execution_time': metrics['execution_time'],
                'true_positives': m['tp'],
                'false_positives': m['fp'],
                'false_negatives': m['fn'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1_score': m['f1_score']
            })
        
        df_metrics = pd.DataFrame(rows_metrics)

        # Salvataggio su PostgreSQL
        try:
            df_metrics.to_sql('metrics_log', engine, if_exists='append', index=False)
            log.info("Metriche comparative salvate in PostgreSQL (2 righe).")
        except Exception as e:
            log.error(f"Errore salvataggio metriche PostgreSQL: {e}")

        # Salvataggio su CSV 
        csv_path = "/opt/airflow/dags/metrics.csv"
        try:
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path, sep=";")
                df_metrics = pd.concat([df_existing, df_metrics], ignore_index=True)
            df_metrics.to_csv(csv_path, index=False, sep=";")
        except Exception as e:
            log.error(f"Errore salvataggio CSV metriche: {e}")
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