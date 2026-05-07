from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from sqlalchemy import create_engine, text
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd 
import numpy as np
import logging
import time 
import sys
import os
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc

sys.path.insert(0, '/opt/airflow/dags') 

from sensor_simulator import run

log = logging.getLogger(__name__) 

DB_CONN = 'postgresql+psycopg2://airflow:airflow@postgres:5432/greenhouse_db' # Stringa di connessione a PostgreSQL 


def calc_performance_metrics(tp, fp, fn):
    #Calcola metriche di performance da una confusion matrix.
    
   # Dato un True Positive, False Positive e False Negative, calcola:
   #    - Precision: tp / (tp + fp)
   #    - Recall: tp / (tp + fn)
   #    - F1-Score: media armonica di precision e recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0 
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def plot_confusion_matrix_heatmap(y_true, y_pred_dbscan, y_pred_iforest, output_dir):
   
    # Crea Confusion Matrix heatmap side-by-side per DBSCAN e Isolation Forest.
   
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    algorithms = ['DBSCAN', 'Isolation Forest']
    predictions = [y_pred_dbscan, y_pred_iforest]
    
    for idx, (ax, algo, y_pred) in enumerate(zip(axes, algorithms, predictions)):
        # Calcola confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[False, True])
        tn, fp, fn, tp = cm.ravel()
        
        # Calcola metriche
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Crea heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=False, annot_kws={'size': 14, 'weight': 'bold'},
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=11, fontweight='bold')
        ax.set_title(f'{algo}\nP={precision:.3f} | R={recall:.3f} | F1={f1:.3f}',
                    fontsize=12, fontweight='bold', pad=10)
    
    plt.suptitle('Confusion Matrix Comparison: DBSCAN vs Isolation Forest',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    log.info("✓ Saved: 01_confusion_matrix.png")
 
 
def plot_algorithm_comparison_table(metrics_dbscan, metrics_iforest, output_dir):
    
    #Crea una tabella che confronta i due algoritmi, mostrando tutte le metriche 
    
    comparison_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'True Positives', 
                   'False Positives', 'False Negatives', 'Execution Time (ms)'],
        'DBSCAN': [
            f"{metrics_dbscan['precision']:.4f}",
            f"{metrics_dbscan['recall']:.4f}",
            f"{metrics_dbscan['f1_score']:.4f}",
            metrics_dbscan['tp'],
            metrics_dbscan['fp'],
            metrics_dbscan['fn'],
            f"{metrics_dbscan.get('execution_time', 0) * 1000:.2f}"
        ],
        'Isolation Forest': [
            f"{metrics_iforest['precision']:.4f}",
            f"{metrics_iforest['recall']:.4f}",
            f"{metrics_iforest['f1_score']:.4f}",
            metrics_iforest['tp'],
            metrics_iforest['fp'],
            metrics_iforest['fn'],
            f"{metrics_iforest.get('execution_time', 0) * 1000:.2f}"
        ],
        'Winner': []
    }
    
    # Determina vincitori
    for i in range(len(comparison_data['DBSCAN'])):
        if i < 3: 
            val_db = float(comparison_data['DBSCAN'][i])
            val_if = float(comparison_data['Isolation Forest'][i])
            
            # Viene considerato un 'pareggio' se differenza <0.01
            if abs(val_db - val_if) < 0.01:
                comparison_data['Winner'].append('Tie')
            elif val_db > val_if:
                comparison_data['Winner'].append('DBSCAN ✓')
            else:
                comparison_data['Winner'].append('I-Forest ✓')
        else:
            comparison_data['Winner'].append('')
    
    df_comparison = pd.DataFrame(comparison_data)
    
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_comparison.values,
                    colLabels=df_comparison.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.20, 0.25, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(len(df_comparison.columns)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    for i in range(1, len(df_comparison) + 1):
        for j in range(len(df_comparison.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            
        
            if j == 3 and '✓' in str(table[(i, j)].get_text().get_text()):
                table[(i, j)].set_facecolor('#d5f4e6')
                table[(i, j)].set_text_props(weight='bold', color='#27ae60')
    
    plt.title('Algorithm Performance Comparison',
             fontweight='bold', fontsize=14, pad=20)
    plt.savefig(f'{output_dir}/02_algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Salvataggio anche in CSV
    df_comparison.to_csv(f'{output_dir}/algorithm_comparison.csv', index=False, sep=';')
    log.info("✓ Saved: 02_algorithm_comparison.png")
 
 
def plot_anomaly_scatter_by_parameter(df, parameter, output_dir):

    # Crea scatter plot mostrando le anomalie rilevate da DBSCAN e Isolation Forest.
    # Codifica colori per ogni punto:
    #   - Verde (TP): Anomalia rilevata correttamente
    #   - Rosso (FP): Falso positivo
    #   - Blu (TN): Valore normale correttamente identificato
    #   - Arancione (FN): Anomalia non rilevata
    df_param = df[df['parameter_name'] == parameter].copy().reset_index(drop=True)
    
    if len(df_param) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    def get_color_map(y_true, y_pred):
        return np.select(
            condlist=[
                (y_true == 1) & (y_pred == 1), # TP
                (y_true == 0) & (y_pred == 1), # FP
                (y_true == 0) & (y_pred == 0), # TN
                (y_true == 1) & (y_pred == 0)  # FN
            ],
            choicelist=['#27ae60', '#e74c3c', '#3498db', '#f39c12'],
            default='#7f8c8d'
        )

    #  DBSCAN
    colors_db = get_color_map(df_param['anomaly'], df_param['anomaly_detected_by_dbscan'])
    ax1.scatter(df_param.index, df_param['value'], c=colors_db, s=60, alpha=0.6, edgecolors='none')
    
    # ISOLATION FOREST
    colors_if = get_color_map(df_param['anomaly'], df_param['anomaly_iforest'])
    ax2.scatter(df_param.index, df_param['value'], c=colors_if, s=60, alpha=0.6, edgecolors='none')

    ax1.set_title(f'DBSCAN - {parameter}', fontweight='bold')
    ax2.set_title(f'Isolation Forest - {parameter}', fontweight='bold')
    
    plt.savefig(f'{output_dir}/03_scatter_{parameter}.png', dpi=150) # DPI 150 è sufficiente e più leggero
    plt.close()
 
def plot_precision_recall_curves(y_true, y_pred_dbscan, y_pred_iforest, output_dir):

    # Crea curve Precision-Recall per confrontare le performance dei due algoritmi.
    # La curva Precision-Recall è utile per dataset sbilanciati (poche anomalie).
    # L'area sotto la curva (AUC) rappresenta la performance media.


    fig, ax = plt.subplots(figsize=(10, 7))
    
    # DBSCAN
    precision_db, recall_db, _ = precision_recall_curve(y_true, y_pred_dbscan.astype(float))
    auc_db = auc(recall_db, precision_db)
    ax.plot(recall_db, precision_db, 'b-', linewidth=5, 
           label=f'DBSCAN (AUC={auc_db:.3f})')
    
    # Isolation Forest
    precision_if, recall_if, _ = precision_recall_curve(y_true, y_pred_iforest.astype(float))
    auc_if = auc(recall_if, precision_if)
    ax.plot(recall_if, precision_if, 'r-', linewidth=2,
           label=f'Isolation Forest (AUC={auc_if:.3f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_precision_recall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
 
 
def plot_parameter_distributions(df, output_dir):
    
    #Crea distribuzione per ogni parametro evidenziando la media .
    
    parameters = df['parameter_name'].unique()
    n_params = len(parameters)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, param in enumerate(sorted(parameters)):
        ax = axes[idx]
        df_param = df[df['parameter_name'] == param]['value'].dropna()
        
        ax.hist(df_param, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_title(f'Distribution - {param}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
       
        mean_val = df_param.mean()
        std_val = df_param.std()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=9)
    
    for idx in range(n_params, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Parameter Distributions', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info("✓ Saved: 05_distributions.png")

def generate_data():
    # Task 1 --> Generazione dataset simulato di sensori tramite la chiamata a sensor_simulator.run() con scale_factor=3.0 
    # per generare (circa)19000 righe di dati (3 giorni * 5 parametri * 2 serre)
    run(scale_factor = 3.0) 


def load_data():

    # Task 2 -->legge il CSV generato dal simulatore e ritorna il DataFrame in formato JSON
    # così che Airflow può passarlo alla task successiva
    df = pd.read_csv("/opt/airflow/data/sensori.csv", sep=";")
    log.info(f"Caricati {len(df)} record dal CSV")
    return df.to_json() 

    # NB: viene utilizzato '**kwargs' perchè la funzione può ricevere un numero variabile di argomenti
def clean_data(**kwargs):

    # Task 3 --> Pulisce i dati rimuovendo valori nulli o invalidi.
    # Recupera DataFrame da Xcom, identifica righe con valori nulli o timestamp mancanti e separa i dati in due insiemi  (df_clean e df_discarded)
    # Ritorna 'clean_df' (DataFrame pulito ) e 'discarded_df' (DataFrame degli scarti)
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

    # Conversione in float, se un valore non è convertibile lo trasforma in NaN  e identifica valori non numerici 
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

    ti.xcom_push(key="clean_df",     value=df_clean.to_json())
    ti.xcom_push(key="discarded_df", value=df_discarded.to_json())




def anomaly_detection(eps= 0.1, min_samples= 2, contamination= 0.05, **kwargs):

    # Task 4 --> Applica due algoritmi di anomaly detection e confronta i risultati. 

    # Algoritmi : 
    #   1- DBSCAN --> Density-based clustering 
    #   2- Isolation Forest --> Ensemble method

    # Per ogni parametro normalizza i valori con StandardScaler, applica entrambi gli algoritmi,
    # confronta con gound truth e calcola metriche (TP, FP, FN, Precision, Recall, F1)
    
    # Ritorna : anomaly_df --> DataFrame con predizioni di entrambi gli algoritmi  e metrics --> dizionario con metriche DBSCAN e IsoF
    start_time = time.time()
    ti = kwargs['ti']

    # Recupero i dati puliti da XCom
    data = ti.xcom_pull(task_ids='clean_data', key="clean_df")
    df = pd.read_json(data)
    df['value'] = df['value'].astype(float)

    # Inizializzazione statistiche per il confronto (Ground Truth)
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
        df_param['value_normalized'] = scaler.fit_transform(df_param[['value']])

        # --- ALGORITMO 1: DBSCAN ---
        dbscan = DBSCAN(eps=eps, min_samples= min_samples)
        df_param['anomaly_detected_by_dbscan'] = (dbscan.fit_predict(df_param[['value_normalized']]) == -1)

        # --- ALGORITMO 2: ISOLATION FOREST ---
        iso_forest = IsolationForest(contamination=contamination , random_state=42)
        df_param['anomaly_iforest'] = (iso_forest.fit_predict(df_param[['value_normalized']]) == -1)

        # ---  CONFRONTO CON GROUND TRUTH ---
        for idx, row in df_param.iterrows():
            real = bool(row['anomaly'])

            if real and row['anomaly_detected_by_dbscan']: 
                stats['dbscan']['tp'] += 1
            elif not real and row['anomaly_detected_by_dbscan']: 
                stats['dbscan']['fp'] += 1
            elif real and not row['anomaly_detected_by_dbscan']:
                stats['dbscan']['fn'] += 1
            
            if real and row['anomaly_iforest']:
                stats['iforest']['tp'] += 1
            elif not real and row['anomaly_iforest']:
                stats['iforest']['fp'] += 1
            elif real and not row['anomaly_iforest']:
                stats['iforest']['fn'] += 1 

        df_param['confidence_score'] = 0.0
        
        n_db = df_param['anomaly_detected_by_dbscan'].sum()
        n_if = df_param['anomaly_iforest'].sum()
        log.info(f"Parametro '{param}': record={len(df_param)}, DBSCAN={n_db}, I-Forest={n_if}")

        result.append(df_param)

    if not result:
        log.error("Nessun parametro processato — controlla il CSV.")
        raise ValueError("anomaly_detection: nessun dato disponibile dopo il filtraggio")

    df_final = pd.concat(result, ignore_index=True)

    # CALCOLO METRICHE FINALI (usando la funzione esterna calc_performance_metrics)
    p_db, r_db, f1_db = calc_performance_metrics(stats['dbscan']['tp'], stats['dbscan']['fp'], stats['dbscan']['fn'])
    p_if, r_if, f1_if = calc_performance_metrics(stats['iforest']['tp'], stats['iforest']['fp'], stats['iforest']['fn'])

    execution_time = time.time() - start_time

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

    ti.xcom_push(key="anomaly_df", value=df_final.to_json())
    ti.xcom_push(key="metrics", value=metrics_summary)

    log.info(f"Analisi completata in {execution_time:.2f}s. DBSCAN F1: {f1_db:.3f}, I-Forest F1: {f1_if:.3f}")

def save_results(**kwargs):

    # Task 5 --> salva i risultati dell' anomaly detection nelle tabelle PostgreSQL e le metriche di performance in 'metrics_log'
    # 3 tabelle :
    #   1- sensor_measurements_clean 
    #   2- sensor_measurements_anomalies
    #   3- sensor_measurements_discarded
    ti = kwargs['ti']

    df_processed = pd.read_json(ti.xcom_pull(task_ids='anomaly_detection', key="anomaly_df"))
    df_discarded = pd.read_json(ti.xcom_pull(task_ids='clean_data', key="discarded_df"))
    metrics = ti.xcom_pull(task_ids='anomaly_detection', key='metrics')
    
    engine = create_engine(DB_CONN)

    columns_base = ['id_sensor', 'day_time', 'parameter_name', 'value',
                    'anomaly', 'anomaly_detected_by_dbscan', 'confidence_score']

    
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
        csv_path = "/opt/airflow/data/metrics.csv"
        try:
            if os.path.exists(csv_path):
                df_existing = pd.read_csv(csv_path, sep=";")
                df_metrics = pd.concat([df_existing, df_metrics], ignore_index=True)
            df_metrics.to_csv(csv_path, index=False, sep=";")
        except Exception as e:
            log.error(f"Errore salvataggio CSV metriche: {e}")


def generate_report(**kwargs):

    # Task 6 --> genera report visuale con grafici comparativi e analitici
    # Crea 5 visualizzazioni e vengono salvate in /opt/airflow/reports come PNG:
    #   1. Confusion Matrix (side-by-side DBSCAN vs Isolation Forest)
    #   2. Tabella comparativa delle metriche
    #   3. Scatter plots per parametro (mostrando TP/FP/TN/FN)
    #   4. Precision-Recall curves
    #   5. Istogrammi di distribuzione dei parametri
    start_time = time.time()
    output_dir = '/opt/airflow/reports'
    os.makedirs(output_dir, exist_ok=True)
    
    ti = kwargs['ti']
    
    # ===== 1. Recupera dati da XCom =====
    log.info("Retrieving data from previous tasks...")
    
    try:
        metrics_summary = ti.xcom_pull(task_ids='anomaly_detection', key='metrics')
        log.info(f"Metrics retrieved: DBSCAN F1={metrics_summary['dbscan']['f1_score']:.4f}, "
                f"I-Forest F1={metrics_summary['iforest']['f1_score']:.4f}")
    except Exception as e:
        log.error(f"Failed to retrieve metrics: {e}")
        raise
    
    # ===== 2. Leggi dati dal database =====
    log.info("Reading data from PostgreSQL...")
    engine = create_engine(DB_CONN)
    
    try:
        df = pd.read_sql(text("""
            SELECT 
                cm.id_sensor,
                cm.day_time,
                cm.parameter_name,
                cm.value,
                cm.anomaly,
                cm.anomaly_detected_by_dbscan,
                COALESCE(cm.anomaly_detected_by_dbscan, FALSE) as anomaly_iforest
            FROM sensor_measurements_clean cm
            ORDER BY cm.day_time
        """), engine)
        
        log.info(f"Loaded {len(df)} records from database")
    except Exception as e:
        log.error(f"Failed to read from database: {e}")
        raise
    
    if df.empty:
        log.error("No data available in database")
        raise ValueError("No data in sensor_measurements_clean")
    
    # ===== 3. Genera GRAFICI ESSENZIALI =====
    
    
    # GRAFICO 1: Confusion Matrix
    log.info("[1/5] Generating Confusion Matrix...")
    try:
        plot_confusion_matrix_heatmap(
            y_true=df['anomaly'].astype(bool),
            y_pred_dbscan=df['anomaly_detected_by_dbscan'].fillna(False).astype(bool),
            y_pred_iforest=df['anomaly_iforest'].fillna(False).astype(bool),
            output_dir=output_dir
        )
    except Exception as e:
        log.error(f"Failed to generate confusion matrix: {e}")
    
    # GRAFICO 2: Algorithm Comparison Table
    log.info("[2/5] Generating Algorithm Comparison Table...")
    try:
        plot_algorithm_comparison_table(
            metrics_dbscan=metrics_summary['dbscan'],
            metrics_iforest=metrics_summary['iforest'],
            output_dir=output_dir
        )
    except Exception as e:
        log.error(f"Failed to generate comparison table: {e}")
    
    # GRAFICO 3: Scatter Plots per Parametro
    log.info("[3/5] Generating Scatter Plots by Parameter...")
    try:
        for param in sorted(df['parameter_name'].unique()):
            df_param = df[df['parameter_name'] == param]
            
            # Campiona se il dataset è molto grande (evita grafici troppo pesanti)
            if len(df_param) > 1000:
                df_to_plot = df_param.sample(n=1000, random_state=42).sort_index()
                log.info(f"   -> {param}: Sampling 1000/{len(df_param)} points")
            else:
                df_to_plot = df_param.sort_index()
                log.info(f"   -> {param}: Using all {len(df_param)} points")
            
            plot_anomaly_scatter_by_parameter(df_to_plot, param, output_dir)
            
    except Exception as e:
        log.error(f"Failed to generate scatter plots: {e}")
    
    log.info("[4/5] Generating Precision-Recall Curves...")
    try:
        plot_precision_recall_curves(
            y_true=df['anomaly'].astype(bool),
            y_pred_dbscan=df['anomaly_detected_by_dbscan'].fillna(False).astype(bool),
            y_pred_iforest=df['anomaly_iforest'].fillna(False).astype(bool),
            output_dir=output_dir
        )
    except Exception as e:
        log.error(f"Failed to generate PR curves: {e}")
    
    log.info("[5/5] Generating Parameter Distributions...")
    try:
        plot_parameter_distributions(df, output_dir)
    except Exception as e:
        log.error(f"Failed to generate distributions: {e}")
    
    #  Salvataggio metriche nel database 
    log.info("=" * 60)
    log.info("SAVING METRICS TO DATABASE")
    log.info("=" * 60)
    
    try:
        insert_query_dbscan = text("""
            INSERT INTO metrics_log 
            (algorithm_name, execution_time, true_positives, false_positives, 
             false_negatives, precision, recall, f1_score)
            VALUES ('DBSCAN', :exec_time, :tp, :fp, :fn, :precision, :recall, :f1)
        """)
        
        with engine.connect() as conn:
            conn.execute(insert_query_dbscan, {
                'exec_time': metrics_summary['execution_time'],
                'tp': metrics_summary['dbscan']['tp'],
                'fp': metrics_summary['dbscan']['fp'],
                'fn': metrics_summary['dbscan']['fn'],
                'precision': metrics_summary['dbscan']['precision'],
                'recall': metrics_summary['dbscan']['recall'],
                'f1': metrics_summary['dbscan']['f1_score']
            })
            conn.commit()
        insert_query_iforest = text("""
            INSERT INTO metrics_log 
            (algorithm_name, execution_time, true_positives, false_positives, 
             false_negatives, precision, recall, f1_score)
            VALUES ('Isolation Forest', :exec_time, :tp, :fp, :fn, :precision, :recall, :f1)
        """)
        
        with engine.connect() as conn:
            conn.execute(insert_query_iforest, {
                'exec_time': metrics_summary['execution_time'],
                'tp': metrics_summary['iforest']['tp'],
                'fp': metrics_summary['iforest']['fp'],
                'fn': metrics_summary['iforest']['fn'],
                'precision': metrics_summary['iforest']['precision'],
                'recall': metrics_summary['iforest']['recall'],
                'f1': metrics_summary['iforest']['f1_score']
            })
            conn.commit()
        
        log.info("✓ Metrics saved to database")
    except Exception as e:
        log.error(f"Failed to save metrics: {e}")
    
  


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
    task_report = PythonOperator(
        task_id="generate_report",
        python_callable= generate_report
    )
    task_generate >> task_load >> task_clean >> task_anomaly >> task_save >> task_report