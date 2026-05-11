# Greenhouse Airflow Pipeline - Istruzioni di Avvio

Questa pipeline utilizza Apache Airflow (via Docker Compose) per simulare, processare e salvare dati di sensori di un sistema di serre a temperatura controllata.

## Avvio
1. Clona o scarica questa cartella sul tuo PC.
2. Da terminale, posizionarsi nella cartella `airflow`, configurare le variabili di ambiente e avviare i servizi con Docker Compose:
   ```bash
   cp .env.example .env
   docker compose up --build
   ```

## Airflow
- URL: http://localhost:8080
- Username: `airflow`
- Password: `airflow`

## Configurazioni e credenziali
- Tutte le credenziali sono già impostate nei file:
  - `docker-compose.yaml` (Postgres, Redis, Airflow)
  - `.env` (variabili ambiente Airflow)
- Database Postgres: user `airflow`, password `airflow`, db `airflow` (default) + `greenhouse_db` (creato da `init-db.sql`)

## File importanti
- `dags/greenhouse_pipeline.py`: DAG principale
- `dags/sensor_simulator.py`: Simulatore dati sensori
- `dags/sensori.csv`: File dati generato
- `init-db.sql`: Crea tabelle su Postgres
- `requirements.txt`: Dipendenze di Python 

## Flusso di elaborazione

1. **Generazione dati** : ~19.000 record di sensori simulati (3 giorni)
2. **Caricamento CSV** : legge i dati in memoria
3. **Pulizia dati** : rimuove valori nulli/invalidi
4. **Rilevazione anomalie** : applica DBSCAN e Isolation Forest
5. **Salvataggio i risultati** : popola le tabelle di PostgreSQL
6. **Generazione report** : crea 5 grafici di analisi 

## Scelta degli algoritmi

### DBSCAN (Density-Based Spatial Clustering)
  Algoritmo eccellente per anomalie di densità, sensibile ad anomalie locali con parametro eps controllabile

### Isolation Forest(Ensemble Method)
algoritmo ottimo per anomalie globali, scalabile a dataset grandi e non richiede una definizione di distanza

### ObiettivoConfrontare 
Due approcci diversi (ma complementari) per validare la robustezza delle anomalie rilevate. 
Se entrambi concordano su un'anomalia, è **probabilmente vera**. Se discordano, è **borderline** e quindi richiede una **verifica manuale**. 


## Risultati e Output

 **Dati in PostgreSQL**
 - `sensor_measurements_clean` : dati validi 
 - `sensor_measurements_anomalies` : anomalie rilevate
 - `sensor_measurements_discarded` : dati scartati
 - `metrics_log` : metriche di performance
 
 **Grafici per il report**
- `01_confusion_matrix.png` : Confronto DBSCAN vs Isolation Forest
- `02_algorithm_comparison.png` : Tabella metriche comparative
- `03_scatter_*.png` : Scatter plot per parametro (temperature, humidity, etc.)
- `04_precision_recall.png` : Curve Precision-Recall
- `05_distributions.png` : Istogrammi distribuzione parametri

