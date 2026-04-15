# Greenhouse Airflow Pipeline - Istruzioni di Avvio

Questa pipeline utilizza Apache Airflow (via Docker Compose) per simulare, processare e salvare dati di sensori di un sistema di serre a temperatura controllata.

## Avvio
1. Clona o scarica questa cartella sul tuo PC.
2. Da terminale, posizionarsi nella cartella `airflow` e lanciare:
   ```powershell
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


- I dati vengono salvati su Postgres in tre tabelle: `sensor_measurements_clean`, `sensor_measurements_anomalies`, `sensor_measurements_discarded`.
