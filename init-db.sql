-- Database greenhouse
CREATE DATABASE greenhouse_db;

\c greenhouse_db

-- Tabella originale mantenuta per compatibilità
CREATE TABLE sensor_measurements (
    id SERIAL PRIMARY KEY,
    id_sensor VARCHAR(50) NOT NULL,
    day_time TIMESTAMP NOT NULL,
    parameter_name VARCHAR(50) NOT NULL,
    value FLOAT,
    anomaly BOOLEAN,
    anomaly_detected_by_dbscan BOOLEAN,
    confidence_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabella 1: dati sani
-- Contiene i record che hanno superato il cleaning e non sono stati
-- marcati come anomalia da DBSCAN né dal simulatore
CREATE TABLE sensor_measurements_clean (
    id SERIAL PRIMARY KEY,
    id_sensor VARCHAR(50) NOT NULL,
    day_time TIMESTAMP NOT NULL,
    parameter_name VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    anomaly BOOLEAN DEFAULT FALSE,
    anomaly_detected_by_dbscan BOOLEAN DEFAULT FALSE,
    confidence_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabella 2: anomalie
-- Contiene i record marcati come anomalia da DBSCAN oppure già
-- flaggati come anomali dal simulatore (outlier intenzionali)
CREATE TABLE sensor_measurements_anomalies (
    id SERIAL PRIMARY KEY,
    id_sensor VARCHAR(50) NOT NULL,
    day_time TIMESTAMP NOT NULL,
    parameter_name VARCHAR(50) NOT NULL,
    value FLOAT,
    anomaly BOOLEAN DEFAULT TRUE,
    anomaly_detected_by_dbscan BOOLEAN DEFAULT TRUE,
    confidence_score FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabella 3: record scartati
-- Contiene i record eliminati durante il cleaning per campi mancanti
-- o non validi. La colonna discard_reason spiega il motivo dello scarto.
-- Questi dati non passano mai ad anomaly_detection.
CREATE TABLE sensor_measurements_discarded (
    id SERIAL PRIMARY KEY,
    id_sensor VARCHAR(50),           -- può essere NULL: era il campo mancante
    day_time TIMESTAMP,              -- può essere NULL: era il campo mancante
    parameter_name VARCHAR(50),
    value VARCHAR(50),               -- VARCHAR per conservare il valore originale
                                     -- anche se non era convertibile a float
    discard_reason VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indici per query frequenti per sensore e per finestra temporale
CREATE INDEX idx_sensor_time       ON sensor_measurements(id_sensor, day_time);
CREATE INDEX idx_clean_sensor_time ON sensor_measurements_clean(id_sensor, day_time);
CREATE INDEX idx_anom_sensor_time  ON sensor_measurements_anomalies(id_sensor, day_time);
CREATE INDEX idx_disc_sensor_time  ON sensor_measurements_discarded(id_sensor, day_time);