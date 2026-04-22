import datetime
import random
import pandas as pd

# Il seguente programma consente di simulare dei sensori (di un sistema di serre a temperatura controllata), 
# per la rilevazione di valori (anche anomali o nulli) per poi salvarli in formato tabellare in un file csv

start = datetime.datetime(2026, 1, 1, 23, 0, 0)  # timestamp iniziale

# Di seguito troviamo i vari range attribuiti ai sensori 
# Range per il GIORNO (6:00 - 22:00)
ranges_day = {
    "temperature": (18, 26.9),
    "humidity": (60, 70),
    "CO2_level": (600, 1200),
    "ventilation": (0.5, 1),
    "ph_level": (6, 7)
}

# Range per la NOTTE (22:00 - 6:00)
ranges_night = {
    "temperature": (10, 16),
    "humidity": (65, 75),
    "CO2_level": (700, 1100),
    "ventilation": (0.2, 0.6),
    "ph_level": (6.2, 6.8)
}

# Range ALTO per il GIORNO
ranges_high_day = {
    "temperature": (27, 33),
    "humidity": (70.1, 80),
    "CO2_level": (1201, 1300),
    "ventilation": (1.1, 2),
    "ph_level": (7.1, 10)
}

# Range ALTO per la NOTTE
ranges_high_night = {
    "temperature": (17, 22),
    "humidity": (76, 85),
    "CO2_level": (1100, 1250),
    "ventilation": (0.7, 1.2),
    "ph_level": (7.0, 7.5)
}

# Range BASSO per il GIORNO
ranges_low_day = {
    "temperature": (-2, 14.9),
    "humidity": (0, 59.9),
    "CO2_level": (200, 599),
    "ventilation": (0, 0.49),
    "ph_level": (2, 5.9)
}

# Range BASSO per la NOTTE
ranges_low_night = {
    "temperature": (5, 9.9),
    "humidity": (40, 64.9),
    "CO2_level": (300, 699),
    "ventilation": (0, 0.19),
    "ph_level": (3, 6.0)
}

def random_mis(min_val, max_val):
    
    #Genera un parametro random nel range preso come parametro iniziale e lo arrotonda di tre cifre dopo la virgola 
    
    result = random.uniform(min_val, max_val)
    return round(result, 3)

def generate_meas(sensor, p, start):
    # NB: Un caso particolare nei sensori è il livello di CO2, che permette di avere solo risultati interi, 
    #   in quanto la misurazione viene fatta in ppm (Parti per milione).
    
    # La funzione resituirà un dizionario con una misurazione, ogni misurazione sarà composta dall'id del sensore, l'ora e la data della misurazione,
    #  il nome del parametro che va a verificare il valore del parametro e anomaly impostato a False perché viene passato ranges[p] 
    # (ovvero i range con le misurazioni ottimali). L'asterisco prima di ranges[p] permette di dividere i valori, in due valori separati

    if p == "CO2_level":
        measurement = {
        "id_sensor": sensor,
        "day_time": start,
        "parameter_name": p,
        "value" : int(random_mis(*ranges[p])),
        "anomaly": False
    }
    else:
        measurement = {
        "id_sensor": sensor,
        "day_time": start,
        "parameter_name": p,
        "value" : random_mis(*ranges[p]),
        "anomaly": False
        }
    return measurement


def outlier_meas(sensor, p, start, outlier_high_prob=0.5):

    # Genera valori outlier con una probabilità del 50% per valori sopra al range ottimale e una probabilità del 50% 
    # per valori sotto al range ottimale. Il caso del livello di CO2 viene separato dagli altri con la stessa logica della funzione precedente.

    # Anomaly è impostato a true in quanto si è verificato un 'errore' da parte del sensore
    if random.random()< outlier_high_prob:

        if p == "CO2_level":
            measurement = {
            "id_sensor": sensor,
            "day_time": start,
            "parameter_name": p,
            "value" : int(random_mis(*ranges_high[p])),
            "anomaly": True
        }
        else:
            measurement = {
            "id_sensor": sensor,
            "day_time": start,
            "parameter_name": p,
            "value" : random_mis(*ranges_high[p]),
            "anomaly": True
            }
    else:
        if p == "CO2_level":
            measurement = {
            "id_sensor": sensor,
            "day_time": start,
            "parameter_name": p,
            "value" : int(random_mis(*ranges_low[p])),
            "anomaly": True
        }
        else:
            measurement = {
            "id_sensor": sensor,
            "day_time": start,
            "parameter_name": p,
            "value" : random_mis(*ranges_low[p]),
            "anomaly": True
            }
    return measurement

def null_meas(sensor, p, start):

    # Crea valori nulli  e restituisce un dizionario con 'anomaly' impostato a True, in quanto il valore restituito è nullo  
    measurement = {
    "id_sensor": sensor,
    "day_time": start,
    "parameter_name": p,
    "value" : None ,
    "anomaly": True
    }
    return measurement

# Definizione di due prototipi di serra, per ogni sensore sono state stabilite le frequenze di calcolo dei valori (in secondi)
# e quante volte ripetere la misurazione, in modo da avere più varietà nel date_time e essere più vicini ad un possibile scenario reale

# temperature, CO2_level, ventilation --> ogni 15 secondi 
# humidity --> ogni 30 secondi
# ph_level --> 1 volta ogni 12 ore ( ogni 43200 secondi)
greenhouses = {
    "first_greenhouse" : {
        "S0001": ["temperature", 15,2880],
        "S0003": ["humidity", 30, 1440],
        "S0004": ["CO2_level", 15, 2880],
        "S0005": ["ventilation", 15, 2880],
        "S0006": ["ph_level", 43200, 2]
    },

    "second_greenhouse" : {
        "S0007": ["temperature", 15, 2880],
        "S0009": ["humidity", 30, 1440],
        "S0010": ["CO2_level", 15, 2880],
        "S0011": ["ventilation", 15, 2880],
        "S0012": ["ph_level", 43200, 2]
    }

}

# Creazione di un dizionario che imposta uno start_time uguale per tutti i sensori, che poi 
# verrà aggiornato man mano che verranno registrate le misurazioni del singolo sensore 
def run(outlier_rate=0.05, null_rate=0.02, outlier_high_prob=0.5, scale_factor=1.0, output_path="sensori.csv"): 
    timestamps= {}

    for greenhouse in greenhouses.values():
        for id_sensor in greenhouse:
            timestamps[id_sensor] = start

    misurazioni = [] # Lista vuota che servià da contenitore per raccogliere tutte le registrazioni e salvare nel file csv

    #Ciclo per la genrazione delle misurazioni

    # Per ogni serra, per ogni sensore, si generano n_misurazioni definite prima


    for key_val, value_val in greenhouses.items():
        for id_sensor, [parametro, intervallo, n_misurazioni] in value_val.items():
            n_misurazioni_scaled= int(n_misurazioni* scale_factor)
            for i in range(n_misurazioni_scaled):
                rand_val= random.random() # Unica estrazione 
                
                if rand_val < outlier_rate:
                    misurazioni.append(outlier_meas(id_sensor, parametro, timestamps[id_sensor], outlier_high_prob))

                elif rand_val < (outlier_rate + null_rate):
                    misurazioni.append(null_meas(id_sensor, parametro, timestamps[id_sensor]))
                else:
                    misurazioni.append(generate_meas(id_sensor, parametro, timestamps[id_sensor]))

                # Aggiorna il timestamps del sensore che ha appena effettuato la misurazione
                timestamps[id_sensor] = timestamps[id_sensor] + datetime.timedelta(seconds= intervallo)
                
    # I valori nulli vengono sostituiti da 'null'
    for riga  in misurazioni:
        for chiave, valore in riga.items():
            if valore is None:
                riga[chiave] = "null"

    # Salva misurazioni in un file csv in ordine cronologico e separando con una ',' i valori.
    # Ogni riga conterrà una misurazione di un sensore 
    df = pd.DataFrame(misurazioni).sort_values(by= ("day_time"))
    df.to_csv(output_path, index=False, sep=";")
    pass