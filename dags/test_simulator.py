# test_simulator.py
from sensor_simulator import run
import pandas as pd
import os

# Test 1: Comportamento di default
print("Test 1: Default (5% outlier, 2% null)")
run(output_path="test_output1.csv")
df = pd.read_csv("test_output1.csv", sep=";")
print(f"Totale righe: {len(df)}")
print(f"Anomalie: {(df['anomaly']=='True').sum()}")
print()

# Test 2: Con più outlier (15%)
print("Test 2: Con 15% outlier")
run(outlier_rate=0.15, output_path="test_output2.csv")
df = pd.read_csv("test_output2.csv", sep=";")
print(f"Totale righe: {len(df)}")
anomalie = (df['anomaly']=='True').sum()
pct = (anomalie / len(df)) * 100
print(f"Anomalie: {anomalie} ({pct:.1f}%)")
print()

# Test 3: Con doppi dati
print("Test 3: Con doppi dati (scale_factor=2.0)")
run(scale_factor=2.0, output_path="test_output3.csv")
df = pd.read_csv("test_output3.csv", sep=";")
print(f"Totale righe: {len(df)}")
print()

# Test 4: Combo
print("Test 4: 20% outlier + 2x dati")
run(outlier_rate=0.20, scale_factor=2.0, output_path="test_output4.csv")
df = pd.read_csv("test_output4.csv", sep=";")
print(f"Totale righe: {len(df)}")
anomalie = (df['anomaly']=='True').sum()
pct = (anomalie / len(df)) * 100
print(f"Anomalie: {anomalie} ({pct:.1f}%)")
print()

print("Tutti i test completati!")

# Pulisci i file di test
for f in ["test_output1.csv", "test_output2.csv", "test_output3.csv", "test_output4.csv"]:
    if os.path.exists(f):
        os.remove(f)