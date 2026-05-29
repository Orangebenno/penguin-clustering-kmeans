# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")

# 2. Preprocessing
# One-Hot-Encoding für kategoriale Variablen
penguins_df1 = pd.get_dummies(penguins_df, columns=['sex'], drop_first=True, dtype=int)

# Standardisieren/Skalieren der Werte
scaler = StandardScaler()
scaler.set_output(transform='pandas')
penguins_df1 = scaler.fit_transform(penguins_df1)

# 3. Ermittlung der optimalen Cluster-Anzahl (Elbow-Methode)
inertia_values = []
k_range = range(2, 10)

for k in k_range:
    kmean = KMeans(n_clusters=k, random_state=42) # random_state macht Ergebnisse reproduzierbar
    kmean.fit(penguins_df1)
    # sklearn nimmt uns die mathematische Arbeit ab:
    inertia_values.append(kmean.inertia_)

# Plot der Elbow-Methode zur Veranschaulichung
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia_values, marker='o')
plt.title('Elbow-Methode zur Bestimmung von K')
plt.xlabel('Anzahl der Cluster (K)')
plt.ylabel('Inertia')
plt.show()

# 4. Das finale Modell mit K=3
kmeans_final = KMeans(n_clusters=3, random_state=42)
penguins_df['cluster'] = kmeans_final.fit_predict(penguins_df1)

# 5. Visualisierung des Ergebnisses (mit Farben für die Cluster)
plt.figure(figsize=(8, 6))
# Der Parameter 'c' sorgt für die farbliche Trennung der Cluster
plt.scatter(penguins_df['culmen_length_mm'], penguins_df['flipper_length_mm'], c=penguins_df['cluster'], cmap='viridis')
plt.title('Pinguin Cluster: Schnabellänge vs. Flossenlänge')
plt.xlabel('Schnabellänge (mm)')
plt.ylabel('Flossenlänge (mm)')
plt.show()

# 6. Statistische Auswertung
stat_penguins = penguins_df.groupby('cluster').mean()
print(stat_penguins)
