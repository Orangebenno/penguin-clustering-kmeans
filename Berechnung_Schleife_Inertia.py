# Ermittlung der Inertia

Inertia = []
Tabelle = {}
for k in range(2,10):
    kmean = KMeans(n_clusters=k)
    kmean = kmean.fit(penguins_df1)
    centroid = kmean.cluster_centers_
    # Berechnung der Abstände der Punkte zu jedem centroid
    for i in range(0,k):
        Tabelle[f'centroid{i}'] = ((penguins_df1-centroid[i,:])**2).sum(axis=1)
    # Berechnung des Abstandes der Punkte zum nächsten centroid (kürzester Abstand)
    Tabellemin = pd.DataFrame(Tabelle).min(axis=1)
    #Summe dieser Abstände (Tabellemin.sum()
    Inertia.append(Tabellemin.sum())

print(Inertia)
