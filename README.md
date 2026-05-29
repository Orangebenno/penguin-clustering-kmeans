# Palmer Penguins Clustering mit K-Means

Dieses Projekt zeigt, wie man die Methode KMeans nutzt, um Pinguin-Arten anhand ihrer körperlichen Merkmale in Gruppen zu unterteilen. Verwendet wurde der bekannte *Palmer Penguins* Datensatz.
KMeans ist einer von mehreren Methoden des 'unüberwachtes Maschinelles Lernen'.

## 🎯 Zielstellung
Das Ziel war es, die Pinguine rein anhand ihrer morphologischen Merkmale (wie Schnabellänge, Flossenlänge und Körpergewicht) zu gruppieren, ohne dem Computer vorher zu verraten, welcher Pinguin zu welcher Art gehört.

## 🛠️ Workflow & Projektschritte

1. **Datenbereinigung:** Fehlende Werte (`NaN`) wurden entfernt, damit der Algorithmus ohne Fehler rechnen kann.
2. **Feature Engineering & Skalierung:**
   - Die Kategorie "Geschlecht" wurde über One-Hot-Encoding (`pd.get_dummies`) in Zahlen umgewandelt.
   - Alle numerischen Werte wurden mit dem `StandardScaler` standardisiert. Dies verhindert, dass das große Körpergewicht (in Gramm) die kleineren Millimeter-Maße der Schnäbel bei der Abstandsberechnung unterdrückt.
3. **Modellierung (K-Means):**
   - Mithilfe der **Elbow-Methode** (Inertia-Kurve) wurde die optimale Clusteranzahl ermittelt.
   - Das finale Modell wurde mit **K=3** Clustern trainiert, was genau den drei echten Pinguin-Arten im Datensatz entspricht.
4. **Auswertung:**
   - Analyse der Gruppen-Mittelwerte mittels `.groupby().mean()`.

## 📊 Wichtigste Erkenntnisse
- Der K-Means-Algorithmus ist in der Lage, die drei Pinguin-Arten allein anhand der Körpermesswerte sehr präzise voneinander zu trennen.
- Ohne das Feature Scaling (Standardisierung) wäre das Clustering fehlgeschlagen, da das Gewicht das dominierende Merkmal gewesen wäre.

## 💻 Verwendete Bibliotheken
- Python 3
- Pandas (Datenanalyse)
- Scikit-Learn (StandardScaler & KMeans)
- Matplotlib (Visualisierung)
