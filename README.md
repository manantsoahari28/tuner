# Détection de Notes et Accordeur Chromatique Temps Réel (TFD / FFT)

Projet d'évaluation en **Traitement du Signal** fondé sur l'application de la **Transformée de Fourier Discrète (TFD / DFT)**.

Ce projet implémente un système complet de détection de notes de musique et un accordeur chromatique haute précision en temps réel, visualisé dynamiquement avec **NumPy** et **Matplotlib**.

---

## 🎵 Nouvelle Mélodie d'Évaluation (`melodie.wav`)

La mélodie générée est un thème musical mélodique composé de 5 notes synthétisées avec harmoniques physiques naturelles et amortissement acoustique :

| Note | Fréquence nominale | Fréquence réelle | Écart en cents | Statut d'accordage |
|:----:|:-----------------:|:----------------:|:--------------:|:------------------:|
| **Do4 (C4)** | 261,63 Hz | 261,63 Hz | 0 cent | Juste (zone verte) |
| **Ré4 (D4)** | 293,66 Hz | 293,66 Hz | 0 cent | Juste (zone verte) |
| **Mi4 (E4)** | 329,63 Hz | 326,22 Hz | **-18 cents** | **Désaccordé (trop bas)** |
| **Sol4 (G4)** | 392,00 Hz | 392,00 Hz | 0 cent | Juste (zone verte) |
| **La4 (A4)** | 440,00 Hz | 440,00 Hz | 0 cent | Diapason universel de référence |

> **Intérêt pédagogique** : La note **Mi4 (E4)** est intentionnellement désaccordée de -18 cents afin de démontrer la déviation de l'aiguille de l'accordeur chromatique en dehors de la zone verte ($\pm 5$ cents).

---

## ⚙️ Installation

Dans votre environnement Python (version 3.10 ou supérieure) :

```bash
pip install -r Requirements.txt
```

*Bibliothèques principales utilisées : `numpy` et `matplotlib` (traitement du signal, synthèse et affichage graphique).*

---

## 🚀 Utilisation

### 1. Lancement rapide (Recommandé)
Pour exécuter l'accordeur directement sur la nouvelle mélodie :
```bash
python main.py
```

### 2. Modes d'analyse avec `tuner.py`
```bash
# Analyse en boucle continue de la mélodie
python tuner.py --wav melodie.wav

# Analyse d'un signal sinusoïdal de test avec harmoniques (ex: La 440 Hz)
python tuner.py --demo 440

# Utilisation du microphone en direct (si micro disponible)
python tuner.py

# Sauvegarde d'une capture d'écran PNG de l'analyse
python tuner.py --wav melodie.wav --save capture_test.png
```

*Dans l'interface graphique, appuyez sur la touche **'s'** pour enregistrer une capture instantanée.*

### 3. Outils pédagogiques & Régénération
```bash
# Démonstration du calcul de la TFD sur l'exemple du cours (x = [1, 6, 8, -4, 9])
python main.py --dft-cours

# Régénération automatique de tous les livrables (mélodie, captures, diagramme, PDF)
python main.py --regenerer-tout
```

---

## 🔬 Chaîne de Traitement du Signal (DSP)

L'extraction de la fréquence fondamentale $f_0$ repose sur une chaîne d'analyse en 5 étapes :

1. **Échantillonnage et Fenêtrage glissant** :
   - Fréquence d'échantillonnage standard $f_s = 44100\text{ Hz}$.
   - Trame temporelle de $N = 8192$ échantillons ($\approx 186\text{ ms}$).
   - Application d'une **fenêtre de Hann** : $w[n] = 0.5 \left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$ pour adoucir les frontières de la trame et supprimer la fuite spectrale (*spectral leakage*).

2. **Sur-échantillonnage fréquentiel (Zero-Padding $\times 2$) & TFD** :
   - Ajout de zéros jusqu'à $N_{\text{FFT}} = 16384$ points pour interpoler le spectre continu.
   - Calcul rapide de la TFD via la Fast Fourier Transform (`np.fft.rfft`), complexité $O(N \log N)$ vs $O(N^2)$ pour la TFD naïve.

3. **Harmonic Product Spectrum (HPS)** :
   - Sur les instruments réels, les harmoniques d'ordre supérieur ($2f_0, 3f_0$) peuvent être plus fortes que la fondamentale.
   - On multiplie le spectre d'amplitude par ses versions sous-échantillonnées :
     $$\text{HPS}[k] = |X[k]| \times |X[2k]| \times |X[3k]| \times |X[4k]|$$
   - Les harmoniques se superposent et renforcent le pic à la fondamentale exacte $f_0$.

4. **Interpolation Parabolique Sub-échantillon** :
   - Ajustement d'une parabole sur le logarithme des 3 bins entourant le pic maximal $(k-1, k, k+1)$ :
     $$\delta = \frac{1}{2} \frac{\alpha - \gamma}{\alpha - 2\beta + \gamma}$$
   - Permet d'obtenir la fréquence $f_0$ avec une précision continue inférieure à $0{,}1\text{ Hz}$.

5. **Modèle Chromatique et Écart en Cents** :
   - Conversion en indice MIDI : $n = 69 + 12 \log_2\left(\frac{f_0}{440}\right)$.
   - Détermination du nom de la note tempérée la plus proche (Do, Ré, Mi, Fa, Sol, La, Si).
   - Calcul de l'écart de justesse en cents : $\text{cents} = 100 \times (n_{\text{exact}} - n_{\text{arrondi}})$.
   - Si $|\text{cents}| \le 5$, la note est considérée juste (zone verte du cadran).

---

## 📂 Livrables de l'Examen

- **`tuner.py`** : Code source complet de détection de notes et de l'accordeur chromatique en temps réel.
- **`make_test_wave.py`** : Script de synthèse physique de la nouvelle mélodie.
- **`main.py`** : Orchestrateur pédagogique et lanceur centralisé.
- **`generate_pdf.py`** : Générateur automatisé du rapport PDF de synthèse.
- **`document_pdf.pdf`** : Document PDF d'évaluation comprenant les captures d'écran, le tableau des résultats, le diagramme du spectre fréquentiel de la note La4 et les explications théoriques.
- **`captures_*.png`** : Captures de l'accordeur chromatique pour les 5 notes de la mélodie.
- **`diagramme_spectre.png`** : Diagramme détaillé du spectre fréquentiel annoté avec les harmoniques $f_0, 2f_0, 3f_0, 4f_0$.
