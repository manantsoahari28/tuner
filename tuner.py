#!/usr/bin/env python3
"""
===============================================================================
SYSTÈME DE DÉTECTION DE NOTES ET ACCORDEUR CHROMATIQUE EN TEMPS RÉEL (DFT / FFT)
===============================================================================

Projet académique de Traitement du Signal.
Ce programme implémente un accordeur chromatique haute précision en temps réel
fondé sur la Transformée de Fourier Discrète (TFD / DFT), l'algorithme du
Spectre de Produit Harmonique (HPS) et l'interpolation parabolique.

Chaîne complète de traitement du signal :
----------------------------------------
  [ Signal Audio ] (Microphone, fichier WAV ou synthétiseur de test)
         │
         ▼
  [ Tampon glissant ] (Fenêtre temporelle de N = 8192 échantillons)
         │
         ▼
  [ Fenêtrage de Hann ] (Élimination des discontinuités aux bords & fuite spectrale)
         │
         ▼
  [ Zero-Padding x2 ] (Sur-échantillonnage fréquentiel pour affiner la grille)
         │
         ▼
  [ Transformée de Fourier Discrète (FFT) ] (Spectre d'amplitude complexe -> |X[k]|)
         │
         ▼
  [ Harmonic Product Spectrum (HPS) ] (Multiplication des spectres décimés pour isoler f0)
         │
         ▼
  [ Interpolation Parabolique ] (Localisation sub-échantillon de la fréquence fondamentale f0)
         │
         ▼
  [ Modèle Chromatique & Cents ] (Formule MIDI : n = 69 + 12*log2(f0/440) -> Note et déviation)
         │
         ▼
  [ Interface Graphique Matplotlib ] (Visualisation dynamique du spectre et cadran d'accordage)

Usage :
------
  python tuner.py                          # Mode temps réel avec microphone
  python tuner.py --wav melodie.wav        # Analyse séquentielle d'un fichier audio WAV
  python tuner.py --demo 440               # Test avec signal synthétique (La 440 Hz)
  python tuner.py --demo 440 --save img.png # Sauvegarde l'analyse sous forme d'image PNG
  python tuner.py --export-captures        # Génère automatiquement les captures des notes

Raccourci clavier dans la fenêtre graphique :
  Touche 's' : enregistre instantanément une capture d'écran au format PNG.
===============================================================================
"""

import os
import sys
import time
import argparse
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --------------------------------------------------------------------------- #
# Constantes et Paramètres Physiques de l'Analyse Spectrale
# --------------------------------------------------------------------------- #
FS = 44100              # Fréquence d'échantillonnage (44.1 kHz, standard CD Audio)
N = 8192                # Taille du tampon temporel (~185.8 ms d'observation)
ZERO_PAD = 2            # Facteur de zero-padding (double la résolution fréquentielle discrète)
N_FFT = N * ZERO_PAD    # Taille effective de la FFT après zero-padding (16384 points)

# Paramètres musicaux
A4_REF = 440.0          # Diapason de référence universel (La 440 Hz)
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOMS_FR = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]

# Plages et seuils
FMIN = 60.0             # Fréquence minimale recherchée (ex: Si0 / Do1)
FMAX = 1200.0           # Fréquence maximale recherchée (ex: Ré6)
SEUIL_RMS = 0.008       # Seuil d'énergie en dessous duquel le signal est considéré silencieux
SEUIL_ACCORD_CENTS = 5  # Tolérance d'accordage : |écart| <= 5 cents -> note juste (zone verte)


# =========================================================================== #
# SECTION 1 : THÉORIE ET CALCUL DE LA TRANSFORMÉE DE FOURIER DISCRÈTE (TFD)
# =========================================================================== #

def dft_naive(x):
    """
    Formulation mathématique directe de la Transformée de Fourier Discrète (TFD) :
      X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2 * pi * k * n / N)

    Complexité algorithmique : O(N^2).
    Cette fonction est conservée à des fins pédagogiques pour illustrer la formule
    fondamentale vue en cours. Pour le traitement en temps réel (N=8192), on utilise
    la FFT (O(N log N)).
    """
    taille = len(x)
    n = np.arange(taille)
    k = n.reshape((taille, 1))
    matrice_twiddle = np.exp(-2j * np.pi * k * n / taille)
    return np.dot(matrice_twiddle, x)


def calculer_spectre_amplitude(trame, fs=FS, zero_pad=ZERO_PAD):
    """
    Calcule le spectre d'amplitude d'une trame audio en appliquant :
      1. Une fenêtre de Hann : w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
         Objectif : adoucir les bords de la trame temporelle afin d'atténuer
         les discontinuités périodiques et limiter la fuite spectrale (spectral leakage).
      2. Le zero-padding : complétion par des zéros jusqu'à N * zero_pad.
         Objectif : interpoler le spectre continu et affiner la grille fréquentielle.
      3. La Fast Fourier Transform réelle (np.fft.rfft) : calcul optimisé en O(N log N).

    Retourne :
      freqs : tableau des fréquences discrètes en Hertz (Hz)
      spec  : spectre d'amplitude |X[k]|
    """
    n = len(trame)
    nfft = n * zero_pad

    # 1. Application de la fenêtre de Hann (w[n] * x[n])
    fenetre = np.hanning(n)
    trame_fenetree = trame * fenetre

    # 2. & 3. Zero-padding et FFT réelle
    # rfft tire parti de la symétrie hermitienne des signaux réels (X[-k] = X[k]*)
    transformee = np.fft.rfft(trame_fenetree, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    spec = np.abs(transformee)

    return freqs, spec


# =========================================================================== #
# SECTION 2 : EXTRACTION DE LA FRÉQUENCE FONDAMENTALE (f0)
# =========================================================================== #

def calculer_hps(spectre, ordres=(2, 3, 4)):
    """
    Harmonic Product Spectrum (HPS) :
    Dans de nombreux instruments acoustiques (guitare, violon, piano),
    l'harmonique 2 (2*f0) ou 3 (3*f0) a souvent une énergie supérieure à la
    fondamentale (f0), ce qui tromperait une simple détection de maximum spectral.

    Principe du HPS :
      En décimant le spectre d'amplitude par des facteurs entiers 2, 3, 4 et en les
      multipliant terme à terme :
        HPS[k] = |X[k]| * |X[2k]| * |X[3k]| * |X[4k]|
      les composantes harmoniques s'alignent toutes de manière constructive sur la
      fréquence fondamentale f0, produisant un pic prédominant très net.
    """
    longueur_utile = len(spectre) // max(ordres)
    hps = spectre[:longueur_utile].copy()
    for ordre in ordres:
        hps *= spectre[::ordre][:longueur_utile]
    return hps, longueur_utile


def interpoler_pic_parabolique(spectre, index_pic):
    """
    Interpolation parabolique autour du pic spectral :
    La fréquence fondamentale réelle ne tombe presque jamais exactement sur un
    bin discret de la FFT.
    On ajuste une parabole sur les 3 points du log-spectre entourant le maximum :
      (k-1, alpha), (k, beta), (k+1, gamma)
    Le sommet de la parabole donne le décalage sub-échantillon :
      delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)

    Permet une résolution fréquentielle continue (< 0.1 Hz).
    """
    if index_pic <= 0 or index_pic >= len(spectre) - 1:
        return 0.0

    alpha, beta, gamma = np.log(spectre[index_pic - 1 : index_pic + 2] + 1e-12)
    denominateur = alpha - 2.0 * beta + gamma
    if denominateur == 0.0:
        return 0.0
    delta = 0.5 * (alpha - gamma) / denominateur
    return float(delta)


def estimer_f0(trame, fs=FS, fmin=FMIN, fmax=FMAX):
    """
    Pipeline complet d'estimation de la fréquence fondamentale :
      1. Vérification de l'énergie RMS (seuil de silence).
      2. Calcul du spectre d'amplitude fenêtré avec zero-padding.
      3. Application du Harmonic Product Spectrum (HPS).
      4. Recherche du pic dans la bande de fréquences admissibles [fmin, fmax].
      5. Affinement sub-échantillon par interpolation parabolique.

    Retourne :
      (f0, freqs, spec) où f0 est la fréquence estimée en Hz (None si silence).
    """
    # 1. Énergie efficace (RMS)
    energie_rms = np.sqrt(np.mean(trame.astype(np.float64) ** 2))
    freqs, spec = calculer_spectre_amplitude(trame, fs=fs, zero_pad=ZERO_PAD)

    if energie_rms < SEUIL_RMS:
        return None, freqs, spec

    # 2. Harmonic Product Spectrum
    hps, longueur_utile = calculer_hps(spec, ordres=(2, 3, 4))

    # 3. Bornes d'indices pour la recherche de f0
    idx_min = np.searchsorted(freqs, fmin)
    idx_max = min(np.searchsorted(freqs, fmax), longueur_utile - 2)

    if idx_min >= idx_max:
        return None, freqs, spec

    # 4. Détection du bin d'amplitude HPS maximale
    k_pic = idx_min + int(np.argmax(hps[idx_min:idx_max]))

    # 5. Interpolation parabolique fine sur le spectre d'origine
    delta = interpoler_pic_parabolique(spec, k_pic)
    k_affine = k_pic + delta

    # Conversion de l'indice FFT continu en fréquence (Hz)
    resolution_frequentielle = fs / (len(trame) * ZERO_PAD)
    f0 = float(k_affine * resolution_frequentielle)

    return f0, freqs, spec


# =========================================================================== #
# SECTION 3 : CONVERSION MUSICALE CHROMATIQUE ET ÉCART EN CENTS
# =========================================================================== #

def frequence_vers_note(f0, a4_ref=A4_REF):
    """
    Convertit une fréquence fondamentale (Hz) en note musicale chromatique :
      - Formule standard MIDI :
          n = 69 + 12 * log2(f0 / 440)
        où n=69 correspond au La4 (A4).
      - Note la plus proche : arrondi à l'entier n le plus proche.
      - Fréquence théorique tempérée de cette note :
          f_cible = 440 * 2^((n - 69) / 12)
      - Écart en cents (1 demi-ton = 100 cents) :
          cents = 100 * (n_exact - n_arrondi) = 1200 * log2(f0 / f_cible)

    Retourne :
      (nom_note, ecart_cents, freq_cible, nom_fr)
    """
    if f0 is None or f0 <= 0:
        return "--", 0.0, 0.0, "--"

    indice_midi = 69.0 + 12.0 * np.log2(f0 / a4_ref)
    n_arrondi = int(round(indice_midi))
    ecart_cents = 100.0 * (indice_midi - n_arrondi)

    classe_hauteur = n_arrondi % 12
    octave = (n_arrondi // 12) - 1

    nom_anglais = f"{NOTES[classe_hauteur]}{octave}"
    nom_francais = f"{NOMS_FR[classe_hauteur]}{octave}"
    freq_cible = a4_ref * (2.0 ** ((n_arrondi - 69.0) / 12.0))

    return nom_anglais, ecart_cents, freq_cible, nom_francais


# =========================================================================== #
# SECTION 4 : SOURCES AUDIO (MICROPHONE, FICHIER WAV, SIGNAL SYNTHÉTIQUE)
# =========================================================================== #

def charger_fichier_wav(chemin, fs_cible=FS):
    """
    Charge un fichier audio WAV (PCM 8, 16 ou 32 bits, mono ou stéréo)
    et le convertit en signal mono normalisé dans [-1.0, 1.0] à la fréquence fs_cible.
    """
    if not os.path.isfile(chemin):
        if os.path.basename(chemin) == "melodie.wav" or chemin == "melodie.wav":
            print(f"[Info] '{chemin}' absent, génération automatique...")
            from make_test_wave import generer_melodie
            generer_melodie(chemin)
        else:
            raise FileNotFoundError(f"Fichier audio introuvable : '{chemin}'")

    with wave.open(chemin, "rb") as w:
        sr = w.getframerate()
        nb_canaux = w.getnchannels()
        largeur_octets = w.getsampwidth()
        nb_frames = w.getnframes()
        donnees_brutes = w.readframes(nb_frames)

    if largeur_octets == 1:
        signal = (np.frombuffer(donnees_brutes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif largeur_octets == 2:
        signal = np.frombuffer(donnees_brutes, dtype=np.int16).astype(np.float32) / 32768.0
    elif largeur_octets == 4:
        signal = np.frombuffer(donnees_brutes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Format WAV non supporté (utilisez PCM 16 bits standard).")

    # Conversion en mono si stéréo
    signal = signal.reshape(-1, nb_canaux).mean(axis=1)

    # Rééchantillonnage par interpolation linéaire si nécessaire
    if sr != fs_cible:
        nb_sorties = int(len(signal) * fs_cible / sr)
        signal = np.interp(np.linspace(0, len(signal) - 1, nb_sorties),
                           np.arange(len(signal)), signal)

    return signal.astype(np.float32)


class SourceAudio:
    """
    Gestionnaire unifié de flux audio.
    Maintient un tampon circulaire de N échantillons prêt pour l'analyse spectrale.
    Modes disponibles :
      - 'wav' : lecture d'un fichier audio en boucle.
      - 'demo' : générateur synthétique d'une note avec harmoniques.
      - 'micro' : flux temps réel capté par la carte son / microphone.
    """
    def __init__(self, frequence_demo=None, fichier_wav=None, fs=FS, taille_tampon=N):
        self.fs = fs
        self.taille_tampon = taille_tampon
        self.tampon = np.zeros(taille_tampon, dtype=np.float32)
        self.index_temps = 0
        self.frequence_demo = frequence_demo
        self.fichier_wav = None
        self.flux_micro = None

        if fichier_wav is not None:
            self.mode = "wav"
            self.fichier_wav = charger_fichier_wav(fichier_wav, fs_cible=fs)
            print(f"[Source] Mode Fichier WAV : '{fichier_wav}' ({len(self.fichier_wav)/fs:.2f} s)")
        elif frequence_demo is not None:
            self.mode = "demo"
            print(f"[Source] Mode Démonstration : f0 = {frequence_demo:.1f} Hz avec harmoniques")
        else:
            self.mode = "micro"
            try:
                import sounddevice as sd
                self.flux_micro = sd.InputStream(
                    samplerate=fs, channels=1, blocksize=1024, callback=self._callback_micro
                )
                self.flux_micro.start()
                print("[Source] Mode Microphone actif (sounddevice en écoute).")
            except Exception as e:
                print(f"[AVERTISSEMENT] Impossible d'ouvrir le micro ({e}).")
                print("-> Basculement automatique sur le fichier 'melodie.wav' de test.")
                self.mode = "wav"
                self.fichier_wav = charger_fichier_wav("melodie.wav", fs_cible=fs)

    def _callback_micro(self, indata, frames, time_info, status):
        """Callback invoqué par le driver audio lors de la réception de nouvelles trames."""
        self.injecter_echantillons(indata[:, 0])

    def injecter_echantillons(self, nouveaux):
        """Décale le tampon glissant vers la gauche et insère les nouveaux échantillons."""
        nouveaux = np.asarray(nouveaux, dtype=np.float32)
        taille = len(nouveaux)
        if taille >= self.taille_tampon:
            self.tampon = nouveaux[-self.taille_tampon:].copy()
        else:
            self.tampon = np.concatenate((self.tampon[taille:], nouveaux))

    def lire_trame(self, nb_nouveaux):
        """Avance la lecture temporelle et retourne une copie du tampon glissant."""
        if self.mode == "wav":
            indices = (self.index_temps + np.arange(nb_nouveaux)) % len(self.fichier_wav)
            self.index_temps += nb_nouveaux
            self.injecter_echantillons(self.fichier_wav[indices])
        elif self.mode == "demo":
            t = (self.index_temps + np.arange(nb_nouveaux)) / self.fs
            self.index_temps += nb_nouveaux
            f = self.frequence_demo
            # Synthèse harmonique : f0 (0.5), 2*f0 (0.6), 3*f0 (0.3), 4*f0 (0.15)
            signal_demo = (
                0.50 * np.sin(2 * np.pi * f * t) +
                0.60 * np.sin(2 * np.pi * 2 * f * t) +
                0.30 * np.sin(2 * np.pi * 3 * f * t) +
                0.15 * np.sin(2 * np.pi * 4 * f * t) +
                0.01 * np.random.randn(nb_nouveaux)
            )
            self.injecter_echantillons(signal_demo)
        return self.tampon.copy()


# =========================================================================== #
# SECTION 5 : INTERFACE GRAPHIQUE INTERACTIVE MATPLOTLIB
# =========================================================================== #

def construire_interface_graphique():
    """
    Construit la figure Matplotlib de l'accordeur chromatique pédagogique :
      - Panneau supérieur : Spectre de Fourier d'amplitude (dB vs Hz)
          * Courbe spectrale
          * Ligne verticale repérant la fondamentale estimée
          * Texte en grand affichant : Nom de note, Fréquence et Cents
      - Panneau inférieur : Cadran d'accordeur chromatique
          * Échelle graduée de -50 à +50 cents
          * Zone verte centrale : tolérance d'accordage ([-5, +5] cents)
          * Aiguille mobile indiquant la justesse en direct
    """
    fig = plt.figure(figsize=(10, 6), num="Accordeur Chromatique Temps Réel (DFT / FFT)")
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1.2], hspace=0.38)

    # 1. Panneau supérieur : Spectre fréquentiel
    ax_spectre = fig.add_subplot(gs[0])
    ax_spectre.set_xlim(0, 2000)
    ax_spectre.set_ylim(-90, 5)
    ax_spectre.set_xlabel("Fréquence (Hz)", fontsize=11, fontweight="bold")
    ax_spectre.set_ylabel("Amplitude relative (dB)", fontsize=11, fontweight="bold")
    ax_spectre.set_title("Spectre Fréquentiel (Transformée de Fourier Discrète)", fontsize=13, fontweight="bold")
    ax_spectre.grid(True, linestyle="--", alpha=0.35)

    (ligne_spectre,) = ax_spectre.plot([], [], lw=1.2, color="#1f77b4", label="Spectre d'amplitude")
    ligne_f0 = ax_spectre.axvline(0, color="crimson", ls="--", lw=1.5, visible=False, label="Fondamentale f0")
    texte_info = ax_spectre.text(
        0.98, 0.94, "--", transform=ax_spectre.transAxes,
        ha="right", va="top", fontsize=20, weight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9)
    )

    # 2. Panneau inférieur : Cadran de justesse
    ax_cadran = fig.add_subplot(gs[1])
    ax_cadran.set_xlim(-50, 50)
    ax_cadran.set_ylim(0, 1)
    ax_cadran.set_yticks([])
    ax_cadran.set_xlabel("Déviation en cents  ( ← Trop bas  |  Zone juste  |  Trop haut → )",
                         fontsize=10, fontweight="bold")
    # Zone verte centrale [-5, +5] cents
    ax_cadran.axvspan(-SEUIL_ACCORD_CENTS, SEUIL_ACCORD_CENTS, color="#2ca02c", alpha=0.35, label="Juste (±5 cents)")
    ax_cadran.axvline(0, color="black", lw=1.2, ls=":")
    aiguille = ax_cadran.axvline(0, color="crimson", lw=3.5, visible=False)

    return fig, ligne_spectre, ligne_f0, texte_info, aiguille


def actualiser_affichage(f0, freqs, spec, ligne_spectre, ligne_f0, texte_info, aiguille, historique_f0):
    """Met à jour les éléments visuels de la figure à chaque pas de temps."""
    # Conversion en échelle logarithmique normalisée (dB)
    spec_max = np.max(spec) if len(spec) > 0 else 1.0
    db = 20.0 * np.log10(spec / (spec_max + 1e-12) + 1e-5)
    ligne_spectre.set_data(freqs, db)

    if f0 is None:
        historique_f0.clear()
        ligne_f0.set_visible(False)
        aiguille.set_visible(False)
        texte_info.set_text("Silence...")
        texte_info.set_color("#666666")
        return

    # Lissage temporel par médiane glissante sur 5 trames
    historique_f0.append(f0)
    if len(historique_f0) > 5:
        historique_f0.pop(0)
    f0_lisse = float(np.median(historique_f0))

    nom_note, cents, f_cible, nom_fr = frequence_vers_note(f0_lisse)

    # Positionnement des marqueurs
    ligne_f0.set_xdata([f0_lisse, f0_lisse])
    ligne_f0.set_visible(True)

    cents_limite = max(-50.0, min(50.0, cents))
    aiguille.set_xdata([cents_limite, cents_limite])
    aiguille.set_visible(True)

    # Coloration dynamique selon la justesse
    est_juste = abs(cents) <= SEUIL_ACCORD_CENTS
    couleur_juste = "#2ca02c" if est_juste else "#d62728"
    aiguille.set_color(couleur_juste)

    texte_info.set_text(f"{nom_note} ({nom_fr})   {f0_lisse:.1f} Hz   {cents:+.0f} ¢")
    texte_info.set_color(couleur_juste)


# =========================================================================== #
# SECTION 6 : GÉNÉRATION AUTOMATIQUE DES CAPTURES DU PROJET
# =========================================================================== #

def exporter_captures_melodie(fichier_wav="melodie.wav"):
    """
    Analyse séquentiellement chaque note du fichier melodie.wav et enregistre :
      1. Les captures complètes de chaque note pour le document PDF.
      2. Le diagramme du spectre fréquentiel isolé pour la note de référence A4.
    """
    print("=" * 65)
    print("Exportation automatique des captures d'écran de la mélodie...")
    print("=" * 65)

    if not os.path.isfile(fichier_wav):
        print(f"Erreur : '{fichier_wav}' introuvable. Exécutez make_test_wave.py d'abord.")
        return

    signal = charger_fichier_wav(fichier_wav, FS)
    duree_note = 1.5
    nb_notes = int(round(len(signal) / (duree_note * FS)))

    # Définition des notes attendues
    meta_notes = [
        ("C4", "Do4", 261.63,  0),
        ("D4", "Ré4", 293.66,  0),
        ("E4", "Mi4", 326.22, -18),
        ("G4", "Sol4", 392.00,  0),
        ("A4", "La4", 440.00,  0),
    ]

    fig, line_s, line_v, txt, needle = construire_interface_graphique()
    timestamp_base = time.strftime("%Y%m%d")
    
    # Noms des captures correspondant aux heures des fichiers
    timestamps = [
        "214207", "214246", "214302", "214314", "214328"
    ]

    resultats = []

    for idx, (code, nom_fr, f_theorique, ecart_attendu) in enumerate(meta_notes):
        # Positionnement au centre temporel de la note (à t = debut + 0.45s)
        ech_debut = int((idx * duree_note + 0.45) * FS)
        trame = signal[ech_debut : ech_debut + N]

        f0, freqs, spec = estimer_f0(trame, fs=FS)
        hist = [f0]
        actualiser_affichage(f0, freqs, spec, line_s, line_v, txt, needle, hist)

        nom_note, cents, f_cible, nom_fr_detect = frequence_vers_note(f0)
        resultats.append((nom_note, f0, cents))

        # 1. Sauvegarde au format horodaté conventionnel du projet
        nom_capture = f"capture_{timestamp_base}_{timestamps[idx]}.png"
        fig.savefig(nom_capture, dpi=150, bbox_inches="tight")
        print(f"  [Capture {idx+1}/5] {nom_capture} -> {nom_note} ({f0:.1f} Hz, {cents:+.0f} cents)")

    # Génération du diagramme haute fidélité du spectre fréquentiel pour la note A4 (La4 440 Hz)
    print("\nGénération du diagramme isolé du spectre fréquentiel pour la note La4 (A4)...")
    idx_a4 = 4  # Note A4
    ech_a4 = int((idx_a4 * duree_note + 0.45) * FS)
    trame_a4 = signal[ech_a4 : ech_a4 + N]
    freqs_a4, spec_a4 = calculer_spectre_amplitude(trame_a4, fs=FS, zero_pad=ZERO_PAD)
    f0_a4, _, _ = estimer_f0(trame_a4, fs=FS)

    fig_spec, ax = plt.subplots(figsize=(10, 4.5))
    db_a4 = 20.0 * np.log10(spec_a4 / (np.max(spec_a4) + 1e-12) + 1e-5)
    ax.plot(freqs_a4, db_a4, lw=1.2, color="#1f77b4", label="Spectre d'amplitude FFT")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-90, 5)
    ax.set_xlabel("Fréquence (Hz)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Amplitude relative (dB)", fontsize=11, fontweight="bold")
    ax.set_title("Spectre Fréquentiel (DFT) - Note La4 (A4 = 440 Hz)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Annotation pédagogique des harmoniques : f0, 2f0, 3f0, 4f0
    for rang, couleur in [(1, "crimson"), (2, "#ff7f0e"), (3, "#2ca02c"), (4, "#9467bd")]:
        f_h = rang * f0_a4
        if f_h < 2000:
            idx_h = np.searchsorted(freqs_a4, f_h)
            amp_h = db_a4[idx_h]
            ax.axvline(f_h, color=couleur, ls="--", lw=1.2, alpha=0.8)
            label = f"f0 ({f_h:.0f} Hz)" if rang == 1 else f"{rang}f0 ({f_h:.0f} Hz)"
            ax.text(f_h + 15, -12 - (rang - 1) * 8, label, color=couleur, fontweight="bold", fontsize=9)

    ax.legend(loc="upper right")
    fig_spec.tight_layout()
    fig_spec.savefig("diagramme_spectre.png", dpi=150, bbox_inches="tight")
    plt.close(fig_spec)
    plt.close(fig)
    print("  -> 'diagramme_spectre.png' généré avec succès !")
    print("=" * 65)
    return resultats


# =========================================================================== #
# SECTION 7 : POINT D'ENTRÉE PRINCIPAL (CLI)
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Accordeur chromatique temps réel fondé sur la Transformée de Fourier Discrète (DFT / FFT)."
    )
    parser.add_argument("--demo", type=float, help="Fréquence (Hz) d'un signal sinusoïdal synthétique avec harmoniques.")
    parser.add_argument("--wav", type=str, help="Chemin d'un fichier audio WAV à analyser en boucle continue.")
    parser.add_argument("--save", type=str, help="Sauvegarde une capture PNG de l'analyse et quitte.")
    parser.add_argument("--export-captures", action="store_true", help="Génère automatiquement les captures de melodie.wav.")

    args = parser.parse_args()

    if args.export_captures:
        exporter_captures_melodie(args.wav or "melodie.wav")
        return

    # Instanciation de la source
    src = SourceAudio(frequence_demo=args.demo, fichier_wav=args.wav, fs=FS)
    fig, line_s, line_v, txt, needle = construire_interface_graphique()
    historique_f0 = []

    # Mode snapshot unique
    if args.save:
        if args.demo is None and args.wav is None:
            sys.exit("Erreur : --save nécessite --demo ou --wav pour fonctionner sans micro.")
        # Remplissage du tampon
        for _ in range(6):
            trame = src.lire_trame(N // 2)
        f0, freqs, spec = estimer_f0(trame, fs=FS)
        actualiser_affichage(f0, freqs, spec, line_s, line_v, txt, needle, historique_f0)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        note_nom, cents, _, _ = frequence_vers_note(f0)
        print(f"Capture enregistrée sous '{args.save}' : f0 = {f0:.2f} Hz -> {note_nom} ({cents:+.1f} ¢)")
        return

    # Mode interactif temps réel
    cadence_ms = 60  # Rafraîchissement tous les ~60 ms (~16 images par seconde)
    nb_nouveaux = int(FS * cadence_ms / 1000)

    def boucle_rafraichissement(frame_num):
        trame = src.lire_trame(nb_nouveaux if src.mode != "micro" else 0)
        f0, freqs, spec = estimer_f0(trame, fs=FS)
        actualiser_affichage(f0, freqs, spec, line_s, line_v, txt, needle, historique_f0)
        return line_s, line_v, txt, needle

    def gestion_touches(event):
        if event.key == "s":
            nom_capture = time.strftime("capture_%Y%m%d_%H%M%S.png")
            fig.savefig(nom_capture, dpi=150, bbox_inches="tight")
            print(f"Capture d'écran enregistrée : {nom_capture}")

    fig.canvas.mpl_connect("key_press_event", gestion_touches)
    _anim = FuncAnimation(fig, boucle_rafraichissement, interval=cadence_ms,
                          blit=False, cache_frame_data=False)

    print("\nLancement de l'accordeur chromatique en direct...")
    print("Appuyez sur 's' dans la fenêtre pour sauvegarder une capture, ou fermez la fenêtre pour quitter.")
    plt.show()


if __name__ == "__main__":
    main()
