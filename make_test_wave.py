#!/usr/bin/env python3
"""
Génération du fichier audio d'évaluation : melodie.wav
======================================================
Ce script génère une mélodie de test musicale pour l'évaluation
du système de détection de notes et de l'accordeur chromatique en temps réel.

Mélodie générée : Thème mélodique (Do4 - Ré4 - Mi4 - Sol4 - La4)
  1. Do4  (C4) : 261.63 Hz, accord parfait (0 cent)
  2. Ré4  (D4) : 293.66 Hz, accord parfait (0 cent)
  3. Mi4  (E4) : 326.22 Hz, désaccordé intentionnellement (-18 cents)
  4. Sol4 (G4) : 392.00 Hz, accord parfait (0 cent)
  5. La4  (A4) : 440.00 Hz, diapason de référence universel (0 cent)

Modélisation physique et traitement du signal :
  - Chaque note est modélisée par sa fréquence fondamentale (f0) et ses harmoniques
    naturelles (2*f0, 3*f0, 4*f0, 5*f0), avec l'harmonique 2 légèrement prédominante
    afin d'illustrer la nécessité de l'algorithme HPS (Harmonic Product Spectrum).
  - Enveloppe temporelle : attaque rapide (15 ms) et amortissement exponentiel réaliste.
  - Léger bruit ambiant pour simuler des conditions de captation réelles.
"""

import wave
import numpy as np

# Constantes d'échantillonnage
FS = 44100            # Fréquence d'échantillonnage standard CD audio (Hz)
DUREE_NOTE = 1.5      # Durée de chaque note en secondes

# Définition des notes de la nouvelle mélodie
# Format : (Nom affiché, Code international, Fréquence nominale Hz, Écart en cents)
NOTES_MELODIE = [
    ("Do4",  "C4", 261.63,   0),  # Juste
    ("Ré4",  "D4", 293.66,   0),  # Juste
    ("Mi4",  "E4", 329.63, -18),  # Désaccordé (-18 cents) pour valider l'accordeur
    ("Sol4", "G4", 392.00,   0),  # Juste
    ("La4",  "A4", 440.00,   0),  # Juste (Référence internationale A440)
]

def generer_melodie(fichier_sortie="melodie.wav"):
    """Synthétise la mélodie et sauvegarde le résultat en fichier WAV PCM 16 bits."""
    morceaux = []
    
    print("=" * 60)
    print("Synthèse de la nouvelle mélodie pour l'accordeur chromatique")
    print("=" * 60)
    
    for nom, code, f_nominale, ecart_cents in NOTES_MELODIE:
        # Fréquence réelle tenant compte du désaccordage en cents
        # Formule : f_reelle = f_nominale * 2^(cents / 1200)
        f_reelle = f_nominale * (2.0 ** (ecart_cents / 1200.0))
        print(f"  • Note {code} ({nom}) : {f_reelle:6.2f} Hz | Décalage : {ecart_cents:+3d} cents")
        
        # Vecteur temporel pour la durée de la note
        nb_echantillons = int(DUREE_NOTE * FS)
        t = np.arange(nb_echantillons, dtype=np.float64) / FS
        
        # Enveloppe temporelle (attaque linéaire douce + extinction exponentielle)
        duree_attaque = 0.015  # 15 ms
        attaque = np.minimum(1.0, t / duree_attaque)
        decroissance = np.exp(-1.1 * t)
        enveloppe = attaque * decroissance
        
        # Synthèse harmonique : f0, 2f0, 3f0, 4f0, 5f0
        # L'harmonique 2 est légèrement renforcée (0.60 vs 0.50) pour simuler
        # un instrument acoustique riche (violon, corde pincée)
        poids_harmoniques = [0.50, 0.60, 0.30, 0.15, 0.05]
        signal_note = np.zeros_like(t)
        for rang, poids in enumerate(poids_harmoniques, start=1):
            signal_note += poids * np.sin(2.0 * np.pi * rang * f_reelle * t)
            
        # Application de l'enveloppe
        signal_note *= enveloppe
        
        # Ajout d'un très léger bruit de fond blanc gaussien réaliste
        signal_note += 0.002 * np.random.randn(len(t))
        morceaux.append(signal_note)
        
    # Concaténation de toutes les notes
    signal_global = np.concatenate(morceaux)
    
    # Normalisation à 80% de la pleine échelle 16 bits (-32768 à +32767)
    amplitude_max = np.max(np.abs(signal_global))
    signal_normalise = (signal_global / amplitude_max) * 0.8 * 32767
    signal_int16 = signal_normalise.astype(np.int16)
    
    # Écriture dans le fichier audio WAV au format PCM 16-bit mono 44.1 kHz
    with wave.open(fichier_sortie, "wb") as wav_out:
        wav_out.setnchannels(1)      # Mono
        wav_out.setsampwidth(2)      # 2 octets = 16 bits
        wav_out.setframerate(FS)     # 44100 Hz
        wav_out.writeframes(signal_int16.tobytes())
        
    duree_totale = len(signal_global) / FS
    print("=" * 60)
    print(f"Fichier '{fichier_sortie}' créé avec succès !")
    print(f"Durée totale : {duree_totale:.1f} secondes ({len(signal_int16)} échantillons)")
    print("=" * 60)

if __name__ == "__main__":
    generer_melodie()
