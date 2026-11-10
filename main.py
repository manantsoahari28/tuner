#!/usr/bin/env python3
"""
===============================================================================
PROJET D'EXAMEN : DÉTECTION DE NOTES ET ACCORDEUR CHROMATIQUE EN TEMPS RÉEL
===============================================================================
Traitement du Signal & Application de la Transformée de Fourier Discrète (TFD)

Ce script est le point d'entrée principal (orchestrateur pédagogique) du projet.
Il permet de :
  1. Lancer l'accordeur chromatique interactif (sur melodie.wav ou micro).
  2. Démontrer le calcul théorique de la TFD / DFT (ex. du cours : x = [1, 6, 8, -4, 9]).
  3. Régénérer l'ensemble des livrables de l'examen en une commande :
     - La mélodie de test synthétique (melodie.wav)
     - Les captures d'écran des notes analysées
     - Le diagramme détaillé du spectre fréquentiel (diagramme_spectre.png)
     - Le document de synthèse complet au format PDF (document_pdf.pdf)

Usage :
------
  python main.py                    # Lance l'accordeur sur la nouvelle melodie.wav
  python main.py --mic              # Lance l'accordeur avec le microphone en direct
  python main.py --demo 440         # Lance l'accordeur avec une note synthétique (La 440 Hz)
  python main.py --dft-cours        # Démonstration du calcul de la TFD sur l'exemple du cours
  python main.py --regenerer-tout   # Régénère la mélodie, les captures et le PDF complet
===============================================================================
"""

import sys
import os
import argparse
import numpy as np

# Importation des modules du projet
from tuner import dft_naive, main as tuner_main, exporter_captures_melodie
from make_test_wave import generer_melodie
from generate_pdf import generer_pdf


def afficher_banniere_pedagogique():
    """Affiche une présentation claire de la chaîne de traitement du signal."""
    print("=" * 75)
    print("  EXAMEN DE TRAITEMENT DU SIGNAL : ACCORDEUR CHROMATIQUE TEMPS RÉEL")
    print("=" * 75)
    print("  Chaîne de traitement implémentée :")
    print("    1. Échantillonnage audio (fs = 44.1 kHz, fenêtre glissante N = 8192)")
    print("    2. Fenêtrage de Hann (réduction de la fuite spectrale / spectral leakage)")
    print("    3. Zero-padding x2 (interpolation fréquentielle fine sur 16384 points)")
    print("    4. Transformée de Fourier Discrète rapide (np.fft.rfft)")
    print("    5. Harmonic Product Spectrum (HPS ordre 4 : élimination des ambiguïtés d'octave)")
    print("    6. Interpolation parabolique sub-échantillon (précision < 0.1 Hz)")
    print("    7. Conversion chromatique MIDI & calcul d'écart en cents (n = 69 + 12*log2(f0/440))")
    print("=" * 75)


def demonstration_dft_cours():
    """
    Démonstration du calcul de la TFD sur l'exemple officiel du cours (exemple lecon.docx) :
      Entrée : x = [1, 6, 8, -4, 9] (N = 5)
      Formule : X[k] = sum_{n=0}^{N-1} x[n] * exp(-j * 2*pi * k * n / N)
    """
    print("\n" + "=" * 65)
    print("DÉMONSTRATION PÉDAGOGIQUE : CALCUL DE LA TFD (EXEMPLE DU COURS)")
    print("=" * 65)
    x = np.array([1, 6, 8, -4, 9], dtype=float)
    n = len(x)
    print(f"Signal d'entrée x (taille N = {n}) : {x.tolist()}\n")

    # 1. Calcul via la formule matricielle directe O(N^2)
    X_dft = dft_naive(x)

    # 2. Calcul via la Fast Fourier Transform numpy en O(N log N)
    X_fft = np.fft.fft(x)

    print("Résultats de la Transformée de Fourier Discrète :")
    for k in range(n):
        val_dft = X_dft[k]
        val_fft = X_fft[k]
        module = np.abs(val_dft)
        phase = np.angle(val_dft)
        print(f"  k = {k} : X[{k}] = {val_dft.real:+8.4f} {val_dft.imag:+8.4f}j "
              f"| Module = {module:7.4f} | Phase = {phase:+6.3f} rad")

    # Vérification pour k=0 : composante continue (somme des échantillons)
    somme_theorique = np.sum(x)
    print(f"\nVérification k = 0 (Composante continue) :")
    print(f"  X[0] = sum(x[n]) = {somme_theorique}  --> Calculé : {X_dft[0].real:.1f}")

    # Écart numérique entre TFD naïve et FFT
    ecart_max = float(np.max(np.abs(X_dft - X_fft)))
    print(f"\nÉcart maximal |X_TFD - X_FFT| : {ecart_max:.2e} (équivalence mathématique parfaite)")
    print("=" * 65 + "\n")


def regenerer_tout():
    """Exécute la chaîne complète de régénération des livrables."""
    print("\n>>> ÉTAPE 1/3 : Synthèse de la nouvelle mélodie (melodie.wav)...")
    generer_melodie("melodie.wav")

    print("\n>>> ÉTAPE 2/3 : Analyse spectrale et export des captures d'écran...")
    exporter_captures_melodie("melodie.wav")

    print("\n>>> ÉTAPE 3/3 : Compilation du document PDF de synthèse (document_pdf.pdf)...")
    generer_pdf("document_pdf.pdf")

    print("\n" + "=" * 65)
    print("RÉGÉNÉRATION COMPLÈTE TERMINÉE AVEC SUCCÈS !")
    print("Tous les livrables d'examen sont à jour :")
    print("  [x] melodie.wav (nouvelle mélodie avec notes accordées et désaccordée)")
    print("  [x] Captures d'écran PNG (spectres et cadrans d'accordage)")
    print("  [x] diagramme_spectre.png (spectre haute résolution de la note La4)")
    print("  [x] document_pdf.pdf (document officiel prêt à rendre)")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Point d'entrée de l'accordeur chromatique temps réel (DFT / FFT)."
    )
    parser.add_argument("--mic", action="store_true", help="Utiliser le microphone en temps réel.")
    parser.add_argument("--wav", type=str, default="melodie.wav", help="Fichier WAV à analyser (défaut: melodie.wav).")
    parser.add_argument("--demo", type=float, help="Fréquence d'un signal sinusoïdal de test (ex: 440).")
    parser.add_argument("--dft-cours", action="store_true", help="Exécute la démonstration du calcul de la TFD du cours.")
    parser.add_argument("--regenerer-tout", action="store_true", help="Régénère la mélodie, les captures et le PDF.")

    args = parser.parse_args()

    afficher_banniere_pedagogique()

    if args.dft-cours if hasattr(args, "dft-cours") else False:
        demonstration_dft_cours()
        return

    # Si l'utilisateur a passé --dft-cours
    if "--dft-cours" in sys.argv:
        demonstration_dft_cours()
        return

    if args.regenerer_tout:
        regenerer_tout()
        return

    # Par défaut : lancer l'accordeur
    print(f"\nLancement de l'accordeur chromatique...")
    if args.mic:
        print("Source sélectionnée : Microphone en direct.")
        sys.argv = ["tuner.py"]
    elif args.demo is not None:
        print(f"Source sélectionnée : Signal synthétique (f0 = {args.demo} Hz).")
        sys.argv = ["tuner.py", "--demo", str(args.demo)]
    else:
        if not os.path.isfile(args.wav):
            print(f"Fichier '{args.wav}' absent, génération automatique...")
            generer_melodie(args.wav)
        print(f"Source sélectionnée : Fichier audio '{args.wav}'.")
        sys.argv = ["tuner.py", "--wav", args.wav]

    tuner_main()


if __name__ == "__main__":
    main()

