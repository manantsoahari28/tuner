#!/usr/bin/env python3
"""
Génération du document de synthèse PDF pour l'évaluation
=========================================================
Ce script crée le fichier 'document_pdf.pdf' contenant :
  - Les 5 captures d'écran de l'accordeur sur la nouvelle mélodie
  - Le tableau récapitulatif des résultats (Notes, Fréquences, Écarts en cents)
  - Le diagramme détaillé du spectre fréquentiel avec ses harmoniques (Note A4)
  - L'explication théorique et algorithmique de l'extraction de f0
  - La remarque méthodologique sur l'utilisation de melodie.wav
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

def generer_pdf(fichier_pdf="document_pdf.pdf"):
    largeur, hauteur = A4  # 595.27557 x 841.88977 points

    c = canvas.Canvas(fichier_pdf, pagesize=A4)
    c.setTitle("Accordeur chromatique - captures et spectre")

    # Fonction utilitaire pour convertir y_top (mesuré depuis le haut) en y_reportlab (mesuré depuis le bas)
    def y_top(y):
        return hauteur - y

    # ----------------------------------------------------------------------- #
    # 1. TITRE PRINCIPAL
    # ----------------------------------------------------------------------- #
    c.setFont("Helvetica-Bold", 15)
    c.setFillColor(colors.black)
    c.drawCentredString(largeur / 2.0, y_top(52), "Accordeur chromatique temps réel - Captures et spectre")

    # ----------------------------------------------------------------------- #
    # 2. SECTION 1 : CAPTURES D'ÉCRAN
    # ----------------------------------------------------------------------- #
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(42.85, y_top(72), "Captures d'écran")

    captures = [
        "capture_20261110_214207.png",
        "capture_20261110_214246.png",
        "capture_20261110_214302.png",
        "capture_20261110_214314.png",
        "capture_20261110_214328.png",
    ]
    spec_img = "diagramme_spectre.png"

    # Vérification et génération à la volée si les images ne sont pas sur le disque
    fichiers_generes_temporairement = []
    if not all(os.path.exists(f) for f in captures) or not os.path.exists(spec_img):
        print("[Info] Captures manquantes, génération automatique pour le PDF...")
        from tuner import exporter_captures_melodie
        exporter_captures_melodie("melodie.wav")
        fichiers_generes_temporairement = captures + [spec_img]

    infos_captures = [
        ("C4", " - 261,6 Hz - 0 cent"),
        ("D4", " - 293,7 Hz - 0 cent"),
        ("E4", " - 326,2 Hz - -18 cents"),
        ("G4", " - 392,0 Hz - 0 cent"),
        ("A4", " - 440,0 Hz - 0 cent"),
    ]

    img_w = 167.24
    img_h = 105.52

    # Ligne 1 : 3 captures
    xs_ligne1 = [38.27, 214.02, 389.76]
    y_img_ligne1_top = 87.02
    y_img_ligne1_rl = y_top(y_img_ligne1_top + img_h)

    for i in range(3):
        chemin = captures[i]
        x = xs_ligne1[i]
        if os.path.exists(chemin):
            c.drawImage(chemin, x, y_img_ligne1_rl, width=img_w, height=img_h)
        # Légende sous l'image
        c.setFont("Helvetica-Bold", 8)
        note_str, suite_str = infos_captures[i]
        larg_totale = c.stringWidth(note_str, "Helvetica-Bold", 8) + c.stringWidth(suite_str, "Helvetica", 8)
        x_texte = x + (img_w - larg_totale) / 2.0
        y_texte = y_top(204)
        c.drawString(x_texte, y_texte, note_str)
        c.setFont("Helvetica", 8)
        c.drawString(x_texte + c.stringWidth(note_str, "Helvetica-Bold", 8), y_texte, suite_str)

    # Ligne 2 : 2 captures + Tableau de résultats
    xs_ligne2 = [38.27, 214.02]
    y_img_ligne2_top = 217.54
    y_img_ligne2_rl = y_top(y_img_ligne2_top + img_h)

    for i in range(2):
        idx = 3 + i
        chemin = captures[idx]
        x = xs_ligne2[i]
        if os.path.exists(chemin):
            c.drawImage(chemin, x, y_img_ligne2_rl, width=img_w, height=img_h)
        c.setFont("Helvetica-Bold", 8)
        note_str, suite_str = infos_captures[idx]
        larg_totale = c.stringWidth(note_str, "Helvetica-Bold", 8) + c.stringWidth(suite_str, "Helvetica", 8)
        x_texte = x + (img_w - larg_totale) / 2.0
        y_texte = y_top(334)
        c.drawString(x_texte, y_texte, note_str)
        c.setFont("Helvetica", 8)
        c.drawString(x_texte + c.stringWidth(note_str, "Helvetica-Bold", 8), y_texte, suite_str)

    # Tableau de résultats à droite de la ligne 2
    # Titre "Résultats"
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.black)
    c.drawCentredString(473.38, y_top(224), "Résultats")

    tbl_x1 = 395.43
    tbl_x2 = 551.34
    tbl_w = tbl_x2 - tbl_x1
    col_w1 = 36.85
    col_w2 = 62.36
    col_w3 = 56.69
    row_h = 17.0
    tbl_y_top = 230.54

    # Remplissage en-tête bleu marine
    c.setFillColor(colors.Color(0.121569, 0.305882, 0.474510))
    c.rect(tbl_x1, y_top(tbl_y_top + row_h), tbl_w, row_h, fill=1, stroke=0)

    # Texte en-tête (blanc)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 8)
    c.drawCentredString(tbl_x1 + col_w1 / 2.0, y_top(tbl_y_top + 12), "Note")
    c.drawCentredString(tbl_x1 + col_w1 + col_w2 / 2.0, y_top(tbl_y_top + 12), "Fréquence")
    c.drawCentredString(tbl_x1 + col_w1 + col_w2 + col_w3 / 2.0, y_top(tbl_y_top + 12), "Écart")

    # Données du tableau
    lignes_tableau = [
        ("C4", "261,6 Hz", "0 cent"),
        ("D4", "293,7 Hz", "0 cent"),
        ("E4", "326,2 Hz", "-18 cents"),
        ("G4", "392,0 Hz", "0 cent"),
        ("A4", "440,0 Hz", "0 cent"),
    ]

    c.setFillColor(colors.black)
    for r_idx, (note, freq, ecart) in enumerate(lignes_tableau):
        curr_y_top = tbl_y_top + (r_idx + 1) * row_h
        y_text = y_top(curr_y_top + 12)
        c.drawCentredString(tbl_x1 + col_w1 / 2.0, y_text, note)
        c.drawCentredString(tbl_x1 + col_w1 + col_w2 / 2.0, y_text, freq)
        c.drawCentredString(tbl_x1 + col_w1 + col_w2 + col_w3 / 2.0, y_text, ecart)

    # Bordures du tableau
    c.setStrokeColor(colors.Color(0.501961, 0.501961, 0.501961))
    c.setLineWidth(0.6)
    # Lignes horizontales
    for r in range(7):
        y_line = y_top(tbl_y_top + r * row_h)
        c.line(tbl_x1, y_line, tbl_x2, y_line)
    # Lignes verticales
    c.line(tbl_x1, y_top(tbl_y_top), tbl_x1, y_top(tbl_y_top + 6 * row_h))
    c.line(tbl_x1 + col_w1, y_top(tbl_y_top), tbl_x1 + col_w1, y_top(tbl_y_top + 6 * row_h))
    c.line(tbl_x1 + col_w1 + col_w2, y_top(tbl_y_top), tbl_x1 + col_w1 + col_w2, y_top(tbl_y_top + 6 * row_h))
    c.line(tbl_x2, y_top(tbl_y_top), tbl_x2, y_top(tbl_y_top + 6 * row_h))

    # ----------------------------------------------------------------------- #
    # 3. SECTION 2 : DIAGRAMME DU SPECTRE FRÉQUENTIEL
    # ----------------------------------------------------------------------- #
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(42.85, y_top(358), "Diagramme du spectre fréquentiel (note A4)")

    spec_img = "diagramme_spectre.png"
    spec_w = 453.54
    spec_h = 206.63
    spec_x = 70.87
    spec_y_top = 371.06
    if os.path.exists(spec_img):
        c.drawImage(spec_img, spec_x, y_top(spec_y_top + spec_h), width=spec_w, height=spec_h)

    # Légende explicative du spectre
    c.setFont("Helvetica", 8)
    caption_spec = "Les pics sont les fréquences présentes : f0 = fondamentale (440 Hz), puis les harmoniques 2f0 (880 Hz), 3f0 (1320 Hz) et 4f0 (1760 Hz)."
    c.drawCentredString(largeur / 2.0, y_top(588), caption_spec)

    # ----------------------------------------------------------------------- #
    # 4. SECTION 3 : EXTRACTION DE LA FRÉQUENCE FONDAMENTALE
    # ----------------------------------------------------------------------- #
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(42.85, y_top(608), "Extraction de la fréquence fondamentale")

    # Texte pédagogique en 5 étapes
    # 1. Hann
    # 2. DFT / FFT
    # 3. HPS
    # 4. Interpolation parabolique
    # 5. Note MIDI et cents
    y_cours = 628
    interligne = 14.0

    # Ligne 1
    c.setFont("Helvetica-Bold", 10)
    c.drawString(42.85, y_top(y_cours), "1.")
    c.setFont("Helvetica", 10)
    c.drawString(53.0, y_top(y_cours), "On prend 8192 échantillons du son et on applique une fenêtre de Hann.")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(380.0, y_top(y_cours), "2.")
    c.setFont("Helvetica", 10)
    c.drawString(390.0, y_top(y_cours), "La DFT (calculée avec la FFT) donne")

    # Ligne 2
    y_cours += interligne
    c.drawString(42.85, y_top(y_cours), "le spectre : chaque pic est une fréquence présente.")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(285.0, y_top(y_cours), "3.")
    c.setFont("Helvetica", 10)
    c.drawString(295.0, y_top(y_cours), "Le pic le plus haut n'est pas toujours la fondamentale (ici,")

    # Ligne 3
    y_cours += interligne
    c.drawString(42.85, y_top(y_cours), "c'est le 2e, à 880 Hz). On utilise donc le ")
    w_avant = 42.85 + c.stringWidth("c'est le 2e, à 880 Hz). On utilise donc le ", "Helvetica", 10)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(w_avant, y_top(y_cours), "Harmonic Product Spectrum")
    w_apres = w_avant + c.stringWidth("Harmonic Product Spectrum", "Helvetica-Oblique", 10)
    c.setFont("Helvetica", 10)
    c.drawString(w_apres, y_top(y_cours), " : on multiplie le spectre par lui-même réduit 2,")

    # Ligne 4
    y_cours += interligne
    c.drawString(42.85, y_top(y_cours), "3 et 4 fois, ce qui fait ressortir la vraie fondamentale.")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(295.0, y_top(y_cours), "4.")
    c.setFont("Helvetica", 10)
    c.drawString(305.0, y_top(y_cours), "Une parabole ajustée autour du pic donne la fréquence")

    # Ligne 5
    y_cours += interligne
    c.drawString(42.85, y_top(y_cours), "précise.")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(85.0, y_top(y_cours), "5.")
    c.setFont("Helvetica", 10)
    c.drawString(95.0, y_top(y_cours), "La note vient de n = 69 + 12 x log2(f0/440), et l'écart en cents mesure la justesse (moins de 5 cents =")

    # Ligne 6
    y_cours += interligne
    c.drawString(42.85, y_top(y_cours), "note juste, zone verte).")

    # ----------------------------------------------------------------------- #
    # 5. REMARQUE FINALE
    # ----------------------------------------------------------------------- #
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(42.85, y_top(725), "Remarque : le micro de mon PC étant en panne, ces captures ont été faites avec le fichier audio melodie.wav (même traitement")
    c.drawString(42.85, y_top(737), "qu'avec le micro).")

    # Sauvegarde du PDF
    c.save()
    print(f"[Succès] Le fichier '{fichier_pdf}' a été généré avec succès !")

if __name__ == "__main__":
    generer_pdf()

