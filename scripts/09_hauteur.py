"""Étape 9 : hauteur des arbres aujourd'hui (LiDAR HD de l'IGN, vols de 2021 à 2025).

La BD Forêt dit quelle essence pousse où, mais elle date : depuis, des parcelles ont été coupées
(scolytes de l'épicéa, tempêtes, coupes rases) et d'autres ont grandi. Le « modèle numérique de hauteur »
du LiDAR HD donne la hauteur réelle de la végétation au mètre près. On le demande à 20 m (le service de
l'IGN fait la moyenne), puis on résume chaque carré de 100 m :
- h_moy : hauteur moyenne (m x 4) ;
- part_haut : part du carré couverte d'arbres de 15 m et plus (%) : futaie adulte ;
- part_bas : part du carré à moins de 3 m (%) : coupe récente, jeune plantation, clairière, pré.

Source : service WMS-Raster de la Géoplateforme (gratuit, sans clé).
Résultat : data/hauteur.tif (255 = pas de mesure)
"""
import numpy as np

from commun import COLS, DATA, LIGNES, PAS, X0, Y1, ecrire_bandes, get, lire_bandes

WMS = "https://data.geopf.fr/wms-r/wms"
COUCHE = "IGNF_LIDAR-HD_MNH_ELEVATION.ELEVATIONGRIDCOVERAGE.LAMB93"
FACTEUR = 5                 # 5 x 5 mesures de 20 m par carré de 100 m
TUILE = 400                 # carrés par tuile : 2000 x 2000 points demandés à la fois
INCONNU = 255


def tuile(l0, c0, nl, nc):
    """Hauteurs (m) d'une tuile, à 20 m : tableau (nl x 5, nc x 5), NaN sans mesure."""
    x0, y1 = X0 + c0 * PAS, Y1 - l0 * PAS
    x1, y0 = x0 + nc * PAS, y1 - nl * PAS
    r = get(WMS, {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": COUCHE, "STYLES": "",
        "CRS": "EPSG:2154", "BBOX": f"{x0},{y0},{x1},{y1}", "WIDTH": nc * FACTEUR, "HEIGHT": nl * FACTEUR,
        "FORMAT": "image/x-bil;bits=32",
    }, timeout=300)
    a = np.frombuffer(r.content, dtype="<f4")
    if a.size != nl * FACTEUR * nc * FACTEUR:
        raise ValueError(f"réponse inattendue ({len(r.content)} octets)")
    a = a.reshape(nl * FACTEUR, nc * FACTEUR).astype("float32")
    return np.where((a < -100) | (a > 100), np.nan, np.clip(a, 0, 60))


def resumer(a, nl, nc):
    """Résumé par carré de 100 m : hauteur moyenne, part haute (>= 15 m), part basse (< 3 m), validité."""
    b = a.reshape(nl, FACTEUR, nc, FACTEUR)
    ok = ~np.isnan(b)
    n = ok.sum(axis=(1, 3))
    d = np.maximum(n, 1)
    moy = np.nansum(b, axis=(1, 3)) / d
    haut = ((b >= 15) & ok).sum(axis=(1, 3)) / d
    bas = ((b < 3) & ok).sum(axis=(1, 3)) / d
    return moy, haut, bas, n >= (FACTEUR * FACTEUR) // 2 + 1


def main():
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0
    h = np.full((LIGNES, COLS), INCONNU, dtype="uint8")
    haut, bas = h.copy(), h.copy()
    tuiles = [(l0, c0) for l0 in range(0, LIGNES, TUILE) for c0 in range(0, COLS, TUILE)
              if masque[l0:l0 + TUILE, c0:c0 + TUILE].any()]
    print(f"Hauteur des arbres (LiDAR HD) : {len(tuiles)} tuiles...")
    rates = 0
    for i, (l0, c0) in enumerate(tuiles, start=1):
        nl, nc = min(TUILE, LIGNES - l0), min(TUILE, COLS - c0)
        try:
            moy, ph, pb, ok = resumer(tuile(l0, c0, nl, nc), nl, nc)
        except Exception as e:          # une tuile manquante laisse « pas de mesure » : le score n'est pas pénalisé
            rates += 1
            print(f"  tuile {i}/{len(tuiles)} indisponible : {e}")
            continue
        sl = (slice(l0, l0 + nl), slice(c0, c0 + nc))
        h[sl] = np.where(ok, np.clip(np.rint(moy * 4), 0, 250), INCONNU)
        haut[sl] = np.where(ok, np.rint(ph * 100), INCONNU)
        bas[sl] = np.where(ok, np.rint(pb * 100), INCONNU)
        if i % 8 == 0 or i == len(tuiles):
            print(f"  tuile {i}/{len(tuiles)}")
    if rates > len(tuiles) // 2:
        raise SystemExit("Trop de tuiles indisponibles : hauteur des arbres non enregistrée.")
    for b in (h, haut, bas):
        b[~masque] = INCONNU
    connu = masque & (h != INCONNU)
    foret = lire_bandes(DATA / "occupation.tif")["foret"] >= 80
    f = connu & foret
    print(f"Mesure disponible sur {connu.sum() / masque.sum() * 100:.1f} % de la zone.")
    if f.any():
        hm = h[f] / 4
        print(f"En forêt (BD Forêt) : hauteur médiane {np.median(hm):.0f} m ; "
              f"{(hm < 5).mean() * 100:.1f} % des carrés sous 5 m (coupes, jeunes plantations), "
              f"{(hm >= 20).mean() * 100:.1f} % à 20 m et plus")
    ecrire_bandes(DATA / "hauteur.tif", {"h_moy": h, "part_haut": haut, "part_bas": bas}, "uint8")
    print("Hauteur des arbres enregistrée.")


if __name__ == "__main__":
    main()
