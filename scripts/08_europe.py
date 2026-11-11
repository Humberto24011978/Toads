"""Étape 8 : cartes européennes à 30 m (EcoDataCube, OpenGeoHub), ramenées aux carrés de 100 m.

Ces cartes couvrent la France et les pays voisins de la même façon, ce qui permet au modèle de l'étape 4
d'apprendre aussi sur les observations faites en Allemagne, en Suisse et au Luxembourg :
- essences d'arbres (probabilité de présence 2018-2020) : épicéa, sapin, pin sylvestre, chêne, hêtre, châtaignier.
  Elles séparent le sapin de l'épicéa, que la BD Forêt regroupe ;
- occupation du sol (probabilité par classe Corine, 2019) : feuillus, conifères, forêt mixte, prairies, cultures ;
- sol : pH et part de sable (0-20 cm), humidité liée au relief (indice topographique, « TWI »).

Fichiers ouverts à distance (lecture partielle), sans compte.
Résultats : data/europe_arbres.tif et data/europe_sol.tif (une bande par carte, valeurs de 0 à 250, 255 = inconnu)
"""
import os

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from commun import COLS, CRS, DATA, LIGNES, TRANSFORM, ecrire_bandes

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "6")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "1024")

SOURCE = "/vsicurl/https://s3.ecodatacube.eu/arco/"
ESSENCE = "veg_{}_anv.eml_p_30m_0..0cm_20180101_20201231_eumap_epsg3035_v0.3.tif"
CORINE = "lcv_landcover.{}_lucas.corine.eml_p_30m_0..0cm_20190101_20191231_eumap_epsg3035_v0.2.tif"
ARBRES = {
    "epicea": ESSENCE.format("picea.abies"), "sapin": ESSENCE.format("abies.alba"),
    "pin_sylvestre": ESSENCE.format("pinus.sylvestris"), "chene_p": ESSENCE.format("quercus.robur"),
    "hetre_p": ESSENCE.format("fagus.sylvatica"), "chataignier_p": ESSENCE.format("castanea.sativa"),
    "lc_feuillus": CORINE.format(311), "lc_coniferes": CORINE.format(312), "lc_mixte": CORINE.format(313),
}
SOL = {
    "lc_prairie": CORINE.format(231), "lc_cultures": CORINE.format(211),
    "twi": "twi_edtm_m_30m_s_20000101_20221231_eu_epsg.3035_v20241230.tif",
    "ph_edc": "ph.h2o_iso.10390.2021.index_m_30m_b0cm..20cm_20200101_20221231_eu_epsg.3035_v20250212.tif",
    "sable": "sand.tot_iso.11277.2020.wpct_m_30m_b0cm..20cm_20200101_20221231_eu_epsg.3035_v20250212.tif",
}
INCONNU = 255


def lire(nom, fichier):
    """Lit une carte à distance, la moyenne sur la grille de 100 m et la ramène entre 0 et 250."""
    with rasterio.open(SOURCE + fichier) as src:
        dst = np.full((LIGNES, COLS), np.nan, dtype="float32")
        reproject(rasterio.band(src, 1), dst, src_nodata=src.nodata, dst_transform=TRANSFORM, dst_crs=CRS,
                  dst_nodata=np.nan, resampling=Resampling.average)
        infos = f"{src.dtypes[0]}, sans donnée = {src.nodata}"
    ok = ~np.isnan(dst)
    if not ok.any():
        raise ValueError("aucune valeur dans l'emprise")
    q = np.percentile(dst[ok], [1, 50, 99])
    # Mise à l'échelle : on vise des valeurs entières entre 0 et 250 sans perdre de finesse
    proba = nom.startswith("lc_") or nom in ARBRES      # probabilités de 0 à 100 : gardées telles quelles
    echelle = 1.0
    if q[2] > 250:
        echelle = 250 / q[2]
    elif q[2] <= 25 and not proba:
        echelle = 10.0
    v = np.clip(np.rint(dst * echelle), 0, 250)
    if proba:
        v = np.where(ok, v, 0)                  # probabilité inconnue : 0
    elif (~ok).any():                           # sol : valeur la plus proche
        idx = ndimage.distance_transform_edt(~ok, return_distances=False, return_indices=True)
        v = v[tuple(idx)]
    print(f"  {nom} : {infos} ; 1 %, médiane, 99 % = {q[0]:.1f}, {q[1]:.1f}, {q[2]:.1f} ; échelle x{echelle:g} ; "
          f"{(~ok).mean() * 100:.1f} % sans donnée")
    return v.astype("uint8"), echelle


def groupe(couches, chemin):
    bandes, echelles = {}, {}
    for nom, fichier in couches.items():
        try:
            bandes[nom], echelles[nom] = lire(nom, fichier)
        except Exception as e:          # une carte manquante n'arrête pas le reste : le modèle fera sans
            print(f"  {nom} : indisponible ({e})")
    if bandes:
        ecrire_bandes(chemin, bandes, "uint8")
    return echelles


def main():
    print("Essences d'arbres et forêts (EcoDataCube, 30 m)...")
    e1 = groupe(ARBRES, DATA / "europe_arbres.tif")
    print("Sol, prairies et humidité (EcoDataCube, 30 m)...")
    e2 = groupe(SOL, DATA / "europe_sol.tif")
    print(f"Cartes européennes enregistrées : {len(e1) + len(e2)} sur {len(ARBRES) + len(SOL)}.")
    if not e1 and not e2:
        raise SystemExit("Aucune carte européenne disponible.")


if __name__ == "__main__":
    main()
