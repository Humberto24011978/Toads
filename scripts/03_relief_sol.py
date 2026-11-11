"""Étape 3 : relief (altitude, pente, exposition) et acidité du sol.

Sources gratuites, sans clé :
- altitude : modèle numérique Copernicus GLO-30 (30 m), hébergé en accès libre sur AWS ;
- pH du sol (0-5 cm) : SoilGrids (ISRIC), maille de 250 m.

Résultat : data/relief_sol.tif (altitude, pente, exposition, creux/crêtes, pH), carrés de 100 m,
sur toute la grille (France et pays voisins)
"""
import os

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from commun import BBOX, COLS, CRS, DATA, LIGNES, PAS, TRANSFORM, ecrire_bandes, get, lire_bandes

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "3")

DEM = "https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"
SOILGRIDS = "https://maps.isric.org/mapserv"


def nom_tuile(lat, lon):
    return f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"


def altitude():
    alt = np.full((LIGNES, COLS), np.nan, dtype="float32")
    for lat in range(int(BBOX[1]), int(np.ceil(BBOX[3]))):
        for lon in range(int(BBOX[0]), int(np.ceil(BBOX[2]))):
            n = nom_tuile(lat, lon)
            print(f"  tuile {n}")
            with rasterio.open("/vsicurl/" + DEM.format(n=n)) as src:
                morceau = np.full((LIGNES, COLS), np.nan, dtype="float32")
                reproject(rasterio.band(src, 1), morceau, dst_transform=TRANSFORM, dst_crs=CRS,
                          dst_nodata=np.nan, resampling=Resampling.average)
            alt = np.where(np.isnan(alt), morceau, alt)
    return alt


def pente_exposition(alt):
    z = np.where(np.isnan(alt), np.nanmean(alt), alt)
    dz_ligne = np.gradient(z, axis=0) / PAS     # positif vers le sud
    dz_est = np.gradient(z, axis=1) / PAS
    dz_nord = -dz_ligne
    pente = np.degrees(np.arctan(np.hypot(dz_est, dz_nord)))
    # Direction vers laquelle la pente est tournée (0 = nord, 90 = est)
    exposition = (np.degrees(np.arctan2(-dz_est, -dz_nord)) + 360) % 360
    nordicite = np.cos(np.radians(exposition)) * np.clip(pente / 5, 0, 1)  # 0 sur terrain plat
    return pente, nordicite


def ph_sol():
    crs = "http://www.opengis.net/def/crs/EPSG/0/4326"
    params = {
        "map": "/map/phh2o.map", "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage",
        "COVERAGEID": "phh2o_0-5cm_mean", "FORMAT": "image/tiff",
        "SUBSET": [f"long({BBOX[0]},{BBOX[2]})", f"lat({BBOX[1]},{BBOX[3]})"],
        "SUBSETTINGCRS": crs, "OUTPUTCRS": crs,
    }
    chemin = DATA / "ph_soilgrids.tif"
    chemin.write_bytes(get(SOILGRIDS, params, timeout=600).content)
    ph = np.full((LIGNES, COLS), np.nan, dtype="float32")
    with rasterio.open(chemin) as src:
        brut = src.read(1).astype("float32")
        brut[(brut <= 0) | (brut > 140)] = np.nan       # pas de donnée
        reproject(brut, ph, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=TRANSFORM, dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.bilinear)
    chemin.unlink()
    ph = ph / 10                                         # SoilGrids stocke pH x 10
    # Les zones urbaines et l'eau n'ont pas de valeur : on prend la plus proche
    vide = np.isnan(ph)
    if vide.any() and (~vide).any():
        idx = ndimage.distance_transform_edt(vide, return_distances=False, return_indices=True)
        ph = ph[tuple(idx)]
    return ph


def position_topographique(alt, rayon_m=300):
    """Altitude du carré moins l'altitude moyenne alentour : négatif = creux, vallon ; positif = crête, butte."""
    z = np.where(np.isnan(alt), np.nanmean(alt), alt)
    taille = 2 * int(rayon_m // PAS) + 1
    return z - ndimage.uniform_filter(z, size=taille)


def calculer(alt, ph):
    """Bandes de relief et de sol sur toute la grille (France et pays voisins) : le modèle de l'étape 4
    apprend aussi sur les observations faites de l'autre côté de la frontière."""
    pente, nordicite = pente_exposition(alt)
    tpi = position_topographique(alt)
    tpi_large = position_topographique(alt, rayon_m=1000)
    alt = np.where(np.isnan(alt), 0, alt)
    return {
        "altitude": np.rint(alt),
        "pente": np.rint(pente * 10),               # dixièmes de degré
        "nordicite": np.rint(nordicite * 100),      # -100 sud, +100 nord
        "tpi": np.rint(tpi),                        # m, sur 300 m : négatif = creux, positif = crête
        "tpi_large": np.rint(tpi_large),            # m, sur 1 km : fond de vallée ou sommet
        "ph": np.rint(ph * 10),                     # pH x 10
    }


def main():
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0
    print("Altitude (Copernicus GLO-30)...")
    alt = altitude()
    print(f"  altitude dans la zone : {np.nanmin(alt[masque]):.0f} à {np.nanmax(alt[masque]):.0f} m")
    print("pH du sol (SoilGrids)...")
    ph = ph_sol()
    print(f"  pH dans la zone : {np.nanpercentile(ph[masque], 5):.1f} à {np.nanpercentile(ph[masque], 95):.1f}")
    bandes = calculer(alt, ph)
    ecrire_bandes(DATA / "relief_sol.tif", bandes, "int16")
    print("Relief et sol enregistrés.")


if __name__ == "__main__":
    main()
