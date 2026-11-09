"""Étape 7 : réserve utile du sol (mm d'eau que le sol peut retenir pour les racines et le mycélium).

Un sol sableux ou caillouteux, mince ou sur une crête, se vide bien plus vite qu'un sol limoneux profond
au fond d'un vallon. Estimation pour chaque carré de 100 m :
- texture (sable, argile) et cailloux entre 15 et 60 cm de profondeur : SoilGrids (ISRIC, 250 m) ;
- eau utile par cm de sol selon la texture (ordres de grandeur des tables françaises de réserve utile) :
  sable 0,7 mm/cm, sable limoneux 1,0, limon sableux 1,35, limon 1,7, sol argileux 1,5 ;
- profondeur explorée de 60 cm, réduite sur les fortes pentes (sols plus minces) ;
- position : un creux reçoit l'eau qui ruisselle (+), une crête la perd (-).

Résultat : data/reserve_utile.tif (mm, entier)
"""
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from commun import BBOX, COLS, CRS, DATA, LIGNES, TRANSFORM, ecrire_bandes, get, lire_bandes

SOILGRIDS = "https://maps.isric.org/mapserv"
PROFONDEURS = ["15-30cm", "30-60cm"]
PROFONDEUR_CM = 60


def couche(variable, profondeur):
    """Une couche SoilGrids reprojetée sur la grille de 100 m (valeurs brutes, trous comblés)."""
    crs = "http://www.opengis.net/def/crs/EPSG/0/4326"
    params = {
        "map": f"/map/{variable}.map", "SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCoverage",
        "COVERAGEID": f"{variable}_{profondeur}_mean", "FORMAT": "image/tiff",
        "SUBSET": [f"long({BBOX[0]},{BBOX[2]})", f"lat({BBOX[1]},{BBOX[3]})"],
        "SUBSETTINGCRS": crs, "OUTPUTCRS": crs,
    }
    chemin = DATA / f"{variable}_{profondeur}.tif"
    chemin.write_bytes(get(SOILGRIDS, params, timeout=600).content)
    sortie = np.full((LIGNES, COLS), np.nan, dtype="float32")
    with rasterio.open(chemin) as src:
        brut = src.read(1).astype("float32")
        brut[(brut < 0) | (brut > 1000)] = np.nan
        if variable != "cfvo":
            brut[brut == 0] = np.nan                     # 0 : pas de donnée (eau, ville)
        reproject(brut, sortie, src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                  dst_transform=TRANSFORM, dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.bilinear)
    chemin.unlink()
    vide = np.isnan(sortie)
    if vide.any() and (~vide).any():
        idx = ndimage.distance_transform_edt(vide, return_distances=False, return_indices=True)
        sortie = sortie[tuple(idx)]
    return sortie / 10                                   # g/kg -> % ; cm3/dm3 -> %


def eau_par_cm(sable, argile):
    """Eau utile (mm par cm de sol) selon la texture."""
    v = np.interp(sable, [15, 30, 50, 70, 85], [1.8, 1.65, 1.35, 1.0, 0.7])
    return np.where(argile > 45, np.minimum(v, 1.5), v)


def main():
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0
    relief = lire_bandes(DATA / "relief_sol.tif")
    pente = relief["pente"].astype("float32") / 10                 # degrés
    tpi = relief["tpi"].astype("float32")                          # m : négatif = creux
    ru = np.zeros((LIGNES, COLS), dtype="float32")
    for p in PROFONDEURS:
        print(f"Sol entre {p} (SoilGrids : sable, argile, cailloux)...")
        sable, argile, cailloux = couche("sand", p), couche("clay", p), couche("cfvo", p)
        print(f"  sable {np.nanmedian(sable[masque]):.0f} %, argile {np.nanmedian(argile[masque]):.0f} %, "
              f"cailloux {np.nanmedian(cailloux[masque]):.0f} % (médianes)")
        ru += eau_par_cm(sable, argile) * (1 - np.clip(cailloux, 0, 80) / 100) / len(PROFONDEURS)
    profondeur = PROFONDEUR_CM * (1 - 0.5 * np.clip((pente - 10) / 25, 0, 1))
    position = np.clip(1 - tpi / 100, 0.7, 1.3)
    ru = np.clip(ru * profondeur * position, 15, 250)
    ru = np.where(masque, np.rint(ru), 0)
    q = np.percentile(ru[masque], [10, 50, 90])
    print(f"Réserve utile : {q[0]:.0f} mm (10 %), {q[1]:.0f} mm (médiane), {q[2]:.0f} mm (90 %)")
    ecrire_bandes(DATA / "reserve_utile.tif", {"ru": ru}, "int16")
    print("Réserve utile enregistrée.")


if __name__ == "__main__":
    main()
