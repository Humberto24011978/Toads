"""Étape 2 : départements, forêts (BD Forêt v2) et occupation du sol (Corine Land Cover 2018).

Source : service WFS public de la Géoplateforme IGN, sans clé.
Les polygones sont rasterisés à 50 m, puis résumés en proportions par carré de 250 m.

Résultat : data/occupation.tif (une bande par catégorie, en % du carré)
"""
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from scipy import ndimage

from commun import (BBOX, COLS, CRS, DATA, DEPARTEMENTS, FIN, LIGNES, TRANSFORM, TRANSFORM_FIN,
                    ecrire_bandes, get)

WFS = "https://data.geopf.fr/wfs/ows"
COUCHE_DEP = "ADMINEXPRESS-COG-CARTO.LATEST:departement"
COUCHE_FORET = "LANDCOVER.FORESTINVENTORY.V2:formation_vegetale"
COUCHE_CLC = "LANDCOVER.CLC18_FR:clc18_fr"
PAGE = 5000
TUILE = 0.25  # degrés

# Essences de la BD Forêt regroupées en catégories utiles pour les champignons
CATEGORIES_FORET = ["chene", "hetre", "chataignier", "feuillus", "mixte", "sapin_epicea",
                    "pin", "douglas_meleze", "coniferes", "peuplier_robinier", "foret_autre", "lande"]


def categorie_essence(essence, code_tfv):
    e = (essence or "").strip().lower()
    code = (code_tfv or "").upper()
    if code.startswith("LA"):
        return "lande"
    if e.startswith("chêne"):
        return "chene"
    if e.startswith("hêtre"):
        return "hetre"
    if e.startswith("châtaignier"):
        return "chataignier"
    if e == "feuillus":
        return "feuillus"
    if e == "mixte":
        return "mixte"
    if "sapin" in e or "épicéa" in e:
        return "sapin_epicea"
    if e.startswith("pin"):
        return "pin"
    if e.startswith("douglas") or e.startswith("mélèze"):
        return "douglas_meleze"
    if e.startswith("conifère"):
        return "coniferes"
    if e.startswith("peuplier") or e.startswith("robinier"):
        return "peuplier_robinier"
    return "foret_autre"


# Codes Corine Land Cover regroupés
GROUPES_CLC = {
    "prairie": ["231"],
    "pelouse": ["321"],
    "agri_naturel": ["243"],
    "transition": ["324"],
    "lande_clc": ["322"],
    "cultures": ["211", "212", "213", "221", "222", "241", "242"],
    "espace_vert": ["141", "142"],
    "urbain": ["111", "112", "121", "122", "123", "124", "131", "132", "133"],
    "foret_clc": ["311", "312", "313"],
    "eau": ["511", "512", "411", "412"],
}


def wfs(couche, bbox, tri=None):
    """Télécharge une couche WFS sur une emprise (lon/lat), page par page, en Lambert-93."""
    morceaux, debut = [], 0
    while True:
        params = {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": couche, "outputFormat": "application/json", "srsName": CRS,
            # Avec le suffixe EPSG:4326, la Géoplateforme attend l'ordre lon,lat
            "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
            "count": PAGE, "startIndex": debut,
        }
        if tri:
            params["sortBy"] = tri
        r = get(WFS, params, timeout=300)
        gdf = gpd.read_file(BytesIO(r.content))
        if gdf.empty:
            break
        morceaux.append(gdf)
        if len(gdf) < PAGE:
            break
        debut += PAGE
    if not morceaux:
        return gpd.GeoDataFrame(geometry=[], crs=CRS)
    gdf = pd.concat(morceaux, ignore_index=True)
    return gpd.GeoDataFrame(gdf, geometry="geometry", crs=CRS)


def tuiles(emprise):
    lon1, lat1, lon2, lat2 = emprise
    for lon in np.arange(lon1, lon2, TUILE):
        for lat in np.arange(lat1, lat2, TUILE):
            yield (round(lon, 4), round(lat, 4), round(min(lon + TUILE, lon2), 4), round(min(lat + TUILE, lat2), 4))


def telecharger_par_tuiles(couche, emprise, nom):
    parties, n = [], 0
    liste = list(tuiles(emprise))
    for i, t in enumerate(liste, start=1):
        gdf = wfs(couche, t, tri="id")
        if not gdf.empty:
            parties.append(gdf)
            n += len(gdf)
        if i % 10 == 0 or i == len(liste):
            print(f"  {nom} : tuile {i}/{len(liste)}, {n} polygones")
    tout = pd.concat(parties, ignore_index=True)
    tout = tout.drop_duplicates(subset="id")
    return gpd.GeoDataFrame(tout, geometry="geometry", crs=CRS)


def departements():
    gdf = wfs(COUCHE_DEP, BBOX)
    gdf = gdf[gdf["code_insee"].astype(str).isin(DEPARTEMENTS)]
    trouves = sorted(gdf["code_insee"].astype(str))
    print(f"Départements trouvés : {trouves}")
    if len(trouves) != len(DEPARTEMENTS):
        raise SystemExit("Il manque des départements, arrêt.")
    return gdf


def rasteriser(gdf, valeurs, fin=True, dtype="uint8"):
    forme = (LIGNES * FIN, COLS * FIN) if fin else (LIGNES, COLS)
    transform = TRANSFORM_FIN if fin else TRANSFORM
    formes = ((g, int(v)) for g, v in zip(gdf.geometry, valeurs) if g is not None and not g.is_empty)
    return features.rasterize(formes, out_shape=forme, transform=transform, fill=0, dtype=dtype)


def proportion(masque_fin):
    """Part (0 à 1) des pixels de 50 m vrais dans chaque carré de 250 m."""
    m = masque_fin.reshape(LIGNES, FIN, COLS, FIN)
    return m.mean(axis=(1, 3))


def calculer(dep, foret, clc):
    """Transforme les polygones en proportions par carré de 250 m."""
    bandes = {}
    masque = rasteriser(dep, np.ones(len(dep)), fin=False) > 0
    bandes["masque"] = masque * 100

    cats = [categorie_essence(e, c) for e, c in zip(foret.get("essence", [None] * len(foret)),
                                                     foret.get("code_tfv", [None] * len(foret)))]
    codes = np.array([CATEGORIES_FORET.index(c) + 1 for c in cats], dtype="uint8")
    r_foret = rasteriser(foret, codes)
    for i, nom in enumerate(CATEGORIES_FORET, start=1):
        bandes[nom] = proportion(r_foret == i) * 100
    est_foret = (r_foret > 0) & (r_foret != CATEGORIES_FORET.index("lande") + 1)
    bandes["foret"] = proportion(est_foret) * 100
    del r_foret

    code_clc = pd.to_numeric(clc["code_18"], errors="coerce").fillna(0).astype(int).to_numpy()
    r_clc = rasteriser(clc, code_clc, dtype="uint16")
    for nom, liste in GROUPES_CLC.items():
        bandes[nom] = proportion(np.isin(r_clc, [int(c) for c in liste])) * 100

    # Lisières : milieux ouverts herbacés en contact avec la forêt (à 50 m près)
    ouvert = np.isin(r_clc, [231, 321, 243, 324, 322]) & ~est_foret
    pres_foret = ndimage.binary_dilation(est_foret, iterations=1)
    pres_ouvert = ndimage.binary_dilation(ouvert, iterations=1)
    lisiere = (ouvert & pres_foret) | (est_foret & pres_ouvert)
    bandes["lisiere"] = proportion(lisiere) * 100

    for nom in bandes:
        bandes[nom] = np.where(masque, np.rint(bandes[nom]), 0)
    return bandes


def main():
    DATA.mkdir(exist_ok=True)
    dep = departements()
    dep[["code_insee", "nom_officiel", "geometry"]].to_file(DATA / "departements.geojson", driver="GeoJSON")
    emprise = tuple(dep.to_crs(4326).total_bounds.round(3))

    print("Forêts (BD Forêt v2)...")
    foret = telecharger_par_tuiles(COUCHE_FORET, emprise, "forêts")
    print(f"  {len(foret)} polygones ; essences : {foret['essence'].value_counts().head(15).to_dict()}")

    print("Occupation du sol (Corine Land Cover 2018)...")
    clc = telecharger_par_tuiles(COUCHE_CLC, emprise, "Corine")
    print(f"  {len(clc)} polygones")

    bandes = calculer(dep, foret, clc)
    ecrire_bandes(DATA / "occupation.tif", bandes, "uint8")
    m = bandes["masque"] > 0
    print(f"Carrés dans la zone : {int(m.sum())}")
    print(f"Part de forêt : {bandes['foret'][m].mean():.1f} %, prairies : {bandes['prairie'][m].mean():.1f} %")


if __name__ == "__main__":
    main()
