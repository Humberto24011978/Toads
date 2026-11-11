"""Étape 10 : où a-t-on le droit de cueillir ? Forêts publiques, zones protégées, terrains militaires.

- Forêts publiques (BD TOPO de l'IGN, d'après l'ONF) : forêts domaniales (État) et autres forêts publiques
  (surtout communales). La cueillette familiale y est en général tolérée ; ailleurs, la forêt est privée et
  les champignons appartiennent au propriétaire.
- Zones exclues par prudence : réserves naturelles, réserves biologiques, arrêtés de protection de biotope,
  sites des conservatoires d'espaces naturels (cueillette interdite ou réglementée selon les lieux),
  camps et enceintes militaires.

Source : service WFS de la Géoplateforme (gratuit, sans clé).
Résultat : data/acces.tif — publique (0 non, 1 forêt domaniale, 2 autre forêt publique), interdit (0 ou 1)
"""
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features

from commun import BBOX, COLS, CRS, DATA, LIGNES, TRANSFORM, ecrire_bandes, get, lire_bandes

WFS = "https://data.geopf.fr/wfs/ows"
PAGE = 1000
PROTEGE = ["Réserve naturelle", "Réserve biologique", "Arrêté de protection",
           "Site acquis ou assimilé des conservatoires d'espaces naturels"]
MILITAIRE = ["Camp militaire non clos", "Enceinte militaire", "Champ de tir"]


def wfs(couche, filtre=None, champs=("nature", "toponyme")):
    """Télécharge une couche de la BD TOPO sur l'emprise, page par page, en Lambert-93."""
    cql = f"BBOX(geometrie,{BBOX[0]},{BBOX[1]},{BBOX[2]},{BBOX[3]},'EPSG:4326')"
    if filtre:
        cql = f"{filtre} AND {cql}"
    morceaux, debut = [], 0
    while True:
        r = get(WFS, {
            "service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeNames": couche,
            "outputFormat": "application/json", "srsName": CRS, "cql_filter": cql,
            "propertyName": ",".join(champs + ("geometrie",)), "sortBy": "cleabs",
            "count": PAGE, "startIndex": debut,
        }, timeout=600)
        gdf = gpd.read_file(BytesIO(r.content))
        if gdf.empty:
            break
        morceaux.append(gdf)
        if len(gdf) < PAGE:
            break
        debut += PAGE
    if not morceaux:
        return gpd.GeoDataFrame({"nature": []}, geometry=[], crs=CRS)
    return gpd.GeoDataFrame(pd.concat(morceaux, ignore_index=True), geometry="geometry", crs=CRS)


def liste(valeurs):
    return "(" + ",".join("'" + v.replace("'", "''") + "'" for v in valeurs) + ")"


def rasteriser(gdf, valeurs):
    formes = [(g, int(v)) for g, v in zip(gdf.geometry, valeurs) if g is not None and not g.is_empty]
    if not formes:
        return np.zeros((LIGNES, COLS), dtype="uint8")
    return features.rasterize(formes, out_shape=(LIGNES, COLS), transform=TRANSFORM, fill=0, dtype="uint8")


def main():
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0

    print("Forêts publiques (BD TOPO)...")
    fp = wfs("BDTOPO_V3:foret_publique")
    domaniale = fp["nature"].astype(str).str.contains("domaniale", case=False)
    # Les forêts domaniales sont dessinées en dernier : elles l'emportent en cas de recouvrement
    fp = pd.concat([fp[~domaniale], fp[domaniale]])
    publique = rasteriser(fp, np.where(fp["nature"].astype(str).str.contains("domaniale", case=False), 1, 2))
    print(f"  {len(fp)} forêts publiques, dont {int(domaniale.sum())} domaniales")

    print("Zones protégées et terrains militaires (BD TOPO)...")
    interdit = np.zeros((LIGNES, COLS), dtype="uint8")
    try:
        prot = wfs("BDTOPO_V3:parc_ou_reserve", f"nature IN {liste(PROTEGE)}")
        interdit |= rasteriser(prot, np.ones(len(prot)))
        print(f"  {len(prot)} zones protégées : {prot['nature'].value_counts().to_dict()}")
    except Exception as e:
        print(f"  zones protégées indisponibles : {e}")
    try:
        mil = wfs("BDTOPO_V3:zone_d_activite_ou_d_interet",
                  f"categorie='Administratif ou militaire' AND nature IN {liste(MILITAIRE)}")
        interdit |= rasteriser(mil, np.ones(len(mil)))
        print(f"  {len(mil)} terrains militaires : {mil['nature'].value_counts().to_dict()}")
    except Exception as e:
        print(f"  terrains militaires indisponibles : {e}")

    publique[~masque] = 0
    interdit[~masque] = 0
    foret = lire_bandes(DATA / "occupation.tif")["foret"] >= 50
    f = masque & foret
    print(f"Forêt de la zone : {((publique > 0) & f).sum() / f.sum() * 100:.0f} % publique "
          f"({((publique == 1) & f).sum() / f.sum() * 100:.0f} % domaniale), "
          f"{(interdit > 0)[f].mean() * 100:.1f} % en zone exclue")
    ecrire_bandes(DATA / "acces.tif", {"publique": publique, "interdit": interdit}, "uint8")
    print("Accès enregistré.")


if __name__ == "__main__":
    main()
