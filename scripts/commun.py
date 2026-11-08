"""Réglages communs à tous les scripts : grille, espèces, téléchargements."""
import json
import time
from pathlib import Path

import numpy as np
import requests
from rasterio.transform import from_origin

RACINE = Path(__file__).resolve().parent.parent
DATA = RACINE / "data"
SITE = RACINE / "site"

# Départements couverts : Bas-Rhin, Haut-Rhin, Meurthe-et-Moselle, Meuse, Moselle, Vosges
DEPARTEMENTS = ["67", "68", "54", "55", "57", "88"]

# Emprise en degrés (lon min, lat min, lon max, lat max)
BBOX = (4.80, 47.40, 8.25, 49.65)

# Grille de calcul en Lambert-93 (EPSG:2154), carrés de 250 m
CRS = "EPSG:2154"
X0, X1 = 830_000, 1_095_000
Y0, Y1 = 6_700_000, 6_965_000
PAS = 250
COLS = (X1 - X0) // PAS          # 1060
LIGNES = (Y1 - Y0) // PAS        # 1060
TRANSFORM = from_origin(X0, Y1, PAS, PAS)

# Grille fine (50 m) pour les forêts et l'occupation du sol
FIN = 5
PAS_FIN = PAS // FIN
TRANSFORM_FIN = from_origin(X0, Y1, PAS_FIN, PAS_FIN)

ESPECES = {
    "cepe": {"nom": "Cèpe", "latin": ["Boletus edulis", "Boletus aereus", "Boletus reticulatus", "Boletus pinophilus"]},
    "girolle": {"nom": "Girolle", "latin": ["Cantharellus cibarius"]},
    "trompette": {"nom": "Trompette de la mort", "latin": ["Craterellus cornucopioides"]},
    "chanterelle_tube": {"nom": "Chanterelle en tube", "latin": ["Craterellus tubaeformis"]},
    "coulemelle": {"nom": "Coulemelle", "latin": ["Macrolepiota procera"]},
    "rose_des_pres": {"nom": "Rosé des prés", "latin": ["Agaricus campestris"]},
    "coprin": {"nom": "Coprin chevelu", "latin": ["Coprinus comatus"]},
    "pied_de_mouton": {"nom": "Pied de mouton", "latin": ["Hydnum repandum", "Hydnum rufescens"]},
}

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Toads/1.0"


def get(url, params=None, essais=6, timeout=180):
    """Requête GET avec nouvelles tentatives en cas d'erreur réseau ou de limite."""
    attente = 5
    for i in range(essais):
        try:
            r = SESSION.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.RequestException, requests.HTTPError) as e:
            if i == essais - 1:
                raise
            print(f"  nouvelle tentative dans {attente} s ({e})")
            time.sleep(attente)
            attente = min(attente * 2, 120)


def profil(nb_bandes, dtype, nodata=None):
    p = {
        "driver": "GTiff", "height": LIGNES, "width": COLS, "count": nb_bandes,
        "dtype": dtype, "crs": CRS, "transform": TRANSFORM,
        "compress": "deflate", "predictor": 2, "tiled": True,
    }
    if nodata is not None:
        p["nodata"] = nodata
    return p


def ecrire_bandes(chemin, bandes: dict, dtype, nodata=None):
    """Écrit un GeoTIFF multibande dont chaque bande porte un nom."""
    import rasterio
    noms = list(bandes)
    with rasterio.open(chemin, "w", **profil(len(noms), dtype, nodata)) as dst:
        for i, nom in enumerate(noms, start=1):
            dst.write(np.asarray(bandes[nom]).astype(dtype), i)
            dst.set_band_description(i, nom)


def lire_bandes(chemin):
    import rasterio
    with rasterio.open(chemin) as src:
        return {src.descriptions[i - 1]: src.read(i) for i in range(1, src.count + 1)}


def ecrire_json(chemin, obj, compact=None):
    compact = Path(chemin).name.startswith("secteurs_") if compact is None else compact
    texte = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) if compact else \
        json.dumps(obj, ensure_ascii=False, indent=1)
    Path(chemin).write_text(texte, encoding="utf-8")
