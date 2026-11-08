"""Étape quotidienne : météo des trois dernières semaines et cartes du jour.

Source : Open-Meteo (gratuit, sans clé).
Pour chaque espèce, un indice « c'est le moment » (saison, pluie, délai depuis
la dernière bonne pluie, température, gel) multiplie le score de potentiel.

Résultats :
- site/cartes/<espece>_jour.png : la carte du jour
- site/cartes/<espece>_potentiel.png : le potentiel toute saison
- site/cartes/info.json : date, emprise et état des conditions par espèce
"""
import datetime as dt
from zoneinfo import ZoneInfo

import ephem
import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from scipy import ndimage

from commun import (BBOX, COLS, CRS, DATA, ESPECES, LIGNES, PAS, SITE, TRANSFORM, X0, X1, Y0, Y1, ecrire_json, get,
                    lire_bandes)

API = "https://api.open-meteo.com/v1/forecast"
PAS_METEO = (0.15, 0.12)  # degrés (lon, lat) : environ 11 x 13 km

# Saison : 1 en pleine saison, valeur réduite en début ou fin de saison
SAISON = {
    "cepe":             {6: .4, 7: 1, 8: 1, 9: 1, 10: 1, 11: .4},
    "girolle":          {6: .5, 7: 1, 8: 1, 9: 1, 10: 1, 11: .3},
    "trompette":        {8: .5, 9: 1, 10: 1, 11: 1, 12: .3},
    "chanterelle_tube": {8: .3, 9: 1, 10: 1, 11: 1, 12: .8, 1: .3},
    "coulemelle":       {6: .4, 7: 1, 8: 1, 9: 1, 10: 1, 11: .3},
    "rose_des_pres":    {6: .4, 7: .6, 8: 1, 9: 1, 10: 1, 11: .4},
    "coprin":           {4: .4, 5: .5, 6: .4, 7: .4, 8: .5, 9: 1, 10: 1, 11: 1, 12: .3},
    "pied_de_mouton":   {7: .3, 8: .6, 9: 1, 10: 1, 11: 1, 12: .6},
}
TEMPERATURE = {  # moyenne sur 7 jours idéale (°C)
    "cepe": (10, 20), "girolle": (12, 22), "trompette": (8, 18), "chanterelle_tube": (4, 15),
    "coulemelle": (10, 20), "rose_des_pres": (10, 20), "coprin": (6, 18),
    "pied_de_mouton": (6, 18),
}
RESISTANCE_GEL = {"chanterelle_tube": .7, "trompette": .6, "pied_de_mouton": .6}


def dans(v, lo, hi, marge):
    d = np.maximum(lo - v, 0) + np.maximum(v - hi, 0)
    return np.clip(1 - d / marge, 0, 1)


def points_meteo():
    lons = np.arange(BBOX[0], BBOX[2] + 1e-9, PAS_METEO[0])
    lats = np.arange(BBOX[3], BBOX[1] - 1e-9, -PAS_METEO[1])  # du nord au sud
    return lons, lats


def meteo():
    lons, lats = points_meteo()
    pts = [(lo, la) for la in lats for lo in lons]
    pluie, tmoy, tmin = [], [], []
    for i in range(0, len(pts), 100):
        lot = pts[i:i + 100]
        data = get(API, {
            "longitude": ",".join(f"{p[0]:.3f}" for p in lot),
            "latitude": ",".join(f"{p[1]:.3f}" for p in lot),
            "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_min",
            "past_days": 21, "forecast_days": 1, "timezone": "Europe/Paris",
        }).json()
        data = data if isinstance(data, list) else [data]
        for d in data:
            j = d["daily"]
            pluie.append(np.array(j["precipitation_sum"], dtype=float))
            tmoy.append(np.array(j["temperature_2m_mean"], dtype=float))
            tmin.append(np.array(j["temperature_2m_min"], dtype=float))
    forme = (len(lats), len(lons))
    return {"pluie": np.nan_to_num(np.array(pluie)), "tmoy": np.array(tmoy), "tmin": np.array(tmin), "forme": forme}


def resume_meteo(m):
    """Résume la météo de chaque point (aujourd'hui = dernier jour de la série)."""
    pluie = m["pluie"]
    passe = pluie[:, :-1]                                   # jours écoulés, sans aujourd'hui
    pluie14 = passe[:, -14:].sum(axis=1)
    # Bonne pluie : au moins 8 mm sur une journée, ou 15 mm sur deux jours
    deux = passe[:, 1:] + passe[:, :-1]
    bonne = (passe[:, 1:] >= 8) | (deux >= 15)
    jours = np.full(len(pluie), 99.0)
    for i in range(len(pluie)):
        idx = np.flatnonzero(bonne[i])
        if len(idx):
            jours[i] = bonne.shape[1] - idx[-1]
    tmoy7 = np.nanmean(m["tmoy"][:, -8:-1], axis=1)
    gel = np.nanmin(m["tmin"][:, -6:], axis=1) < -2
    return {"pluie14": pluie14, "jours": jours, "tmoy7": tmoy7, "gel": gel}


def indice(espece, r, mois):
    f_saison = SAISON[espece].get(mois, 0)
    if f_saison == 0:
        return np.zeros(len(r["pluie14"]))
    f_pluie = np.clip(r["pluie14"] / 40, 0, 1)
    f_delai = dans(r["jours"], 5, 20, 8)
    f_temp = dans(r["tmoy7"], *TEMPERATURE[espece], 6)
    f_gel = np.where(r["gel"], RESISTANCE_GEL.get(espece, .3), 1)
    return f_saison * f_gel * f_temp * (0.5 * f_pluie + 0.5 * f_delai)


def vers_grille(valeurs, forme):
    """Interpole un champ météo (grille en degrés) sur la grille Lambert-93 de 250 m."""
    lons, lats = points_meteo()
    src = valeurs.reshape(forme).astype("float32")
    t = from_origin(lons[0] - PAS_METEO[0] / 2, lats[0] + PAS_METEO[1] / 2, *PAS_METEO)
    dst = np.zeros((LIGNES, COLS), dtype="float32")
    reproject(src, dst, src_transform=t, src_crs="EPSG:4326", dst_transform=TRANSFORM, dst_crs=CRS,
              resampling=Resampling.bilinear)
    return dst


# Palette : jaune pâle -> orange -> rouge -> prune ; transparent sous le seuil
PALETTE = [(12, (246, 226, 122, 120)), (40, (244, 163, 64, 170)), (70, (217, 69, 43, 205)), (100, (110, 26, 70, 230))]
SEUIL = 12


def table_couleurs():
    lut = np.zeros((256, 4), dtype=np.uint8)
    for v in range(SEUIL, 256):
        x = min(v, 100)
        for (a, ca), (b, cb) in zip(PALETTE[:-1], PALETTE[1:]):
            if a <= x <= b:
                t = (x - a) / (b - a)
                lut[v] = [round(ca[k] + t * (cb[k] - ca[k])) for k in range(4)]
                break
    return lut


def preparer_mercator():
    """Grille Web Mercator (celle des cartes en ligne) couvrant la zone."""
    t, w, h = calculate_default_transform(CRS, "EPSG:3857", COLS, LIGNES, left=X0, bottom=Y0, right=X1, top=Y1,
                                          resolution=340)
    gauche, haut = t.c, t.f
    droite, bas = gauche + w * t.a, haut + h * t.e
    ouest, sud, est, nord = transform_bounds("EPSG:3857", "EPSG:4326", gauche, bas, droite, haut)
    return t, w, h, [[sud, ouest], [nord, est]]


def image(grille, t, w, h, lut, chemin):
    dst = np.zeros((h, w), dtype="float32")
    reproject(grille.astype("float32"), dst, src_transform=TRANSFORM, src_crs=CRS, src_nodata=0,
              dst_transform=t, dst_crs="EPSG:3857", dst_nodata=0, resampling=Resampling.bilinear)
    Image.fromarray(lut[np.clip(np.rint(dst), 0, 255).astype(np.uint8)], "RGBA").save(chemin, optimize=True)


def secteurs(grille, seuil, rayon=3):
    """Meilleurs secteurs d'une carte : moyenne sur 750 m, puis un seul point par voisinage de ~1,75 km.

    Renvoie [[lat, lon, score], ...] du meilleur au moins bon.
    """
    lisse = ndimage.uniform_filter(grille.astype("float32"), size=3)
    lig, col = np.nonzero(lisse >= seuil)
    ordre = np.argsort(-lisse[lig, col], kind="stable")
    pris = np.zeros(lisse.shape, dtype=bool)
    choisis = []
    for i in ordre:
        li, c = lig[i], col[i]
        if pris[li, c]:
            continue
        choisis.append(i)
        pris[max(li - rayon, 0):li + rayon + 1, max(c - rayon, 0):c + rayon + 1] = True
    choisis = np.array(choisis, dtype=int)
    if len(choisis) == 0:
        return []
    x = X0 + (col[choisis] + 0.5) * PAS
    y = Y1 - (lig[choisis] + 0.5) * PAS
    lon, lat = Transformer.from_crs(CRS, 4326, always_xy=True).transform(x, y)
    valeurs = lisse[lig[choisis], col[choisis]]
    return [[round(float(a), 4), round(float(o), 4), int(round(float(v)))] for a, o, v in zip(lat, lon, valeurs)]


PHASES = [("Nouvelle lune", "🌑"), ("Premier croissant", "🌒"), ("Premier quartier", "🌓"),
          ("Gibbeuse croissante", "🌔"), ("Pleine lune", "🌕"), ("Gibbeuse décroissante", "🌖"),
          ("Dernier quartier", "🌗"), ("Dernier croissant", "🌘")]


def lune(jour):
    """Phase de la lune à midi (heure de Paris) et dates des prochaines nouvelle et pleine lunes."""
    paris = ZoneInfo("Europe/Paris")
    midi = dt.datetime(jour.year, jour.month, jour.day, 12, tzinfo=paris).astimezone(dt.timezone.utc)
    d = ephem.Date(midi.replace(tzinfo=None))
    avant, apres = ephem.previous_new_moon(d), ephem.next_new_moon(d)
    fraction = (d - avant) / (apres - avant)
    nom, emoji = PHASES[int(round(fraction * 8)) % 8]
    date_locale = lambda e: e.datetime().replace(tzinfo=dt.timezone.utc).astimezone(paris).date().isoformat()
    return {"phase": nom, "emoji": emoji, "illumination": round(ephem.Moon(d).phase),
            "croissante": ephem.next_full_moon(d) < apres,
            "prochaine_nouvelle": date_locale(apres), "prochaine_pleine": date_locale(ephem.next_full_moon(d))}


def statut(valeur, f_saison):
    if f_saison == 0:
        return "hors saison"
    if valeur >= 0.5:
        return "bon moment"
    if valeur >= 0.25:
        return "conditions moyennes"
    return "peu probable en ce moment"


def main():
    if not (DATA / "scores.tif").exists():
        print("Pas encore de scores : lancez d'abord la préparation des données.")
        return
    scores = lire_bandes(DATA / "scores.tif")
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0
    aujourdhui = dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    mois = aujourdhui.month
    print(f"Météo du {aujourdhui}...")
    m = meteo()
    r = resume_meteo(m)

    # Points météo situés dans la zone, pour l'état régional
    lons, lats = points_meteo()
    lon_g, lat_g = np.meshgrid(lons, lats)
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(lon_g.ravel(), lat_g.ravel())
    c = ((x - X0) // PAS).astype(int)
    li = ((Y1 - y) // PAS).astype(int)
    ok = (c >= 0) & (c < COLS) & (li >= 0) & (li < LIGNES)
    dans_zone = np.zeros(lon_g.size, dtype=bool)
    dans_zone[ok] = masque[li[ok], c[ok]]
    if not dans_zone.any():
        dans_zone[:] = True

    sortie = SITE / "cartes"
    sortie.mkdir(parents=True, exist_ok=True)
    t, w, h, bornes = preparer_mercator()
    lut = table_couleurs()
    infos = {}
    for espece, nom in ((k, v["nom"]) for k, v in ESPECES.items()):
        if espece not in scores:
            print(f"{nom} : pas encore de score, ignorée")
            continue
        ind = indice(espece, r, mois)
        ind_grille = np.clip(vers_grille(ind, m["forme"]), 0, 1)
        jour = scores[espece].astype("float32") * ind_grille
        image(jour, t, w, h, lut, sortie / f"{espece}_jour.png")
        image(scores[espece], t, w, h, lut, sortie / f"{espece}_potentiel.png")
        ecrire_json(sortie / f"secteurs_{espece}.json", {
            "jour": secteurs(jour, 15), "potentiel": secteurs(scores[espece], 35)})
        regional = float(np.mean(ind[dans_zone]))
        meilleur = float(np.percentile(ind[dans_zone], 90))
        infos[espece] = {"nom": nom, "indice": round(regional * 100), "indice_meilleur": round(meilleur * 100),
                         "statut": statut(meilleur, SAISON[espece].get(mois, 0)),
                         "saison": SAISON[espece].get(mois, 0) > 0}
        print(f"{nom} : indice {infos[espece]['indice']} (meilleurs secteurs {infos[espece]['indice_meilleur']}) "
              f"-> {infos[espece]['statut']}")

    ecrire_json(sortie / "info.json", {
        "date": aujourdhui.isoformat(),
        "bornes": bornes,
        "meteo": {
            "pluie_14j_mm": round(float(np.mean(r["pluie14"][dans_zone])), 1),
            "temperature_7j": round(float(np.nanmean(r["tmoy7"][dans_zone])), 1),
            "jours_depuis_bonne_pluie": int(np.median(r["jours"][dans_zone])),
            "gel_recent": bool(np.mean(r["gel"][dans_zone]) > 0.3),
        },
        "lune": lune(aujourdhui),
        "especes": infos,
    })
    print("Cartes du jour prêtes.")


if __name__ == "__main__":
    main()
