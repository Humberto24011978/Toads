"""Étape quotidienne : météo des trois dernières semaines et cartes du jour.

Source : Open-Meteo (gratuit, sans clé) : pluie, température de l'air et du sol sur 30 jours.
L'indice de chaque espèce (voir indice_meteo.py) est calculé à plusieurs altitudes,
puis chaque carré de 250 m prend la valeur de son altitude. Il multiplie le score de potentiel.

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

import indice_meteo as im
from commun import (BBOX, COLS, CRS, DATA, ESPECES, LIGNES, PAS, SITE, TRANSFORM, X0, X1, Y0, Y1, ecrire_json, get,
                    lire_bandes)

API = "https://api.open-meteo.com/v1/forecast"
PAS_METEO = (0.15, 0.12)  # degrés (lon, lat) : environ 11 x 13 km
JOURS_PASSES = 30
# Baisse de température avec l'altitude (°C par mètre)
GRADIENT_AIR, GRADIENT_SOL = 0.0065, 0.0055
# Altitudes de calcul : l'indice est calculé à chaque niveau, puis chaque carré prend celui de son altitude
NIVEAUX = np.arange(100, 1501, 100)
# Points de contrôle (trouvailles réelles) : leur indice est affiché dans le journal de calcul
CONTROLES = {"Sandweg, Oberhaslach (cèpes le 7 novembre 2026)": (48.556, 7.315)}


def points_meteo():
    lons = np.arange(BBOX[0], BBOX[2] + 1e-9, PAS_METEO[0])
    lats = np.arange(BBOX[3], BBOX[1] - 1e-9, -PAS_METEO[1])  # du nord au sud
    return lons, lats


def meteo():
    lons, lats = points_meteo()
    pts = [(lo, la) for la in lats for lo in lons]
    pluie, tmoy, tmin, tsol, alt = [], [], [], [], []
    for i in range(0, len(pts), 100):
        lot = pts[i:i + 100]
        data = get(API, {
            "longitude": ",".join(f"{p[0]:.3f}" for p in lot),
            "latitude": ",".join(f"{p[1]:.3f}" for p in lot),
            # Même variable de sol que les archives utilisées pour l'apprentissage (06_phenologie.py)
            "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_min,soil_temperature_0_to_7cm_mean",
            "past_days": JOURS_PASSES, "forecast_days": 1, "timezone": "Europe/Paris",
        }).json()
        data = data if isinstance(data, list) else [data]
        for d in data:
            j = d["daily"]
            pluie.append(np.array(j["precipitation_sum"], dtype=float))
            tmoy.append(np.array(j["temperature_2m_mean"], dtype=float))
            tmin.append(np.array(j["temperature_2m_min"], dtype=float))
            tsol.append(np.array(j["soil_temperature_0_to_7cm_mean"], dtype=float))
            alt.append(float(d.get("elevation") or 0))
    return {"pluie": np.array(pluie), "tmoy": np.array(tmoy), "tmin": np.array(tmin), "tsol": np.array(tsol),
            "alt": np.array(alt), "forme": (len(lats), len(lons))}


def vers_grille(valeurs, forme):
    """Interpole un champ météo (grille en degrés) sur la grille Lambert-93 de 250 m."""
    lons, lats = points_meteo()
    src = valeurs.reshape(forme).astype("float32")
    t = from_origin(lons[0] - PAS_METEO[0] / 2, lats[0] + PAS_METEO[1] / 2, *PAS_METEO)
    dst = np.zeros((LIGNES, COLS), dtype="float32")
    reproject(src, dst, src_transform=t, src_crs="EPSG:4326", dst_transform=TRANSFORM, dst_crs=CRS,
              resampling=Resampling.bilinear)
    return dst


def indice_grille(espece, m, mois, altitude):
    """Indice sur la grille de 250 m, en tenant compte de l'altitude de chaque carré."""
    niveaux = []
    for z in NIVEAUX:
        ecart = (m["alt"] - z)[:, None]
        ind = im.indice(espece, m["pluie"], m["tsol"] + GRADIENT_SOL * ecart, m["tmin"] + GRADIENT_AIR * ecart, mois)
        niveaux.append(vers_grille(ind, m["forme"]))
    niveaux = np.stack(niveaux)
    pos = np.clip((altitude - NIVEAUX[0]) / (NIVEAUX[1] - NIVEAUX[0]), 0, len(NIVEAUX) - 1)
    bas = np.floor(pos).astype(int)
    haut = np.minimum(bas + 1, len(NIVEAUX) - 1)
    f = pos - bas
    lig, col = np.indices(altitude.shape)
    return np.clip(niveaux[bas, lig, col] * (1 - f) + niveaux[haut, lig, col] * f, 0, 1)


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


def case(lat, lon):
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(lon, lat)
    return int((Y1 - y) // PAS), int((x - X0) // PAS)


def main():
    if not (DATA / "scores.tif").exists():
        print("Pas encore de scores : lancez d'abord la préparation des données.")
        return
    scores = lire_bandes(DATA / "scores.tif")
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0
    altitude = lire_bandes(DATA / "relief_sol.tif")["altitude"].astype("float32")
    aujourdhui = dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    mois = aujourdhui.month
    print(f"Météo du {aujourdhui}...")
    m = meteo()

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
    controles = {nom: case(*pos) for nom, pos in CONTROLES.items()}
    for espece, nom in ((k, v["nom"]) for k, v in ESPECES.items()):
        if espece not in scores:
            print(f"{nom} : pas encore de score, ignorée")
            continue
        ind_grille = indice_grille(espece, m, mois, altitude)
        jour = scores[espece].astype("float32") * ind_grille
        image(jour, t, w, h, lut, sortie / f"{espece}_jour.png")
        image(scores[espece], t, w, h, lut, sortie / f"{espece}_potentiel.png")
        ecrire_json(sortie / f"secteurs_{espece}.json", {
            "jour": secteurs(jour, 15), "potentiel": secteurs(scores[espece], 35)})
        # État régional : indice des carrés de la zone où l'espèce a un bon potentiel
        bons = masque & (scores[espece] >= 50)
        valeurs = ind_grille[bons] if bons.any() else ind_grille[masque]
        regional, meilleur = float(np.mean(valeurs)), float(np.percentile(valeurs, 90))
        f_saison = im.SAISON[espece].get(mois, 0)
        infos[espece] = {"nom": nom, "indice": round(regional * 100), "indice_meilleur": round(meilleur * 100),
                         "statut": statut(meilleur, f_saison), "saison": f_saison > 0}
        print(f"{nom} : indice {infos[espece]['indice']} (meilleurs secteurs {infos[espece]['indice_meilleur']}) "
              f"-> {infos[espece]['statut']}")
        for nom_c, (lc, cc) in controles.items():
            print(f"  contrôle {nom_c} : indice {ind_grille[lc, cc]:.2f}, potentiel {scores[espece][lc, cc]}, "
                  f"carte du jour {jour[lc, cc]:.0f}")

    with np.errstate(invalid="ignore"):
        tmoy7 = np.nanmean(m["tmoy"][:, -8:-1], axis=1)
        tsol7 = np.nanmean(m["tsol"][:, -8:-1], axis=1)
        gel = np.nanmin(m["tmin"][:, -4:-1], axis=1) < im.GEL
    ecrire_json(sortie / "info.json", {
        "date": aujourdhui.isoformat(),
        "bornes": bornes,
        "meteo": {
            "pluie_14j_mm": round(float(np.mean(np.nansum(m["pluie"][:, -15:-1], axis=1)[dans_zone])), 1),
            "temperature_7j": round(float(np.nanmean(tmoy7[dans_zone])), 1),
            "temperature_sol_7j": round(float(np.nanmean(tsol7[dans_zone])), 1),
            "jours_depuis_bonne_pluie": int(np.median(im.jours_depuis_bonne_pluie(m["pluie"])[dans_zone])),
            "gel_recent": bool(np.mean(gel[dans_zone]) > 0.3),
        },
        "lune": lune(aujourdhui),
        "especes": infos,
    })
    print("Cartes du jour prêtes.")


if __name__ == "__main__":
    main()
