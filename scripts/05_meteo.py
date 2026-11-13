"""Étape quotidienne : météo des trente derniers jours, prévisions et cartes du jour.

Source : Open-Meteo (gratuit, sans clé) : pluie, température de l'air et du sol, évaporation.
L'indice de chaque espèce (voir indice_meteo.py) est calculé à plusieurs altitudes,
puis chaque carré de 100 m prend la valeur de son altitude. Il multiplie le score de potentiel.

Résultats (dossier site/cartes) :
- <espece>_potentiel.png et <espece>_j0.png à _j7.png : potentiel toute saison, carte du jour et des 7 jours suivants
  (une couleur par tranche de 4 points : le site retrouve le score d'un point à partir de sa couleur) ;
- secteurs_<espece>_*.json : meilleurs secteurs, avec leur statut (forêt publique ou non) ;
- info.json : date, emprise, météo, prévisions, groupes d'espèces, fiabilité des cartes.
"""
import datetime as dt
import json
import os
from zoneinfo import ZoneInfo

import ephem
import math

import numpy as np
from PIL import Image
from pyproj import Transformer
from rasterio.warp import calculate_default_transform, transform_bounds
from scipy import ndimage

import indice_meteo as im
from commun import (BBOX, COLS, CRS, DATA, ESPECES, GROUPES, LIGNES, PAS, SITE, X0, X1, Y0, Y1, ecrire_json, get,
                    lire_bandes, membres)

API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VIGIEAU = "https://api.vigieau.gouv.fr/api/departements"
DEPS = ["54", "55", "57", "67", "68", "88"]
GROS_PAS = 3            # bilan hydrique depuis janvier : un point sur 3 dans chaque sens (environ 35 x 40 km)
PAS_METEO = (0.15, 0.12)  # degrés (lon, lat) : environ 11 x 13 km
JOURS_PASSES = 30
JOURS_PREVUS = 7                 # prévisions : aujourd'hui + 7 jours
# Zone surveillée pour les prévisions (centre, rayon)
RAYON_TERRE = 6378137          # sphère de la projection Web Mercator
ZONE = {"nom": "autour de Marlenheim", "lat": 48.621, "lon": 7.491, "rayon_km": 40}
# Baisse de température avec l'altitude (°C par mètre)
GRADIENT_AIR, GRADIENT_SOL = 0.0065, 0.0055
# Altitudes de calcul : l'indice est calculé à chaque niveau, puis chaque carré prend celui de son altitude
NIVEAUX = np.arange(100, 1501, 100)
# Trouvailles personnelles (secret GitHub TROUVAILLES) : leur indice est affiché, sans le lieu
CONTROLES = [(t["espece"], t["lat"], t["lon"]) for t in json.loads(os.environ.get("TROUVAILLES", "").strip() or "[]")
             if not t.get("absent")]


def points_meteo():
    lons = np.arange(BBOX[0], BBOX[2] + 1e-9, PAS_METEO[0])
    lats = np.arange(BBOX[3], BBOX[1] - 1e-9, -PAS_METEO[1])  # du nord au sud
    return lons, lats


def meteo():
    lons, lats = points_meteo()
    pts = [(lo, la) for la in lats for lo in lons]
    pluie, tmoy, tmin, tsol, et0, alt = [], [], [], [], [], []
    for i in range(0, len(pts), 100):
        lot = pts[i:i + 100]
        data = get(API, {
            "longitude": ",".join(f"{p[0]:.3f}" for p in lot),
            "latitude": ",".join(f"{p[1]:.3f}" for p in lot),
            # Même variable de sol que les archives utilisées pour l'apprentissage (06_phenologie.py)
            "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_min,soil_temperature_0_to_7cm_mean,"
                     "et0_fao_evapotranspiration",
            "past_days": JOURS_PASSES, "forecast_days": JOURS_PREVUS + 1, "timezone": "Europe/Paris",
        }).json()
        data = data if isinstance(data, list) else [data]
        for d in data:
            j = d["daily"]
            pluie.append(np.array(j["precipitation_sum"], dtype=float))
            tmoy.append(np.array(j["temperature_2m_mean"], dtype=float))
            tmin.append(np.array(j["temperature_2m_min"], dtype=float))
            tsol.append(np.array(j["soil_temperature_0_to_7cm_mean"], dtype=float))
            et0.append(np.array(j["et0_fao_evapotranspiration"], dtype=float))
            alt.append(float(d.get("elevation") or 0))
    return {"pluie": np.array(pluie), "tmoy": np.array(tmoy), "tmin": np.array(tmin), "tsol": np.array(tsol),
            "et0": np.array(et0), "alt": np.array(alt), "forme": (len(lats), len(lons))}


def reserve_sol(m, aujourdhui):
    """Réserve en eau du sol (0 à 1) pour chaque classe de sol, chaque point et chaque jour de la série météo.

    Bilan pluie moins évaporation depuis le 1er janvier (sol plein en hiver) : archives Open-Meteo sur une grille
    plus lâche jusqu'au début de la série, puis la série du jour (30 jours passés et prévisions) point par point.
    Résultat : tableau (classes de réserve utile, points, jours).
    """
    lons, lats = points_meteo()
    ny, nx = m["forme"]
    debut = dt.date(aujourdhui.year, 1, 1)
    fin = aujourdhui - dt.timedelta(days=JOURS_PASSES + 1)
    departs = np.ones((len(im.CLASSES_RU), ny * nx))
    if fin > debut:
        gl, gL = np.arange(0, nx, GROS_PAS), np.arange(0, ny, GROS_PAS)
        pts = [(lons[c], lats[r]) for r in gL for c in gl]
        pl, ev = [], []
        for i in range(0, len(pts), 60):
            lot = pts[i:i + 60]
            data = get(ARCHIVE, {
                "longitude": ",".join(f"{p[0]:.3f}" for p in lot), "latitude": ",".join(f"{p[1]:.3f}" for p in lot),
                "start_date": debut.isoformat(), "end_date": fin.isoformat(),
                "daily": "precipitation_sum,et0_fao_evapotranspiration", "timezone": "Europe/Paris"}, timeout=300).json()
            for d in (data if isinstance(data, list) else [data]):
                pl.append(np.array(d["daily"]["precipitation_sum"], dtype=float))
                ev.append(np.array(d["daily"]["et0_fao_evapotranspiration"], dtype=float))
        pl, ev = np.array(pl), np.array(ev)
        # Chaque point fin part de la réserve du point lâche le plus proche
        rr = np.clip(np.rint(np.arange(ny) / GROS_PAS).astype(int), 0, len(gL) - 1)
        cc = np.clip(np.rint(np.arange(nx) / GROS_PAS).astype(int), 0, len(gl) - 1)
        for c, ru in enumerate(im.CLASSES_RU):
            gros = im.bilan_hydrique(pl, ev, ru=ru)[:, -1].reshape(len(gL), len(gl))
            departs[c] = gros[rr][:, cc].ravel()
        print(f"  réserve au {fin} (bilan depuis le {debut}) : "
              + ", ".join(f"sol de {ru} mm {np.median(departs[c]) * 100:.0f} %" for c, ru in enumerate(im.CLASSES_RU)))
    return np.stack([im.bilan_hydrique(m["pluie"], m["et0"], departs[c], ru=ru) for c, ru in enumerate(im.CLASSES_RU)])


def preparer_sol():
    """Classe de réserve utile de chaque carré de 100 m (interpolation entre deux classes)."""
    chemin = DATA / "reserve_utile.tif"
    ru = lire_bandes(chemin)["ru"].astype("float32") if chemin.exists() else np.full((LIGNES, COLS), im.RESERVE_UTILE, "float32")
    ru = np.where(ru <= 0, im.RESERVE_UTILE, ru)
    cl = np.array(im.CLASSES_RU, dtype="float32")
    pos = np.clip(np.interp(ru, cl, np.arange(len(cl))), 0, len(cl) - 1)
    bas = np.floor(pos).astype("int8")
    return bas, np.minimum(bas + 1, len(cl) - 1).astype("int8"), (pos - bas).astype("float32"), ru


def humidite_grille(m, k, sol, geo):
    """État du sol pour la pousse au jour k (0 sec à 1 bien pourvu), carré par carré, selon sa réserve utile."""
    bas, haut, f, _ = sol
    col = JOURS_PASSES + k
    par_classe = np.stack([vers_grille(im.humidite_sol(m["reserve"][c], col), m["forme"], geo)
                           for c in range(len(im.CLASSES_RU))])
    lig, cols = np.indices(bas.shape)
    return par_classe[bas, lig, cols] * (1 - f) + par_classe[haut, lig, cols] * f


def resume_sol(m, sol, dans_zone, masque, zone):
    """Réserve en eau du sol dans la zone : sol typique des bois de la zone, et sol sableux ou caillouteux."""
    if m.get("reserve") is None:
        return {}
    a = JOURS_PASSES
    ru_zone = float(np.median(sol[3][masque & zone])) if (masque & zone).any() else im.RESERVE_UTILE
    c = int(np.argmin(np.abs(np.array(im.CLASSES_RU) - ru_zone)))
    r = m["reserve"][c][dans_zone]
    leger = m["reserve"][1][dans_zone]           # classe de 50 mm : sol sableux, caillouteux ou mince
    return {"reserve_utile_mm": round(ru_zone), "reserve_sol_pct": round(float(np.median(r[:, a])) * 100),
            "reserve_sol_30j_pct": round(float(np.median(r[:, 0])) * 100),
            "reserve_sol_prevue_pct": round(float(np.median(r[:, -1])) * 100),
            "reserve_sol_leger_pct": round(float(np.median(leger[:, a])) * 100)}


def secheresse():
    """Niveau des arrêtés sécheresse en vigueur par département (VigiEau)."""
    try:
        deps = get(VIGIEAU, {}, essais=2, timeout=60).json()
        return {str(d["code"]): d.get("niveauGraviteMax") for d in deps if str(d.get("code")) in DEPS}
    except Exception as e:
        print(f"  VigiEau indisponible : {e}")
        return {}


def preparer_geometrie(altitude):
    """Position de chaque carré de 100 m dans la grille météo (ligne, colonne) et dans les niveaux d'altitude.

    Calculée une seule fois : chaque carte du jour se fait ensuite en une interpolation.
    """
    lons, lats = points_meteo()
    lig, col = np.indices((LIGNES, COLS), dtype="float32")
    x = X0 + (col + 0.5) * PAS
    y = Y1 - (lig + 0.5) * PAS
    lon, lat = Transformer.from_crs(CRS, 4326, always_xy=True).transform(x, y)
    fc = ((lon - lons[0]) / PAS_METEO[0]).astype("float32")
    fl = ((lats[0] - lat) / PAS_METEO[1]).astype("float32")
    niv = np.clip((altitude - NIVEAUX[0]) / (NIVEAUX[1] - NIVEAUX[0]), 0, len(NIVEAUX) - 1).astype("float32")
    return np.stack([niv, fl, fc])


def vers_grille(valeurs, forme, geo):
    """Interpole un champ météo (grille en degrés) sur la grille Lambert-93 de 100 m."""
    return ndimage.map_coordinates(valeurs.reshape(forme).astype("float32"), geo[1:], order=1, mode="nearest",
                                   output=np.float32)


def series_jour(m, k):
    """Les 31 jours qui se terminent au jour k (0 = aujourd'hui, 1 = demain...)."""
    i = JOURS_PASSES + k
    return m["pluie"][:, i - 30:i + 1], m["tsol"][:, i - 30:i + 1], m["tmin"][:, i - 30:i + 1]


def indice_grille(espece, m, k, mois, geo):
    """Indice du jour k sur la grille de 100 m, en tenant compte de l'altitude de chaque carré."""
    pluie, tsol, tmin = series_jour(m, k)
    cube = []
    for z in NIVEAUX:
        ecart = (m["alt"] - z)[:, None]
        cube.append(im.indice(espece, pluie, tsol + GRADIENT_SOL * ecart, tmin + GRADIENT_AIR * ecart, mois)
                    .reshape(m["forme"]))
    cube = np.stack(cube).astype("float32")
    return np.clip(ndimage.map_coordinates(cube, geo, order=1, mode="nearest", output=np.float32), 0, 1)


# Palette sur 1000 : jaune pâle -> orange -> rouge -> prune ; transparent sous le seuil
PALETTE = [(80, (246, 226, 122, 115)), (200, (244, 163, 64, 165)), (400, (217, 69, 43, 200)),
           (650, (130, 30, 75, 225)), (1000, (70, 10, 45, 240))]
SEUIL = 80
MAX = 1000
PAS_COULEUR = 4            # une couleur par tranche de 4 points : 250 couleurs, images 3 à 4 fois plus légères


def couleur(v):
    for (a, ca), (b, cb) in zip(PALETTE[:-1], PALETTE[1:]):
        if a <= v <= b:
            t = (v - a) / (b - a)
            return [round(ca[k] + t * (cb[k] - ca[k])) for k in range(4)]
    return [0, 0, 0, 0]


def table_couleurs():
    """Couleur (rouge, vert, bleu, opacité) de chaque tranche ; la tranche 0 est transparente."""
    return [[0, 0, 0, 0]] + [couleur(min(i * PAS_COULEUR, MAX)) if i * PAS_COULEUR >= SEUIL else [0, 0, 0, 0]
                             for i in range(1, MAX // PAS_COULEUR + 1)]


def preparer_mercator():
    """Grille Web Mercator (celle des cartes en ligne) couvrant la zone, et le carré de 100 m de chaque pixel."""
    t, w, h = calculate_default_transform(CRS, "EPSG:3857", COLS, LIGNES, left=X0, bottom=Y0, right=X1, top=Y1,
                                          resolution=150)
    gauche, haut = t.c, t.f
    droite, bas = gauche + w * t.a, haut + h * t.e
    ouest, sud, est, nord = transform_bounds("EPSG:3857", "EPSG:4326", gauche, bas, droite, haut)
    lig, col = np.indices((h, w), dtype="float64")
    x, y = Transformer.from_crs("EPSG:3857", CRS, always_xy=True).transform(gauche + (col + 0.5) * t.a,
                                                                             haut + (lig + 0.5) * t.e)
    c = np.floor((x - X0) / PAS).astype(np.int64)
    li = np.floor((Y1 - y) / PAS).astype(np.int64)
    dedans = (c >= 0) & (c < COLS) & (li >= 0) & (li < LIGNES)
    indices = np.where(dedans, li * COLS + c, -1)
    return {"indices": indices, "dedans": dedans, "bornes": [[sud, ouest], [nord, est]],
            "mercator": {"gauche": gauche, "haut": haut, "pas": t.a, "largeur": int(w), "hauteur": int(h)}}


def image(grille, merc, table, chemin):
    """Carte en PNG à palette : chaque pixel porte le score de son carré de 100 m, arrondi à 4 points près."""
    v = np.zeros(merc["indices"].shape, dtype="float32")
    v[merc["dedans"]] = grille.ravel()[merc["indices"][merc["dedans"]]]
    tranche = np.where(v >= SEUIL, np.clip(np.ceil(v / PAS_COULEUR), 1, MAX // PAS_COULEUR), 0).astype("uint8")
    img = Image.fromarray(tranche, "P")
    img.putpalette([x for c in table for x in c[:3]])
    img.save(chemin, transparency=bytes(c[3] for c in table))


def voile(prive, merc, chemin):
    """Voile clair sur les carrés hors forêt publique (affiché quand on ne veut que les forêts publiques)."""
    v = np.zeros(merc["indices"].shape, dtype="uint8")
    v[merc["dedans"]] = prive.ravel()[merc["indices"][merc["dedans"]]]
    img = Image.fromarray(v, "P")
    img.putpalette([0, 0, 0, 247, 246, 241])
    img.save(chemin, transparency=bytes([0, 215]))


def secteurs(grille, seuil, statut, merc, rayon=6):
    """Meilleurs secteurs d'une carte : le meilleur pixel de chaque voisinage de ~1,2 km, pris sur l'image publiée.

    On travaille sur les pixels de l'image elle-même (un pixel = le score d'un carré de 100 m, par tranches de
    4 points) : toucher la carte à l'endroit indiqué redonne exactement ce score, et aucun pixel à moins de
    600 m n'en a un plus élevé.

    Renvoie [[lat, lon, score, statut], ...] du meilleur au moins bon ; statut : 0 forêt privée ou statut
    inconnu, 1 forêt domaniale, 2 autre forêt publique.
    """
    dedans, indices, m = merc["dedans"], merc["indices"], merc["mercator"]
    v = np.zeros(indices.shape, dtype="float32")
    v[dedans] = grille.ravel()[indices[dedans]]
    v = np.minimum(np.ceil(v / PAS_COULEUR) * PAS_COULEUR, MAX)
    lig, col = np.nonzero(v >= seuil)
    ordre = np.argsort(-v[lig, col], kind="stable")
    pris = np.zeros(v.shape, dtype=bool)
    choisis = []
    for i in ordre:
        li, c = lig[i], col[i]
        if pris[li, c]:
            continue
        zone = (slice(max(li - rayon, 0), li + rayon + 1), slice(max(c - rayon, 0), c + rayon + 1))
        if v[zone].max() > v[li, c]:        # un pixel voisin fait mieux : il appartient à un meilleur secteur
            continue
        choisis.append(i)
        pris[zone] = True
    choisis = np.array(choisis, dtype=int)
    if len(choisis) == 0:
        return []
    li, c = lig[choisis], col[choisis]
    x = m["gauche"] + (c + 0.5) * m["pas"]
    y = m["haut"] - (li + 0.5) * m["pas"]
    lon = np.degrees(x / RAYON_TERRE)
    lat = np.degrees(2 * np.arctan(np.exp(y / RAYON_TERRE)) - np.pi / 2)
    st = statut.ravel()[indices[li, c]] if statut is not None else np.zeros(len(choisis), int)
    return [[round(float(a), 5), round(float(o), 5), int(val), int(e)]
            for a, o, val, e in zip(lat, lon, v[li, c], st)]


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
    if valeur >= 0.3:
        return "bon moment"
    if valeur >= 0.12:
        return "conditions moyennes"
    return "peu probable en ce moment"


def case(lat, lon):
    x, y = Transformer.from_crs(4326, CRS, always_xy=True).transform(lon, lat)
    return int((Y1 - y) // PAS), int((x - X0) // PAS)


def masque_zone(masque):
    lig, col = np.indices(masque.shape)
    x = X0 + (col + 0.5) * PAS
    y = Y1 - (lig + 0.5) * PAS
    cx, cy = Transformer.from_crs(4326, CRS, always_xy=True).transform(ZONE["lon"], ZONE["lat"])
    return masque & ((x - cx) ** 2 + (y - cy) ** 2 <= (ZONE["rayon_km"] * 1000) ** 2)


def meilleur_coin(liste, publics=False):
    """Score du meilleur secteur à moins de ZONE["rayon_km"] du centre : le chiffre que la page affiche pour le
    coin n° 1 « autour de moi » avec le même rayon."""
    r = math.radians
    for lat, lon, score, st in liste:                 # liste triée du meilleur au moins bon
        if publics and not st:
            continue
        h = (math.sin(r(lat - ZONE["lat"]) / 2) ** 2
             + math.cos(r(lat)) * math.cos(r(ZONE["lat"])) * math.sin(r(lon - ZONE["lon"]) / 2) ** 2)
        if 12742 * math.asin(math.sqrt(h)) <= ZONE["rayon_km"]:
            return score
    return 0


def niveau(ind_grille, potentiel, zone):
    """Indice des meilleurs secteurs d'une zone (90e centile parmi les carrés à bon potentiel)."""
    bons = zone & (potentiel >= 500)
    valeurs = ind_grille[bons] if bons.sum() >= 20 else ind_grille[zone]
    return float(np.mean(valeurs)), float(np.percentile(valeurs, 90))


ORDRE_STATUTS = ["hors saison", "peu probable en ce moment", "conditions moyennes", "bon moment"]


def fiabilite():
    """Fiabilité de la carte de potentiel de chaque espèce, d'après la validation de l'étape 4."""
    chemin = DATA / "scores_resume.json"
    if not chemin.exists():
        return {}
    sortie = {}
    for e, r in json.loads(chemin.read_text(encoding="utf-8")).items():
        auc, n = r.get("auc"), r.get("observations", 0)
        if auc is None or n < 40:
            niveau_txt = "indicative"
        elif auc >= 0.75:
            niveau_txt = "bonne"
        elif auc >= 0.65:
            niveau_txt = "correcte"
        else:
            niveau_txt = "indicative"
        sortie[e] = {"niveau": niveau_txt, "auc": auc, "observations": n,
                     "observations_region": r.get("observations_region")}
    return sortie


def main():
    if not (DATA / "scores.tif").exists():
        print("Pas encore de scores : lancez d'abord la préparation des données.")
        return
    scores = lire_bandes(DATA / "scores.tif")
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0
    zone = masque_zone(masque)
    altitude = lire_bandes(DATA / "relief_sol.tif")["altitude"].astype("float32")
    publique = lire_bandes(DATA / "acces.tif")["publique"] if (DATA / "acces.tif").exists() else None
    geo = preparer_geometrie(altitude)
    aujourdhui = dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    dates = [aujourdhui + dt.timedelta(days=k) for k in range(JOURS_PREVUS + 1)]
    print(f"Météo du {aujourdhui} et prévisions à {JOURS_PREVUS} jours...")
    m = meteo()
    print("Réserve en eau du sol (bilan depuis janvier, selon le sol de chaque carré)...")
    try:
        m["reserve"] = reserve_sol(m, aujourdhui)
        sol = preparer_sol()
        humidite = {k: humidite_grille(m, k, sol, geo) for k in range(len(dates))}
    except Exception as e:      # sans archives, on garde l'indice sans réserve en eau
        print(f"  indisponible : {e}")
        m["reserve"], sol, humidite = None, None, {}
    niveaux_secheresse = secheresse()

    sortie = SITE / "cartes"
    sortie.mkdir(parents=True, exist_ok=True)
    for vieux in sortie.glob("*.png"):
        vieux.unlink()
    for vieux in sortie.glob("secteurs_*.json"):
        vieux.unlink()
    merc = preparer_mercator()
    table = table_couleurs()
    if publique is not None:
        voile(masque & (publique == 0), merc, sortie / "prive.png")
    infos, previsions = {}, {}
    groupe_de = {e: g for g in GROUPES for e in membres(g)}
    # Cartes des groupes (« tous les cèpes ») : meilleur score pondéré de leurs membres, carré par carré
    cumul = {g: {} for g in GROUPES}
    controles = [(e2, case(lat, lon), n) for n, (e, lat, lon) in enumerate(CONTROLES, start=1) for e2 in membres(e)]

    def publier(cle, nom_carte, carte, seuil):
        image(carte, merc, table, sortie / f"{cle}_{nom_carte}.png")
        liste = secteurs(carte, seuil, publique, merc)
        ecrire_json(sortie / f"secteurs_{cle}_{nom_carte}.json", liste)
        return liste

    for espece, e in ESPECES.items():
        nom = e["nom"]
        if espece not in scores:
            print(f"{nom} : pas encore de score, ignorée")
            continue
        potentiel = scores[espece]
        g = groupe_de.get(espece)
        poids = GROUPES[g]["poids"][espece] if g else 1
        publier(espece, "potentiel", potentiel, 400)
        if g:
            cumul[g]["potentiel"] = np.maximum(cumul[g].get("potentiel", 0), potentiel.astype("float32") * poids)
        prev = {"zone": [], "zone_pub": [], "indice_zone": [], "region": [], "statut": []}
        for k, jour_k in enumerate(dates):
            ind_grille = indice_grille(espece, m, k, jour_k.month, geo)
            if k in humidite:          # réserve en eau du sol en profondeur, selon le sol de chaque carré
                ind_grille = ind_grille * im.effet_sol(espece, humidite[k]).astype("float32")
            carte = potentiel.astype("float32") * ind_grille        # sur 1000
            liste = publier(espece, f"j{k}", carte, 120)
            prev["zone"].append(meilleur_coin(liste))
            prev["zone_pub"].append(meilleur_coin(liste, publics=True))
            if g:
                cumul[g][k] = np.maximum(cumul[g].get(k, 0), carte * poids)
            moy_r, p90_r = niveau(ind_grille, potentiel, masque)
            _, p90_z = niveau(ind_grille, potentiel, zone)
            f_saison = im.SAISON[espece].get(jour_k.month, 0)
            prev["indice_zone"].append(round(p90_z * 1000))
            prev["region"].append(round(p90_r * 1000))
            prev["statut"].append(statut(p90_r, f_saison))
            if k == 0:
                infos[espece] = {"nom": nom, "indice": round(moy_r * 1000), "indice_meilleur": round(p90_r * 1000),
                                 "statut": statut(p90_r, f_saison), "saison": f_saison > 0}
                if g:
                    infos[espece].update(groupe=g, court=e.get("court", nom))
                for e2, (lc, cc), n in controles:
                    if e2 == espece and 0 <= lc < LIGNES and 0 <= cc < COLS:
                        print(f"  trouvaille personnelle n° {n} : indice {ind_grille[lc, cc]:.2f}, "
                              f"potentiel {potentiel[lc, cc]}, carte du jour {carte[lc, cc]:.0f}")
        previsions[espece] = prev
        print(f"{nom} : aujourd'hui {infos[espece]['statut']} ; région {prev['region']} ; {ZONE['nom']} {prev['zone']}")

    # Groupes : cartes combinées et résumé (meilleur de leurs membres)
    groupes = {}
    for g, def_g in GROUPES.items():
        presents = [e for e in membres(g) if e in infos]
        if not presents or "potentiel" not in cumul[g]:
            continue
        publier(g, "potentiel", cumul[g]["potentiel"], 400)
        listes = [publier(g, f"j{k}", cumul[g][k], 120) for k in range(len(dates))]
        pond = lambda e, cle, k: previsions[e][cle][k] * def_g["poids"][e]
        previsions[g] = {
            "zone": [meilleur_coin(l) for l in listes],
            "zone_pub": [meilleur_coin(l, publics=True) for l in listes],
            "indice_zone": [round(max(pond(e, "indice_zone", k) for e in presents)) for k in range(len(dates))],
            "region": [round(max(pond(e, "region", k) for e in presents)) for k in range(len(dates))],
            "statut": [max((previsions[e]["statut"][k] for e in presents), key=ORDRE_STATUTS.index)
                       for k in range(len(dates))],
        }
        infos[g] = {"nom": def_g["nom"], "indice": max(infos[e]["indice"] for e in presents),
                    "indice_meilleur": max(infos[e]["indice_meilleur"] for e in presents),
                    "statut": previsions[g]["statut"][0], "saison": any(infos[e]["saison"] for e in presents)}
        groupes[g] = {"nom": def_g["nom"], "tous": def_g["tous"], "membres": presents}
        print(f"{def_g['nom']} (groupe) : aujourd'hui {infos[g]['statut']} ; {ZONE['nom']} {previsions[g]['zone']}")

    # Résumé météo des points dans la zone couverte (aujourd'hui = colonne JOURS_PASSES)
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
    a = JOURS_PASSES
    with np.errstate(invalid="ignore"):
        tmoy7 = np.nanmean(m["tmoy"][:, a - 7:a], axis=1)
        tsol7 = np.nanmean(m["tsol"][:, a - 7:a], axis=1)
        gel = np.nanmin(m["tmin"][:, a - 3:a], axis=1) < im.GEL
        pluie_prevue = np.nansum(m["pluie"][:, a + 1:], axis=1)
    pluie_passee = m["pluie"][:, :a + 1]
    ecrire_json(sortie / "info.json", {
        "date": aujourdhui.isoformat(),
        "bornes": merc["bornes"],
        "mercator": merc["mercator"],
        "couleurs": {"pas": PAS_COULEUR, "seuil": SEUIL, "table": table},
        "meteo": {
            "pluie_14j_mm": round(float(np.mean(np.nansum(m["pluie"][:, a - 14:a], axis=1)[dans_zone])), 1),
            "temperature_7j": round(float(np.nanmean(tmoy7[dans_zone])), 1),
            "temperature_sol_7j": round(float(np.nanmean(tsol7[dans_zone])), 1),
            "jours_depuis_bonne_pluie": int(np.median(im.jours_depuis_bonne_pluie(pluie_passee)[dans_zone])),
            "gel_recent": bool(np.mean(gel[dans_zone]) > 0.3),
            "pluie_prevue_7j_mm": round(float(np.mean(pluie_prevue[dans_zone])), 1),
            **resume_sol(m, sol, dans_zone, masque, zone),
        },
        "secheresse": niveaux_secheresse,
        "lune": lune(aujourdhui),
        "previsions": {"dates": [d.isoformat() for d in dates], "zone": ZONE, "especes": previsions},
        "especes": infos,
        "groupes": groupes,
        "fiabilite": fiabilite(),
        "acces": publique is not None,
        "voile": publique is not None,
    })
    print("Cartes du jour et prévisions prêtes.")


if __name__ == "__main__":
    main()
