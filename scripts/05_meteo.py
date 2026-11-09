"""Étape quotidienne : météo des trois dernières semaines et cartes du jour.

Source : Open-Meteo (gratuit, sans clé) : pluie, température de l'air et du sol sur 30 jours.
L'indice de chaque espèce (voir indice_meteo.py) est calculé à plusieurs altitudes,
puis chaque carré de 100 m prend la valeur de son altitude. Il multiplie le score de potentiel.

Résultats :
- site/cartes/<espece>_jour.png : la carte du jour
- site/cartes/<espece>_potentiel.png : le potentiel toute saison
- site/cartes/info.json : date, emprise et état des conditions par espèce
"""
import datetime as dt
import json
import os
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
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VIGIEAU = "https://api.vigieau.gouv.fr/api/departements"
DEPS = ["54", "55", "57", "67", "68", "88"]
GROS_PAS = 3            # bilan hydrique depuis janvier : un point sur 3 dans chaque sens (environ 35 x 40 km)
PAS_METEO = (0.15, 0.12)  # degrés (lon, lat) : environ 11 x 13 km
JOURS_PASSES = 30
JOURS_PREVUS = 7                 # prévisions : aujourd'hui + 7 jours
# Zone surveillée pour les prévisions (centre, rayon)
ZONE = {"nom": "autour de Marlenheim", "lat": 48.621, "lon": 7.491, "rayon_km": 40}
# Baisse de température avec l'altitude (°C par mètre)
GRADIENT_AIR, GRADIENT_SOL = 0.0065, 0.0055
# Altitudes de calcul : l'indice est calculé à chaque niveau, puis chaque carré prend celui de son altitude
NIVEAUX = np.arange(100, 1501, 100)
# Trouvailles personnelles (secret GitHub TROUVAILLES) : leur indice est affiché, sans le lieu
CONTROLES = [(t["espece"], t["lat"], t["lon"]) for t in json.loads(os.environ.get("TROUVAILLES", "").strip() or "[]")]


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


def humidite_grille(m, k, sol):
    """État du sol pour la pousse au jour k (0 sec à 1 bien pourvu), carré par carré, selon sa réserve utile."""
    bas, haut, f, _ = sol
    col = JOURS_PASSES + k
    par_classe = np.stack([vers_grille(im.humidite_sol(m["reserve"][c], col), m["forme"])
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


def vers_grille(valeurs, forme):
    """Interpole un champ météo (grille en degrés) sur la grille Lambert-93 de 100 m."""
    lons, lats = points_meteo()
    src = valeurs.reshape(forme).astype("float32")
    t = from_origin(lons[0] - PAS_METEO[0] / 2, lats[0] + PAS_METEO[1] / 2, *PAS_METEO)
    dst = np.zeros((LIGNES, COLS), dtype="float32")
    reproject(src, dst, src_transform=t, src_crs="EPSG:4326", dst_transform=TRANSFORM, dst_crs=CRS,
              resampling=Resampling.bilinear)
    return dst


def interpolation_altitude(altitude):
    """Prépare le passage des niveaux d'altitude de calcul à l'altitude de chaque carré."""
    pos = np.clip((altitude - NIVEAUX[0]) / (NIVEAUX[1] - NIVEAUX[0]), 0, len(NIVEAUX) - 1)
    bas = np.floor(pos).astype(int)
    return bas, np.minimum(bas + 1, len(NIVEAUX) - 1), (pos - bas).astype("float32"), np.indices(altitude.shape)


def series_jour(m, k):
    """Les 31 jours qui se terminent au jour k (0 = aujourd'hui, 1 = demain...)."""
    i = JOURS_PASSES + k
    return m["pluie"][:, i - 30:i + 1], m["tsol"][:, i - 30:i + 1], m["tmin"][:, i - 30:i + 1]


def indice_grille(espece, m, k, mois, interp):
    """Indice du jour k sur la grille de 100 m, en tenant compte de l'altitude de chaque carré."""
    pluie, tsol, tmin = series_jour(m, k)
    niveaux = []
    for z in NIVEAUX:
        ecart = (m["alt"] - z)[:, None]
        ind = im.indice(espece, pluie, tsol + GRADIENT_SOL * ecart, tmin + GRADIENT_AIR * ecart, mois)
        niveaux.append(vers_grille(ind, m["forme"]))
    niveaux = np.stack(niveaux)
    bas, haut, f, (lig, col) = interp
    return np.clip(niveaux[bas, lig, col] * (1 - f) + niveaux[haut, lig, col] * f, 0, 1)


# Palette sur 1000 : jaune pâle -> orange -> rouge -> prune ; transparent sous le seuil
PALETTE = [(80, (246, 226, 122, 115)), (200, (244, 163, 64, 165)), (400, (217, 69, 43, 200)),
           (650, (130, 30, 75, 225)), (1000, (70, 10, 45, 240))]
SEUIL = 80
MAX = 1000


def table_couleurs():
    lut = np.zeros((MAX + 1, 4), dtype=np.uint8)
    for v in range(SEUIL, MAX + 1):
        x = v
        for (a, ca), (b, cb) in zip(PALETTE[:-1], PALETTE[1:]):
            if a <= x <= b:
                t = (x - a) / (b - a)
                lut[v] = [round(ca[k] + t * (cb[k] - ca[k])) for k in range(4)]
                break
    return lut


def preparer_mercator():
    """Grille Web Mercator (celle des cartes en ligne) couvrant la zone."""
    t, w, h = calculate_default_transform(CRS, "EPSG:3857", COLS, LIGNES, left=X0, bottom=Y0, right=X1, top=Y1,
                                          resolution=150)
    gauche, haut = t.c, t.f
    droite, bas = gauche + w * t.a, haut + h * t.e
    ouest, sud, est, nord = transform_bounds("EPSG:3857", "EPSG:4326", gauche, bas, droite, haut)
    return t, w, h, [[sud, ouest], [nord, est]]


def image(grille, t, w, h, lut, chemin):
    dst = np.zeros((h, w), dtype="float32")
    reproject(grille.astype("float32"), dst, src_transform=TRANSFORM, src_crs=CRS, src_nodata=0,
              dst_transform=t, dst_crs="EPSG:3857", dst_nodata=0, resampling=Resampling.bilinear)
    Image.fromarray(lut[np.clip(np.rint(dst), 0, MAX).astype(np.int32)], "RGBA").save(chemin, optimize=True)


def secteurs(grille, seuil, rayon=6):
    """Meilleurs secteurs d'une carte : moyenne sur 300 m, puis un seul point par voisinage de ~1,2 km.

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


def niveau(ind_grille, potentiel, zone):
    """Indice des meilleurs secteurs d'une zone (90e centile parmi les carrés à bon potentiel)."""
    bons = zone & (potentiel >= 500)
    valeurs = ind_grille[bons] if bons.sum() >= 20 else ind_grille[zone]
    return float(np.mean(valeurs)), float(np.percentile(valeurs, 90))


def main():
    if not (DATA / "scores.tif").exists():
        print("Pas encore de scores : lancez d'abord la préparation des données.")
        return
    scores = lire_bandes(DATA / "scores.tif")
    masque = lire_bandes(DATA / "occupation.tif")["masque"] > 0
    zone = masque_zone(masque)
    altitude = lire_bandes(DATA / "relief_sol.tif")["altitude"].astype("float32")
    interp = interpolation_altitude(altitude)
    aujourdhui = dt.datetime.now(ZoneInfo("Europe/Paris")).date()
    dates = [aujourdhui + dt.timedelta(days=k) for k in range(JOURS_PREVUS + 1)]
    print(f"Météo du {aujourdhui} et prévisions à {JOURS_PREVUS} jours...")
    m = meteo()
    print("Réserve en eau du sol (bilan depuis janvier, selon le sol de chaque carré)...")
    try:
        m["reserve"] = reserve_sol(m, aujourdhui)
        sol = preparer_sol()
        humidite = {k: humidite_grille(m, k, sol) for k in range(len(dates))}
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
    t, w, h, bornes = preparer_mercator()
    lut = table_couleurs()
    infos, previsions = {}, {}
    controles = [(e, case(lat, lon)) for e, lat, lon in CONTROLES]
    for espece, nom in ((k, v["nom"]) for k, v in ESPECES.items()):
        if espece not in scores:
            print(f"{nom} : pas encore de score, ignorée")
            continue
        potentiel = scores[espece]
        image(potentiel, t, w, h, lut, sortie / f"{espece}_potentiel.png")
        ecrire_json(sortie / f"secteurs_{espece}_potentiel.json", secteurs(potentiel, 400))
        prev = {"zone": [], "region": [], "statut": []}
        for k, jour_k in enumerate(dates):
            ind_grille = indice_grille(espece, m, k, jour_k.month, interp)
            if k in humidite:          # réserve en eau du sol en profondeur, selon le sol de chaque carré
                ind_grille = ind_grille * im.effet_sol(espece, humidite[k]).astype("float32")
            carte = potentiel.astype("float32") * ind_grille        # sur 1000
            image(carte, t, w, h, lut, sortie / f"{espece}_j{k}.png")
            ecrire_json(sortie / f"secteurs_{espece}_j{k}.json", secteurs(carte, 120))
            moy_r, p90_r = niveau(ind_grille, potentiel, masque)
            _, p90_z = niveau(ind_grille, potentiel, zone)
            f_saison = im.SAISON[espece].get(jour_k.month, 0)
            prev["zone"].append(round(p90_z * 1000))
            prev["region"].append(round(p90_r * 1000))
            prev["statut"].append(statut(p90_r, f_saison))
            if k == 0:
                infos[espece] = {"nom": nom, "indice": round(moy_r * 1000), "indice_meilleur": round(p90_r * 1000),
                                 "statut": statut(p90_r, f_saison), "saison": f_saison > 0}
                for n, (e, (lc, cc)) in enumerate(controles, start=1):
                    if e == espece and 0 <= lc < LIGNES and 0 <= cc < COLS:
                        print(f"  trouvaille personnelle n° {n} : indice {ind_grille[lc, cc]:.2f}, "
                              f"potentiel {potentiel[lc, cc]}, carte du jour {carte[lc, cc]:.0f}")
        previsions[espece] = prev
        print(f"{nom} : aujourd'hui {infos[espece]['statut']} ; région {prev['region']} ; {ZONE['nom']} {prev['zone']}")

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
        "bornes": bornes,
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
    })
    print("Cartes du jour et prévisions prêtes.")


if __name__ == "__main__":
    main()
