"""Étape 6 : apprendre les réglages météo de chaque espèce à partir des observations datées.

Pour chaque observation GBIF datée, on récupère la météo réelle des 30 jours précédents
(archives Open-Meteo, gratuites) : pluie, température de l'air et du sol.
On compare ensuite les dates où l'espèce a été trouvée à des dates « témoins » :
- les dates où d'autres champignons ont été signalés (les mycologues étaient sortis, sans noter l'espèce) :
  autres espèces suivies et « groupe cible » de l'étape 1, là où la météo est déjà téléchargée ;
- des dates tirées au hasard de mars à décembre.
Les réglages (délai après la pluie, température du sol pendant la pousse) qui séparent le
mieux les deux groupes sont retenus, s'ils font nettement mieux que les réglages de départ.

Résultats :
- data/meteo_historique.csv : la météo téléchargée (gardée pour ne pas la redemander)
- data/phenologie.json : réglages retenus et qualité (AUC) avant / après
"""
import datetime as dt
import itertools
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer

import indice_meteo as im
from commun import COLS, CRS, DATA, GROUPES, LIGNES, PAS, X0, Y1, ecrire_json, get, lire_bandes, membres

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
CELLULE = 0.25                  # degrés : résolution des archives ERA5
AVANT = 35                      # jours de météo nécessaires avant chaque date
ANNEE_MIN = 2010
TEMOINS_HASARD = 2              # dates au hasard par observation
TEMOINS_FOND = 4000             # observations du groupe cible tirées au sort comme témoins, par espèce
MIN_PRESENCES = 60
GRADIENT_AIR, GRADIENT_SOL = 0.0065, 0.0055
SEUIL_TROUVAILLE = 0.2          # une trouvaille réelle doit ressortir au moins à ce niveau (indice sans plafond)


def trouvailles():
    """Trouvailles personnelles, lues dans le secret GitHub TROUVAILLES (jamais écrites dans le dépôt).

    Format : [{"espece": "cepe_bordeaux", "lat": 48.5, "lon": 7.3, "date": "2026-11-07"}, ...]
    « espece » peut être un groupe (« cepe ») quand l'espèce exacte n'est pas connue ; une ligne avec
    "absent": true note une sortie sans rien trouver (gardée pour mémoire, pas utilisée ici) ;
    "date_approx": true marque une date incertaine : la trouvaille sert à la carte des lieux, pas au réglage de la saison.
    """
    brut = os.environ.get("TROUVAILLES", "").strip()
    if not brut:
        return pd.DataFrame(columns=["espece", "lat", "lon", "date"])
    t = pd.DataFrame(json.loads(brut))
    for drapeau in ("absent", "date_approx"):      # sorties bredouilles, dates incertaines : pas pour régler la saison
        if drapeau in t:
            t = t[~t[drapeau].fillna(False).astype(bool)]
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    return t[["espece", "lat", "lon", "date"]].reset_index(drop=True)


def attribuer_groupes(t, meteo):
    """Une trouvaille notée « cèpe » sans précision est attribuée au cèpe que la météo de ce jour-là
    explique le mieux (en novembre en montagne, ce sera le cèpe de Bordeaux)."""
    for i in t.index[t.espece.isin(list(GROUPES))]:
        candidats = membres(t.loc[i, "espece"])
        P1, T1, M1, ok = series(meteo, t.loc[[i]])
        if len(ok):
            mois = int(t.loc[i, "date"].month)
            valeurs = {e: float(im.indice(e, P1, T1, M1, mois)[0]) for e in candidats}
            t.loc[i, "espece"] = max(valeurs, key=valeurs.get)
        else:
            t.loc[i, "espece"] = candidats[0]
    return t


def preparer(o):
    o["cel_lat"] = (o.lat / CELLULE).round() * CELLULE
    o["cel_lon"] = (o.lon / CELLULE).round() * CELLULE
    o["an"] = o.date.dt.year
    # Altitude de chaque observation (grille de 250 m), pour corriger les températures
    t = Transformer.from_crs(4326, CRS, always_xy=True)
    x, y = t.transform(o.lon.to_numpy(), o.lat.to_numpy())
    col = ((x - X0) // PAS).astype(int)
    lig = ((Y1 - y) // PAS).astype(int)
    alt = lire_bandes(DATA / "relief_sol.tif")["altitude"]
    ok = (col >= 0) & (col < COLS) & (lig >= 0) & (lig < LIGNES)
    o["alt"] = np.nan
    o.loc[ok, "alt"] = alt[lig[ok], col[ok]]
    o.loc[o.alt <= 0, "alt"] = np.nan
    return o.reset_index(drop=True)


def charger_observations():
    o = pd.read_csv(DATA / "observations.csv")
    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    o = o.dropna(subset=["date"])
    o = o[(o.date.dt.year >= ANNEE_MIN) & (o.date < pd.Timestamp.today().normalize())]
    return preparer(o).drop_duplicates(subset=["espece", "cel_lat", "cel_lon", "date"]).reset_index(drop=True)


def charger_fond():
    """Observations datées du groupe cible (tous champignons) : elles servent de témoins."""
    chemin = DATA / "fond.csv"
    if not chemin.exists():
        return None
    f = pd.read_csv(chemin)
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    f = f.dropna(subset=["date"])
    f = f[(f.date.dt.year >= ANNEE_MIN) & (f.date < pd.Timestamp.today().normalize())]
    f["espece"] = f["espece"].fillna("").astype(str)
    return preparer(f).drop_duplicates(subset=["espece", "cel_lat", "cel_lon", "date"]).reset_index(drop=True)


def telecharger_meteo(o):
    """Météo quotidienne par cellule et par année, sur la période couverte par les observations."""
    chemin = DATA / "meteo_historique.csv"
    deja = pd.read_csv(chemin, parse_dates=["date"]) if chemin.exists() else pd.DataFrame()
    groupes = o.groupby(["cel_lat", "cel_lon", "an"]).date.agg(["min", "max"]).reset_index()
    nouveaux = []
    for i, g in enumerate(groupes.itertuples(), start=1):
        debut = (g.min - pd.Timedelta(days=AVANT)).date()
        fin = g.max.date()
        if not deja.empty:
            ici = deja[(deja.cel_lat == g.cel_lat) & (deja.cel_lon == g.cel_lon)]
            if not ici.empty and ici.date.min().date() <= debut and ici.date.max().date() >= fin and \
                    ici.date.between(pd.Timestamp(debut), pd.Timestamp(fin)).sum() >= (fin - debut).days:
                continue
        j = get(ARCHIVE, {
            "latitude": g.cel_lat, "longitude": g.cel_lon,
            "start_date": debut.isoformat(), "end_date": fin.isoformat(), "timezone": "Europe/Paris",
            "daily": "precipitation_sum,temperature_2m_mean,temperature_2m_min,soil_temperature_0_to_7cm_mean",
        }, timeout=120).json()
        d = j["daily"]
        nouveaux.append(pd.DataFrame({
            "cel_lat": g.cel_lat, "cel_lon": g.cel_lon, "elev": j.get("elevation", np.nan),
            "date": pd.to_datetime(d["time"]), "pluie": d["precipitation_sum"], "tmoy": d["temperature_2m_mean"],
            "tmin": d["temperature_2m_min"], "tsol": d["soil_temperature_0_to_7cm_mean"],
        }))
        if i % 50 == 0:
            print(f"  météo : {i}/{len(groupes)} périodes")
    tout = pd.concat([deja] + nouveaux, ignore_index=True) if nouveaux else deja
    tout = tout.drop_duplicates(subset=["cel_lat", "cel_lon", "date"], keep="last").sort_values(["cel_lat", "cel_lon", "date"])
    tout.to_csv(chemin, index=False, float_format="%.2f")
    print(f"  {len(nouveaux)} périodes téléchargées, {len(tout)} jours de météo en tout")
    return tout


def series(meteo, echantillons):
    """Séries de 31 jours (30 passés + le jour J) pour chaque échantillon (cellule, date, altitude)."""
    index = {k: g.set_index("date") for k, g in meteo.groupby(["cel_lat", "cel_lon"])}
    P, T, M, garder = [], [], [], []
    for i, e in enumerate(echantillons.itertuples()):
        g = index.get((e.cel_lat, e.cel_lon))
        if g is None:
            continue
        jours = pd.date_range(e.date - pd.Timedelta(days=30), e.date)
        bloc = g.reindex(jours)
        if bloc.pluie.isna().sum() > 2 or bloc.tsol.isna().sum() > 2:
            continue
        ecart = (bloc.elev.iloc[-1] - e.alt) if not np.isnan(e.alt) and not np.isnan(bloc.elev.iloc[-1]) else 0
        P.append(bloc.pluie.to_numpy())
        T.append(bloc.tsol.to_numpy() + GRADIENT_SOL * ecart)
        M.append(bloc.tmin.to_numpy() + GRADIENT_AIR * ecart)
        garder.append(i)
    return np.array(P), np.array(T), np.array(M), np.array(garder, dtype=int)


def auc(scores, y):
    """Aire sous la courbe ROC : 0,5 = pas mieux que le hasard, 1 = séparation parfaite."""
    rangs = pd.Series(scores).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((rangs[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def indice_sans_saison(espece, params, P, T, M):
    sauvegarde = im.PARAMS[espece], im.SAISON[espece]
    im.PARAMS[espece], im.SAISON[espece] = params, {m: 1 for m in range(1, 13)}
    try:
        return im.indice(espece, P, T, M, 1)
    finally:
        im.PARAMS[espece], im.SAISON[espece] = sauvegarde


def echantillons_espece(o, espece, rng, fond=None):
    pres = o[o.espece == espece][["cel_lat", "cel_lon", "date", "alt"]].assign(y=1)
    # Témoins 1 : autres champignons signalés (même effort d'observation)
    autres = o[o.espece != espece][["cel_lat", "cel_lon", "date", "alt"]]
    if fond is not None:
        f = fond[fond.espece != espece][["cel_lat", "cel_lon", "date", "alt"]]
        if len(f) > TEMOINS_FOND:
            f = f.iloc[rng.choice(len(f), TEMOINS_FOND, replace=False)]
        autres = pd.concat([autres, f], ignore_index=True)
    autres = autres.merge(pres[["cel_lat", "cel_lon", "date"]], how="left", indicator=True)
    autres = autres[autres._merge == "left_only"].drop(columns="_merge").assign(y=0)
    # Témoins 2 : dates au hasard, même cellule, du 1er mars au 31 décembre
    hasard = pres.loc[pres.index.repeat(TEMOINS_HASARD)].copy()
    debut = pd.to_datetime(hasard.date.dt.year.astype(str) + "-03-01")
    hasard["date"] = debut + pd.to_timedelta(rng.integers(0, 306, len(hasard)), unit="D")
    hasard["y"] = 0
    return pd.concat([pres, autres, hasard], ignore_index=True)


def grille_params(defaut):
    for dlo, dhi, tlo, thi in itertools.product([2, 3, 5, 7, 9], [10, 14, 18, 22, 26, 30],
                                                [3, 5, 7, 9, 11, 13], [11, 13, 15, 17, 19, 21]):
        if dhi - dlo >= 5 and thi - tlo >= 3:
            yield dict(defaut, delai=(dlo, dhi), tsol=(tlo, thi))


def main():
    rng = np.random.default_rng(7)
    o = charger_observations()
    t = trouvailles()
    t = preparer(t) if len(t) else t
    print(f"{len(o)} observations datées depuis {ANNEE_MIN}, {len(t)} trouvailles personnelles")
    print("Météo historique (archives Open-Meteo)...")
    meteo = telecharger_meteo(pd.concat([o, t], ignore_index=True) if len(t) else o)
    if len(t):
        t = attribuer_groupes(t, meteo)
    fond = charger_fond()
    print("Témoins : " + (f"{len(fond)} observations datées du groupe cible" if fond is not None else "autres espèces suivies"))
    precedent = json.loads((DATA / "phenologie.json").read_text()) if (DATA / "phenologie.json").exists() else {}
    resultats = {}
    for espece, defaut in im.PARAMS_DEFAUT.items():
        ech = echantillons_espece(o, espece, rng, fond)
        P, T, M, garder = series(meteo, ech)
        if len(garder) == 0:
            continue
        y = ech.y.to_numpy()[garder]
        n_pres = int(y.sum())
        a_defaut = auc(indice_sans_saison(espece, defaut, P, T, M), y)
        # Trouvailles personnelles de cette espèce : garde-fou
        tr = t[t.espece == espece] if len(t) else t
        Pt, Tt, Mt, _ = series(meteo, tr) if len(tr) else (None, None, None, [])
        mois_t = tr.date.dt.month.to_numpy() if len(tr) else []

        def respecte(prm):
            if Pt is None or len(Pt) == 0:
                return True
            sauvegarde = im.PARAMS[espece]
            im.PARAMS[espece] = prm
            try:
                return all(im.indice(espece, Pt[i:i + 1], Tt[i:i + 1], Mt[i:i + 1], int(mois_t[i]))[0] >= SEUIL_TROUVAILLE
                           for i in range(len(Pt)))
            finally:
                im.PARAMS[espece] = sauvegarde
        if n_pres < MIN_PRESENCES:
            print(f"{espece} : {n_pres} présences, trop peu pour apprendre (AUC des réglages de départ {a_defaut:.2f})")
            resultats[espece] = {"presences": n_pres, "auc_defaut": round(a_defaut, 3), "retenu": False}
            continue
        # Moitié des cellules pour apprendre, l'autre moitié pour vérifier
        cel = ech.iloc[garder][["cel_lat", "cel_lon"]].astype(str).agg("_".join, axis=1).to_numpy()
        uniques = np.unique(cel)
        test_cel = set(rng.choice(uniques, size=len(uniques) // 3, replace=False))
        test = np.array([c in test_cel for c in cel])
        meilleur, a_meilleur = None, -1
        refuses = 0
        for prm in grille_params(defaut):
            if not respecte(prm):
                refuses += 1
                continue
            a = auc(indice_sans_saison(espece, prm, P[~test], T[~test], M[~test]), y[~test])
            if a > a_meilleur:
                meilleur, a_meilleur = prm, a
        a_test_appris = auc(indice_sans_saison(espece, meilleur, P[test], T[test], M[test]), y[test])
        a_test_defaut = auc(indice_sans_saison(espece, defaut, P[test], T[test], M[test]), y[test])
        retenu = a_test_appris >= a_test_defaut + 0.02 and respecte(meilleur)
        resultats[espece] = {
            "presences": n_pres, "temoins": int(len(y) - n_pres),
            "auc_defaut": round(a_test_defaut, 3), "auc_appris": round(a_test_appris, 3), "retenu": bool(retenu),
            "delai": list(meilleur["delai"]), "tsol": list(meilleur["tsol"]),
            "trouvailles": int(len(tr)), "reglages_refuses_par_trouvailles": refuses,
        }
        print(f"{espece} : {n_pres} présences ; réglages appris : délai {meilleur['delai']} jours, "
              f"sol {meilleur['tsol']} °C ; AUC sur cellules de test {a_test_defaut:.2f} -> {a_test_appris:.2f} "
              f"({'retenu' if retenu else 'pas assez net, réglages de départ gardés'})"
              + (f" ; {len(tr)} trouvaille(s) personnelle(s) respectée(s), {refuses} réglages écartés" if len(tr) else ""))

    ecrire_json(DATA / "phenologie.json", {"date": dt.date.today().isoformat(), "especes": resultats})

    # Contrôle final avec les réglages retenus (sans jamais afficher le lieu)
    im.charger_reglages(DATA / "phenologie.json")
    for i, tr in enumerate(t.itertuples(), start=1) if len(t) else []:
        P1, T1, M1, ok = series(meteo, t.iloc[[i - 1]])
        if len(ok):
            v = im.indice(tr.espece, P1, T1, M1, tr.date.month)[0]
            print(f"Trouvaille personnelle n° {i} ({tr.espece}) : indice {v:.2f}")


if __name__ == "__main__":
    main()
