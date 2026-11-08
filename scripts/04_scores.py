"""Étape 4 : score de potentiel (0 à 100) par espèce et par carré de 250 m.

Le score combine :
- des règles écologiques (arbres, milieu, acidité du sol, altitude) ;
- un modèle (forêt aléatoire) appris sur les observations GBIF, qui affine les règles.
Les règles servent de garde-fou : un champ de maïs reste à zéro même si des
observations voisines existent.

Résultat : data/scores.tif (une bande par espèce)
"""
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier

from commun import COLS, DATA, ESPECES, LIGNES, PAS, X0, Y1, ecrire_bandes, ecrire_json, lire_bandes

# Affinité de chaque espèce forestière pour chaque type de peuplement (0 à 1)
ARBRES = {
    "cepe":             dict(chene=1, hetre=1, chataignier=1, feuillus=.8, mixte=.9, sapin_epicea=.9, pin=.8,
                             douglas_meleze=.6, coniferes=.7, peuplier_robinier=.1, foret_autre=.4, lande=.2),
    "girolle":          dict(chene=1, hetre=1, chataignier=1, feuillus=.8, mixte=.9, sapin_epicea=.8, pin=.7,
                             douglas_meleze=.5, coniferes=.6, peuplier_robinier=.1, foret_autre=.4, lande=.2),
    "trompette":        dict(chene=.9, hetre=1, chataignier=.6, feuillus=.8, mixte=.6, sapin_epicea=.3, pin=.2,
                             douglas_meleze=.2, coniferes=.2, peuplier_robinier=.1, foret_autre=.3, lande=0),
    "chanterelle_tube": dict(chene=.4, hetre=.6, chataignier=.4, feuillus=.4, mixte=.9, sapin_epicea=1, pin=.7,
                             douglas_meleze=.6, coniferes=.8, peuplier_robinier=0, foret_autre=.3, lande=.2),
    "pied_de_mouton":   dict(chene=.9, hetre=1, chataignier=.7, feuillus=.8, mixte=.9, sapin_epicea=.9, pin=.6,
                             douglas_meleze=.5, coniferes=.7, peuplier_robinier=.1, foret_autre=.4, lande=.1),
}

# Affinité des espèces de prairie pour chaque milieu ouvert (0 à 1)
MILIEUX = {
    "coulemelle":    dict(prairie=.6, pelouse=.8, transition=.8, lande_clc=.6, agri_naturel=.4, lande=.6),
    "rose_des_pres": dict(prairie=1, pelouse=.9, agri_naturel=.5, transition=.2),
    "coprin":        dict(prairie=.9, espace_vert=1, agri_naturel=.6, pelouse=.5, urbain=.15),
}

# pH préféré, altitude préférée (m)
SOL_ALT = {
    "cepe":             dict(ph=(3.8, 6.5), alt=(100, 1200)),
    "girolle":          dict(ph=(3.5, 6.0), alt=(100, 1200)),
    "trompette":        dict(ph=(5.0, 8.0), alt=(100, 1000)),
    "chanterelle_tube": dict(ph=(3.5, 5.5), alt=(250, 1350)),
    "coulemelle":       dict(ph=(4.5, 8.0), alt=(100, 1100)),
    "rose_des_pres":    dict(ph=(5.5, 8.0), alt=(100, 1100)),
    "coprin":           dict(ph=(5.5, 8.5), alt=(100, 1000)),
    "pied_de_mouton":   dict(ph=(4.5, 7.5), alt=(100, 1300)),
}

VARIABLES_OCC = ["chene", "hetre", "chataignier", "feuillus", "mixte", "sapin_epicea", "pin", "douglas_meleze",
                 "coniferes", "peuplier_robinier", "lande", "foret", "prairie", "pelouse", "agri_naturel",
                 "transition", "cultures", "espace_vert", "urbain", "lisiere"]
VARIABLES_RELIEF = ["altitude", "pente", "nordicite", "ph"]
MIN_OBS = 25


def dans(v, lo, hi, marge):
    """1 dans l'intervalle, puis décroît jusqu'à 0 à « marge » en dehors."""
    d = np.maximum(lo - v, 0) + np.maximum(v - hi, 0)
    return np.clip(1 - d / marge, 0, 1)


def score_regles(espece, occ, rel):
    f = {k: v.astype("float32") / 100 for k, v in occ.items()}
    ph = rel["ph"] / 10
    alt = rel["altitude"].astype("float32")
    if espece in ARBRES:
        hab = sum(a * f[cat] for cat, a in ARBRES[espece].items())
        hab = np.clip(hab / 0.6, 0, 1)                       # 60 % de bon peuplement suffit
    else:
        hab = sum(a * f[m] for m, a in MILIEUX[espece].items())
        hab = np.clip(hab / 0.5, 0, 1)
        if espece == "coulemelle":                           # elle adore les lisières
            hab = 0.5 * hab + 0.5 * np.clip(f["lisiere"] / 0.12, 0, 1)
    p = SOL_ALT[espece]
    f_ph = dans(ph, *p["ph"], 1.0)
    f_alt = dans(alt, *p["alt"], 250)
    return hab * (0.4 + 0.6 * f_ph) * f_alt


def cases_observations(obs):
    """Convertit les points (lon, lat) en indices de carré de la grille."""
    t = Transformer.from_crs(4326, 2154, always_xy=True)
    x, y = t.transform(obs["lon"].to_numpy(), obs["lat"].to_numpy())
    col = ((x - X0) // PAS).astype(int)
    lig = ((Y1 - y) // PAS).astype(int)
    ok = (col >= 0) & (col < COLS) & (lig >= 0) & (lig < LIGNES)
    out = obs[ok].copy()
    out["case"] = lig[ok] * COLS + col[ok]
    return out


def rang(v):
    """Rang centile (0 à 1)."""
    return pd.Series(v).rank(pct=True).to_numpy()


def main():
    occ = lire_bandes(DATA / "occupation.tif")
    rel = lire_bandes(DATA / "relief_sol.tif")
    masque = occ["masque"].ravel() > 0
    zone = np.flatnonzero(masque)
    X = np.column_stack([occ[v].ravel()[zone] for v in VARIABLES_OCC] +
                        [rel[v].ravel()[zone] for v in VARIABLES_RELIEF]).astype("float32")
    pos = np.full(LIGNES * COLS, -1, dtype=int)
    pos[zone] = np.arange(len(zone))

    obs = pd.read_csv(DATA / "observations.csv")
    obs = cases_observations(obs)
    obs = obs[masque[obs["case"]]]
    rng = np.random.default_rng(42)
    resultats, resume = {}, {}

    for espece, infos in ESPECES.items():
        regles = score_regles(espece, occ, rel).ravel()[zone]
        presences = np.unique(obs.loc[obs.espece == espece, "case"])
        n = len(presences)
        if n >= MIN_OBS:
            # Fond : carrés au hasard + carrés où d'autres espèces ont été vues
            # (cela compense le fait que les gens observent surtout près des chemins)
            autres = np.unique(obs.loc[obs.espece != espece, "case"])
            autres = np.setdiff1d(autres, presences)
            hasard = rng.choice(zone, size=min(len(zone), max(5000, 10 * n)), replace=False)
            fond = np.setdiff1d(np.union1d(hasard, autres), presences)
            lignes = np.concatenate([pos[presences], pos[fond]])
            y = np.concatenate([np.ones(n), np.zeros(len(fond))])
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                        class_weight="balanced_subsample", n_jobs=-1, random_state=42)
            rf.fit(X[lignes], y)
            modele = rang(rf.predict_proba(X)[:, 1])
            final = regles * (0.5 + 0.5 * modele)
            importances = sorted(zip(rf.feature_importances_, VARIABLES_OCC + VARIABLES_RELIEF), reverse=True)[:5]
            info_modele = [nom for _, nom in importances]
        else:
            final = regles * 0.75
            info_modele = None
        # Mise à l'échelle : les 1 % meilleurs carrés atteignent 100
        positifs = final[final > 0]
        ref = np.percentile(positifs, 99) if len(positifs) else 1
        score = np.clip(final / max(ref, 1e-6), 0, 1) * 100
        grille = np.zeros(LIGNES * COLS, dtype="float32")
        grille[zone] = score
        resultats[espece] = np.rint(grille.reshape(LIGNES, COLS))
        resume[espece] = {"observations": int(n), "modele": info_modele is not None,
                          "variables_cles": info_modele,
                          "part_favorable": round(float((score >= 50).mean() * 100), 1)}
        print(f"{infos['nom']} : {n} carrés observés, modèle {'oui' if info_modele else 'non'}, "
              f"{resume[espece]['part_favorable']} % de la zone favorable")
        if info_modele:
            print(f"  variables les plus utiles : {', '.join(info_modele)}")

    ecrire_bandes(DATA / "scores.tif", resultats, "uint8")
    ecrire_json(DATA / "scores_resume.json", resume)


if __name__ == "__main__":
    main()
