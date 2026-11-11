"""Étape 4 : score de potentiel (0 à 1000) par espèce et par carré de 100 m.

Le score combine :
- des règles écologiques (arbres de la BD Forêt, milieu, acidité du sol, altitude, exposition pour les espèces
  qui aiment la chaleur, hauteur réelle des arbres mesurée par LiDAR) : elles disent où l'habitat existe ;
- un modèle (forêt aléatoire) qui classe les carrés où l'habitat existe. Il apprend sur les observations
  précises de toute l'emprise (France, Allemagne, Suisse, Luxembourg), avec des cartes identiques des deux
  côtés de la frontière : relief, sol, essences d'arbres à 30 m, humidité liée au relief.
  Il compare les carrés où l'espèce a été vue à ceux où d'autres champignons ont été notés (« groupe cible ») :
  il apprend ainsi ce que l'espèce préfère, et non les endroits où les gens se promènent.
Moins une espèce a d'observations, moins le modèle pèse face aux règles.
La qualité du modèle est mesurée sur des secteurs de 20 km qu'il n'a pas vus (AUC : 0,5 = hasard, 1 = parfait).

Résultats : data/scores.tif (une bande par espèce) et data/scores_resume.json
"""
import json
import os

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from commun import COLS, DATA, ESPECES, LIGNES, PAS, X0, Y1, ecrire_bandes, ecrire_json, lire_bandes

# Affinité de chaque espèce forestière pour chaque type de peuplement de la BD Forêt (0 à 1)
ARBRES = {
    # Cèpe de Bordeaux : le plus répandu, sous épicéas, sapins, hêtres, chênes ; sols acides, plaine et montagne
    "cepe_bordeaux":    dict(chene=.8, hetre=1, chataignier=.8, feuillus=.6, mixte=1, sapin_epicea=1, pin=.7,
                             douglas_meleze=.5, coniferes=.7, peuplier_robinier=.05, foret_autre=.4, lande=.2),
    # Cèpe d'été : feuillus (chênes, hêtres, châtaigniers, charmes), lisières et bois clairs, plaine et collines
    "cepe_ete":         dict(chene=1, hetre=.9, chataignier=1, feuillus=.8, mixte=.6, sapin_epicea=.2, pin=.3,
                             douglas_meleze=.15, coniferes=.2, peuplier_robinier=.1, foret_autre=.4, lande=.1),
    # Cèpe bronzé : chênes et châtaigniers des coteaux chauds, rare dans la région
    "cepe_bronze":      dict(chene=1, hetre=.4, chataignier=1, feuillus=.5, mixte=.3, sapin_epicea=0, pin=.15,
                             douglas_meleze=0, coniferes=.05, peuplier_robinier=.05, foret_autre=.3, lande=.1),
    # Cèpe des pins : pins sylvestres sur sable et grès, sapins et épicéas d'altitude, parfois hêtres
    "cepe_pins":        dict(chene=.3, hetre=.45, chataignier=.4, feuillus=.2, mixte=.7, sapin_epicea=.7, pin=1,
                             douglas_meleze=.4, coniferes=.7, peuplier_robinier=0, foret_autre=.3, lande=.3),
    "girolle":          dict(chene=1, hetre=1, chataignier=1, feuillus=.8, mixte=.9, sapin_epicea=.8, pin=.7,
                             douglas_meleze=.5, coniferes=.6, peuplier_robinier=.1, foret_autre=.4, lande=.2),
    "trompette":        dict(chene=.9, hetre=1, chataignier=.6, feuillus=.8, mixte=.6, sapin_epicea=.3, pin=.2,
                             douglas_meleze=.2, coniferes=.2, peuplier_robinier=.1, foret_autre=.3, lande=0),
    "chanterelle_tube": dict(chene=.4, hetre=.6, chataignier=.4, feuillus=.4, mixte=.9, sapin_epicea=1, pin=.7,
                             douglas_meleze=.6, coniferes=.8, peuplier_robinier=0, foret_autre=.3, lande=.2),
    "pied_de_mouton":   dict(chene=.9, hetre=1, chataignier=.7, feuillus=.8, mixte=.9, sapin_epicea=.9, pin=.6,
                             douglas_meleze=.5, coniferes=.7, peuplier_robinier=.1, foret_autre=.4, lande=.1),
    # Morille : forêts alluviales (frênes, ormes, peupliers), feuillus frais, vergers et jardins, lisières
    "morille":          dict(chene=.6, hetre=.3, chataignier=.1, feuillus=1, mixte=.4, sapin_epicea=.15, pin=.2,
                             douglas_meleze=.1, coniferes=.1, peuplier_robinier=1, foret_autre=.6, lande=.1,
                             agri_naturel=.35, espace_vert=.3),
}

# Affinité des espèces de prairie pour chaque milieu ouvert (0 à 1)
MILIEUX = {
    "coulemelle":    dict(prairie=.6, pelouse=.8, transition=.8, lande_clc=.6, agri_naturel=.4, lande=.6),
    "rose_des_pres": dict(prairie=1, pelouse=.9, agri_naturel=.5, transition=.2),
    "coprin":        dict(prairie=.9, espace_vert=1, agri_naturel=.6, pelouse=.5, urbain=.15),
}

# pH préféré (avec la marge au-delà de laquelle il ne convient plus), altitude préférée (m) et sa marge.
# Les valeurs de SoilGrids sont lissées : en forêt elles vont d'environ 5,0 (grès des Vosges)
# à 6,5 (calcaires lorrains). Les seuils sont réglés sur cette échelle.
SOL_ALT = {
    "cepe_bordeaux":    dict(ph=(3.5, 5.6), marge=1.1, alt=(150, 1300)),
    "cepe_ete":         dict(ph=(4.5, 6.6), marge=1.0, alt=(100, 700)),
    "cepe_bronze":      dict(ph=(4.3, 6.2), marge=0.9, alt=(100, 450), marge_alt=200),
    "cepe_pins":        dict(ph=(3.5, 5.3), marge=0.9, alt=(150, 1350)),
    "girolle":          dict(ph=(3.5, 5.4), marge=1.1, alt=(100, 1200)),
    "trompette":        dict(ph=(5.8, 8.5), marge=0.9, alt=(100, 1000)),
    "chanterelle_tube": dict(ph=(3.5, 5.3), marge=0.9, alt=(300, 1350)),
    "coulemelle":       dict(ph=(4.5, 8.0), marge=1.0, alt=(100, 1100)),
    "rose_des_pres":    dict(ph=(5.7, 8.5), marge=0.8, alt=(100, 1100)),
    "coprin":           dict(ph=(5.6, 8.5), marge=0.8, alt=(100, 1000)),
    "pied_de_mouton":   dict(ph=(5.2, 6.8), marge=0.8, alt=(100, 1300)),
    "morille":          dict(ph=(6.2, 8.5), marge=1.0, alt=(100, 500)),
}

# Espèces qui aiment la chaleur : baisse maximale sur un versant nord bien marqué
THERMOPHILE = {"cepe_ete": .25, "cepe_bronze": .5}
# Hauteur moyenne des arbres (m) à partir de laquelle le peuplement convient pleinement.
# En dessous de 3 m (coupe rase, jeune plantation), il ne reste presque rien pour les espèces liées aux arbres.
HAUTEUR_PLEINE = {"cepe_bordeaux": 10, "cepe_ete": 12, "cepe_bronze": 12, "cepe_pins": 10, "girolle": 10,
                  "trompette": 15, "chanterelle_tube": 12, "pied_de_mouton": 12}

VARIABLES_RELIEF = ["altitude", "pente", "nordicite", "tpi", "tpi_large", "ph"]
MIN_OBS = 25                # carrés observés : en dessous, pas de modèle, les règles seules
PRECISION_MAX = 300         # mètres : précision exigée d'une observation pour apprendre à 100 m
BLOC = 200                  # carrés : secteurs de 20 km pour la validation
PLIS = 5


def dans(v, lo, hi, marge):
    """1 dans l'intervalle, puis décroît jusqu'à 0 à « marge » en dehors."""
    d = np.maximum(lo - v, 0) + np.maximum(v - hi, 0)
    return np.clip(1 - d / marge, 0, 1)


def facteur_hauteur(espece, occ, hauteur):
    """Coupes rases et jeunes plantations : la BD Forêt date, le LiDAR dit ce qu'il reste vraiment."""
    if hauteur is None or espece not in HAUTEUR_PLEINE:
        return 1.0
    connu = hauteur["h_moy"] < 255
    boise = np.clip(occ["foret"].astype("float32") / 100, 0.25, 1)
    h = np.minimum(hauteur["h_moy"].astype("float32") / 4 / boise, 45)     # hauteur de la partie boisée du carré
    f = np.clip((h - 3) / (HAUTEUR_PLEINE[espece] - 3), 0.1, 1)
    return np.where(connu, f, 1.0)


def score_regles(espece, occ, rel, hauteur=None):
    f = {k: v.astype("float32") / 100 for k, v in occ.items()}
    ph = rel["ph"] / 10
    alt = rel["altitude"].astype("float32")
    if espece in ARBRES:
        hab = sum(a * f[cat] for cat, a in ARBRES[espece].items())
        hab = np.clip(hab / 0.8, 0, 1)                       # 80 % de bon peuplement suffit
    else:
        hab = sum(a * f[m] for m, a in MILIEUX[espece].items())
        hab = np.clip(hab / 0.5, 0, 1)
        if espece == "coulemelle":                           # elle adore les lisières
            hab = 0.5 * hab + 0.5 * np.clip(f["lisiere"] / 0.12, 0, 1)
    if espece == "morille":                                  # fonds de vallée plats et bords de cours d'eau
        pente = rel["pente"].astype("float32") / 10
        creux = np.clip(-rel["tpi"].astype("float32") / 15, 0, 1)
        hab = hab * dans(pente, 0, 6, 14) * (0.85 + 0.15 * creux)
    if espece in THERMOPHILE:                                # versants nord moins favorables
        nord = np.clip(rel["nordicite"].astype("float32") / 100, 0, 1)
        hab = hab * (1 - THERMOPHILE[espece] * nord)
    hab = hab * facteur_hauteur(espece, occ, hauteur)
    p = SOL_ALT[espece]
    f_ph = dans(ph, *p["ph"], p["marge"])
    f_alt = dans(alt, *p["alt"], p.get("marge_alt", 250))
    if espece == "morille":                                  # sols calcaires ou alluvions : condition forte
        return hab * f_ph * f_alt
    return hab * (0.3 + 0.7 * f_ph) * f_alt


def cases(lon, lat):
    """Indices de carré de la grille pour des points (lon, lat) ; -1 hors de la grille."""
    t = Transformer.from_crs(4326, 2154, always_xy=True)
    x, y = t.transform(np.asarray(lon, float), np.asarray(lat, float))
    col = np.floor((x - X0) / PAS).astype(int)
    lig = np.floor((Y1 - y) / PAS).astype(int)
    ok = (col >= 0) & (col < COLS) & (lig >= 0) & (lig < LIGNES)
    return np.where(ok, lig * COLS + col, -1)


def rang(v):
    """Rang centile (0 à 1)."""
    return pd.Series(v).rank(pct=True).to_numpy()


def charger_variables(rel):
    """Variables du modèle, identiques en France et chez les voisins : relief, sol, cartes européennes."""
    noms, colonnes = [], []
    for v in VARIABLES_RELIEF:
        if v in rel:
            noms.append(v)
            colonnes.append(rel[v].ravel())
    for fichier in ("europe_arbres.tif", "europe_sol.tif"):
        if (DATA / fichier).exists():
            for nom, bande in lire_bandes(DATA / fichier).items():
                noms.append(nom)
                colonnes.append(bande.ravel())
    return noms, colonnes


def trouvailles_personnelles():
    """Trouvailles personnelles (secret TROUVAILLES, jamais écrites dans le dépôt ni affichées)."""
    brut = os.environ.get("TROUVAILLES", "").strip()
    if not brut:
        return pd.DataFrame(columns=["espece", "lat", "lon"])
    t = pd.DataFrame(json.loads(brut))
    for drapeau in ("absent", "approx"):        # sorties bredouilles et lieux approximatifs : pas pour la carte à 100 m
        if drapeau in t:
            t = t[~t[drapeau].fillna(False).astype(bool)]
    return t


def validation(X, y, blocs, rng):
    """AUC sur des secteurs de 20 km tenus à l'écart de l'apprentissage."""
    uniques = np.unique(blocs)
    pli_de = dict(zip(uniques, rng.permutation(len(uniques)) % PLIS))
    plis = np.array([pli_de[b] for b in blocs])
    hors = np.full(len(y), np.nan)
    for k in range(PLIS):
        test = plis == k
        if y[~test].sum() < 10 or y[test].sum() == 0:
            continue
        rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, max_features="sqrt",
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=k)
        rf.fit(X[~test], y[~test])
        hors[test] = rf.predict_proba(X[test])[:, 1]
    ok = ~np.isnan(hors)
    if y[ok].sum() < 10 or (1 - y[ok]).sum() < 10:
        return None
    return float(roc_auc_score(y[ok], hors[ok]))


def main():
    occ = lire_bandes(DATA / "occupation.tif")
    rel = lire_bandes(DATA / "relief_sol.tif")
    hauteur = lire_bandes(DATA / "hauteur.tif") if (DATA / "hauteur.tif").exists() else None
    acces = lire_bandes(DATA / "acces.tif") if (DATA / "acces.tif").exists() else None
    masque = occ["masque"].ravel() > 0
    zone = np.flatnonzero(masque)
    interdit = acces["interdit"].ravel()[zone] > 0 if acces else np.zeros(len(zone), bool)
    noms, colonnes = charger_variables(rel)
    print(f"Variables du modèle : {', '.join(noms)}")
    print("Hauteur des arbres : " + ("prise en compte" if hauteur else "pas encore disponible"))
    X_zone = np.column_stack([c[zone] for c in colonnes]).astype("float32")

    def variables(cellules):
        return np.column_stack([c[cellules] for c in colonnes]).astype("float32")

    obs = pd.read_csv(DATA / "observations.csv")
    obs["case"] = cases(obs["lon"], obs["lat"])
    prec = pd.to_numeric(obs.get("precision_m"), errors="coerce")
    obs = obs[(obs["case"] >= 0) & (prec.isna() | (prec <= PRECISION_MAX))]
    perso = trouvailles_personnelles()
    if len(perso):
        perso = perso.assign(case=cases(perso["lon"], perso["lat"]))
        perso = perso[perso["case"] >= 0]
        print(f"{len(perso)} trouvaille(s) personnelle(s) ajoutée(s) aux observations")

    rng = np.random.default_rng(42)
    # Fond : carrés où d'autres champignons ont été notés (groupe cible), plus quelques carrés au hasard
    if (DATA / "fond.csv").exists():
        fond = pd.read_csv(DATA / "fond.csv")
        cible = np.unique(cases(fond["lon"], fond["lat"]))
        cible = cible[cible >= 0]
        source_fond = f"groupe cible ({len(cible)} carrés)"
    else:
        cible = np.unique(obs["case"])
        source_fond = "autres espèces suivies (groupe cible pas encore téléchargé)"
    hasard = rng.choice(LIGNES * COLS, size=max(3000, len(cible) // 5), replace=False)
    print(f"Fond de comparaison : {source_fond} + {len(hasard)} carrés au hasard")

    resultats, resume = {}, {}
    for espece, infos in ESPECES.items():
        regles = score_regles(espece, occ, rel, hauteur).ravel()[zone]
        regles[interdit] = 0
        presences = np.unique(np.concatenate([
            obs.loc[obs.espece == espece, "case"].to_numpy(),
            perso.loc[perso.espece == espece, "case"].to_numpy() if len(perso) else np.array([], int)]).astype(int))
        n = len(presences)
        n_fr = int(masque[presences].sum()) if n else 0
        poids, auc, cles = 0.0, None, None
        if n >= MIN_OBS:
            fond_e = np.setdiff1d(np.union1d(cible, hasard), presences)
            cellules = np.concatenate([presences, fond_e])
            y = np.concatenate([np.ones(n), np.zeros(len(fond_e))])
            X = variables(cellules)
            blocs = (cellules // COLS // BLOC) * 1000 + (cellules % COLS) // BLOC
            auc = validation(X, y, blocs, np.random.default_rng(7))
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                        class_weight="balanced_subsample", n_jobs=-1, random_state=42)
            rf.fit(X, y)
            proba = np.concatenate([rf.predict_proba(X_zone[i:i + 500_000])[:, 1]
                                    for i in range(0, len(zone), 500_000)])
            # Le modèle départage les carrés où l'habitat existe ; il pèse d'autant plus qu'il a d'observations
            # et qu'il se vérifie sur les secteurs tenus à l'écart
            poids = 0.8 * min(1.0, n / 150) ** 0.5
            if auc is not None:
                poids *= float(np.clip((auc - 0.5) / 0.2, 0.25, 1))
            habitat = regles > 0.05
            modele = np.zeros(len(zone))
            modele[habitat] = rang(proba[habitat])
            final = regles * ((1 - poids) + poids * modele)
            cles = [nom for _, nom in sorted(zip(rf.feature_importances_, noms), reverse=True)[:6]]
        else:
            final = regles
        # Mise à l'échelle sur 1000 : seuls les 0,1 % meilleurs carrés de la zone atteignent 1000,
        # et une courbe accentue les écarts pour réserver les hautes valeurs aux meilleurs coins
        ref = np.percentile(final, 99.9)
        score = np.clip(final / max(ref, 1e-6), 0, 1) ** 1.3 * 1000
        grille = np.zeros(LIGNES * COLS, dtype="float32")
        grille[zone] = score
        resultats[espece] = np.rint(grille.reshape(LIGNES, COLS))
        resume[espece] = {"observations": int(n), "observations_region": n_fr,
                          "auc": None if auc is None else round(auc, 3), "poids_modele": round(poids, 2),
                          "variables_cles": cles, "part_favorable": round(float((score >= 500).mean() * 100), 1)}
        print(f"{infos['nom']} : {n} carrés observés (dont {n_fr} dans la région), "
              + (f"AUC {auc:.2f} sur secteurs non vus, " if auc is not None else "")
              + f"poids du modèle {poids:.2f}, {resume[espece]['part_favorable']} % de la zone favorable")
        if cles:
            print(f"  variables les plus utiles : {', '.join(cles)}")

    ecrire_bandes(DATA / "scores.tif", resultats, "uint16")
    ecrire_json(DATA / "scores_resume.json", resume)


if __name__ == "__main__":
    main()
