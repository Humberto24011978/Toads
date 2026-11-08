"""Indice « c'est le moment » : la météo récente est-elle favorable à la pousse ?

Principe, pour chaque espèce :
1. une pluie déclenchante (au moins 20 mm sur deux jours pour un effet plein) ;
2. un délai depuis cette pluie, propre à l'espèce (le temps que les champignons sortent) ;
3. une température du sol favorable pendant la pousse, c'est-à-dire dans les jours qui suivent la pluie
   (et non au moment de la cueillette : des cèpes sortis par temps doux restent là quand il fraîchit) ;
4. un sol pas trop froid aujourd'hui, pas de gel marqué, et un coefficient de saison souple.

Toutes les séries sont quotidiennes, du plus ancien au plus récent ; la dernière colonne est aujourd'hui.
"""
import copy
import json
import warnings
from pathlib import Path

import numpy as np

PARAMS = {
    #                 délai (jours)  marge  T sol pousse (°C)  marge  T sol min aujourd'hui  effet du gel
    "cepe":             dict(delai=(6, 20), marge_delai=8, tsol=(9, 17), marge_t=4, tsol_actuel=4, gel=.3),
    "girolle":          dict(delai=(5, 21), marge_delai=8, tsol=(11, 19), marge_t=4, tsol_actuel=6, gel=.3),
    "trompette":        dict(delai=(7, 25), marge_delai=8, tsol=(7, 15), marge_t=4, tsol_actuel=3, gel=.6),
    "chanterelle_tube": dict(delai=(7, 28), marge_delai=10, tsol=(4, 13), marge_t=4, tsol_actuel=1, gel=.8),
    "coulemelle":       dict(delai=(5, 18), marge_delai=7, tsol=(10, 19), marge_t=4, tsol_actuel=6, gel=.3),
    "rose_des_pres":    dict(delai=(4, 14), marge_delai=6, tsol=(11, 20), marge_t=4, tsol_actuel=7, gel=.3),
    "coprin":           dict(delai=(2, 10), marge_delai=5, tsol=(6, 16), marge_t=4, tsol_actuel=3, gel=.4),
    "pied_de_mouton":   dict(delai=(7, 25), marge_delai=8, tsol=(6, 14), marge_t=4, tsol_actuel=2, gel=.7),
}

# Coefficient de saison (mois -> 0 à 1) : souple, c'est surtout la météo qui décide
SAISON = {
    "cepe":             {5: .2, 6: .6, 7: 1, 8: 1, 9: 1, 10: 1, 11: .8, 12: .3},
    "girolle":          {5: .3, 6: .8, 7: 1, 8: 1, 9: 1, 10: 1, 11: .5, 12: .1},
    "trompette":        {7: .3, 8: .7, 9: 1, 10: 1, 11: 1, 12: .6, 1: .2},
    "chanterelle_tube": {8: .4, 9: 1, 10: 1, 11: 1, 12: 1, 1: .5, 2: .2},
    "coulemelle":       {6: .5, 7: 1, 8: 1, 9: 1, 10: 1, 11: .5},
    "rose_des_pres":    {5: .3, 6: .6, 7: .8, 8: 1, 9: 1, 10: 1, 11: .5},
    "coprin":           {3: .3, 4: .6, 5: .7, 6: .5, 7: .5, 8: .6, 9: 1, 10: 1, 11: 1, 12: .5},
    "pied_de_mouton":   {7: .4, 8: .8, 9: 1, 10: 1, 11: 1, 12: .8, 1: .3},
}

PLUIE_MIN, PLUIE_ECHELLE = 5.0, 18.0  # mm sur deux jours : 20 mm -> 0,57 ; 40 mm -> 0,86 ; 60 mm -> 0,95
CUMUL_ECHELLE = 35.0                   # pluie des 14 derniers jours : 35 mm -> 0,63 ; 70 mm -> 0,86
DUREE_POUSSE = 10                      # jours après la pluie pris en compte pour la température du sol
GEL = -3.0                             # °C (minimum de l'air) sur les 3 derniers jours


def dans(v, lo, hi, marge):
    """1 dans l'intervalle, puis décroît jusqu'à 0 à « marge » en dehors."""
    d = np.maximum(lo - v, 0) + np.maximum(v - hi, 0)
    return np.clip(1 - d / marge, 0, 1)


def pic(v, lo, hi, marge, bord=0.75):
    """1 au centre de [lo, hi], « bord » aux extrémités, puis décroît jusqu'à 0 à « marge » en dehors.

    Contrairement à un simple intervalle, seul l'optimum atteint 1 : le score ne plafonne pas.
    """
    v = np.asarray(v, dtype=float)
    c, demi = (lo + hi) / 2, max((hi - lo) / 2, 1e-6)
    dedans = 1 - (1 - bord) * ((v - c) / demi) ** 2
    d = np.maximum(lo - v, 0) + np.maximum(v - hi, 0)
    dehors = bord * np.clip(1 - d / marge, 0, 1)
    return np.where((v >= lo) & (v <= hi), dedans, dehors)


def moyenne(a, axis=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(a, axis=axis)
    return np.where(np.isnan(m), 0, m)


def indice(espece, pluie, tsol, tmin, mois):
    """Indice de 0 à 1 pour chaque point. pluie, tsol, tmin : tableaux (points, jours)."""
    p = PARAMS[espece]
    f_saison = SAISON[espece].get(mois, 0)
    n, d = pluie.shape
    if f_saison == 0:
        return np.zeros(n)
    passe = np.nan_to_num(pluie[:, :-1])
    deux = passe.copy()
    deux[:, 1:] += passe[:, :-1]
    meilleur = np.zeros(n)
    for j in range(d - 1):
        jours = (d - 1) - j                        # il y a combien de jours
        w = float(pic(jours, *p["delai"], p["marge_delai"]))
        if w == 0:
            continue
        r = 1 - np.exp(-np.maximum(deux[:, j] - PLUIE_MIN, 0) / PLUIE_ECHELLE)
        debut, fin = j + 1, min(j + 1 + DUREE_POUSSE, d)
        t_pousse = moyenne(tsol[:, debut:fin]) if fin > debut else tsol[:, -1]
        f_t = pic(t_pousse, *p["tsol"], p["marge_t"])
        meilleur = np.maximum(meilleur, r * w * f_t)
    # Un sol bien arrosé sur deux semaines aide (plusieurs pluies valent mieux qu'une seule)
    f_cumul = 1 - np.exp(-passe[:, -14:].sum(axis=1) / CUMUL_ECHELLE)
    t_actuel = moyenne(tsol[:, -3:])
    f_actuel = dans(t_actuel, p["tsol_actuel"], 40, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        gel = np.nanmin(tmin[:, -4:-1], axis=1) < GEL
    f_gel = np.where(gel, p["gel"], 1)
    return f_saison * meilleur * (0.6 + 0.4 * f_cumul) * f_actuel * f_gel


def jours_depuis_bonne_pluie(pluie):
    passe = np.nan_to_num(pluie[:, :-1])
    deux = passe.copy()
    deux[:, 1:] += passe[:, :-1]
    bonne = deux >= 15
    jours = np.full(len(pluie), 99)
    for i in range(len(pluie)):
        idx = np.flatnonzero(bonne[i])
        if len(idx):
            jours[i] = passe.shape[1] - idx[-1]
    return jours


PARAMS_DEFAUT = copy.deepcopy(PARAMS)


def charger_reglages(chemin=None):
    """Applique les réglages appris sur les observations (data/phenologie.json), quand ils sont retenus."""
    chemin = Path(chemin) if chemin else Path(__file__).resolve().parent.parent / "data" / "phenologie.json"
    if not chemin.exists():
        return {}
    especes = json.loads(chemin.read_text(encoding="utf-8")).get("especes", {})
    for e, r in especes.items():
        if e not in PARAMS:
            continue
        PARAMS[e] = dict(PARAMS_DEFAUT[e])
        if r.get("retenu"):
            PARAMS[e].update(delai=tuple(r["delai"]), tsol=tuple(r["tsol"]))
    return especes


charger_reglages()
