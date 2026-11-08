# Toads · Coins à champignons

Carte gratuite des zones favorables aux champignons en **Alsace, Lorraine et Vosges**
(Bas-Rhin, Haut-Rhin, Meurthe-et-Moselle, Meuse, Moselle, Vosges), mise à jour chaque matin.

**Site :** https://humberto24011978.github.io/Toads/

Espèces : cèpe, girolle, trompette de la mort, chanterelle en tube, coulemelle, rosé des prés, coprin chevelu, pied de mouton.

## Comment ça marche

1. **Potentiel** (calculé une fois, dans `data/scores.tif`) : pour chaque carré de 250 m, un score de 0 à 100
   combine des règles écologiques (essences d'arbres, milieu, acidité du sol, altitude) et un modèle appris
   sur les observations publiques GBIF.
2. **Aujourd'hui** (chaque matin) : le potentiel est multiplié par un indice météo propre à chaque espèce :
   saison, pluie des 14 derniers jours, délai depuis la dernière bonne pluie, température, gel.

## Données (toutes ouvertes et gratuites)

| Donnée | Source |
| --- | --- |
| Observations | [GBIF](https://www.gbif.org) |
| Essences forestières | BD Forêt v2, IGN (Géoplateforme) |
| Occupation du sol | Corine Land Cover 2018 (Géoplateforme) |
| Limites départementales | ADMIN EXPRESS, IGN |
| Altitude | Copernicus DEM GLO-30 |
| pH du sol | SoilGrids, ISRIC |
| Météo | [Open-Meteo](https://open-meteo.com) |

## Lancer les calculs

Tout tourne sur GitHub, rien à installer :

- **Préparation des données** (onglet *Actions*, à lancer à la main) : télécharge les données et recalcule le potentiel.
- **Cartes du jour** : se lance seule chaque matin, ou à la main.

## Avertissement

La carte indique des zones favorables, jamais une certitude. Elle ne remplace pas l'identification :
en cas de doute, faites vérifier votre cueillette par un pharmacien ou une société mycologique.
