"""Étape 1 : observations depuis GBIF (API publique, sans compte).

Deux fichiers :
- data/observations.csv : les observations des espèces suivies, dans toute l'emprise (France, Allemagne, Suisse,
  Luxembourg, Belgique) : les mêmes forêts continuent de l'autre côté de la frontière, et les observations
  précises y sont bien plus nombreuses qu'en France ;
- data/fond.csv : toutes les observations précises (à 250 m près) de champignons à chapeau et de morilles
  (« groupe cible »). Elles disent où et quand des mycologues ont cherché : cela sert de point de comparaison
  pour ne pas confondre « bon coin » et « coin souvent visité ».
"""
import csv
import time

from commun import BBOX, DATA, ESPECES, get

API = "https://api.gbif.org/v1"
PRECISION_MAX = 5000       # mètres : au-delà, le point ne sert plus à rien (le tri fin se fait à l'étape 4)
PRECISION_FOND = 250       # mètres : précision exigée pour le groupe cible
GROUPE_CIBLE = {"Agaricomycetes": "CLASS", "Pezizomycetes": "CLASS"}


def cle_taxon(nom, rang=None):
    """Clé GBIF d'un nom latin. Refuse les correspondances approximatives ou remontées à un rang supérieur."""
    params = {"name": nom, "kingdom": "Fungi", "strict": "true"}
    if rang:
        params["rank"] = rang
    j = get(f"{API}/species/match", params).json()
    attendu = rang or ("GENUS" if " " not in nom.strip() else "SPECIES")
    if j.get("matchType") != "EXACT" or j.get("rank") != attendu:
        return None
    return j.get("acceptedUsageKey") or j.get("usageKey")


def observations(cle, **filtres):
    lon1, lat1, lon2, lat2 = BBOX
    debut = 0
    while True:
        page = get(f"{API}/occurrence/search", {
            "taxonKey": cle,
            "hasCoordinate": "true",
            "hasGeospatialIssue": "false",
            "occurrenceStatus": "PRESENT",
            "decimalLongitude": f"{lon1},{lon2}",
            "decimalLatitude": f"{lat1},{lat2}",
            "limit": 300,
            "offset": debut,
            **filtres,
        }).json()
        yield from page["results"]
        if page.get("endOfRecords", True) or debut + 300 >= 99_000:
            break
        debut += 300
        time.sleep(0.2)


def especes_suivies():
    total_general, cles_suivies = 0, {}
    with open(DATA / "observations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["espece", "nom_latin", "lon", "lat", "date", "precision_m", "pays"])
        for espece, infos in ESPECES.items():
            vus, total, precis = set(), 0, 0
            for nom in infos["latin"]:
                cle = cle_taxon(nom)
                if not cle:
                    print(f"  introuvable dans GBIF : {nom}")
                    continue
                cles_suivies[cle] = espece
                for o in observations(cle):
                    if o["key"] in vus:
                        continue
                    vus.add(o["key"])
                    prec = o.get("coordinateUncertaintyInMeters")
                    if prec and prec > PRECISION_MAX:
                        continue
                    w.writerow([espece, nom, round(o["decimalLongitude"], 5), round(o["decimalLatitude"], 5),
                                (o.get("eventDate") or "")[:10], prec or "", o.get("countryCode") or ""])
                    total += 1
                    precis += bool(prec and prec <= PRECISION_FOND)
            print(f"{infos['nom']} : {total} observations, dont {precis} précises à {PRECISION_FOND} m")
            total_general += total
    print(f"Total : {total_general} observations")
    return cles_suivies


def groupe_cible(cles_suivies):
    """Toutes les observations précises de champignons de l'emprise : où et quand on a cherché."""
    n, vus = 0, set()
    with open(DATA / "fond.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat", "date", "espece", "pays"])
        for nom, rang in GROUPE_CIBLE.items():
            cle = cle_taxon(nom, rang)
            if not cle:
                print(f"  introuvable dans GBIF : {nom}")
                continue
            for o in observations(cle, coordinateUncertaintyInMeters=f"0,{PRECISION_FOND}"):
                if o["key"] in vus:
                    continue
                vus.add(o["key"])
                suivie = cles_suivies.get(o.get("speciesKey")) or cles_suivies.get(o.get("genusKey")) or ""
                w.writerow([round(o["decimalLongitude"], 5), round(o["decimalLatitude"], 5),
                            (o.get("eventDate") or "")[:10], suivie, o.get("countryCode") or ""])
                n += 1
                if n % 6000 == 0:
                    print(f"  groupe cible : {n} observations")
    print(f"Groupe cible : {n} observations précises de champignons dans l'emprise")
    return n


def main():
    DATA.mkdir(exist_ok=True)
    cles = especes_suivies()
    try:
        groupe_cible(cles)
    except Exception as e:      # complément : l'étape 4 sait s'en passer
        print(f"Groupe cible indisponible : {e}")
        (DATA / "fond.csv").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
