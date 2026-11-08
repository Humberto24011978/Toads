"""Étape 1 : observations des sept espèces depuis GBIF (API publique, sans compte).

Résultat : data/observations.csv
"""
import csv
import time

from commun import BBOX, DATA, ESPECES, get

API = "https://api.gbif.org/v1"
PRECISION_MAX = 1000  # mètres : au-delà, le point est trop vague pour des carrés de 250 m


def cle_taxon(nom):
    j = get(f"{API}/species/match", {"name": nom, "kingdom": "Fungi", "strict": "true"}).json()
    if j.get("matchType") == "NONE":
        return None
    return j.get("acceptedUsageKey") or j.get("usageKey")


def observations(cle):
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
        }).json()
        yield from page["results"]
        if page.get("endOfRecords", True) or debut + 300 >= 99_000:
            break
        debut += 300
        time.sleep(0.3)


def main():
    DATA.mkdir(exist_ok=True)
    total_general = 0
    with open(DATA / "observations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["espece", "nom_latin", "lon", "lat", "date", "precision_m"])
        for espece, infos in ESPECES.items():
            vus, total = set(), 0
            for nom in infos["latin"]:
                cle = cle_taxon(nom)
                if not cle:
                    print(f"  introuvable dans GBIF : {nom}")
                    continue
                for o in observations(cle):
                    if o["key"] in vus:
                        continue
                    vus.add(o["key"])
                    prec = o.get("coordinateUncertaintyInMeters")
                    if prec and prec > PRECISION_MAX:
                        continue
                    w.writerow([espece, nom, round(o["decimalLongitude"], 5), round(o["decimalLatitude"], 5),
                                (o.get("eventDate") or "")[:10], prec or ""])
                    total += 1
            print(f"{infos['nom']} : {total} observations")
            total_general += total
    print(f"Total : {total_general} observations")


if __name__ == "__main__":
    main()
