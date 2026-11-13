"""Étape 1 : observations depuis GBIF, complétées par iNaturalist (API publiques, sans compte).

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
INAT = "https://api.inaturalist.org/v1/observations"   # complément : observations validées absentes de GBIF
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


def inaturalist(nom, **filtres):
    """Observations iNaturalist validées (« niveau recherche ») d'un taxon dans l'emprise, position non floutée.

    GBIF ne reçoit d'iNaturalist que les observations sous licence ouverte : les autres (un quart environ)
    ne se trouvent qu'ici. Seules la position, la date et la précision sont reprises.
    """
    lon1, lat1, lon2, lat2 = BBOX
    dernier = 0
    while True:
        page = get(INAT, {
            **({} if "taxon_id" in filtres else {"taxon_name": nom}), "quality_grade": "research", "geoprivacy": "open", "taxon_geoprivacy": "open",
            "swlng": lon1, "swlat": lat1, "nelng": lon2, "nelat": lat2,
            "per_page": 200, "order_by": "id", "order": "asc", "id_above": dernier, **filtres,
        }).json()
        resultats = page.get("results", [])
        for o in resultats:
            g = (o.get("geojson") or {}).get("coordinates")
            if g and len(g) == 2:
                yield {"lon": float(g[0]), "lat": float(g[1]), "date": (o.get("observed_on") or "")[:10],
                       "precision": o.get("positional_accuracy"), "taxon": ((o.get("taxon") or {}).get("name") or "")}
        if len(resultats) < 200:
            break
        dernier = resultats[-1]["id"]
        time.sleep(1.1)                     # iNaturalist demande de rester autour d'une requête par seconde


def cle_doublon(lon, lat, date):
    """Même endroit (à ~100 m près) le même jour : la même observation vue par deux sources."""
    return (round(float(lon), 3), round(float(lat), 3), date)


def especes_suivies():
    total_general, cles_suivies = 0, {}
    with open(DATA / "observations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["espece", "nom_latin", "lon", "lat", "date", "precision_m", "pays"])
        for espece, infos in ESPECES.items():
            vus, total, precis, deja, ajouts = set(), 0, 0, set(), 0
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
                    deja.add(cle_doublon(o["decimalLongitude"], o["decimalLatitude"], (o.get("eventDate") or "")[:10]))
                    total += 1
                    precis += bool(prec and prec <= PRECISION_FOND)
            for nom in infos["latin"]:
                try:
                    for o in inaturalist(nom):
                        cle = cle_doublon(o["lon"], o["lat"], o["date"])
                        prec = o["precision"]
                        if cle in deja or (prec and prec > PRECISION_MAX):
                            continue
                        deja.add(cle)
                        w.writerow([espece, nom, round(o["lon"], 5), round(o["lat"], 5), o["date"], prec or "", ""])
                        total += 1
                        ajouts += 1
                        precis += bool(prec and prec <= PRECISION_FOND)
                except Exception as e:          # complément : son absence n'arrête rien
                    print(f"  iNaturalist indisponible pour {nom} : {e}")
            if ajouts:
                print(f"  dont {ajouts} observations iNaturalist absentes de GBIF")
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
    gbif = n
    latins = {nom: espece for espece, infos in ESPECES.items() for nom in infos["latin"]}
    try:
        with open(DATA / "fond.csv", encoding="utf-8") as f:
            deja = {cle_doublon(l["lon"], l["lat"], l["date"]) for l in csv.DictReader(f)}
        with open(DATA / "fond.csv", "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for nom, rang in GROUPE_CIBLE.items():
                # Les grands groupes se désignent par leur numéro : la recherche par nom ne les trouve pas
                taxons = get(INAT.replace("observations", "taxa"), {"q": nom, "rank": rang.lower()}).json()["results"]
                numero = next(t["id"] for t in taxons if t["name"] == nom)
                for o in inaturalist(nom, taxon_id=numero, acc_below=PRECISION_FOND + 1):
                    cle = cle_doublon(o["lon"], o["lat"], o["date"])
                    if cle in deja or not o["precision"]:
                        continue
                    deja.add(cle)
                    suivie = next((e for latin, e in latins.items() if o["taxon"].startswith(latin)), "")
                    w.writerow([round(o["lon"], 5), round(o["lat"], 5), o["date"], suivie, ""])
                    n += 1
                    if n - gbif >= 40_000:
                        break
    except Exception as e:
        print(f"  iNaturalist indisponible pour le groupe cible : {e}")
    print(f"Groupe cible : {n} observations précises de champignons dans l'emprise (dont {n - gbif} d'iNaturalist seul)")
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
