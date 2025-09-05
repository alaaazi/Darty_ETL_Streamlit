# Generate static tables
"""
Génère l'ensemble des tables CSV nécessaires pour le projet Darties ainsi
que la dimension des magasins enrichie avec un nom lisible.

Ce script crée à la volée les fichiers suivants dans le dossier où il est
exécuté :

* ``ListeTypeMesure.csv`` : deux types de mesure ("Mois" et "Cumul de janvier au mois").
* ``ListeTypeVal.csv`` : trois types de valeurs ("CA", "Ventes", "Marge").
* ``ParamètreHistorique.csv`` : paramètres pour l'affichage de l'historique
  avec un ordre logique.
* ``ParamètrePalmarès.csv`` : paramètres pour le classement/palmarès
  (chiffre d'affaires, ventes, marge).
* ``Type_produit.csv`` : liste des familles de produits (DVD, Fours, Hifi).
* ``Calendrier.csv`` : dimension calendrier générée de manière
  déterministe pour toutes les dates du 1er janvier 2019 au 31 décembre 2030.
  La table comporte les colonnes ``Date``, ``Année``, ``Mois``, ``Numéro mois``,
  ``Semestre``, ``Numéro semaine``, ``Jour``, ``Numéro jour``, ``Est jour semaine``
  et ``Mois/Année``.
* ``dim_magasin_avec_nom.csv`` : version enrichie de ``dim_magasin.csv`` avec
  une colonne ``Nom_magasin`` générée à partir de l'identifiant du magasin.

Exécution :

```
python3 generate_static_tables.py
```

Assurez‑vous que ``dim_magasin.csv`` se trouve dans le même répertoire si
vous souhaitez produire le fichier enrichi.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def generate_liste_type_mesure(path: Path) -> None:
    df = pd.DataFrame({'Valeur': ['Mois', 'Cumul de janvier au mois']})
    df.to_csv(path / 'ListeTypeMesure.csv', index=False)


def generate_liste_type_val(path: Path) -> None:
    df = pd.DataFrame({'Valeur': ['CA', 'Ventes', 'Marge']})
    df.to_csv(path / 'ListeTypeVal.csv', index=False)


def generate_parametre_historique(path: Path) -> None:
    df = pd.DataFrame({
        'NomTechnique': ['Par date', 'Cumul depuis Janvier'],
        'Ordre': [0, 1]
    })
    df.to_csv(path / 'ParamètreHistorique.csv', index=False)


def generate_parametre_palmares(path: Path) -> None:
    df = pd.DataFrame({
        'NomTechnique': ['CA Réel', 'Ventes Réel', 'Marge Réel'],
        'Ordre': [0, 1, 2]
    })
    df.to_csv(path / 'ParamètrePalmarès.csv', index=False)


def generate_type_produit(path: Path) -> None:
    df = pd.DataFrame({'type_produit': ['DVD', 'Fours', 'Hifi']})
    df.to_csv(path / 'Type_produit.csv', index=False)


def generate_calendrier(path: Path, start_date: str = '2019-01-01', end_date: str = '2030-12-31') -> None:
    """Génère une dimension calendrier quotidienne.

    Args:
        path: répertoire de sortie.
        start_date: date de début incluse (format AAAA-MM-JJ).
        end_date: date de fin incluse (format AAAA-MM-JJ).
    """
    # Construire la plage de dates
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame({'Date': dates})
    df['Année'] = df['Date'].dt.year
    # Nom des mois et des jours en français (en minuscules)
    month_names_fr = {
        1: 'janvier', 2: 'fevrier', 3: 'mars', 4: 'avril', 5: 'mai', 6: 'juin',
        7: 'juillet', 8: 'aout', 9: 'septembre', 10: 'octobre', 11: 'novembre', 12: 'decembre'
    }
    day_names_fr = {
        0: 'lundi', 1: 'mardi', 2: 'mercredi', 3: 'jeudi', 4: 'vendredi', 5: 'samedi', 6: 'dimanche'
    }
    df['Mois'] = df['Date'].dt.month.map(month_names_fr)
    df['Numéro mois'] = df['Date'].dt.month
    df['Semestre'] = ((df['Date'].dt.month - 1) // 6) + 1
    # ISO semaine
    df['Numéro semaine'] = df['Date'].dt.isocalendar().week
    df['Jour'] = df['Date'].dt.weekday.map(day_names_fr)
    df['Numéro jour'] = df['Date'].dt.day
    df['Est jour semaine'] = df['Date'].dt.weekday < 5  # lundi=0 ... vendredi=4
    df['Mois/Année'] = df['Mois'] + ' (' + df['Année'].astype(str) + ')'
    # Formater la date comme string "YYYY-MM-DD 00:00:00.000" pour cohérence avec l'exemple
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d 00:00:00.000')
    df.to_csv(path / 'Calendrier.csv', index=False)


def enrichir_dim_magasin(path: Path, dim_filename: str = 'dim_magasin.csv') -> None:
    dim_path = path / dim_filename
    if not dim_path.exists():
        print(f"Attention : {dim_filename} introuvable. La dimension magasin ne sera pas enrichie.")
        return
    df = pd.read_csv(dim_path)
    # Chercher la colonne identifiant du magasin
    possible_cols = ['id_magasin', 'Id_magasin', 'ID_magasin']
    id_col = next((col for col in possible_cols if col in df.columns), None)
    if id_col is None:
        print(f"Impossible de trouver une colonne d'identifiant parmi {possible_cols} dans {dim_filename}.")
        return
    df['Nom_magasin'] = 'Magasin ' + df[id_col].astype(str)
    df.to_csv(path / 'dim_magasin_avec_nom.csv', index=False)


if __name__ == '__main__':
    # Répertoire courant
    base_path = Path(__file__).parent
    # Générer les tables statiques
    generate_liste_type_mesure(base_path)
    generate_liste_type_val(base_path)
    generate_parametre_historique(base_path)
    generate_parametre_palmares(base_path)
    generate_type_produit(base_path)
    generate_calendrier(base_path)
    # Enrichir la dimension magasin si elle existe
    enrichir_dim_magasin(base_path)
    print("Les tables statiques et la dimension magasin enrichie ont été générées avec succès.")