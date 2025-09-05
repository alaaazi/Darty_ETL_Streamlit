# Enrichment
"""
Script utilitaire pour charger des tables supplémentaires nécessaires au
projet Darties et enrichir la dimension des magasins avec un nom
lisible (``Magasin 1``, ``Magasin 2``, etc.).

Ce module propose deux fonctionnalités principales :

1. **Chargement des tables de paramètres** :
   - ``ListeTypeMesure.csv``
   - ``ListeTypeVal.csv``
   - ``ParamètreHistorique.csv``
   - ``ParamètrePalmarès.csv``
   - ``Type_produit.csv``
   - ``Calendrier.csv``
   Ces fichiers sont lues tels quels et retournés sous forme de
   ``pandas.DataFrame`` pour une intégration simple dans vos outils
   d’analyse ou de visualisation.

2. **Ajout du nom de magasin dans ``dim_magasin``** :
   Le fichier ``dim_magasin.csv`` généré par le pipeline contient
   actuellement quatre colonnes (identifiant magasin, identifiant ville,
   enseigne et publicité).  Pour faciliter la lecture et l’affichage
   (par exemple dans un tableau de bord), on ajoute une cinquième
   colonne ``Nom_magasin`` qui prend la forme ``Magasin X``, où ``X``
   correspond à l’identifiant du magasin.

Pour exécuter ce script, placez‑le dans le même dossier que vos
fichiers CSV et lancez :

```
python3 extra_tables.py
```

Les nouvelles tables seront sauvegardées dans le dossier courant et
pourront être utilisées directement dans votre application.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple


def load_static_tables(base_path: Path) -> Dict[str, pd.DataFrame]:
    """Charge les tables de paramètres statiques dans un dictionnaire.

    Args:
        base_path: répertoire où se trouvent les fichiers CSV.

    Returns:
        Un dictionnaire où les clés sont des noms parlants et les
        valeurs sont les ``DataFrame`` associés.
    """
    tables: Dict[str, Tuple[str, str]] = {
        'liste_type_mesure': 'ListeTypeMesure.csv',
        'liste_type_val': 'ListeTypeVal.csv',
        'parametre_historique': 'ParamètreHistorique.csv',
        'parametre_palmares': 'ParamètrePalmarès.csv',
        'type_produit_ext': 'Type_produit.csv',
        'calendrier': 'Calendrier.csv',
    }
    result: Dict[str, pd.DataFrame] = {}
    for key, filename in tables.items():
        path = base_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Le fichier {filename} est introuvable dans {base_path}")
        df = pd.read_csv(path)
        result[key] = df
    return result


def add_nom_magasin_to_dim(base_path: Path, dim_filename: str = 'dim_magasin.csv') -> pd.DataFrame:
    """Ajoute une colonne ``Nom_magasin`` à la dimension des magasins.

    Cette fonction lit le fichier ``dim_magasin.csv`` (ou un autre nom
    passé en argument), identifie la colonne contenant l'identifiant du
    magasin et crée une nouvelle colonne ``Nom_magasin`` dont le
    contenu est ``Magasin <id_magasin>``.  Le résultat est sauvegardé
    dans un nouveau fichier ``dim_magasin_avec_nom.csv`` et renvoyé.

    Args:
        base_path: répertoire où se trouve le fichier ``dim_magasin``.
        dim_filename: nom du fichier de dimension magasin à traiter.

    Returns:
        Un ``DataFrame`` représentant la table enrichie.
    """
    dim_path = base_path / dim_filename
    if not dim_path.exists():
        raise FileNotFoundError(f"Le fichier {dim_filename} est introuvable dans {base_path}")
    df = pd.read_csv(dim_path)
    # Déterminer la colonne identifiant le magasin (sensibilité à la casse)
    possible_cols = ['id_magasin', 'Id_magasin', 'ID_magasin']
    id_col = None
    for col in possible_cols:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        raise KeyError(
            "Impossible de déterminer la colonne identifiant le magasin dans "
            f"{dim_filename}. Les colonnes disponibles sont : {list(df.columns)}"
        )
    # Ajouter la colonne Nom_magasin
    df['Nom_magasin'] = 'Magasin ' + df[id_col].astype(str)
    # Sauvegarder sous un nouveau nom pour éviter d'écraser le fichier d'origine
    output_path = base_path / 'dim_magasin_avec_nom.csv'
    df.to_csv(output_path, index=False)
    return df


if __name__ == '__main__':
    # Définir le dossier contenant les CSV ; ici on prend le dossier du script
    base_path = Path(__file__).parent
    # Charger les tables de paramètres
    tables = load_static_tables(base_path)
    print("Tables de paramètres chargées :", ', '.join(tables.keys()))
    # Enrichir la dimension magasin
    dim_magasin_enrichie = add_nom_magasin_to_dim(base_path)
    print("La colonne 'Nom_magasin' a été ajoutée à dim_magasin. Le fichier sauvegardé est dim_magasin_avec_nom.csv.")