# Main pipeline
"""
Ce module contient un pipeline Python complet pour reproduire
les transformations effectuées dans le projet GitHub
``inzabdou/Data-Pipeline-Reporting-Modernization-avec-GCP-dbt-Power-BI``.

L'objectif est de partir des fichiers Excel bruts fournis (2024_HISTO.xlsx
et 2025_BUDGET.xlsx) et d'obtenir des tables propres équivalentes à celles
générées par les scripts originaux (nettoyage, dénormalisation et
unpivot).  Les principales étapes de transformation sont :

1. Chargement et nettoyage de la table ``darties`` (villes) :
   - Normalisation des noms de ville en majuscule sans espaces superflus.
   - Correction de la région à l'aide d'un référentiel fourni
     (``classification_villes.csv``).  Lorsque la région du fichier d'origine
     diffère de la région de référence, la valeur est remplacée par la
     valeur du référentiel.
   - Renommage de certaines colonnes pour harmoniser le modèle de données.
   - Création d'une clé technique ``id_ville``.

2. Création de la dimension ``magasin`` à partir des colonnes ``Villes``,
   ``Enseignes`` et ``Publicité``.  Chaque magasin correspond à une
   combinaison unique (ville, enseigne).  La clé technique ``id_magasin``
   est un identifiant séquentiel et ``id_ville`` est repris de la
   dimension ``localite``.

3. Définition de la dimension ``produit``.  Comme il n'existe que trois
   familles de produits dans le périmètre (DVD, Fours et HiFi), la
   dimension est codée en dur avec les identifiants 1, 2 et 3.

4. Nettoyage des feuilles de mesures (CA, Ventes et Marge brute).  Les
   valeurs négatives sont remplacées par des ``NaN`` pour les mesures CA
   (chiffre d'affaires) et V (volume), conformément au script
   ``external_tables_creation.py`` du projet initial.  Les valeurs de
   marge brute peuvent être négatives et ne sont donc pas modifiées.

5. Unpivot des feuilles de mesures afin de convertir chaque table large
   (colonnes ``O_Janvier``, ``R_Janvier``, etc.) en un format long
   (ville, type_conso, mois, année, montant).  Cette étape reproduit
   exactement la macro dbt ``generate_unpivot_select``.

6. Fusion des trois mesures par produit et par période afin d'obtenir
   l'équivalent de la table fact ``fait_conso``.  Après fusion, les
   identifiants de dimensions sont joints pour produire les clés
   techniques utilisées dans le modèle en étoile.

Les fonctions de ce module peuvent être exécutées directement en appelant
la fonction ``run_pipeline`` en fin de fichier.  Les tables finales
``dim_localite``, ``dim_magasin``, ``dim_produit`` et ``fait_conso`` sont
retournées sous forme de DataFrame Pandas.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List


def load_classification_mapping(path: Path) -> Dict[str, str]:
    """Charge un fichier CSV contenant la correspondance ville → région.

    Le fichier attendu comporte deux colonnes : ``Villes`` et ``REGION``.
    Les valeurs sont mises en majuscules et épurées des espaces avant d'être
    insérées dans le dictionnaire.

    Args:
        path: chemin du fichier CSV de classification.

    Returns:
        Un dictionnaire mapping ville vers région.
    """
    df = pd.read_csv(path)
    # Normaliser les clés et valeurs en majuscule
    mapping = {
        str(row['Villes']).strip().upper(): str(row['REGION']).strip().upper()
        for _, row in df.iterrows()
    }
    return mapping


def clean_darties(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Nettoie la table des localités ``darties``.

    - Met en majuscule la colonne ``Villes`` et supprime les espaces
      superflus.
    - Corrige la colonne ``REGION`` en utilisant le dictionnaire fourni.
      Si la région du fichier diffère de celle du référentiel, on remplace
      par la valeur du référentiel.
    - Renomme les colonnes pour harmoniser la nomenclature.
    - Retourne une nouvelle table avec un identifiant ``id_ville``.

    Args:
        df: table ``darties`` brute chargée depuis Excel.
        mapping: dictionnaire ville → région.

    Returns:
        Table nettoyée avec id_ville et colonnes renommées.
    """
    df_clean = df.copy()
    # Normaliser les noms de ville
    df_clean['Villes'] = df_clean['Villes'].astype(str).str.strip().str.upper()
    # Corriger les régions selon le mapping
    def correct_region(row):
        ville = row['Villes']
        region_orig = str(row['REGION']).strip().upper()
        region_ref = mapping.get(ville)
        if region_ref and region_ref != region_orig:
            return region_ref
        return region_orig

    df_clean['REGION'] = df_clean.apply(correct_region, axis=1)
    # Renommer les colonnes pour cohérence avec le modèle
    rename_cols = {
        'Villes': 'Ville',
        'Enseignes': 'Enseigne',
        'Publicité': 'Publicite',
        'Population': 'Population',
        'Taux_Ouvri': 'Taux_Ouvriers',
        'Taux_Cadre': 'Taux_Cadres',
        'Taux_Inact': 'Taux_Inactifs',
        'Moins_25an': 'Moins_25ans',
        'Les_25_35a': 'Les_25_35ans',
        'Plus_35ans': 'Plus_35ans'
    }
    df_clean = df_clean.rename(columns=rename_cols)
    # Créer l'identifiant de ville (clé technique) en ordonnant par nom de ville
    df_clean = df_clean.sort_values('Ville').reset_index(drop=True)
    df_clean.insert(0, 'id_ville', range(1, len(df_clean) + 1))
    return df_clean


def build_dim_magasin(df_darties: pd.DataFrame) -> pd.DataFrame:
    """Construit la dimension ``magasin`` à partir de la table darties.

    Chaque magasin est défini par le couple (Ville, Enseigne).  La clé
    ``id_magasin`` est un identifiant séquentiel.  L'identifiant de ville
    est repris de ``df_darties``.

    Args:
        df_darties: table ``darties`` nettoyée avec ``id_ville``.

    Returns:
        Table ``dim_magasin`` avec ``id_magasin``, ``id_ville``, ``Enseigne`` et
        ``Publicite``.
    """
    magasins = df_darties[['id_ville', 'Ville', 'Enseigne', 'Publicite']].copy()
    # Il peut y avoir plusieurs enseignes par ville ; on dédoublonne
    magasins = magasins.drop_duplicates(subset=['Ville', 'Enseigne']).reset_index(drop=True)
    magasins = magasins.sort_values(['Ville', 'Enseigne']).reset_index(drop=True)
    magasins.insert(0, 'id_magasin', range(1, len(magasins) + 1))
    # Supprimer la colonne Ville pour ne garder que l'identifiant
    magasins = magasins[['id_magasin', 'id_ville', 'Enseigne', 'Publicite']]
    return magasins


def build_dim_produit() -> pd.DataFrame:
    """Crée la dimension des produits.

    Il n'existe que trois familles de produits : DVD, Fours et HiFi.
    Chaque entrée possède un identifiant technique ``id_produit``.

    Returns:
        Table ``dim_produit`` avec ``id_produit`` et ``type_produit``.
    """
    data = {
        'id_produit': [1, 2, 3],
        'type_produit': ['dvd', 'fours', 'hifi']
    }
    return pd.DataFrame(data)


def clean_measure_sheet(df: pd.DataFrame, replace_negative: bool = True) -> pd.DataFrame:
    """Nettoie une feuille de mesures.

    Les feuilles de mesures (CA, V, MB) contiennent une colonne 'Villes'
    suivie de 24 colonnes correspondant aux couples (O/R, mois).  Cette
    fonction supprime les espaces superflus dans les noms de ville et,
    éventuellement, remplace les valeurs négatives par ``NaN`` pour
    certaines mesures.

    Args:
        df: DataFrame chargé depuis Excel pour une mesure particulière.
        replace_negative: si ``True``, toute valeur négative est remplacée
            par ``NaN``.  Conformément au script original, cette option doit
            être activée pour les mesures CA et V (chiffre d'affaires et
            ventes) et désactivée pour la marge brute.

    Returns:
        DataFrame nettoyé.
    """
    df_clean = df.copy()
    # Normaliser la colonne Villes : majuscules et suppression des espaces
    df_clean['Villes'] = df_clean['Villes'].astype(str).str.strip().str.upper()
    # Remplacement des négatifs par NaN si demandé
    if replace_negative:
        value_cols = df_clean.columns[1:]
        numeric = df_clean[value_cols].apply(pd.to_numeric, errors='coerce')
        mask = numeric < 0
        # Remplacer uniquement les valeurs négatives par NaN
        df_clean.loc[:, value_cols] = numeric.where(~mask, other=pd.NA)
    return df_clean


def unpivot_measure(df: pd.DataFrame, annee: int, amount_col_name: str) -> pd.DataFrame:
    """Transforme une table de mesures large en format long.

    Les colonnes de la forme ``O_Janvier``, ``R_Janvier``, ..., ``R_Decembre``
    sont converties en lignes.  La colonne ``Villes`` est normalisée en
    majuscule et devient ``ville``.  La fonction renvoie un DataFrame
    comportant les colonnes : ``ville``, ``type_conso`` (O ou R), ``mois``
    (en minuscules), ``annee`` et ``amount_col_name`` (nom du montant).

    Args:
        df: DataFrame de départ nettoyé.
        annee: année associée aux données (ex. 2024 ou 2025).
        amount_col_name: nom de la colonne de montant souhaité dans
            le DataFrame retourné (par exemple ``montant_ca``, ``montant_vente``
            ou ``montant_marge_brute``).

    Returns:
        DataFrame unpivoté avec les colonnes [ville, type_conso, mois,
        annee, amount_col_name].
    """
    # Identifier les couples (type, mois) à partir des intitulés de colonnes
    columns = [col for col in df.columns if col != 'Villes']
    records: List[pd.DataFrame] = []
    for col in columns:
        # Le nom de colonne est de la forme O_Janvier ou R_Decembre
        try:
            type_conso, mois = col.split('_', 1)
        except ValueError:
            # Colonne inattendue, on l'ignore
            continue
        subset = df[['Villes', col]].copy()
        subset['type_conso'] = type_conso
        subset['mois'] = mois.lower()
        subset['annee'] = annee
        subset = subset.rename(columns={'Villes': 'ville', col: amount_col_name})
        records.append(subset)
    result = pd.concat(records, ignore_index=True)
    return result


def join_measures(
    df_ca: pd.DataFrame,
    df_mb: pd.DataFrame,
    df_v: pd.DataFrame
) -> pd.DataFrame:
    """Effectue la jointure entre les mesures CA, marge brute et ventes.

    Les DataFrames d'entrée doivent partager les colonnes clés suivantes :
    ``ville``, ``type_conso``, ``mois`` et ``annee``.  La jointure se fait
    sur ces colonnes en mode ``outer`` afin de conserver toutes les
    combinaisons.  En cas de doublon (par exemple lorsqu'une valeur existe
    dans ``df_ca`` mais pas dans ``df_v``), la valeur manquante restera
    ``NaN``.

    Args:
        df_ca: mesures de chiffre d'affaires.
        df_mb: mesures de marge brute.
        df_v: mesures de volume (ventes).

    Returns:
        DataFrame fusionné avec colonnes clés et mesures.
    """
    # Jointure CA ↔ MB
    merged = pd.merge(df_ca, df_mb, how='outer', on=['ville', 'type_conso', 'mois', 'annee'])
    # Jointure avec Ventes
    merged = pd.merge(merged, df_v, how='outer', on=['ville', 'type_conso', 'mois', 'annee'])
    return merged


def prepare_fact(
    df_measures: pd.DataFrame,
    dim_localite: pd.DataFrame,
    dim_magasin: pd.DataFrame,
    dim_produit: pd.DataFrame,
    product_type: str
) -> pd.DataFrame:
    """Enrichit les mesures avec les clés techniques et formatte la date.

    Cette fonction associe les identifiants ``id_ville`` et ``id_magasin``
    via la dimension localite/magasin, ajoute l'identifiant de produit et
    formate la date ``date_conso`` au format ``MM/YYYY`` en utilisant la
    colonne ``mois`` et ``annee``.  Les noms de mois sont traduits en
    numéros en respectant la correspondance décrite dans la macro dbt.

    Args:
        df_measures: DataFrame contenant les mesures fusionnées.
        dim_localite: table des villes.
        dim_magasin: table des magasins.
        dim_produit: table des produits.
        product_type: nom du produit (``dvd``, ``fours`` ou ``hifi``).

    Returns:
        DataFrame de faits enrichi.
    """
    fact = df_measures.copy()
    # Associer l'id_ville
    fact = pd.merge(fact, dim_localite[['id_ville', 'Ville']], how='left', left_on='ville', right_on='Ville')
    fact = fact.drop(columns=['Ville'])
    # Associer l'id_magasin via id_ville
    fact = pd.merge(fact, dim_magasin[['id_magasin', 'id_ville']], how='left', on='id_ville', suffixes=('', '_mag'))
    # Ajouter id_produit
    id_produit = dim_produit.loc[dim_produit['type_produit'] == product_type.lower(), 'id_produit'].values[0]
    fact['id_produit'] = id_produit
    # Créer la date sous forme MM/YYYY à partir du mois en lettres
    mois_mapping = {
        'janvier': '01', 'fevrier': '02', 'mars': '03', 'avril': '04', 'mai': '05',
        'juin': '06', 'juillet': '07', 'aout': '08', 'septembre': '09',
        'octobre': '10', 'novembre': '11', 'decembre': '12'
    }
    fact['date_conso'] = fact['mois'].map(mois_mapping) + '/' + fact['annee'].astype(str)
    # Réordonner et sélectionner les colonnes finales
    fact = fact[['id_magasin', 'id_produit', 'date_conso', 'montant_ca', 'montant_vente', 'montant_marge_brute', 'type_conso']].copy()
    # Générer l'identifiant technique id_conso (clé primaire) en se basant sur l'ordre
    fact.insert(0, 'id_conso', range(1, len(fact) + 1))
    return fact


def run_pipeline(
    path_2024: Path,
    path_2025: Path,
    classification_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Exécute l'ensemble du pipeline pour les années 2024 et 2025.

    Args:
        path_2024: chemin vers le fichier Excel ``2024_HISTO.xlsx``.
        path_2025: chemin vers le fichier Excel ``2025_BUDGET.xlsx``.
        classification_path: chemin vers le fichier ``classification_villes.csv``.

    Returns:
        Un tuple contenant : (dim_localite, dim_magasin, dim_produit, fait_conso)
    """
    # 1. Charger le mapping de régions
    ville_region_mapping = load_classification_mapping(classification_path)
    # 2. Charger les fichiers Excel
    xls_2024 = pd.ExcelFile(path_2024)
    xls_2025 = pd.ExcelFile(path_2025)
    # 3. Nettoyer la table darties (identique dans les deux fichiers)
    df_darties_raw = pd.read_excel(xls_2024, 'darties')
    dim_localite = clean_darties(df_darties_raw, ville_region_mapping)
    # 4. Construire les dimensions magasin et produit
    dim_magasin = build_dim_magasin(dim_localite)
    dim_produit = build_dim_produit()
    # Liste des produits et des feuilles associées
    produits = ['DVD', 'Fours', 'Hifi']
    measures = {'CA': ('montant_ca', True), 'V': ('montant_vente', True), 'MB': ('montant_marge_brute', False)}
    # DataFrame final pour les faits
    fact_frames: List[pd.DataFrame] = []
    # 5. Traiter chaque année et chaque produit
    for annee, xls in [(2024, xls_2024), (2025, xls_2025)]:
        for prod in produits:
            # Chargement et nettoyage des trois mesures
            unpivoted: Dict[str, pd.DataFrame] = {}
            for measure_prefix, (amount_col, replace_neg) in measures.items():
                sheet_name = f"{measure_prefix}_{prod}"
                if sheet_name not in xls.sheet_names:
                    # Si la feuille n'existe pas (ex: certaines années), on passe
                    continue
                df_raw = pd.read_excel(xls, sheet_name)
                df_clean = clean_measure_sheet(df_raw, replace_negative=replace_neg)
                df_unpivot = unpivot_measure(df_clean, annee, amount_col)
                unpivoted[amount_col] = df_unpivot
            # Vérifier que nous avons au moins une mesure
            if not unpivoted:
                continue
            # Fusionner les mesures disponibles
            df_ca = unpivoted.get('montant_ca')
            df_mb = unpivoted.get('montant_marge_brute')
            df_v = unpivoted.get('montant_vente')
            # Certaines années ou produits peuvent ne pas avoir toutes les mesures
            # On crée des DataFrames vides si nécessaire avec les colonnes clés
            keys = ['ville', 'type_conso', 'mois', 'annee']
            if df_ca is None:
                df_ca = pd.DataFrame(columns=keys + ['montant_ca'])
            if df_mb is None:
                df_mb = pd.DataFrame(columns=keys + ['montant_marge_brute'])
            if df_v is None:
                df_v = pd.DataFrame(columns=keys + ['montant_vente'])
            df_joined = join_measures(df_ca, df_mb, df_v)
            # Enrichir avec les clés techniques et produit
            fact_df = prepare_fact(df_joined, dim_localite, dim_magasin, dim_produit, prod.lower())
            fact_frames.append(fact_df)
    # 6. Concaténer l'ensemble des faits
    fait_conso = pd.concat(fact_frames, ignore_index=True)
    return dim_localite, dim_magasin, dim_produit, fait_conso


if __name__ == '__main__':
    """
    Point d'entrée lorsqu'on exécute le module en tant que script.

    Ce bloc prépare les chemins nécessaires et exécute le pipeline.  Afin
    d'être robuste, il cherche le fichier `classification_villes.csv`
    dans le répertoire du script.  S'il n'est pas trouvé, une erreur
    explicite est levée.  Vous pouvez adapter ces chemins selon l'emplacement
    de vos fichiers Excel et CSV.
    """
    # Définir le répertoire de base comme étant le dossier contenant ce script
    base_path = Path(__file__).parent
    # Chemins vers les fichiers Excel
    path_2024 = base_path / '2024_HISTO.xlsx'
    path_2025 = base_path / '2025_BUDGET.xlsx'
    # Chemin vers le fichier de classification ; on vérifie qu'il existe
    classification_path = base_path / 'classification_villes.csv'
    if not classification_path.exists():
        raise FileNotFoundError(
            f"Le fichier de classification des villes est introuvable à l'emplacement {classification_path}. "
            "Placez `classification_villes.csv` dans le même dossier que ce script ou indiquez le chemin correct."
        )
    # Exécuter le pipeline
    dim_localite, dim_magasin, dim_produit, fait_conso = run_pipeline(
        path_2024,
        path_2025,
        classification_path
    )
    # Sauvegarder les résultats dans le même dossier que le script
    dim_localite.to_csv(base_path / 'dim_localite.csv', index=False)
    dim_magasin.to_csv(base_path / 'dim_magasin.csv', index=False)
    dim_produit.to_csv(base_path / 'dim_produit.csv', index=False)
    fait_conso.to_csv(base_path / 'fait_conso.csv', index=False)
    print('Pipeline terminé. Fichiers générés dans le dossier share.')
