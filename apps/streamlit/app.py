# app.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime

# ========= Config Streamlit (toujours en premier) =========
st.set_page_config(
    page_title="Darties — Accueil",
    page_icon="🏠",
    layout="wide",
)

# ========= Styles (landing avec 3 tuiles) =========
st.markdown("""
<style>
.banner{
  background:#eef3fb; border:1px solid #cdd9f0; border-radius:12px;
  padding:16px 20px; margin-bottom:14px;
}
.banner h1{margin:0;font-size:24px;letter-spacing:.5px}
.badge{float:right;text-align:right;font-size:12px;color:#334155}
.small{font-size:12px;color:#475569}

.grid{display:grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap:18px; margin-top:10px}
.card{
  position:relative; background:linear-gradient(180deg,#0b1220 0%, #0e1626 100%);
  border:1px solid #223047; border-radius:16px; padding:22px; min-height:160px;
  color:#e5e7eb; box-shadow:0 6px 24px rgba(3,8,20,.18), inset 0 0 0 1px rgba(34,48,71,.15);
  transition:transform .12s ease, box-shadow .12s ease, border-color .12s ease;
}
.card:hover{ transform: translateY(-2px); border-color:#3b82f6; box-shadow:0 10px 28px rgba(2,8,23,.35)}
.card .icon{font-size:28px; line-height:1; margin-bottom:8px;}
.card h3{margin:.2rem 0 .6rem; font-size:20px}
.card p{opacity:.88; font-size:14px; margin:0 0 16px}
.card a.btn{
  display:inline-block; text-decoration:none; font-weight:600; letter-spacing:.3px;
  background:#1f2937; border:1px solid #334155; color:#e5e7eb;
  padding:.55rem .9rem; border-radius:999px;
}
.card a.btn:hover{background:#0b1220; border-color:#3b82f6}

.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.dot.ok{background:#22c55e}
.dot.ko{background:#ef4444}
</style>
""", unsafe_allow_html=True)

now = datetime.now()
st.markdown(f"""
<div class="banner">
  <span class="badge">
    <div>Date du jour : {now.strftime("%d / %m / %y - %Hh%M")}</div>
  </span>
  <h1>🏠 Accueil — Darties BI</h1>
  <div class="small">Connecte la base une seule fois ici. Les pages <b>National / Régional / Magasin</b> réutiliseront la même connexion.</div>
</div>
""", unsafe_allow_html=True)

# ========= Connexion BDD (une seule fois) =========
@st.cache_resource(show_spinner=False)
def get_engine(conn_str: str):
    return create_engine(conn_str, pool_pre_ping=True)

@st.cache_data(show_spinner=False)
def load_data(conn_str: str) -> pd.DataFrame:
    eng = get_engine(conn_str)
    return get_data(eng)

def safe_to_datetime(x):
    try: return pd.to_datetime(x, errors="coerce")
    except Exception: return pd.NaT

def get_data(engine):
    # ⚠️ Adapter si ton schéma diffère (noms de tables/colonnes)
    q = text("""
        SELECT
            f.date_conso,
            f.montant_ca,
            f.montant_marge_brute,
            f.montant_vente,
            f.type_conso,
            f.Id_magasin,
            f.Id_produit,
            dm.Enseigne,
            dm.Nom_magasin,
            dl.Ville,
            dl.Region,
            dp.Type_produit,
            c.Annee,
            c.Mois AS mois_txt,
            c.Numero_mois
        FROM fait_conso f
        LEFT JOIN Dim_Magasin dm ON dm.Id_magasin = f.Id_magasin
        LEFT JOIN dim_localite dl ON dl.Id_ville = dm.Id_ville
        LEFT JOIN dim_produit dp ON dp.Id_produit = f.Id_produit
        LEFT JOIN calendrier c ON DATE(c.Date) = DATE(f.date_conso)
    """)
    df = pd.read_sql(q, engine)

    # Defaults
    for c, dft in {
        "Enseigne":"", "Nom_magasin":"", "Ville":"", "Region":"", "Type_produit":"",
        "Annee":None, "mois_txt":"", "Numero_mois":None,
    }.items():
        if c not in df.columns: df[c] = dft

    # Types
    df["date_conso"] = df["date_conso"].apply(safe_to_datetime)
    for c in ["montant_ca","montant_marge_brute","montant_vente"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)

    # Année fallback
    if df["Annee"].isna().any():
        has_date = df["date_conso"].notna()
        df.loc[df["Annee"].isna() & has_date, "Annee"] = df.loc[df["Annee"].isna() & has_date, "date_conso"].dt.year

    # Clés utiles
    df["mois"] = df["date_conso"].dt.to_period("M").astype(str)
    df["Id_magasin"] = df["Id_magasin"].fillna(-1).astype(int)
    df["Magasin"] = df.apply(
        lambda r: f"Magasin {r['Id_magasin']}" if (pd.isna(r.get("Nom_magasin")) or str(r.get("Nom_magasin")).strip()=="")
        else str(r.get("Nom_magasin")), axis=1)
    df["Region"] = df["Region"].where(df["Region"].notna() & (df["Region"].astype(str)!=""), "Inconnu")
    df["Type_produit"] = df["Type_produit"].fillna("")
    return df

# ----- Sidebar : paramètres de connexion -----
with st.sidebar:
    st.header("⚙️ Connexion à la base")
    load_dotenv()
    host = st.text_input("Host", os.getenv("DB_HOST", "127.0.0.1"))
    port = st.number_input("Port", 1, 65535, int(os.getenv("DB_PORT", "3306")))
    user = st.text_input("Utilisateur", os.getenv("DB_USER", "root"))
    password = st.text_input("Mot de passe", os.getenv("DB_PASSWORD", ""), type="password")
    dbname = st.text_input("Base de données", os.getenv("DB_NAME", "darties"))
    connect_clicked = st.button("Se connecter / Rafraîchir", use_container_width=True)

# ----- État global -----
if "connected" not in st.session_state: st.session_state.connected = False
if "conn_key"  not in st.session_state: st.session_state.conn_key = None
if "df"        not in st.session_state: st.session_state.df = pd.DataFrame()

conn_key = f"{user}@{host}:{port}/{dbname}"
conn_str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=utf8mb4"
need_reconnect = (not st.session_state.connected) or (st.session_state.conn_key != conn_key)

if connect_clicked or need_reconnect:
    try:
        with st.spinner("Connexion et chargement des données…"):
            df_loaded = load_data(conn_str)   # @cache_data => recharge seulement si conn_str change
        st.session_state.df = df_loaded
        st.session_state.connected = True
        st.session_state.conn_key = conn_key
        st.success(f"✅ Connecté à `{dbname}` — {len(df_loaded):,} lignes".replace(",", " "))
    except Exception as e:
        st.session_state.connected = False
        st.error(f"❌ Erreur de connexion : {e}")

# ----- Statut & navigation (aucune visualisation) -----
if st.session_state.connected and not st.session_state.df.empty:
    df = st.session_state.df
    dmin = df["date_conso"].min()
    dmax = df["date_conso"].max()
    st.markdown(
        f"<span class='dot ok'></span>Connexion OK — "
        f"<b>{len(df):,}</b> lignes • Période: "
        f"<b>{dmin.date() if pd.notna(dmin) else '—'}</b> → <b>{dmax.date() if pd.notna(dmax) else '—'}</b>",
        unsafe_allow_html=True
    )
else:
    st.markdown("<span class='dot ko'></span>Non connecté — renseigne les paramètres à gauche et clique <b>Se connecter / Rafraîchir</b>.",
                unsafe_allow_html=True)

st.markdown("---")
st.subheader("Navigation")

# Tuiles cliquables
st.markdown(
    """
<div class="grid">
  <div class="card">
    <div class="icon">📌</div>
    <h3>Vue Nationale</h3>
    <p>Vision France entière : carte régions, récap par régions, palmarès magasins, exports.</p>
    <a class="btn" href="?page=National">Ouvrir la vue Nationale</a>
  </div>
  <div class="card">
    <div class="icon">📍</div>
    <h3>Vue Régionale</h3>
    <p>Vue d’une région : tableaux par enseigne, historique 24 mois, détails et palmarès régionaux.</p>
    <a class="btn" href="?page=Regional">Ouvrir la vue Régionale</a>
  </div>
  <div class="card">
    <div class="icon">🏬</div>
    <h3>Vue Magasin</h3>
    <p>Focus magasin : résultats, familles, zone de chalandise (ventes), classements national & régional.</p>
    <a class="btn" href="?page=Magasin">Ouvrir la vue Magasin</a>
  </div>
</div>
""",
    unsafe_allow_html=True
)

# Liens Streamlit (optionnels)
st.write("")
c1,c2,c3 = st.columns(3)
with c1: st.page_link("pages/national.py", label="📌 Ouvrir National")
with c2: st.page_link("pages/regional.py", label="📍 Ouvrir Régional")
with c3: st.page_link("pages/magasin.py",  label="🏬 Ouvrir Magasin")

st.caption("Les pages réutilisent automatiquement la DF en mémoire (st.session_state.df). Aucune visualisation sur cette page.")
