# pages/magasin.py
# -*- coding: utf-8 -*-
import io
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Config en premier
st.set_page_config(page_title="TABLEAU DE BORD - DARTIES - MAGASIN", page_icon="🏬", layout="wide")

# ---------- Styles avec couleurs améliorées ----------
def inject_styles():
    st.markdown("""
    <style>
    .banner{
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        border: 1px solid #1e40af;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
        color: white;
    }
    .banner h1{margin:0;padding:0;font-size:22px;letter-spacing:1px;}
    .badge{float:right;text-align:right;font-size:12px;color:rgba(255,255,255,0.9);}
    .card-lite{
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border:1px solid #334155;
        border-radius:10px;
        padding:12px;
        color:#e5e7eb;
        margin-bottom:10px;
        box-shadow:0 0 0 1px rgba(51,65,85,.15) inset;
    }
    .card-lite b{color:#f8fafc;}
    table.tbl-small{border-collapse:collapse;font-size:13px;width:auto;color:#e5e7eb;}
    table.tbl-small th,table.tbl-small td{border:1px solid #334155;padding:6px 8px;}
    table.tbl-small th{
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color:#f8fafc;
        font-weight:700;
    }
    table.tbl-small tr:nth-child(odd) td{background:#0f172a;}
    table.tbl-small tr:nth-child(even) td{background:#1e293b;}
    .ratio-neg{color:#ef4444;font-weight:800;}
    .ratio-pos{color:#10b981;font-weight:800;}
    .pill{
        display:inline-block;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        border:1px solid #1e40af;
        border-radius:999px;
        padding:.25rem .6rem;
        margin-right:.35rem;
    }
    </style>
    """, unsafe_allow_html=True)

def top_banner():
    now = datetime.now()
    st.markdown(f"""
    <div class="banner">
        <span class="badge">
            <div>Date mise à jour : {now.strftime("%d / %m / %y")}</div>
            <div>Date du jour : {now.strftime("%d / %m / %y - %Hh%M")}</div>
        </span>
        <h1>TABLEAU DE BORD - DARTIES - MAGASIN</h1>
    </div>
    """, unsafe_allow_html=True)

inject_styles()
top_banner()

# ---------- Données ----------
if "df" not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
    st.warning("Aucune donnée en mémoire. Ouvre d'abord la page **Accueil** pour te connecter à la base.")
    st.stop()

df = st.session_state.df.copy()

# Normalisation colonnes
expected = ["date_conso","mois","Region","Ville","Magasin","Enseigne","Type_produit",
            "montant_ca","montant_marge_brute","montant_vente","type_conso"]
for c in expected:
    if c not in df.columns:
        df[c] = 0.0 if c in {"montant_ca","montant_marge_brute","montant_vente"} else ""
df["date_conso"] = pd.to_datetime(df["date_conso"], errors="coerce")
df["mois"] = df["date_conso"].dt.to_period("M").astype(str)
for c in ["montant_ca","montant_marge_brute","montant_vente"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

def _nature(x:str):
    x = (str(x) or "").upper()
    if x in {"O","OBJ","OBJECTIF","BUDGET"}: return "Budget"
    return "Réalisé"
df["Nature"] = df["type_conso"].apply(_nature)

# ---------- Helpers communs ----------
def small_table_html(dfx: pd.DataFrame) -> str:
    bud = dfx[dfx["Nature"]=="Budget"].agg({"montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"})
    rea = dfx[dfx["Nature"]=="Réalisé"].agg({"montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"})
    bCA,bV,bM = float(bud.get("montant_ca",0)), float(bud.get("montant_vente",0)), float(bud.get("montant_marge_brute",0))
    rCA,rV,rM = float(rea.get("montant_ca",0)), float(rea.get("montant_vente",0)), float(rea.get("montant_marge_brute",0))
    tab = pd.DataFrame({"": ["Budget","Réel","Ratio"],
                        "CA":[bCA,rCA,(rCA/bCA if bCA>0 else 0.0)],
                        "Vente":[bV,rV,(rV/bV if bV>0 else 0.0)],
                        "Marge":[bM,rM,(rM/bM if bM>0 else 0.0)]})
    rows = []; th = "<tr><th></th><th>CA</th><th>Vente</th><th>Marge</th></tr>"
    for _, row in tab.iterrows():
        lab = row[""]
        def cell(v, ratio):
            if ratio:
                cls = "ratio-pos" if v>=1 else "ratio-neg"
                return f"<td class='{cls}'>{v:.2f}</td>"
            return f"<td>{f'{v:,.2f}'.replace(',', ' ')}</td>"
        rows.append("<tr><td>"+lab+"</td>"+cell(row["CA"],lab=="Ratio")+cell(row["Vente"],lab=="Ratio")+cell(row["Marge"],lab=="Ratio")+"</tr>")
    return "<table class='tbl-small'>" + th + "".join(rows) + "</table>"

@st.cache_data(show_spinner=False)
def scope_store(_df: pd.DataFrame, magasin: str, mois_fin: str, mois_window: int):
    fin = pd.Period(mois_fin,"M").to_timestamp("M")
    deb = fin - pd.DateOffset(months=mois_window-1)
    d = _df[(_df["Magasin"]==magasin) & (_df["date_conso"].between(deb, fin))].copy()
    d["mois"] = d["date_conso"].dt.to_period("M").astype(str)
    return d

# ---------- Sidebar (Filtres + Reset + Export PDF) ----------
with st.sidebar:
    st.markdown("### Filtres")

    # Clés de widgets pour reset
    MAGASIN_KEYS = [
        "mag_mois", "mag_window", "mag_region", "mag_magasin",
        "mag_lines", "mag_metric", "mag_rank_metric"
    ]

    # Bouton reset
    if st.button("🔄 Effacer filtres", use_container_width=True, key="btn_reset_mag"):
        for k in MAGASIN_KEYS:
            st.session_state.pop(k, None)
        st.rerun()

    # Filtres avec valeurs par défaut vides
    mois_dispo = sorted(df["mois"].dropna().unique())
    mois_sel = st.selectbox("Période fin (mois)", [""] + mois_dispo,
                              index=0, key="mag_mois")

    if not mois_sel:
        st.info("Sélectionnez d'abord une période")
        st.stop()

    window_n = st.slider("Fenêtre (mois) pour les graphes", 6, 36, 18, 6, key="mag_window")

    regions_all = ["NORD_OUEST","NORD_EST","REGION_PARISIENNE","SUD_OUEST","SUD_EST"]
    regions_present = [r for r in regions_all if r in df["Region"].dropna().unique().tolist()] or \
                      sorted(df["Region"].dropna().unique().tolist())
    region_sel = st.selectbox("Région", [""] + regions_present, index=0, key="mag_region")

    if not region_sel:
        st.info("Sélectionnez d'abord une région")
        st.stop()

    mags_region = sorted(df.loc[df["Region"]==region_sel, "Magasin"].dropna().unique().tolist())
    mag_sel = st.selectbox("Magasin", [""] + mags_region, index=0, key="mag_magasin")

    if not mag_sel:
        st.info("Sélectionnez d'abord un magasin")
        st.stop()

    st.divider()
    st.markdown("### Export")

    # Export PDF du dashboard
    export_pdf_clicked = st.button("📄 Générer dashboard PDF", use_container_width=True, 
                                  key="btn_pdf_mag", type="primary")

# ---------- Scope filtré ----------
dff = scope_store(df, mag_sel, mois_sel, window_n)
if dff.empty:
    st.info("Aucune donnée pour ce magasin avec les filtres courants.")
    st.stop()

# ---------- En-tête magasin ----------
info = df[(df["Magasin"]==mag_sel)][["Ville","Enseigne"]].dropna().head(1)
ville = info["Ville"].iat[0] if not info.empty else "—"
ens = info["Enseigne"].iat[0] if not info.empty else "—"
st.markdown(f"**Magasin :** <span class='pill'>{mag_sel}</span>  **Ville :** <span class='pill'>{ville}</span>  **Enseigne :** <span class='pill'>{ens}</span>", unsafe_allow_html=True)

# ---------- Onglets ----------
tab_acc, tab_zone, tab_rank = st.tabs(["Magasin", "Zone de chalandise", "Classements"])

# Préparer objets de figures pour l'export PDF
figures_for_pdf = []

# =========================================================
# 1) MAGASIN — Global + chefs (barres Budget/Réel)
# =========================================================
with tab_acc:
    g1, g2 = st.columns([1,2])
    with g1:
        st.markdown("**Résultat du magasin**")
        st.markdown(f"<div class='card-lite'>{small_table_html(dff)}</div>", unsafe_allow_html=True)

    with g2:
        st.markdown("**Chef de produits (famille)**")
        metric = st.radio("Indicateur", ["CA","Ventes","Marge"], horizontal=True, key="mag_metric")
        col_map = {"CA":"montant_ca", "Ventes":"montant_vente", "Marge":"montant_marge_brute"}
        ag = (dff.groupby(["Type_produit","Nature"], as_index=False)
                 .agg(val=(col_map[metric],"sum"))
                 .sort_values(["Type_produit","Nature"]))
        if ag.empty:
            st.info("Aucune donnée à afficher par famille.")
        else:
            fig_bar = px.bar(ag, x="Type_produit", y="val", color="Nature", barmode="group",
                             template="plotly_dark", 
                             color_discrete_map={"Budget": "#64748b", "Réalisé": "#2563eb"},
                             labels={"val":metric, "Type_produit":"Ligne de produit"})
            fig_bar.update_layout(height=400, margin=dict(l=10,r=10,t=10,b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            figures_for_pdf.append(fig_bar)

# =========================================================
# 2) ZONE DE CHALANDISE — 2 visuels ventes par mois
# =========================================================
with tab_zone:
    st.write("🗂️ **Ventes par mois** (magasin, Réel) — avec filtre **ligne de produit**")
    lignes = sorted([x for x in dff["Type_produit"].dropna().unique().tolist() if str(x)!=""])
    sel_lines = st.multiselect("Lignes de produit", options=lignes, default=lignes, key="mag_lines")
    dfl = dff[dff["Type_produit"].isin(sel_lines)] if sel_lines else dff.head(0)

    # Courbe
    st.subheader("📈 Courbe par ligne de produit")
    hist = (dfl.groupby(["mois","Type_produit","Nature"], as_index=False)
                .agg(Ventes=("montant_vente","sum"))
                .sort_values("mois"))
    hist = hist[hist["Nature"]=="Réalisé"]
    if hist.empty:
        st.info("Aucune donnée à tracer.")
    else:
        mois_order = sorted(hist["mois"].unique())
        fig_line = px.line(hist, x="mois", y="Ventes", color="Type_produit",
                           markers=True, template="plotly_dark",
                           category_orders={"mois": mois_order})
        fig_line.update_xaxes(type="category")
        fig_line.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_line, use_container_width=True)
        figures_for_pdf.append(fig_line)

    # Aire empilée
    st.subheader("🟪 Aire empilée — composition mensuelle")
    comp = (dfl[dfl["Nature"]=="Réalisé"]
            .groupby(["mois","Type_produit"], as_index=False)
            .agg(Ventes=("montant_vente","sum"))
            .sort_values("mois"))
    if comp.empty:
        st.info("Aucune donnée pour l'aire empilée.")
    else:
        mois_order = sorted(comp["mois"].unique())
        fig_area = px.area(comp, x="mois", y="Ventes", color="Type_produit",
                           template="plotly_dark", category_orders={"mois": mois_order})
        fig_area.update_xaxes(type="category")
        fig_area.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_area, use_container_width=True)
        figures_for_pdf.append(fig_area)

# =========================================================
# 3) CLASSEMENTS — complet : régional + national
# =========================================================
with tab_rank:
    st.write("🏆 **Classements du magasin** (régional & national) — mois sélectionné.")
    metric_label = st.radio("Métrique de classement :", ["CA Réel","Ventes Réel","Marge Réel"],
                            horizontal=True, key="mag_rank_metric")
    metric_map = {"CA Réel":"montant_ca","Ventes Réel":"montant_vente","Marge Réel":"montant_marge_brute"}
    col_m = metric_map[metric_label]

    # Périmètres courant / précédent
    def base_period(mois_str):
        if mois_str is None or len(mois_str)!=7:
            return df.iloc[0:0].copy()
        b = df[df["mois"]==mois_str]
        return b

    cur  = base_period(mois_sel)
    prev = base_period(str(pd.Period(mois_sel,"M")-1)) if len(mois_sel)==7 else df.iloc[0:0].copy()

    # Agrégats réalisés par magasin/region
    def agg_rank(d):
        if d.empty: return d
        g = (d[d["Nature"]=="Réalisé"].groupby(["Magasin","Region"], as_index=False)
                    .agg(CA=("montant_ca","sum"),
                         Ventes=("montant_vente","sum"),
                         Marge=("montant_marge_brute","sum")))
        return g

    a_cur  = agg_rank(cur)
    a_prev = agg_rank(prev) if not prev.empty else a_cur.head(0).copy()

    if a_cur.empty or mag_sel not in a_cur["Magasin"].values:
        st.info("Pas de données de classement pour ce magasin.")
    else:
        value_col = {"montant_ca":"CA", "montant_vente":"Ventes", "montant_marge_brute":"Marge"}[col_m]

        # Rangs actuels
        a_cur = a_cur.sort_values(value_col, ascending=False).reset_index(drop=True)
        a_cur["Rang National"] = a_cur.index + 1
        a_cur["Rang Régional"] = a_cur.groupby("Region")[value_col].rank(ascending=False, method="min").astype(int)

        # Rangs t-1
        if not a_prev.empty:
            a_prev = a_prev.sort_values(value_col, ascending=False).reset_index(drop=True)
            a_prev["Rang National (t-1)"] = a_prev.index + 1
            a_prev["Rang Régional (t-1)"] = a_prev.groupby("Region")[value_col].rank(ascending=False, method="min").astype(int)
            a_cur = a_cur.merge(a_prev[["Magasin","Rang National (t-1)","Rang Régional (t-1)"]], on="Magasin", how="left")
        else:
            a_cur["Rang National (t-1)"] = np.nan
            a_cur["Rang Régional (t-1)"] = np.nan

        def evol_fmt(x):
            if pd.isna(x): return "—"
            if x == 0: return "="
            return f"{int(x):+d}"

        a_cur["Évol. National"] = (a_cur["Rang National (t-1)"] - a_cur["Rang National"]).apply(evol_fmt)
        a_cur["Évol. Régional"] = (a_cur["Rang Régional (t-1)"] - a_cur["Rang Régional"]).apply(evol_fmt)

        # Carte d'identité du magasin sélectionné
        row = a_cur[a_cur["Magasin"]==mag_sel].iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rang National", int(row["Rang National"]))
        c2.metric("Évol. National", row["Évol. National"])
        c3.metric("Rang Régional", int(row["Rang Régional"]))
        c4.metric("Évol. Régional", row["Évol. Régional"])

        st.markdown("### 🏅 Classement régional (liste complète)")
        reg_table = a_cur[a_cur["Region"]==region_sel] \
                        .sort_values(value_col, ascending=False) \
                        .reset_index(drop=True)
        reg_table["#"] = reg_table.index + 1
        reg_cols = ["#","Magasin","Region","CA","Ventes","Marge","Rang Régional","Évol. Régional","Rang National","Évol. National"]
        st.dataframe(reg_table[reg_cols], use_container_width=True, height=420)

        st.markdown("### 🇫🇷 Classement national (liste complète)")
        nat_table = a_cur.sort_values(value_col, ascending=False).reset_index(drop=True)
        nat_table["#"] = nat_table.index + 1
        nat_cols = ["#","Magasin","Region","CA","Ventes","Marge","Rang National","Évol. National","Rang Régional","Évol. Régional"]
        st.dataframe(nat_table[nat_cols], use_container_width=True, height=560)

        # Export Excel (classements)
        buf_xlsx = io.BytesIO()
        try:
            with pd.ExcelWriter(buf_xlsx, engine="openpyxl") as w:
                a_cur.to_excel(w, sheet_name="Classements_raw", index=False)
                reg_table[reg_cols].to_excel(w, sheet_name="Classement_regional", index=False)
                nat_table[nat_cols].to_excel(w, sheet_name="Classement_national", index=False)
            xbytes = buf_xlsx.getvalue()
        except Exception:
            xbytes = b""
        st.download_button("Exporter Classements (Excel)", data=xbytes,
                           file_name=f"classements_{mag_sel}_{mois_sel}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           disabled=(xbytes==b""))

# ---------- Génération PDF si demandé ----------
def build_pdf_from_figs(store_name: str, region_name: str, ville: str, enseigne: str,
                        mois_sel: str, figures: list, dff_scope: pd.DataFrame) -> bytes:
    """Construit un PDF avec ReportLab."""
    # Résumé KPI (réel)
    rea = dff_scope[dff_scope["Nature"]=="Réalisé"].agg({
        "montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"
    })
    rCA = float(rea.get("montant_ca",0)); rV=float(rea.get("montant_vente",0)); rM=float(rea.get("montant_marge_brute",0))

    # Essayer d'importer reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
    except Exception as e:
        st.error(f"Module manquant pour PDF : {e}. Ajoute `reportlab` à requirements.txt.")
        return b""

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    W, H = A4

    # En-tête
    title = f"DARTIES — Dashboard Magasin\n{store_name} • {ville} • {enseigne}\nRégion: {region_name} • Période: {mois_sel}"
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, H-2.5*cm, "TABLEAU DE BORD - MAGASIN")
    c.setFont("Helvetica", 9)
    for i, line in enumerate(title.split("\n")):
        c.drawString(2*cm, H-(3.2+i*0.5)*cm, line)

    # KPI
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, H-5*cm, "KPI (Réel) :")
    c.setFont("Helvetica", 10)
    c.drawString(2.6*cm, H-5.7*cm, f"CA : {rCA:,.0f} €".replace(",", " "))
    c.drawString(2.6*cm, H-6.3*cm, f"Ventes : {rV:,.0f}".replace(",", " "))
    c.drawString(2.6*cm, H-6.9*cm, f"Marge : {rM:,.0f} €".replace(",", " "))

    y = H-8.5*cm
    max_w_img = W-4*cm
    max_h_img = 8*cm

    # Export de chaque figure en PNG et insertion
    for fig in figures:
        try:
            # Nécessite kaleido
            png_bytes = fig.to_image(format="png", scale=2)
        except Exception as e:
            st.error(f"Export image échoué (kaleido manquant ?): {e}. Ajoute `kaleido` à requirements.txt.")
            return b""

        img = ImageReader(io.BytesIO(png_bytes))
        iw, ih = img.getSize()
        # Ratio pour tenir dans le cadre
        scale = min(max_w_img/iw, max_h_img/ih)
        w_draw = iw*scale
        h_draw = ih*scale

        if y - h_draw < 2.5*cm:  # nouvelle page si pas assez de place
            c.showPage()
            y = H-2.5*cm

        c.setFont("Helvetica-Bold", 9)
        c.drawString(2*cm, y, f"Graphique {len([f for f in figures if f == fig]) + 1}")
        y -= 0.5*cm
        c.drawImage(img, 2*cm, y-h_draw, width=w_draw, height=h_draw, preserveAspectRatio=True, mask='auto')
        y -= (h_draw + 0.8*cm)

    c.showPage()
    c.save()
    return pdf_buf.getvalue()

if export_pdf_clicked:
    pdf_bytes = build_pdf_from_figs(
        store_name=mag_sel, region_name=region_sel, ville=ville, enseigne=ens,
        mois_sel=mois_sel, figures=figures_for_pdf, dff_scope=dff
    )
    st.download_button(
        "📥 Télécharger le PDF",
        data=pdf_bytes,
        file_name=f"dashboard_magasin_{mag_sel}_{mois_sel}.pdf",
        mime="application/pdf",
        disabled=(pdf_bytes==b"")
    )