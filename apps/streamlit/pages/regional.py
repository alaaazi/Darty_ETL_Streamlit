# pages/regional.py
# -*- coding: utf-8 -*-
import io
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TABLEAU DE BORD - DARTIES - REGIONAL", page_icon="📍", layout="wide")

# ---------- Bandeau + styles avec couleurs améliorées ----------
def top_banner():
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
    .banner h1{ margin:0; padding:0; font-size:22px; letter-spacing:1px; }
    .badge{ float:right; text-align:right; font-size:12px; color: rgba(255, 255, 255, 0.9); }

    .card-lite{
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border:1px solid #334155;
        border-radius:10px;
        padding:12px;
        color:#e5e7eb;
        margin-bottom:10px;
        box-shadow: 0 0 0 1px rgba(51,65,85,.15) inset;
    }
    .card-lite b{ color:#f8fafc; }

    table.tbl-small{ border-collapse:collapse; font-size:13px; width:auto; color:#e5e7eb; }
    table.tbl-small th, table.tbl-small td{ border:1px solid #334155; padding:6px 8px; }
    table.tbl-small th{ 
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color:#f8fafc; font-weight:700; 
    }
    table.tbl-small tr:nth-child(odd) td{ background:#0f172a; }
    table.tbl-small tr:nth-child(even) td{ background:#1e293b; }

    .ratio-neg{ color:#ef4444; font-weight:800; }
    .ratio-pos{ color:#10b981; font-weight:800; }
    </style>
    """, unsafe_allow_html=True)

    now = datetime.now()
    st.markdown(f"""
    <div class="banner">
        <span class="badge">
            <div>Date mise à jour : {now.strftime("%d / %m / %y")}</div>
            <div>Date du jour : {now.strftime("%d / %m / %y - %Hh%M")}</div>
        </span>
        <h1>TABLEAU DE BORD - DARTIES - REGIONAL</h1>
    </div>
    """, unsafe_allow_html=True)

top_banner()

# ---------- Sécurité DF ----------
if "df" not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
    st.warning("Aucune donnée en mémoire. Ouvre d'abord la page **Accueil** pour te connecter à la base.")
    st.stop()

df = st.session_state.df.copy()

# Normalisation colonnes
for col in ["date_conso","mois","Annee","Mois","Region","Ville","Magasin","Enseigne","Type_produit",
            "montant_ca","montant_marge_brute","montant_vente","type_conso","lat","lon"]:
    if col not in df.columns:
        df[col] = np.nan if col in {"lat","lon"} else (0.0 if col in {"montant_ca","montant_marge_brute","montant_vente"} else "")
df["date_conso"] = pd.to_datetime(df["date_conso"], errors="coerce")
for c in ["montant_ca","montant_marge_brute","montant_vente","lat","lon"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["mois"] = df["date_conso"].dt.to_period("M").astype(str).fillna("")

def label_nature(x):
    x = str(x).strip().upper()
    if x in {"R","REALISE","RÉALISÉ","REEL","RÉEL"}: return "Réalisé"
    if x in {"O","OBJ","OBJECTIF","BUDGET"}:        return "Budget"
    return "Réalisé"
df["Nature"] = df["type_conso"].fillna("").apply(label_nature)

# ---------- Sidebar (filtres avec valeurs par défaut vides) ----------
with st.sidebar:
    st.markdown("### Filtres")
    
    # Clés pour reset
    REGIONAL_KEYS = ["reg_region", "reg_mois", "reg_fam", "reg_ens", "reg_mags"]
    
    # Bouton reset
    if st.button("🔄 Effacer filtres", use_container_width=True, key="btn_reset_reg"):
        for k in REGIONAL_KEYS:
            st.session_state.pop(k, None)
        st.rerun()
    
    devise = st.selectbox("Devises", ["Euros"], index=0)
    taux   = st.selectbox("Taux", ["Taux 1"], index=0)

    regions_all = ["NORD_OUEST","NORD_EST","REGION_PARISIENNE","SUD_OUEST","SUD_EST"]
    regions_present = [r for r in regions_all if r in df["Region"].dropna().unique().tolist()] or sorted(df["Region"].dropna().unique())
    sel_region = st.selectbox("Région commerciale", options=[""] + regions_present, index=0, key="reg_region")
    
    if not sel_region:
        st.info("Sélectionnez d'abord une région")
        st.stop()

    mois_dispo = sorted(df["mois"].dropna().unique())
    mois_sel = st.selectbox("Période (mois)", [""] + mois_dispo, index=0, key="reg_mois")
    
    if not mois_sel:
        st.info("Sélectionnez d'abord une période")
        st.stop()

    familles = sorted([x for x in df["Type_produit"].fillna("").unique() if str(x)!=""])
    fam_sel  = st.selectbox("Famille", ["Toutes"] + familles, index=0, key="reg_fam")

    enseignes = sorted([x for x in df["Enseigne"].fillna("").unique() if str(x)!=""])
    ens_sel   = st.selectbox("Enseigne", ["Toutes"] + enseignes, index=0, key="reg_ens")

    # Filtre Magasin (multi) limité à la région sélectionnée
    mags_region_all = sorted(df.loc[df["Region"]==sel_region, "Magasin"].dropna().unique().tolist())
    sel_mags = st.multiselect(
        "Magasins (dans la région)",
        options=mags_region_all,
        default=[],
        key="reg_mags",
        help="Laissez vide pour inclure tous les magasins de la région"
    )

    # SUPPRESSION DU CUMUL
    st.write("")
    
    st.divider()
    st.markdown("### Export")
    export_pdf_clicked = st.button("📄 Générer dashboard PDF", use_container_width=True, 
                                  key="btn_pdf_reg", type="primary")

# ---------- Application des filtres ----------
def scope(df0, mois_str, region):
    d = df0[(df0["Region"] == region)].copy()
    if fam_sel != "Toutes": d = d[d["Type_produit"] == fam_sel]
    if ens_sel != "Toutes": d = d[d["Enseigne"] == ens_sel]
    # appliquer filtre magasins si spécifié
    if sel_mags:
        d = d[d["Magasin"].isin(sel_mags)]
    d = d[d["mois"] == mois_str]
    return d

dfr = scope(df, mois_sel, sel_region)
if dfr.empty:
    st.warning("Aucune donnée avec ces filtres pour la région sélectionnée.")
    st.stop()

dfr_real = dfr[dfr["Nature"]=="Réalisé"]
dfr_budg = dfr[dfr["Nature"]=="Budget"]

# ---------- Helpers ----------
def small_table(dfx) -> pd.DataFrame:
    bud = dfx[dfx["Nature"]=="Budget"].agg({"montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"})
    rea = dfx[dfx["Nature"]=="Réalisé"].agg({"montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"})
    bCA,bV,bM = float(bud.get("montant_ca",0)), float(bud.get("montant_vente",0)), float(bud.get("montant_marge_brute",0))
    rCA,rV,rM = float(rea.get("montant_ca",0)), float(rea.get("montant_vente",0)), float(rea.get("montant_marge_brute",0))
    return pd.DataFrame({
        "": ["Budget","Réel","Ratio"],
        "CA":[bCA, rCA, (rCA/bCA if bCA>0 else 0.0)],
        "Vente":[bV, rV, (rV/bV if bV>0 else 0.0)],
        "Marge":[bM, rM, (rM/bM if bM>0 else 0.0)],
    })

def render_tbl(df_tab: pd.DataFrame) -> str:
    rows = []
    th = "<tr><th></th><th>CA</th><th>Vente</th><th>Marge</th></tr>"
    for _, row in df_tab.iterrows():
        label = row[""]
        def cell(val, is_ratio):
            if is_ratio:
                css = "ratio-pos" if val>=1 else "ratio-neg"
                return f"<td class='{css}'>{val:.2f}</td>"
            return f"<td>{f'{val:,.2f}'.replace(',', ' ')}</td>"
        rows.append(
            "<tr><td>"+label+"</td>"+cell(row["CA"], label=="Ratio")+cell(row["Vente"], label=="Ratio")+cell(row["Marge"], label=="Ratio")+"</tr>"
        )
    return "<table class='tbl-small'>" + th + "".join(rows) + "</table>"

# ---------- Variables pour PDF ----------
figures_for_pdf = []

# ---------- Onglets ----------
tab_acc, tab_hist, tab_det, tab_pal = st.tabs(["Accueil", "Historique", "Détails", "Palmarès"])

# =========================================================
# ACCUEIL — tableaux PAR ENSEIGNE + carte
# =========================================================
with tab_acc:
    st.write(f"Résultat **{sel_region}** en {devise}, pour **{mois_sel}** — familles **{fam_sel}**, enseignes **{ens_sel}**.")
    if sel_mags:
        st.write(f"**Magasins sélectionnés:** {', '.join(sel_mags)}")
    
    left, right = st.columns([1, 2])

    # Tables par enseigne
    with left:
        st.markdown("**Tableaux par Enseigne (Budget / Réel / Ratio)**")
        enseignes_region = dfr["Enseigne"].dropna().unique().tolist()
        if not enseignes_region:
            st.info("Aucune enseigne détectée dans cette région pour les filtres.")
        else:
            for ens in sorted(enseignes_region):
                st.markdown(f"**{ens}**")
                st.markdown(f"<div class='card-lite'>{render_tbl(small_table(dfr[dfr['Enseigne']==ens]))}</div>", unsafe_allow_html=True)

    # Carte magasins
    with right:
        st.subheader("🗺️ Magasins de la région (couleur = enseigne)")
        color_map = {"Darty":"#ef4444", "Boulanger":"#2563eb", "Leroy_merlin":"#10b981"}
        REGION_TO_LATLON = {
            "NORD_OUEST": (48.8, -1.5),
            "NORD_EST": (48.7, 6.2),
            "REGION_PARISIENNE": (48.86, 2.35),
            "SUD_OUEST": (44.0, 0.0),
            "SUD_EST": (44.5, 5.5),
        }
        shops = (
            dfr_real.groupby(["Magasin","Ville","Enseigne"], as_index=False)
                    .agg(CA=("montant_ca","sum"),
                         Ventes=("montant_vente","sum"),
                         Marge=("montant_marge_brute","sum"),
                         lat=("lat","max"), lon=("lon","max"))
        )
        rng = np.random.default_rng(42)
        lat0, lon0 = REGION_TO_LATLON.get(sel_region, (46.8, 2.5))
        shops["lat"] = pd.to_numeric(shops["lat"], errors="coerce")
        shops["lon"] = pd.to_numeric(shops["lon"], errors="coerce")
        jitter_lat = pd.Series(lat0, index=shops.index) + pd.Series(rng.normal(0, 0.25, len(shops)), index=shops.index)
        jitter_lon = pd.Series(lon0, index=shops.index) + pd.Series(rng.normal(0, 0.35, len(shops)), index=shops.index)
        shops["lat"] = shops["lat"].where(~shops["lat"].isna(), jitter_lat)
        shops["lon"] = shops["lon"].where(~shops["lon"].isna(), jitter_lon)

        if shops.empty:
            st.info("Aucun magasin à cartographier pour ces filtres.")
        else:
            vals = shops["CA"].astype(float).values
            vmin, vmax = (vals.min(), vals.max()) if len(vals)>0 else (0,1)
            def rpx(v): 
                if vmax==vmin: return 18
                return 10 + (v - vmin)/(vmax - vmin)*28
            figm = go.Figure()
            for ens, sub in shops.groupby("Enseigne", dropna=False):
                color = color_map.get(str(ens), "#8b5cf6")
                figm.add_trace(go.Scattermapbox(
                    lat=sub["lat"], lon=sub["lon"], mode="markers",
                    marker=dict(size=[rpx(v) for v in sub["CA"].values], color=color, opacity=0.85),
                    name=str(ens),
                    hovertemplate="<b>%{text}</b><br>CA: %{customdata[0]:,.0f} €"
                                  "<br>Ventes: %{customdata[1]:,.0f}"
                                  "<br>Marge: %{customdata[2]:,.0f} €<extra></extra>",
                    text=sub["Magasin"],
                    customdata=np.c_[sub["CA"].values, sub["Ventes"].values, sub["Marge"].values]
                ))
            figm.update_layout(
                template="plotly_dark", mapbox_style="carto-darkmatter",
                mapbox=dict(center=dict(lat=float(shops['lat'].mean()),
                                        lon=float(shops['lon'].mean())),
                            zoom=6.2),
                margin=dict(l=10,r=10,t=10,b=10), height=480
            )
            st.plotly_chart(figm, use_container_width=True)
            figures_for_pdf.append(figm)

# =========================================================
# HISTORIQUE — 24 mois + comparaison inter-magasins dynamique
# =========================================================
with tab_hist:
    st.write("Historique **2 ans** — Région : Budget vs Réalisé.")
    if len(mois_sel)==7:
        yyyy, mm = map(int, mois_sel.split("-"))
        max_d = pd.Timestamp(year=yyyy, month=mm, day=1) + pd.offsets.MonthEnd(0)
    else:
        max_d = dfr["date_conso"].max() if dfr["date_conso"].notna().any() else pd.Timestamp.today()
    min_24 = (max_d - pd.DateOffset(months=24)).normalize()

    hist = df[(df["Region"]==sel_region) & (df["date_conso"].between(min_24, max_d))].copy()
    if fam_sel != "Toutes": hist = hist[hist["Type_produit"]==fam_sel]
    if ens_sel != "Toutes": hist = hist[hist["Enseigne"]==ens_sel]
    # appliquer filtre magasins si spécifié
    if sel_mags:
        hist = hist[hist["Magasin"].isin(sel_mags)]
    hist["mois"] = hist["date_conso"].dt.to_period("M").astype(str)
    hist = (hist.groupby(["mois","Nature"], as_index=False)
                 .agg(CA=("montant_ca","sum"),
                      Marge=("montant_marge_brute","sum"),
                      Ventes=("montant_vente","sum")))
    month_order = sorted(hist["mois"].unique())
    t1, t2, t3 = st.tabs(["CA","Ventes","Marge"])
    with t1:
        fig = px.line(hist, x="mois", y="CA", color="Nature", markers=True, template="plotly_dark",
                      color_discrete_map={"Budget": "#64748b", "Réalisé": "#2563eb"},
                      category_orders={"mois": month_order})
        fig.update_xaxes(type="category")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
        figures_for_pdf.append(fig)
    with t2:
        fig = px.line(hist, x="mois", y="Ventes", color="Nature", markers=True, template="plotly_dark",
                      color_discrete_map={"Budget": "#64748b", "Réalisé": "#2563eb"},
                      category_orders={"mois": month_order})
        fig.update_xaxes(type="category")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        fig = px.line(hist, x="mois", y="Marge", color="Nature", markers=True, template="plotly_dark",
                      color_discrete_map={"Budget": "#64748b", "Réalisé": "#2563eb"},
                      category_orders={"mois": month_order})
        fig.update_xaxes(type="category")
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Comparaison inter-magasins dynamique
    st.subheader("🏬 Comparaison inter-magasins — métrique dynamique")
    metric_sel = st.radio("Choix de la métrique :", ["CA","Marge","Ventes"], horizontal=True, index=0, key="metric_shop")
    col_map = {"CA":"montant_ca","Marge":"montant_marge_brute","Ventes":"montant_vente"}
    by_shop = (dfr_real.groupby("Magasin", as_index=False)
                      .agg(val=(col_map[metric_sel], "sum"))
                      .sort_values("val", ascending=True))
    fig_shop = px.bar(by_shop, x="val", y="Magasin", orientation="h", template="plotly_dark",
                      color_discrete_sequence=["#2563eb"],
                      labels={"val": metric_sel})
    fig_shop.update_layout(height=520, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig_shop, use_container_width=True)
    figures_for_pdf.append(fig_shop)

# =========================================================
# DÉTAILS — Écarts & Poids par produit + matrice Magasin x Produit
# =========================================================
with tab_det:
    st.write("Détails région — Écarts Budget vs Réalisé & Poids par produit.")
    pvt = (dfr.groupby(["Type_produit","Nature"], as_index=False)
               .agg(CA=("montant_ca","sum"),
                    Ventes=("montant_vente","sum"),
                    Marge=("montant_marge_brute","sum")))
    rea = pvt[pvt["Nature"]=="Réalisé"].drop(columns=["Nature"]).rename(columns={"CA":"CA_R","Ventes":"Ventes_R","Marge":"Marge_R"})
    bud = pvt[pvt["Nature"]=="Budget"].drop(columns=["Nature"]).rename(columns={"CA":"CA_B","Ventes":"Ventes_B","Marge":"Marge_B"})
    det = rea.merge(bud, on="Type_produit", how="outer").fillna(0.0)
    for m in ["CA","Ventes","Marge"]: det[f"Écart_{m}"] = det[f"{m}_R"] - det[f"{m}_B"]
    totals = {"CA":max(det["CA_R"].sum(),1e-9), "Ventes":max(det["Ventes_R"].sum(),1e-9), "Marge":max(det["Marge_R"].sum(),1e-9)}
    det["Poids_CA_%"] = (det["CA_R"]/totals["CA"]*100).round(2)
    det["Poids_Ventes_%"] = (det["Ventes_R"]/totals["Ventes"]*100).round(2)
    det["Poids_Marge_%"] = (det["Marge_R"]/totals["Marge"]*100).round(2)
    det = det[["Type_produit","CA_B","CA_R","Écart_CA","Ventes_B","Ventes_R","Écart_Ventes","Marge_B","Marge_R","Écart_Marge","Poids_CA_%","Poids_Ventes_%","Poids_Marge_%"]]
    st.dataframe(det.sort_values("CA_R", ascending=False), use_container_width=True, height=420)

    st.subheader("Matrice CA — Magasin x Produit (Réalisé)")
    mat = (dfr_real.pivot_table(index="Magasin", columns="Type_produit", values="montant_ca", aggfunc="sum").fillna(0.0))
    st.dataframe(mat.sort_values(by=mat.columns.tolist(), ascending=False), use_container_width=True, height=420)

# =========================================================
# PALMARÈS — Classement magasins (régional) + position région vs autres
# =========================================================
with tab_pal:
    st.write("Palmarès des magasins **de la région** (classement & évolutions).")
    metric_label = st.selectbox("Indicateur", ["CA Réel","Ventes Réel","Marge Réel"], index=0, key="pal_reg_metric")

    cur_month = mois_sel
    prev_month = str(pd.Period(cur_month, "M") - 1) if len(cur_month)==7 else None

    def build_scope(mois_str):
        if mois_str is None or len(mois_str)!=7:
            return df.iloc[0:0].copy()
        return scope(df, mois_str, sel_region)

    cur = build_scope(cur_month)
    prv = build_scope(prev_month)

    def agg_pal(d):
        if d.empty: return d
        g = (d.groupby(["Magasin","Region","Nature"], as_index=False)
               .agg(CA=("montant_ca","sum"), Ventes=("montant_vente","sum"), Marge=("montant_marge_brute","sum")))
        r = g[g["Nature"]=="Réalisé"].drop(columns=["Nature"]).rename(columns={"CA":"CA_R","Ventes":"Ventes_R","Marge":"Marge_R"})
        b = g[g["Nature"]=="Budget"].drop(columns=["Nature"]).rename(columns={"CA":"CA_B","Ventes":"Ventes_B","Marge":"Marge_B"})
        x = r.merge(b, on=["Magasin","Region"], how="outer").fillna(0.0)
        for k in [("Ratio_CA","CA_R","CA_B"),("Ratio_Ventes","Ventes_R","Ventes_B"),("Ratio_Marge","Marge_R","Marge_B")]:
            x[k[0]] = (x[k[1]]/x[k[2]]).replace([np.inf,-np.inf,np.nan], 0.0)
        return x

    pal_c = agg_pal(cur)
    pal_p = agg_pal(prv) if not prv.empty else pal_c.head(0).copy()

    metric_map = {"CA Réel":"CA_R","Ventes Réel":"Ventes_R","Marge Réel":"Marge_R"}
    mcol = metric_map[metric_label]

    if pal_c.empty:
        st.info("Aucune donnée pour le palmarès avec ces filtres.")
    else:
        pal_c = pal_c.sort_values(mcol, ascending=False).reset_index(drop=True)
        pal_c["Classement Régional"] = pal_c.index + 1

        if not pal_p.empty:
            pal_p = pal_p.sort_values(mcol, ascending=False).reset_index(drop=True)
            pal_p["Classement Régional (t-1)"] = pal_p.index + 1
            pal_c = pal_c.merge(pal_p[["Magasin","Classement Régional (t-1)"]], on="Magasin", how="left")
        else:
            pal_c["Classement Régional (t-1)"] = np.nan

        def evol_fmt(x):
            if pd.isna(x): return "—"
            if x == 0: return "="
            return f"{int(x):+d}"
        pal_c["Évolution Régional"] = (pal_c["Classement Régional (t-1)"] - pal_c["Classement Régional"]).apply(evol_fmt)

        by_reg = (df[df["Nature"]=="Réalisé"].groupby("Region", as_index=False).agg(CA=("montant_ca","sum")))
        by_reg = by_reg.sort_values("CA", ascending=False).reset_index(drop=True)
        pos_region = int(by_reg.index[by_reg["Region"]==sel_region][0]) + 1 if sel_region in by_reg["Region"].values else None

        cols = ["Magasin","CA_B","CA_R","Ratio_CA","Ventes_B","Ventes_R","Ratio_Ventes","Marge_B","Marge_R","Ratio_Marge","Classement Régional","Évolution Régional"]
        st.dataframe(pal_c[cols], use_container_width=True, height=560)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rang régional — CA", "1" if not pal_c.empty else "—")
        c2.metric("Position de la région vs autres (CA)", f"{pos_region}" if pos_region else "—")
        c3.metric("Magasins dans la région", f"{pal_c.shape[0]}")

        b = io.BytesIO()
        try:
            with pd.ExcelWriter(b, engine="openpyxl") as w:
                pal_c[cols].to_excel(w, sheet_name="Palmares_region", index=False)
                by_reg.to_excel(w, sheet_name="CA_regions", index=False)
            xbytes = b.getvalue()
        except Exception:
            xbytes = b""
        st.download_button("Exporter (Excel)", data=xbytes,
                           file_name=f"palmares_region_{sel_region}_{mois_sel}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           disabled=(xbytes==b""))

# ---------- Génération PDF si demandé ----------
def build_regional_pdf(region_name: str, mois_sel: str, fam_sel: str, ens_sel: str, 
                      figures: list, dff_scope: pd.DataFrame) -> bytes:
    """Construit un PDF pour le dashboard régional."""
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
    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, H-2*cm, f"DARTIES — Dashboard Régional - {region_name}")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H-2.7*cm, f"Période: {mois_sel}")
    c.drawString(2*cm, H-3.2*cm, f"Famille: {fam_sel} | Enseigne: {ens_sel}")
    if sel_mags:
        c.drawString(2*cm, H-3.7*cm, f"Magasins: {', '.join(sel_mags)}")

    # KPI régionaux
    reg_data = dff_scope[dff_scope["Nature"]=="Réalisé"].agg({
        "montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"
    })
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, H-4.5*cm, f"KPI {region_name} (Réel) :")
    c.setFont("Helvetica", 9)
    c.drawString(2.5*cm, H-5*cm, f"CA: {reg_data['montant_ca']:,.0f} €".replace(",", " "))
    c.drawString(2.5*cm, H-5.5*cm, f"Ventes: {reg_data['montant_vente']:,.0f}".replace(",", " "))
    c.drawString(2.5*cm, H-6*cm, f"Marge: {reg_data['montant_marge_brute']:,.0f} €".replace(",", " "))

    # Export des figures
    y = H-7.5*cm
    max_w_img = W-4*cm
    max_h_img = 7*cm

    for fig in figures:
        if fig is None:
            continue
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            img = ImageReader(io.BytesIO(png_bytes))
            
            iw, ih = img.getSize()
            scale = min(max_w_img/iw, max_h_img/ih)
            w_draw = iw*scale
            h_draw = ih*scale
            
            if y - h_draw < 2*cm:
                c.showPage()
                y = H-2*cm
            
            c.drawImage(img, 2*cm, y-h_draw, width=w_draw, height=h_draw)
            y -= (h_draw + 1*cm)
            
        except Exception as e:
            st.error(f"Erreur export graphique: {e}")

    c.save()
    return pdf_buf.getvalue()

if export_pdf_clicked:
    pdf_bytes = build_regional_pdf(
        region_name=sel_region, mois_sel=mois_sel, 
        fam_sel=fam_sel, ens_sel=ens_sel,
        figures=figures_for_pdf, dff_scope=dfr
    )
    st.download_button(
        "📥 Télécharger le PDF",
        data=pdf_bytes,
        file_name=f"dashboard_regional_{sel_region}_{mois_sel}.pdf",
        mime="application/pdf",
        disabled=(pdf_bytes==b"")
    )