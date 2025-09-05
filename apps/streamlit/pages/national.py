# pages/national.py
# -*- coding: utf-8 -*-
import io
from datetime import datetime, date
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TABLEAU DE BORD - DARTIES - NATIONAL", page_icon="📌", layout="wide")

# ---------- Bandeau + styles avec couleurs améliorées ----------
def top_banner():
    st.markdown("""
    <style>
    /* Bandeau avec gradient bleu */
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

    /* Cartes avec gradient amélioré */
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

    /* Tableau avec header bleu */
    table.tbl-small{
        border-collapse:collapse; font-size:13px; width:auto; color:#e5e7eb;
    }
    table.tbl-small th, table.tbl-small td{
        border:1px solid #334155; padding:6px 8px;
    }
    table.tbl-small th{
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color:#f8fafc; font-weight:700;
    }
    table.tbl-small tr:nth-child(odd) td{ background:#0f172a; }
    table.tbl-small tr:nth-child(even) td{ background:#1e293b; }

    /* Ratios avec couleurs améliorées */
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
        <h1>TABLEAU DE BORD - DARTIES - NATIONAL</h1>
    </div>
    """, unsafe_allow_html=True)

top_banner()

# ---------- Sécurité : DF depuis app.py ----------
if "df" not in st.session_state or st.session_state.df is None or st.session_state.df.empty:
    st.warning("Aucune donnée en mémoire. Va d'abord sur **Accueil** (app.py) pour te connecter à la base, puis reviens.")
    st.stop()

df = st.session_state.df.copy()

# colonnes attendues
for col in ["date_conso","Annee","Mois","mois","Region","Ville","Enseigne","Type_produit","Magasin",
            "montant_ca","montant_marge_brute","montant_vente","type_conso"]:
    if col not in df.columns:
        df[col] = "" if col not in ["montant_ca","montant_marge_brute","montant_vente"] else 0.0

df["date_conso"] = pd.to_datetime(df["date_conso"], errors="coerce")
for c in ["montant_ca","montant_marge_brute","montant_vente"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
df["mois"] = df["date_conso"].dt.to_period("M").astype(str)

def label_nature(x):
    x = str(x).strip().upper()
    if x in {"R","REALISE","RÉALISÉ","REEL","RÉEL"}: return "Réalisé"
    if x in {"O","OBJ","OBJECTIF","BUDGET"}:        return "Budget"
    return "Réalisé"
df["Nature"] = df["type_conso"].fillna("").apply(label_nature)

# ---------- Panneau de filtres avec valeurs par défaut vides ----------
with st.sidebar:
    st.markdown("### Filtres")
    
    # Clés pour reset
    NATIONAL_KEYS = ["nat_mois", "nat_zones", "nat_fam", "nat_ens"]
    
    # Bouton reset
    if st.button("🔄 Effacer filtres", use_container_width=True, key="btn_reset_nat"):
        for k in NATIONAL_KEYS:
            st.session_state.pop(k, None)
        st.rerun()
    
    devise = st.selectbox("Devises", ["Euros"], index=0)
    taux = st.selectbox("Taux", ["Taux 1"], index=0)

    # période (mois) — commence vide
    mois_dispo = sorted(df["mois"].dropna().unique())
    mois_sel = st.selectbox("Période (mois)", [""] + mois_dispo, index=0, key="nat_mois")
    
    if not mois_sel:
        st.info("Sélectionnez d'abord une période")
        st.stop()

    # régions commerciales — commence vide
    st.markdown("**Régions commerciales**")
    zones = ["NORD_OUEST","NORD_EST","REGION_PARISIENNE","SUD_OUEST","SUD_EST"]
    zones_selected = st.multiselect("Zones", options=zones, default=[], key="nat_zones")
    
    if not zones_selected:
        st.info("Sélectionnez au moins une région")
        st.stop()

    indicateur = st.selectbox("Tous les indicateurs", ["CA / Ventes / Marge"], index=0)
    familles = sorted([x for x in df["Type_produit"].fillna("").unique() if str(x)!=""])
    fam_sel = st.selectbox("Toutes les familles", ["Toutes"] + familles, index=0, key="nat_fam")

    enseignes = sorted([x for x in df["Enseigne"].fillna("").unique() if str(x)!=""])
    ens_sel = st.selectbox("Toutes les enseignes", ["Toutes"] + enseignes, index=0, key="nat_ens")

    # SUPPRESSION DU CUMUL
    st.write("")
    
    st.divider()
    st.markdown("### Export")
    export_pdf_clicked = st.button("📄 Générer dashboard PDF", use_container_width=True, 
                                  key="btn_pdf_nat", type="primary")

# application filtres (SUPPRESSION DU CUMUL)
dff = df.copy()
if mois_sel:
    dff = dff[dff["mois"] == mois_sel]          # mois courant uniquement
dff = dff[dff["Region"].isin(zones_selected)]
if fam_sel != "Toutes":
    dff = dff[dff["Type_produit"] == fam_sel]
if ens_sel != "Toutes":
    dff = dff[dff["Enseigne"] == ens_sel]

# ---------- Helpers calculs ----------
def table_zone(dfx, region_name):
    sub = dfx[dfx["Region"] == region_name] if region_name is not None else dfx
    bud = sub[sub["Nature"]=="Budget"].agg({
        "montant_ca":"sum", "montant_vente":"sum", "montant_marge_brute":"sum"
    })
    rea = sub[sub["Nature"]=="Réalisé"].agg({
        "montant_ca":"sum", "montant_vente":"sum", "montant_marge_brute":"sum"
    })
    bCA,bV,bM = float(bud.get("montant_ca",0)), float(bud.get("montant_vente",0)), float(bud.get("montant_marge_brute",0))
    rCA,rV,rM = float(rea.get("montant_ca",0)), float(rea.get("montant_vente",0)), float(rea.get("montant_marge_brute",0))
    ratioCA = (rCA/bCA) if bCA>0 else 0.0
    ratioV  = (rV /bV ) if bV >0 else 0.0
    ratioM  = (rM /bM ) if bM >0 else 0.0
    tab = pd.DataFrame({
        "": ["Budget","Réel","Ratio"],
        "CA":  [bCA, rCA, ratioCA],
        "Ventes":[bV, rV, ratioV],
        "Marge":[bM, rM, ratioM],
    })
    return tab

def style_small_table(df_tab):
    rows = []
    th = "<tr><th></th><th>CA</th><th>Vente</th><th>Marge</th></tr>"
    for _, row in df_tab.iterrows():
        label = row[""]
        def cell(val, is_ratio):
            if is_ratio:
                css = "ratio-pos" if val>=1 else "ratio-neg"
                return f"<td class='{css}'>{val:.2f}</td>"
            if isinstance(val,(int,float)):
                return f"<td>{f'{val:,.2f}'.replace(',', ' ')}</td>"
            return f"<td>{val}</td>"
        row_html = f"<tr><td>{label}</td>"+\
                   cell(row['CA'], label=="Ratio")+\
                   cell(row['Ventes'], label=="Ratio")+\
                   cell(row['Marge'], label=="Ratio")+\
                   "</tr>"
        rows.append(row_html)
    html = "<table class='tbl-small'>" + th + "".join(rows) + "</table>"
    return html

# ---------- Variables pour PDF ----------
figures_for_pdf = []

# ---------- TABS ----------
tab_acc, tab_hist, tab_detail, tab_rank = st.tabs(["Accueil", "Historique", "Détails", "Palmarès"])

# ======== ACCUEIL ========
with tab_acc:
    st.write(f"Résultat en France en **{devise}**, pour **{', '.join(zones_selected)}**, "
             f"pour le **{mois_sel}**, familles **{fam_sel}**, enseignes **{ens_sel}**.")

    # Carte (centroïdes des 5 zones)
    COLORS = {
        "NORD_OUEST":"#2563eb",
        "NORD_EST":"#10b981",
        "REGION_PARISIENNE":"#8b5cf6",
        "SUD_OUEST":"#f59e0b",
        "SUD_EST":"#ef4444"
    }
    REGION_TO_LATLON = {
        "NORD_OUEST": (48.8, -1.5),
        "NORD_EST": (48.7, 6.2),
        "REGION_PARISIENNE": (48.86, 2.35),
        "SUD_OUEST": (44.0, 0.0),
        "SUD_EST": (44.5, 5.5),
    }

    reg_data = []
    for r in zones_selected:
        lat, lon = REGION_TO_LATLON.get(r, (np.nan, np.nan))
        if np.isnan(lat): continue
        tab = table_zone(dff, r)
        rCA = float(tab.loc[tab[""]=="Réel", "CA"].values[0] if not tab.empty else 0)
        reg_data.append({"Region": r, "lat": lat, "lon": lon, "CA": rCA, "color": COLORS.get(r, "#1f77b4")})

    fig_map = go.Figure()
    if reg_data:
        vals = np.array([x["CA"] for x in reg_data], dtype=float)
        vmin, vmax = (vals.min(), vals.max()) if len(vals)>0 else (0, 1)
        def size_for(v):
            if vmax == vmin: return 20
            return 12 + (v - vmin) / (vmax - vmin) * 40  # px
        for x in reg_data:
            fig_map.add_trace(go.Scattermapbox(
                lat=[x["lat"]], lon=[x["lon"]],
                mode="markers+text",
                marker=dict(size=size_for(x["CA"]), color=x["color"], opacity=0.85),
                text=[x["Region"]], textposition="top center",
                textfont=dict(color="#f8fafc"),
                hovertemplate=f"<b>{x['Region']}</b><br>CA (réel) : {x['CA']:,.0f} €<extra></extra>"
            ))

    fig_map.update_layout(
        template="plotly_dark",
        mapbox_style="carto-darkmatter",
        mapbox=dict(center=dict(lat=46.8, lon=2.5), zoom=4.7),
        margin=dict(l=10,r=10,t=10,b=10), height=520,
        font=dict(color="#e5e7eb")
    )

    g1, g2 = st.columns([2,1])
    with g1:
        st.plotly_chart(fig_map, use_container_width=True)
        figures_for_pdf.append(fig_map)
    with g2:
        st.markdown("**Fiches régionales (Budget / Réel / Ratio)**")
        for r in zones_selected:
            tabz = table_zone(dff, r)
            html = style_small_table(tabz)
            st.markdown(f"<div class='card-lite'><b>{r}</b><br/>{html}</div>", unsafe_allow_html=True)

        # total national
        st.markdown("**National**")
        bud = dff[dff["Nature"]=="Budget"].agg({"montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"})
        rea = dff[dff["Nature"]=="Réalisé"].agg({"montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"})
        bCA,bV,bM = float(bud["montant_ca"]), float(bud["montant_vente"]), float(bud["montant_marge_brute"])
        rCA,rV,rM = float(rea["montant_ca"]), float(rea["montant_vente"]), float(rea["montant_marge_brute"])
        nat = pd.DataFrame({"": ["Budget","Réel","Ratio"],
                            "CA":[bCA,rCA,(rCA/bCA if bCA>0 else 0.0)],
                            "Ventes":[bV,rV,(rV/bV if bV>0 else 0.0)],
                            "Marge":[bM,rM,(rM/bM if bM>0 else 0.0)]})
        st.markdown(f"<div class='card-lite'>{style_small_table(nat)}</div>", unsafe_allow_html=True)

# ======== HISTORIQUE ========
with tab_hist:
    st.write("Budget réel et estimé pour les différentes familles de produits **sur 2 ans**.")
    base = df.copy()
    if fam_sel != "Toutes": base = base[base["Type_produit"]==fam_sel]
    if ens_sel != "Toutes": base = base[base["Enseigne"]==ens_sel]
    base = base[base["Region"].isin(zones_selected)]

    if len(mois_sel)==7:
        yyyy, mm = map(int, mois_sel.split("-"))
        max_d = datetime(yyyy, mm, 28) + pd.offsets.MonthEnd(0)
    else:
        max_d = base["date_conso"].max() if base["date_conso"].notna().any() else pd.Timestamp.today()
    min_24 = (pd.Timestamp(max_d) - pd.DateOffset(months=24)).normalize()

    hist = base[(base["date_conso"]>=min_24) & (base["date_conso"]<=max_d)].copy()
    hist["Annee"] = hist["date_conso"].dt.year
    hist["MoisNum"] = hist["date_conso"].dt.month

    fams = hist["Type_produit"].dropna().unique().tolist()
    fams_used = sorted(fams)[:3] if fams else []
    met = st.selectbox("Indicateur", ["CA","Ventes","Marge"], index=0)

    hist["Val"] = np.select(
        [met=="CA", met=="Ventes", met=="Marge"],
        [hist["montant_ca"], hist["montant_vente"], hist["montant_marge_brute"]],
        default=hist["montant_ca"]
    )

    rows = []
    month_sel_dt = pd.Period(mois_sel, "M").to_timestamp("M")
    for fam in fams_used:
        sub = hist[hist["Type_produit"]==fam]
        sub_month = sub[(sub["date_conso"].dt.year==month_sel_dt.year) & (sub["date_conso"].dt.month==month_sel_dt.month)]
        val_b = sub_month[sub_month["Nature"]=="Budget"]["Val"].sum()
        val_r = sub_month[sub_month["Nature"]=="Réalisé"]["Val"].sum()
        ecart = (val_r - val_b) / val_b if val_b>0 else 0.0

        sub_ytd = sub[(sub["date_conso"].dt.year==month_sel_dt.year) & (sub["date_conso"].dt.month<=month_sel_dt.month)]
        ytd_b = sub_ytd[sub_ytd["Nature"]=="Budget"]["Val"].sum()
        ytd_r = sub_ytd[sub_ytd["Nature"]=="Réalisé"]["Val"].sum()
        rows.append([fam, val_b, val_r, ecart, ytd_b, ytd_r])

    tbl = pd.DataFrame(rows, columns=["Famille", "Budget (mois)", "Réel (mois)", "Écart (mois)", "Budget (YTD)", "Réel (YTD)"])
    tot = pd.DataFrame([["Total",
                         tbl["Budget (mois)"].sum(),
                         tbl["Réel (mois)"].sum(),
                         (tbl["Réel (mois)"].sum()/tbl["Budget (mois)"].sum()-1) if tbl["Budget (mois)"].sum()>0 else 0.0,
                         tbl["Budget (YTD)"].sum(),
                         tbl["Réel (YTD)"].sum()]], columns=tbl.columns)
    tbl = pd.concat([tbl, tot], ignore_index=True)
    st.dataframe(tbl, use_container_width=True, height=420)

# ======== DÉTAILS ========
with tab_detail:
    st.write("Détails en France, **tous indicateurs**, toutes familles, toutes enseignes, pour le mois sélectionné.")
    fam_agg = (
        dff.groupby(["Type_produit","Nature"], as_index=False)
           .agg(CA=("montant_ca","sum"),
                Ventes=("montant_vente","sum"),
                Marge=("montant_marge_brute","sum"))
    )
    rea = fam_agg[fam_agg["Nature"]=="Réalisé"].drop(columns=["Nature"]).rename(columns={"CA":"CA_R","Ventes":"Ventes_R","Marge":"Marge_R"})
    bud = fam_agg[fam_agg["Nature"]=="Budget"].drop(columns=["Nature"]).rename(columns={"CA":"CA_B","Ventes":"Ventes_B","Marge":"Marge_B"})
    det = rea.merge(bud, on="Type_produit", how="outer").fillna(0.0)
    for m in ["CA","Ventes","Marge"]:
        det[f"Écart_{m}"] = det[f"{m}_R"] - det[f"{m}_B"]
    totR = {
        "CA": max(det["CA_R"].sum(),1e-9),
        "Ventes": max(det["Ventes_R"].sum(),1e-9),
        "Marge": max(det["Marge_R"].sum(),1e-9),
    }
    det["Poids_CA_%"] = (det["CA_R"]/totR["CA"]*100).round(2)
    det["Poids_Ventes_%"] = (det["Ventes_R"]/totR["Ventes"]*100).round(2)
    det["Poids_Marge_%"] = (det["Marge_R"]/totR["Marge"]*100).round(2)
    det = det[["Type_produit","CA_B","CA_R","Écart_CA","Ventes_B","Ventes_R","Écart_Ventes","Marge_B","Marge_R","Écart_Marge","Poids_CA_%","Poids_Ventes_%","Poids_Marge_%"]]
    st.dataframe(det.sort_values("CA_R", ascending=False), use_container_width=True, height=500)

# ======== PALMARÈS (avec classements & évolutions) ========
with tab_rank:
    st.write("Palmarès des magasins — toutes familles / enseignes — pour le mois sélectionné.")
    metric_label = st.selectbox("Indicateur palmarès", ["CA Réel","Ventes Réel","Marge Réel"], index=0)

    # -- helpers scope courant / précédent
    def apply_scope(base: pd.DataFrame, mois_str: str) -> pd.DataFrame:
        d = base.copy()
        d = d[d["Region"].isin(zones_selected)]
        if fam_sel != "Toutes": d = d[d["Type_produit"] == fam_sel]
        if ens_sel != "Toutes": d = d[d["Enseigne"] == ens_sel]
        if mois_str is None or len(mois_str)!=7:
            return d.iloc[0:0].copy()
        d = d[d["mois"] == mois_str]
        return d

    cur_month = mois_sel
    try:
        prev_month = str(pd.Period(cur_month, "M") - 1)
    except Exception:
        prev_month = None

    base_cur  = apply_scope(df, cur_month)
    base_prev = apply_scope(df, prev_month) if prev_month else df.iloc[0:0].copy()

    def agg_pal(d: pd.DataFrame) -> pd.DataFrame:
        if d.empty:
            return d
        g = (
            d.groupby(["Magasin","Region","Nature"], as_index=False)
             .agg(CA=("montant_ca","sum"),
                  Ventes=("montant_vente","sum"),
                  Marge=("montant_marge_brute","sum"))
        )
        rea = g[g["Nature"]=="Réalisé"].drop(columns=["Nature"]).rename(
            columns={"CA":"CA_R","Ventes":"Ventes_R","Marge":"Marge_R"}
        )
        bud = g[g["Nature"]=="Budget"].drop(columns=["Nature"]).rename(
            columns={"CA":"CA_B","Ventes":"Ventes_B","Marge":"Marge_B"}
        )
        x = rea.merge(bud, on=["Magasin","Region"], how="outer").fillna(0.0)
        x["Ratio_CA"]     = (x["CA_R"]/x["CA_B"]).replace([np.inf,-np.inf,np.nan], 0.0)
        x["Ratio_Ventes"] = (x["Ventes_R"]/x["Ventes_B"]).replace([np.inf,-np.inf,np.nan], 0.0)
        x["Ratio_Marge"]  = (x["Marge_R"]/x["Marge_B"]).replace([np.inf,-np.inf,np.nan], 0.0)
        return x

    pal_cur  = agg_pal(base_cur)
    pal_prev = agg_pal(base_prev) if not base_prev.empty else pal_cur.head(0).copy()

    metric_map = {"CA Réel":"CA_R", "Ventes Réel":"Ventes_R", "Marge Réel":"Marge_R"}
    metric_col = metric_map[metric_label]

    if pal_cur.empty:
        st.info("Aucune donnée pour le palmarès avec les filtres/mois sélectionnés.")
    else:
        pal_cur = pal_cur.sort_values(metric_col, ascending=False).reset_index(drop=True)
        pal_cur["Classement National"] = pal_cur.index + 1

        if not pal_prev.empty:
            pal_prev = pal_prev.sort_values(metric_col, ascending=False).reset_index(drop=True)
            pal_prev["Classement National (t-1)"] = pal_prev.index + 1
            pal_cur = pal_cur.merge(
                pal_prev[["Magasin","Classement National (t-1)"]],
                on="Magasin", how="left"
            )
        else:
            pal_cur["Classement National (t-1)"] = np.nan

        def evol_fmt(x):
            if pd.isna(x): return "—"
            if x == 0:     return "="
            return f"{int(x):+d}"

        pal_cur["Évolution National"] = (
            pal_cur["Classement National (t-1)"] - pal_cur["Classement National"]
        ).apply(evol_fmt)

        pal_cur["Classement Régional"] = (
            pal_cur.groupby("Region")[metric_col]
                   .rank(ascending=False, method="min")
                   .astype(int)
        )
        if not pal_prev.empty:
            pal_prev["Classement Régional (t-1)"] = (
                pal_prev.groupby("Region")[metric_col]
                        .rank(ascending=False, method="min")
                        .astype(int)
            )
            pal_cur = pal_cur.merge(
                pal_prev[["Magasin","Classement Régional (t-1)"]],
                on="Magasin", how="left"
            )
        else:
            pal_cur["Classement Régional (t-1)"] = np.nan

        pal_cur["Évolution Régional"] = (
            pal_cur["Classement Régional (t-1)"] - pal_cur["Classement Régional"]
        ).apply(evol_fmt)

        pal_cur = pal_cur.sort_values(metric_col, ascending=False)

        cols_show = [
            "Magasin","Region",
            "CA_B","CA_R","Ratio_CA",
            "Ventes_B","Ventes_R","Ratio_Ventes",
            "Marge_B","Marge_R","Ratio_Marge",
            "Classement National","Évolution National",
            "Classement Régional","Évolution Régional",
        ]
        st.dataframe(pal_cur[cols_show], use_container_width=True, height=600)

        # Export Excel
        b = io.BytesIO()
        try:
            with pd.ExcelWriter(b, engine="openpyxl") as w:
                pal_cur[cols_show].to_excel(w, sheet_name="Palmares", index=False)
            xbytes = b.getvalue()
        except Exception:
            xbytes = b""
        st.download_button(
            "Exporter le palmarès (Excel)",
            data=xbytes,
            file_name=f"palmares_{cur_month}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=(xbytes==b"")
        )

# ---------- Génération PDF si demandé ----------
def build_national_pdf(mois_sel: str, zones_selected: list, fam_sel: str, ens_sel: str, 
                      figures: list, dff_scope: pd.DataFrame) -> bytes:
    """Construit un PDF pour le dashboard national."""
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
    c.drawString(2*cm, H-2*cm, "DARTIES — Dashboard National")
    c.setFont("Helvetica", 10)
    c.drawString(2*cm, H-2.7*cm, f"Période: {mois_sel}")
    c.drawString(2*cm, H-3.2*cm, f"Régions: {', '.join(zones_selected)}")
    c.drawString(2*cm, H-3.7*cm, f"Famille: {fam_sel} | Enseigne: {ens_sel}")

    # KPI nationaux
    nat_data = dff_scope[dff_scope["Nature"]=="Réalisé"].agg({
        "montant_ca":"sum","montant_vente":"sum","montant_marge_brute":"sum"
    })
    c.setFont("Helvetica-Bold", 10)
    c.drawString(2*cm, H-4.5*cm, "KPI Nationaux (Réel) :")
    c.setFont("Helvetica", 9)
    c.drawString(2.5*cm, H-5*cm, f"CA Total: {nat_data['montant_ca']:,.0f} €".replace(",", " "))
    c.drawString(2.5*cm, H-5.5*cm, f"Ventes: {nat_data['montant_vente']:,.0f}".replace(",", " "))
    c.drawString(2.5*cm, H-6*cm, f"Marge: {nat_data['montant_marge_brute']:,.0f} €".replace(",", " "))

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
    pdf_bytes = build_national_pdf(
        mois_sel=mois_sel, zones_selected=zones_selected, 
        fam_sel=fam_sel, ens_sel=ens_sel,
        figures=figures_for_pdf, dff_scope=dff
    )
    st.download_button(
        "📥 Télécharger le PDF",
        data=pdf_bytes,
        file_name=f"dashboard_national_{mois_sel}.pdf",
        mime="application/pdf",
        disabled=(pdf_bytes==b"")
    )