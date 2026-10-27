# -*- coding: utf-8 -*-
"""
Observatorio Economico y Social por Departamentos - Streamlit
Datos reales de datos.gov.co
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import unicodedata
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.stattools import adfuller
from scipy import stats

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CONFIGURACION DE LA PAGINA
# ============================================================
st.set_page_config(
    page_title="Observatorio Departamental Colombia",
    page_icon="🇨🇴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# RASTREO Y LOGGING
# ============================================================
LOG_CARGA = []

def log(msg, nivel="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    entrada = f"[{ts}] [{nivel}] {msg}"
    LOG_CARGA.append(entrada)

# ============================================================
# NORMALIZACION
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = t.replace(",", "").replace(".", " ").replace("  ", " ").strip()
    return t

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "bolivar": "Bolívar", "boyaca": "Boyacá",
    "caldas": "Caldas", "caqueta": "Caquetá", "casanare": "Casanare",
    "cauca": "Cauca", "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca", "guainia": "Guainía", "guaviare": "Guaviare",
    "huila": "Huila", "la guajira": "La Guajira", "magdalena": "Magdalena",
    "meta": "Meta", "narino": "Nariño", "norte de santander": "Norte de Santander",
    "putumayo": "Putumayo", "quindio": "Quindío", "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia", "san andres y providencia": "San Andrés y Providencia",
    "archipielago de san andres": "San Andrés y Providencia", "santander": "Santander",
    "sucre": "Sucre", "tolima": "Tolima", "valle": "Valle del Cauca",
    "valle del cauca": "Valle del Cauca", "vaupes": "Vaupés", "vichada": "Vichada",
}

def normalizar_departamento(nombre):
    clave = quitar_tildes(nombre)
    return DEPARTAMENTOS_CANONICOS.get(clave, str(nombre).strip().title() if nombre else None)

# ============================================================
# CARGA DE DATOS
# ============================================================
def _cargar_socrata(url, limit=50000, timeout=90):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        log(f"Error al cargar {url}: {e}", "ERROR")
        return pd.DataFrame()

def cargar_pib():
    log("Cargando PIB departamental (kgyi-qc7j)...")
    url = "https://www.datos.gov.co/resource/kgyi-qc7j.json?$limit=50000&$order=a_o DESC"
    df = _cargar_socrata(url)
    if df.empty:
        return pd.DataFrame()
    df["anio"] = pd.to_numeric(df["a_o"], errors="coerce")
    df["valor"] = pd.to_numeric(df["valor_miles_de_millones_de"], errors="coerce")
    df["departamento"] = df["departamento"].apply(normalizar_departamento)
    df = df[df["tipo_de_precios"].str.contains("constantes", case=False, na=False)]
    agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
    agg["indicador"] = "PIB (miles de millones, constantes 2015)"
    agg["fuente"] = "datos.gov.co/kgyi-qc7j"
    agg["categoria"] = "Economía"
    log(f"PIB cargado: {len(agg)} registros")
    return agg

def cargar_homicidios():
    log("Cargando Homicidios (m8fd-ahd9)...")
    url = "https://www.datos.gov.co/resource/m8fd-ahd9.json?$limit=150000"
    df = _cargar_socrata(url)
    if df.empty:
        return pd.DataFrame()
    df["fecha"] = pd.to_datetime(df["fecha_hecho"], errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["departamento"] = df["departamento"].apply(normalizar_departamento)
    df["cantidad"] = pd.to_numeric(df.get("cantidad"), errors="coerce").fillna(1)
    agg = (df.groupby(["departamento", "anio"], as_index=False)["cantidad"]
           .sum().rename(columns={"cantidad": "valor"}).dropna())
    agg["indicador"] = "Homicidios (víctimas)"
    agg["fuente"] = "MinDefensa/m8fd-ahd9"
    agg["categoria"] = "Seguridad"
    log(f"Homicidios cargados: {len(agg)} registros")
    return agg

def cargar_lesiones():
    log("Cargando Lesiones Personales (72sg-cybi)...")
    url = "https://www.datos.gov.co/resource/72sg-cybi.json?$limit=100000"
    df = _cargar_socrata(url)
    if df.empty:
        return pd.DataFrame()
    fecha_col = next((c for c in ["fecha_hecho", "fecha", "ano"] if c in df.columns), None)
    if fecha_col:
        if "ano" in fecha_col or df[fecha_col].dtype in [int, float]:
            df["anio"] = pd.to_numeric(df[fecha_col], errors="coerce")
        else:
            df["fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
            df["anio"] = df["fecha"].dt.year
    else:
        return pd.DataFrame()
    depto_col = next((c for c in ["departamento", "depto", "cod_depto"] if c in df.columns), None)
    if not depto_col:
        return pd.DataFrame()
    df["departamento"] = df[depto_col].apply(normalizar_departamento)
    cant_col = next((c for c in ["cantidad", "casos", "victimas"] if c in df.columns), None)
    df["cantidad"] = pd.to_numeric(df[cant_col], errors="coerce").fillna(1) if cant_col else 1
    agg = (df.groupby(["departamento", "anio"], as_index=False)["cantidad"]
           .sum().rename(columns={"cantidad": "valor"}).dropna())
    agg["indicador"] = "Lesiones personales"
    agg["fuente"] = "MinDefensa/72sg-cybi"
    agg["categoria"] = "Seguridad"
    log(f"Lesiones cargadas: {len(agg)} registros")
    return agg

def cargar_hurto_personas():
    log("Cargando Hurto a personas (4rxi-8m8d)...")
    url = "https://www.datos.gov.co/resource/4rxi-8m8d.json?$limit=100000"
    df = _cargar_socrata(url)
    if df.empty:
        return pd.DataFrame()
    fecha_col = next((c for c in ["fecha_hecho", "fecha", "ano"] if c in df.columns), None)
    if fecha_col:
        if "ano" in str(fecha_col).lower() or df[fecha_col].dtype in [int, float]:
            df["anio"] = pd.to_numeric(df[fecha_col], errors="coerce")
        else:
            df["fecha"] = pd.to_datetime(df[fecha_col], errors="coerce")
            df["anio"] = df["fecha"].dt.year
    else:
        return pd.DataFrame()
    depto_col = next((c for c in ["departamento", "depto"] if c in df.columns), None)
    if not depto_col:
        return pd.DataFrame()
    df["departamento"] = df[depto_col].apply(normalizar_departamento)
    cant_col = next((c for c in ["cantidad", "casos"] if c in df.columns), None)
    df["cantidad"] = pd.to_numeric(df[cant_col], errors="coerce").fillna(1) if cant_col else 1
    agg = (df.groupby(["departamento", "anio"], as_index=False)["cantidad"]
           .sum().rename(columns={"cantidad": "valor"}).dropna())
    agg["indicador"] = "Hurto a personas"
    agg["fuente"] = "MinDefensa/4rxi-8m8d"
    agg["categoria"] = "Seguridad"
    log(f"Hurto a personas cargado: {len(agg)} registros")
    return agg

def cargar_dnp():
    log("Cargando Recursos DNP (wc86-9j4a)...")
    url = "https://www.datos.gov.co/resource/wc86-9j4a.json?$limit=80000"
    df = _cargar_socrata(url)
    if df.empty:
        return pd.DataFrame()
    df["departamento"] = df["departamento"].apply(normalizar_departamento)
    df["anio"] = pd.to_numeric(df.get("vigencia"), errors="coerce")
    valor_col = next((c for c in ["valorpagado", "valorobligado", "valorcomprometido"] if c in df.columns), None)
    if not valor_col:
        return pd.DataFrame()
    df["valor"] = pd.to_numeric(df[valor_col], errors="coerce")
    agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
    agg["indicador"] = "Recursos públicos DNP (COP)"
    agg["fuente"] = "DNP/wc86-9j4a"
    agg["categoria"] = "Finanzas públicas"
    log(f"DNP cargado: {len(agg)} registros")
    return agg

@st.cache_data(show_spinner="Cargando datos oficiales de datos.gov.co...")
def construir_maestro():
    log("=== INICIANDO CARGA DE FUENTES OFICIALES ===")
    partes = []
    for fn in [cargar_pib, cargar_homicidios, cargar_lesiones, cargar_hurto_personas, cargar_dnp]:
        d = fn()
        if not d.empty:
            partes.append(d)
    if not partes:
        log("Ninguna fuente cargó datos", "ERROR")
        return pd.DataFrame(columns=["departamento", "anio", "valor", "indicador", "fuente", "categoria"])
    maestro = pd.concat(partes, ignore_index=True)
    maestro["anio"] = maestro["anio"].astype(int)
    maestro = maestro.dropna(subset=["departamento", "anio", "valor"])
    log(f"=== MAESTRO FINAL: {len(maestro)} registros | {maestro['indicador'].nunique()} indicadores ===")
    return maestro.sort_values(["categoria", "indicador", "departamento", "anio"]).reset_index(drop=True)

def crear_panel_ancho(df):
    if df.empty:
        return pd.DataFrame()
    pivot = df.pivot_table(
        index=["departamento", "anio"],
        columns="indicador",
        values="valor",
        aggfunc="sum"
    ).reset_index()
    pivot.columns.name = None
    return pivot.dropna(how="all", subset=[c for c in pivot.columns if c not in ["departamento", "anio"]])

# ============================================================
# UTILIDADES
# ============================================================
def filtrar(df, deptos, indicador, anio_min, anio_max):
    d = df.copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    if indicador and indicador != "Todos":
        d = d[d["indicador"] == indicador]
    d = d[(d["anio"] >= anio_min) & (d["anio"] <= anio_max)]
    return d

def get_series(df, depto, indicador):
    d = df[(df["departamento"] == depto) & (df["indicador"] == indicador)].sort_values("anio")
    return d[["anio", "valor"]].dropna()

CENTROIDES = {
    "Amazonas": (-71.5, -1.5), "Antioquia": (-75.5, 6.5), "Arauca": (-70.8, 7.0),
    "Atlántico": (-74.9, 10.7), "Bogotá D.C.": (-74.1, 4.6), "Bolívar": (-75.0, 9.0),
    "Boyacá": (-73.0, 5.5), "Caldas": (-75.5, 5.3), "Caquetá": (-73.5, 1.0),
    "Casanare": (-71.5, 5.5), "Cauca": (-76.5, 2.5), "Cesar": (-73.5, 9.5),
    "Chocó": (-77.0, 6.0), "Córdoba": (-75.8, 8.5), "Cundinamarca": (-74.2, 5.0),
    "Guainía": (-68.5, 2.5), "Guaviare": (-72.5, 2.0), "Huila": (-75.5, 2.5),
    "La Guajira": (-72.5, 11.5), "Magdalena": (-74.2, 10.0), "Meta": (-73.0, 3.5),
    "Nariño": (-77.5, 1.5), "Norte de Santander": (-72.5, 8.0), "Putumayo": (-76.5, 0.5),
    "Quindío": (-75.7, 4.5), "Risaralda": (-75.8, 5.0), "San Andrés y Providencia": (-81.7, 12.5),
    "Santander": (-73.5, 7.0), "Sucre": (-75.3, 9.0), "Tolima": (-75.2, 4.0),
    "Valle del Cauca": (-76.5, 3.8), "Vaupés": (-70.5, 1.0), "Vichada": (-69.5, 5.0),
}

# ============================================================
# CARGA PRINCIPAL DE DATOS
# ============================================================
DF = construir_maestro()
PANEL = crear_panel_ancho(DF)

lista_deptos = sorted(DF["departamento"].dropna().unique()) if not DF.empty else []
lista_inds = sorted(DF["indicador"].unique()) if not DF.empty else []
lista_inds_todos = ["Todos"] + lista_inds
amin = int(DF["anio"].min()) if not DF.empty else 2005
amax = int(DF["anio"].max()) if not DF.empty else 2024
cols_panel = [c for c in PANEL.columns if c not in ["departamento", "anio"]] if not PANEL.empty else []

# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #FFCD00 0%, #003893 50%, #CE1126 100%);
            padding: 1.5rem 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;">
    <h1 style="margin:0; font-size:1.8rem;">🇨🇴 Observatorio de Actividad Económica y Social por Departamentos</h1>
    <p style="margin:0.4rem 0 0 0; opacity:0.95;">Suite metodológica completa · Datos reales de datos.gov.co</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "🏠 Dashboard",
    "🔍 EDA",
    "⛏️ Clustering",
    "📉 PCA",
    "🤖 Machine Learning",
    "📈 VAR / VECM",
    "🗺️ Mapas",
    "🧠 LSTM",
    "🎯 Simulación",
    "🔎 Interpretabilidad"
])

# -------------------- TAB 1: DASHBOARD --------------------
with tab1:
    st.subheader("Estado de la carga de datos")
    if not DF.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total registros", f"{len(DF):,}")
        col2.metric("Indicadores", DF["indicador"].nunique())
        col3.metric("Departamentos", DF["departamento"].nunique())
        col4.metric("Rango años", f"{DF['anio'].min()} – {DF['anio'].max()}")

        st.markdown("#### Cobertura por indicador")
        cov = DF.groupby("indicador").agg(
            registros=("valor", "count"),
            deptos=("departamento", "nunique"),
            anio_min=("anio", "min"),
            anio_max=("anio", "max"),
            fuente=("fuente", "first")
        ).reset_index()
        st.dataframe(cov, use_container_width=True)

        with st.expander("Log de carga"):
            st.code("\n".join(LOG_CARGA[-30:]))
    else:
        st.error("No se pudieron cargar datos. Verifique su conexión a internet.")

    st.markdown("---")
    st.subheader("Exploración rápida")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        depto0 = st.multiselect("Departamento(s)", ["Todos"] + lista_deptos,
                                default=["Antioquia", "Bogotá D.C.", "Valle del Cauca"] if lista_deptos else ["Todos"])
    with c2:
        ind0 = st.selectbox("Indicador", lista_inds_todos, index=1 if lista_inds else 0)
    with c3:
        s_min0 = st.slider("Año inicial", amin, amax, max(amin, amax - 10), key="smin0")
    with c4:
        s_max0 = st.slider("Año final", amin, amax, amax, key="smax0")

    if st.button("Actualizar visualizaciones", key="btn0"):
        st_data = filtrar(DF, depto0, ind0, s_min0, s_max0)
        if st_data.empty:
            st.warning("Sin datos para los filtros seleccionados.")
        else:
            fig_t = px.line(st_data, x="anio", y="valor", color="departamento", markers=True,
                            title=f"Tendencia – {ind0}", template="plotly_white")
            st.plotly_chart(fig_t, use_container_width=True)

            ind_b = ind0 if ind0 != "Todos" else (lista_inds[0] if lista_inds else ind0)
            d_b = DF[(DF["indicador"] == ind_b) & (DF["anio"] == s_max0)].nlargest(12, "valor")
            fig_b = px.bar(d_b, x="valor", y="departamento", orientation="h",
                           title=f"Top 12 – {ind_b} ({s_max0})", template="plotly_white",
                           color="valor", color_continuous_scale="YlOrRd")
            fig_b.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True)
            st.dataframe(st_data.head(200), use_container_width=True)

# -------------------- TAB 2: EDA --------------------
with tab2:
    st.subheader("Análisis Exploratorio de Datos")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        depto1 = st.multiselect("Departamentos", ["Todos"] + lista_deptos, default=["Todos"], key="depto1")
    with c2:
        ind1 = st.selectbox("Indicador", lista_inds_todos, index=1 if lista_inds else 0, key="ind1")
    with c3:
        amin1 = st.slider("Año min", amin, amax, amin, key="amin1")
    with c4:
        amax1 = st.slider("Año max", amin, amax, amax, key="amax1")

    if st.button("Ejecutar EDA descriptivo", key="btn_eda"):
        d = filtrar(DF, depto1, ind1, amin1, amax1)
        if d.empty:
            st.warning("Sin datos.")
        else:
            desc = d.groupby("indicador")["valor"].describe().round(2)
            desc["skew"] = d.groupby("indicador")["valor"].skew().round(3)
            st.markdown("**Estadísticos descriptivos**")
            st.dataframe(desc)

            fig_hist = px.histogram(d, x="valor", color="indicador" if ind1 == "Todos" else None,
                                    marginal="box", nbins=30, title="Distribución de valores",
                                    template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)

            top_deptos = d.groupby("departamento")["valor"].mean().nlargest(12).index
            d_top = d[d["departamento"].isin(top_deptos)]
            fig_box = px.box(d_top, x="departamento", y="valor", color="departamento",
                             title="Boxplot por departamento (Top 12)", template="plotly_white")
            fig_box.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Calcular correlación", key="btn_corr"):
            if PANEL.empty:
                st.warning("Panel vacío.")
            else:
                d = PANEL[(PANEL["anio"] >= amin1) & (PANEL["anio"] <= amax1)].copy()
                cols_num = [c for c in d.columns if c not in ["departamento", "anio"] and d[c].notna().sum() > 10]
                if len(cols_num) < 2:
                    st.warning("Insuficientes indicadores.")
                else:
                    corr = d[cols_num].corr()
                    st.dataframe(corr.round(3))
                    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                    title="Matriz de correlación")
                    st.plotly_chart(fig, use_container_width=True)

    with col_b:
        metodo_out = st.radio("Método outliers", ["IQR", "Z-score"], horizontal=True)
        if st.button("Detectar outliers", key="btn_out"):
            d = filtrar(DF, None, ind1, amin1, amax1)
            if d.empty or ind1 == "Todos":
                st.warning("Seleccione un indicador específico.")
            else:
                vals = d["valor"].dropna()
                if metodo_out == "IQR":
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outliers = d[(d["valor"] < lower) | (d["valor"] > upper)]
                else:
                    z = np.abs(stats.zscore(vals))
                    outliers = d.loc[vals.index[z > 3]]
                st.write(f"**Outliers detectados ({metodo_out}): {len(outliers)}**")
                st.dataframe(outliers.head(20))
                fig = px.scatter(d, x="anio", y="valor", color="departamento",
                                 title=f"Valores y outliers – {ind1}", template="plotly_white")
                if not outliers.empty:
                    fig.add_trace(go.Scatter(x=outliers["anio"], y=outliers["valor"],
                                             mode="markers", marker=dict(symbol="x", size=12, color="red"),
                                             name="Outlier"))
                st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 3: CLUSTERING --------------------
with tab3:
    st.subheader("Clustering de departamentos")
    c1, c2, c3 = st.columns(3)
    with c1:
        anio_cl = st.slider("Año de referencia", amin, amax, amax, key="anio_cl")
    with c2:
        n_cl = st.slider("Nº clusters", 2, 8, 4, key="n_cl")
    with c3:
        met_cl = st.radio("Método", ["KMeans", "Hierarchical", "DBSCAN"], horizontal=True)

    if st.button("Ejecutar clustering", key="btn_cl"):
        if PANEL.empty:
            st.warning("Panel vacío.")
        else:
            d = PANEL[PANEL["anio"] == anio_cl].copy()
            cols = [c for c in d.columns if c not in ["departamento", "anio"] and d[c].notna().sum() > 5]
            if len(cols) < 2 or len(d) < n_cl:
                st.warning("Datos insuficientes.")
            else:
                X = d[cols].fillna(d[cols].median())
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                if met_cl == "KMeans":
                    model = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
                    labels = model.fit_predict(Xs)
                    sil = silhouette_score(Xs, labels) if n_cl > 1 else 0
                elif met_cl == "Hierarchical":
                    model = AgglomerativeClustering(n_clusters=n_cl)
                    labels = model.fit_predict(Xs)
                    sil = silhouette_score(Xs, labels) if n_cl > 1 else 0
                else:
                    model = DBSCAN(eps=1.2, min_samples=2)
                    labels = model.fit_predict(Xs)
                    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
                    sil = silhouette_score(Xs, labels) if n_cl > 1 and -1 not in labels else 0
                d["cluster"] = labels
                st.write(f"**Método: {met_cl} | Clusters: {n_cl} | Silhouette: {sil:.3f}**")
                st.dataframe(d.groupby("cluster")[cols].mean().round(2))

                pca = PCA(n_components=2)
                coords = pca.fit_transform(Xs)
                d["PC1"] = coords[:, 0]
                d["PC2"] = coords[:, 1]
                fig_sc = px.scatter(d, x="PC1", y="PC2", color="cluster", text="departamento",
                                    title=f"Clustering {met_cl} (PCA) – {anio_cl}",
                                    template="plotly_white", color_continuous_scale="Viridis")
                fig_sc.update_traces(textposition="top center", textfont_size=9)
                st.plotly_chart(fig_sc, use_container_width=True)

# -------------------- TAB 4: PCA --------------------
with tab4:
    st.subheader("Análisis de Componentes Principales")
    c1, c2, c3 = st.columns(3)
    with c1:
        amin_pca = st.slider("Año min", amin, amax, amin, key="amin_pca")
    with c2:
        amax_pca = st.slider("Año max", amin, amax, amax, key="amax_pca")
    with c3:
        ncomp = st.slider("Nº componentes", 2, 5, 3, key="ncomp")

    if st.button("Ejecutar PCA", key="btn_pca"):
        if PANEL.empty:
            st.warning("Panel vacío.")
        else:
            d = PANEL[(PANEL["anio"] >= amin_pca) & (PANEL["anio"] <= amax_pca)].copy()
            cols = [c for c in d.columns if c not in ["departamento", "anio"] and d[c].notna().sum() > 10]
            if len(cols) < 2:
                st.warning("Insuficientes variables.")
            else:
                X = d[cols].fillna(d[cols].median())
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                ncomp = min(ncomp, len(cols), Xs.shape[0])
                pca = PCA(n_components=ncomp)
                scores = pca.fit_transform(Xs)
                var_exp = pca.explained_variance_ratio_
                st.write(f"**Varianza explicada acumulada:** {var_exp.cumsum().round(3).tolist()}")
                loadings = pd.DataFrame(pca.components_.T, index=cols,
                                        columns=[f"PC{i+1}" for i in range(ncomp)])
                st.dataframe(loadings.round(3))

                fig_scree = go.Figure()
                fig_scree.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(var_exp))], y=var_exp, name="Var. individual"))
                fig_scree.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(len(var_exp))], y=var_exp.cumsum(),
                                               mode="lines+markers", name="Acumulada", yaxis="y2"))
                fig_scree.update_layout(title="Scree plot", yaxis=dict(title="Proporción"),
                                        yaxis2=dict(title="Acumulada", overlaying="y", side="right"),
                                        template="plotly_white")
                st.plotly_chart(fig_scree, use_container_width=True)

                d_plot = d.copy()
                d_plot["PC1"] = scores[:, 0]
                d_plot["PC2"] = scores[:, 1] if ncomp > 1 else 0
                fig_bi = px.scatter(d_plot, x="PC1", y="PC2",
                                    color="departamento" if d_plot["departamento"].nunique() <= 15 else None,
                                    title="Proyección PC1 vs PC2", template="plotly_white")
                st.plotly_chart(fig_bi, use_container_width=True)

# -------------------- TAB 5: MACHINE LEARNING --------------------
with tab5:
    st.subheader("Modelos de Machine Learning")
    c1, c2 = st.columns(2)
    with c1:
        target_ml = st.selectbox("Variable objetivo (Y)", cols_panel, key="target_ml")
    with c2:
        feats_ml = st.multiselect("Features (X)", cols_panel,
                                  default=cols_panel[1:min(4, len(cols_panel))] if len(cols_panel) > 1 else cols_panel)

    c3, c4, c5, c6 = st.columns(4)
    with c3:
        modelo_ml = st.selectbox("Modelo", ["Ridge", "Lasso", "ElasticNet", "SVR", "RandomForest", "GradientBoosting", "OLS"])
    with c4:
        alpha_ml = st.slider("Alpha", 0.01, 10.0, 1.0, 0.01)
    with c5:
        amin_ml = st.slider("Año min", amin, amax, amin, key="amin_ml")
    with c6:
        amax_ml = st.slider("Año max", amin, amax, amax, key="amax_ml")

    if st.button("Entrenar modelo", key="btn_ml"):
        if PANEL.empty or not target_ml or not feats_ml:
            st.warning("Configure target y features.")
        else:
            d = PANEL[(PANEL["anio"] >= amin_ml) & (PANEL["anio"] <= amax_ml)].copy()
            cols = [target_ml] + feats_ml
            d = d.dropna(subset=cols)
            if len(d) < 20:
                st.warning("Muestra insuficiente (<20 observaciones).")
            else:
                X = d[feats_ml]
                y = d[target_ml]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                models = {
                    "Ridge": Ridge(alpha=alpha_ml),
                    "Lasso": Lasso(alpha=alpha_ml, max_iter=5000),
                    "ElasticNet": ElasticNet(alpha=alpha_ml, l1_ratio=0.5, max_iter=5000),
                    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
                    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6),
                    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=4),
                    "OLS": LinearRegression()
                }
                model = models.get(modelo_ml, Ridge(alpha=alpha_ml))
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"**Modelo: {modelo_ml}**")
                st.write(f"- RMSE: {rmse:,.2f}  |  MAE: {mae:,.2f}  |  R²: {r2:.4f}")
                st.write(f"- N train/test: {len(X_train)}/{len(X_test)}")

                if hasattr(model, "coef_"):
                    coefs = pd.Series(model.coef_, index=feats_ml).sort_values(key=abs, ascending=False)
                    st.dataframe(coefs.round(4).to_frame("Coeficiente"))
                elif hasattr(model, "feature_importances_"):
                    imp = pd.Series(model.feature_importances_, index=feats_ml).sort_values(ascending=False)
                    st.dataframe(imp.round(4).to_frame("Importancia"))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predichos",
                                         marker=dict(size=8, opacity=0.7)))
                lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Ideal",
                                         line=dict(dash="dash", color="red")))
                fig.update_layout(title=f"{modelo_ml}: Real vs Predicho", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 6: VAR / VECM --------------------
with tab6:
    st.subheader("Modelos VAR / VECM")
    c1, c2, c3 = st.columns(3)
    with c1:
        vars_var = st.multiselect("Variables del sistema", cols_panel,
                                  default=cols_panel[:min(3, len(cols_panel))] if cols_panel else [])
    with c2:
        tipo_var = st.radio("Tipo", ["VAR", "VECM"], horizontal=True)
    with c3:
        lags_var = st.slider("Máx. rezagos", 1, 4, 2)

    if st.button("Estimar modelo", key="btn_var"):
        if PANEL.empty or len(vars_var) < 2:
            st.warning("Seleccione al menos 2 variables.")
        else:
            d = PANEL.copy()
            ts = d.groupby("anio")[vars_var].mean().dropna().sort_index()
            if len(ts) < lags_var + 5:
                st.warning(f"Serie demasiado corta ({len(ts)} obs).")
            else:
                def is_stationary(series):
                    try:
                        return adfuller(series.dropna())[1] < 0.05
                    except Exception:
                        return False
                ts_diff = ts.copy()
                for col in vars_var:
                    if not is_stationary(ts[col]):
                        ts_diff[col] = ts[col].diff()
                ts_diff = ts_diff.dropna()
                st.write(f"**Observaciones usadas:** {len(ts_diff)}")

                if tipo_var == "VAR":
                    try:
                        model = VAR(ts_diff)
                        res = model.fit(maxlags=lags_var, ic="aic")
                        st.write(f"**Orden seleccionado (AIC):** {res.k_ar}")
                        st.text(str(res.summary())[:6000])
                        irf = res.irf(10)
                        fig = go.Figure()
                        for i, shock in enumerate(vars_var):
                            for j, resp in enumerate(vars_var):
                                fig.add_trace(go.Scatter(y=irf.irfs[:, j, i],
                                                         name=f"{resp} ← {shock}", line=dict(width=1.5)))
                        fig.update_layout(title="Funciones de Impulso-Respuesta (IRF)",
                                          template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error VAR: {e}")
                else:
                    try:
                        rank_test = select_coint_rank(ts, det_order=0, k_ar_diff=lags_var, method="trace")
                        rank = rank_test.rank
                        st.write(f"**Rango de cointegración (Johansen):** {rank}")
                        if rank == 0:
                            st.warning("No hay cointegración → use VAR en diferencias.")
                        else:
                            model = VECM(ts, k_ar_diff=lags_var, coint_rank=rank, deterministic="ci")
                            res = model.fit()
                            st.text(str(res.summary())[:6000])
                    except Exception as e:
                        st.error(f"Error VECM: {e}")

# -------------------- TAB 7: MAPAS --------------------
with tab7:
    st.subheader("Mapas y análisis espacial")
    c1, c2 = st.columns(2)
    with c1:
        ind_map = st.selectbox("Indicador", lista_inds, key="ind_map")
    with c2:
        anio_map = st.slider("Año", amin, amax, amax, key="anio_map")

    if st.button("Generar mapa + Moran's I", key="btn_map"):
        d = DF[(DF["indicador"] == ind_map) & (DF["anio"] == anio_map)].copy()
        if d.empty:
            st.warning("Sin datos.")
        else:
            d["lon"] = d["departamento"].map(lambda x: CENTROIDES.get(x, (None, None))[0])
            d["lat"] = d["departamento"].map(lambda x: CENTROIDES.get(x, (None, None))[1])
            d = d.dropna(subset=["lon", "lat"])
            fig = px.scatter_geo(d, lat="lat", lon="lon", size="valor", color="valor",
                                 hover_name="departamento",
                                 hover_data={"valor": ":.2f", "lon": False, "lat": False},
                                 color_continuous_scale="YlOrRd", size_max=45,
                                 title=f"{ind_map} – {anio_map}",
                                 projection="natural earth", scope="south america")
            fig.update_geos(center=dict(lon=-74, lat=4.5), lataxis_range=[-5, 13],
                            lonaxis_range=[-82, -66], showland=True,
                            landcolor="rgb(245,245,245)", showcountries=True, countrycolor="gray")
            fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=520)
            st.plotly_chart(fig, use_container_width=True)

            try:
                from scipy.spatial.distance import pdist, squareform
                coords = d[["lon", "lat"]].values
                dist = squareform(pdist(coords))
                w = 1 / (dist + 1e-6)
                np.fill_diagonal(w, 0)
                w = w / w.sum(axis=1, keepdims=True)
                z = (d["valor"] - d["valor"].mean()) / d["valor"].std()
                I = (len(z) / w.sum()) * (z @ w @ z) / (z @ z)
                st.write(f"**Moran's I (global):** {I:.4f}")
                st.caption("Valores > 0 → clustering espacial positivo.")
            except Exception as e:
                st.write(f"No se pudo calcular Moran's I: {e}")

# -------------------- TAB 8: LSTM --------------------
with tab8:
    st.subheader("Deep Learning – LSTM Forecast")
    if not HAS_TF:
        st.warning("TensorFlow no está disponible. Instale tensorflow para usar esta pestaña.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            depto_dl = st.selectbox("Departamento", lista_deptos,
                                    index=lista_deptos.index("Antioquia") if "Antioquia" in lista_deptos else 0)
        with c2:
            ind_dl = st.selectbox("Indicador", lista_inds, key="ind_dl")
        with c3:
            epochs_dl = st.slider("Epochs", 10, 60, 25, 5)
        with c4:
            look_dl = st.slider("Lookback", 2, 5, 3)

        if st.button("Entrenar LSTM + Forecast", key="btn_dl"):
            serie = get_series(DF, depto_dl, ind_dl)
            if len(serie) < look_dl + 8:
                st.warning(f"Serie demasiado corta ({len(serie)} obs).")
            else:
                values = serie["valor"].values.astype(float)
                scaler = MinMaxScaler()
                scaled = scaler.fit_transform(values.reshape(-1, 1)).flatten()
                X, y = [], []
                for i in range(look_dl, len(scaled)):
                    X.append(scaled[i - look_dl:i])
                    y.append(scaled[i])
                X, y = np.array(X), np.array(y)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                split = int(len(X) * 0.75)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                model = Sequential([
                    Input(shape=(look_dl, 1)),
                    LSTM(32, return_sequences=True),
                    Dropout(0.2),
                    LSTM(16),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer="adam", loss="mse")
                es = EarlyStopping(patience=5, restore_best_weights=True)
                with st.spinner("Entrenando LSTM..."):
                    model.fit(X_train, y_train, epochs=epochs_dl, batch_size=4,
                              validation_split=0.15, callbacks=[es], verbose=0)

                pred_scaled = model.predict(X_test, verbose=0).flatten()
                pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                rmse = np.sqrt(mean_squared_error(real, pred))

                last = scaled[-look_dl:].tolist()
                future = []
                for _ in range(5):
                    x_in = np.array(last[-look_dl:]).reshape(1, look_dl, 1)
                    p = model.predict(x_in, verbose=0)[0, 0]
                    future.append(p)
                    last.append(p)
                future = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
                anios_fut = list(range(int(serie["anio"].max()) + 1, int(serie["anio"].max()) + 6))

                st.write(f"**LSTM – {depto_dl} | {ind_dl}**")
                st.write(f"RMSE test: {rmse:,.2f}")
                st.write(f"Predicción 5 años: {dict(zip(anios_fut, future.round(1)))}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=serie["anio"], y=serie["valor"], name="Histórico", mode="lines+markers"))
                anios_test = serie["anio"].values[-len(real):]
                fig.add_trace(go.Scatter(x=anios_test, y=pred, name="Predicción test", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=anios_fut, y=future, name="Forecast futuro",
                                         mode="lines+markers", line=dict(dash="dash")))
                fig.update_layout(title=f"LSTM Forecast – {depto_dl}", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

# -------------------- TAB 9: SIMULACION --------------------
with tab9:
    st.subheader("Simulación de Política")
    c1, c2 = st.columns(2)
    with c1:
        target_sim = st.selectbox("Target", cols_panel, key="target_sim")
    with c2:
        feats_sim = st.multiselect("Features", cols_panel,
                                   default=cols_panel[1:min(4, len(cols_panel))] if len(cols_panel) > 1 else cols_panel,
                                   key="feats_sim")

    c3, c4, c5, c6, c7 = st.columns(5)
    with c3:
        shock_var = st.selectbox("Variable a shokear", cols_panel,
                                 index=1 if len(cols_panel) > 1 else 0, key="shock_var")
    with c4:
        shock_pct = st.slider("Shock (%)", -40, 40, 10, 5)
    with c5:
        horizon = st.slider("Horizonte (años)", 3, 8, 5)
    with c6:
        amin_sim = st.slider("Año min", amin, amax, amin, key="amin_sim")
    with c7:
        amax_sim = st.slider("Año max", amin, amax, amax, key="amax_sim")

    if st.button("Simular política", key="btn_sim"):
        if PANEL.empty or not target_sim or not feats_sim:
            st.warning("Configure target y features.")
        else:
            d = PANEL[(PANEL["anio"] >= amin_sim) & (PANEL["anio"] <= amax_sim)].dropna(
                subset=[target_sim] + feats_sim)
            if len(d) < 25:
                st.warning("Muestra insuficiente.")
            else:
                X = d[feats_sim]
                y = d[target_sim]
                model = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)
                model.fit(X, y)
                last = d.sort_values("anio").groupby("anio")[feats_sim + [target_sim]].mean().iloc[-1]
                baseline = []
                current = last[feats_sim].copy()
                for h in range(horizon):
                    pred = model.predict(current.values.reshape(1, -1))[0]
                    baseline.append(pred)
                    current = current * 1.01
                shocked = last[feats_sim].copy()
                if shock_var in shocked.index:
                    shocked[shock_var] *= (1 + shock_pct / 100)
                scenario = []
                current = shocked.copy()
                for h in range(horizon):
                    pred = model.predict(current.values.reshape(1, -1))[0]
                    scenario.append(pred)
                    current = current * 1.01
                anios = list(range(int(d["anio"].max()) + 1, int(d["anio"].max()) + horizon + 1))
                res_df = pd.DataFrame({
                    "Año": anios,
                    "Baseline": baseline,
                    "Con shock": scenario,
                    "Diferencia": np.array(scenario) - np.array(baseline)
                }).round(2)
                st.write(f"**Shock: {shock_var} {shock_pct:+.1f}% | Target: {target_sim}**")
                st.dataframe(res_df)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=anios, y=baseline, name="Baseline", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=anios, y=scenario, name=f"Shock {shock_var} {shock_pct:+.0f}%",
                                         mode="lines+markers", line=dict(dash="dash")))
                fig.update_layout(title="Simulación de política", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

                imp = pd.Series(model.feature_importances_, index=feats_sim).sort_values(ascending=True)
                fig_imp = px.bar(x=imp.values, y=imp.index, orientation="h",
                                 title="Importancia de variables (GBM)", template="plotly_white")
                st.plotly_chart(fig_imp, use_container_width=True)

# -------------------- TAB 10: INTERPRETABILIDAD --------------------
with tab10:
    st.subheader("Interpretabilidad de modelos")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        target_int = st.selectbox("Target", cols_panel, key="target_int")
    with c2:
        feats_int = st.multiselect("Features", cols_panel,
                                   default=cols_panel[1:min(5, len(cols_panel))] if len(cols_panel) > 1 else cols_panel,
                                   key="feats_int")
    with c3:
        amin_int = st.slider("Año min", amin, amax, amin, key="amin_int")
    with c4:
        amax_int = st.slider("Año max", amin, amax, amax, key="amax_int")

    if st.button("Calcular importancia", key="btn_int"):
        if PANEL.empty or not target_int or not feats_int:
            st.warning("Configure variables.")
        else:
            d = PANEL[(PANEL["anio"] >= amin_int) & (PANEL["anio"] <= amax_int)].dropna(
                subset=[target_int] + feats_int)
            if len(d) < 20:
                st.warning("Muestra insuficiente.")
            else:
                X = d[feats_int]
                y = d[target_int]
                model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                model.fit(X, y)
                perm = permutation_importance(model, X, y, n_repeats=15, random_state=42)
                imp_df = pd.DataFrame({
                    "feature": feats_int,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std
                }).sort_values("importance_mean", ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=imp_df["importance_mean"], y=imp_df["feature"], orientation="h",
                    error_x=dict(type="data", array=imp_df["importance_std"]),
                    marker_color="#1f77b4"
                ))
                fig.update_layout(title="Permutation Importance", template="plotly_white", height=420)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(imp_df.round(4))

                if HAS_SHAP:
                    try:
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X)
                        mean_abs = np.abs(shap_values).mean(axis=0)
                        st.write("**SHAP mean |valor|:**")
                        st.dataframe(pd.Series(mean_abs, index=feats_int).sort_values(ascending=False).round(4))
                    except Exception:
                        pass

st.markdown("---")
st.caption("Fuentes: datos.gov.co (PIB, Homicidios, Lesiones, Hurtos, DNP) · Streamlit + Plotly · Datos reales")
