# 🇨🇴 Observatorio de Actividad Económica y Social por Departamentos

Suite metodológica completa de análisis exploratorio, minería de datos, machine learning, econometría de series de tiempo y deep learning aplicada a indicadores departamentales de Colombia.

**Datos reales** provenientes de [datos.gov.co](https://www.datos.gov.co).

---

## 📋 Descripción

Esta aplicación web (desarrollada con **Gradio**) permite:

- Cargar y rastrear automáticamente indicadores oficiales
- Análisis exploratorio de datos (EDA)
- Detección de outliers
- Matrices de correlación
- Clustering (KMeans, Hierarchical, DBSCAN)
- Análisis de Componentes Principales (PCA)
- Modelos de Machine Learning (Ridge, Lasso, ElasticNet, SVR, Random Forest, Gradient Boosting, OLS)
- Modelos VAR y VECM
- Mapas coropléticos + Moran's I
- Pronósticos con LSTM (Deep Learning)
- Simulación de políticas (shocks)
- Interpretabilidad (Permutation Importance + SHAP)

---

## 📦 Instalación

### 1. Clonar o copiar los archivos

Asegúrate de tener en la misma carpeta:

- `observatorio_economico_y_social.py`
- `requirements.txt`
- `README.md`

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota:** TensorFlow y SHAP son opcionales. La aplicación funciona sin ellos (desactiva automáticamente las pestañas de Deep Learning y parte de Interpretabilidad).

---

## 🚀 Ejecución

```bash
python observatorio_economico_y_social.py
```

La aplicación se lanzará automáticamente y generará un enlace público (`share=True`).

También puedes ejecutarla en Google Colab (el código original proviene de un notebook de Colab).

---

## 📊 Fuentes de datos utilizadas

| Indicador                        | Dataset (datos.gov.co) | Categoría          |
|----------------------------------|------------------------|--------------------|
| PIB departamental (constantes)   | kgyi-qc7j             | Economía           |
| Homicidios                       | m8fd-ahd9             | Seguridad          |
| Lesiones personales              | 72sg-cybi             | Seguridad          |
| Hurto a personas                 | 4rxi-8m8d             | Seguridad          |
| Recursos públicos DNP            | wc86-9j4a             | Finanzas públicas  |

Los datos se cargan en tiempo real al iniciar la aplicación.

---

## 🗂️ Estructura de pestañas

1. **🏠 Dashboard & Rastreo** — Estado de carga + exploración rápida
2. **🔍 Análisis Exploratorio** — Descriptivos, correlación y outliers
3. **⛏️ Clustering** — Agrupamiento de departamentos
4. **📉 PCA** — Reducción de dimensionalidad
5. **🤖 Machine Learning** — Modelos de regresión supervisada
6. **📈 VAR / VECM** — Modelos de series de tiempo multivariados
7. **🗺️ Mapas & Espacial** — Visualización geográfica + Moran's I
8. **🧠 Deep Learning (LSTM)** — Pronósticos con redes neuronales
9. **🎯 Simulación de Política** — Análisis contrafactual con shocks
10. **🔎 Interpretabilidad** — Importancia de variables (Permutation + SHAP)

---

## ⚙️ Requisitos del sistema

- Python 3.9 o superior (recomendado 3.10–3.11)
- Conexión a internet (para descargar datos de datos.gov.co)
- Al menos 4 GB de RAM (8 GB recomendados si se usa TensorFlow)

---

## 📝 Notas metodológicas

- Normalización robusta de nombres de departamentos (manejo de tildes y variantes).
- Solo se utilizan precios constantes para el PIB.
- Los modelos de series de tiempo realizan pruebas de estacionariedad (ADF) y diferencia automáticamente si es necesario.
- El clustering y PCA usan estandarización (StandardScaler).
- La simulación de política se basa en Gradient Boosting.

---

## 🛠️ Personalización

Puedes agregar fácilmente nuevas fuentes de datos modificando la función `construir_maestro()` y creando nuevas funciones `cargar_*()`.

---

## 📄 Licencia

Proyecto académico / de investigación. Uso libre con atribución.

---

**Desarrollado para Seminario de Investigación**  
Datos: datos.gov.co · Visualización: Plotly + Gradio
