# Observatorio Economico y Social por Departamentos (Streamlit)

Aplicacion web de analisis de datos departamentales de Colombia.

---

## Archivos necesarios

Debes tener estos 3 archivos en la misma carpeta:

- app.py
- requirements.txt
- README.md

---

## Como ejecutar en Windows (CMD)

1. Abre la carpeta donde estan los archivos.
2. En la barra de direcciones escribe: cmd y presiona Enter.
3. Ejecuta estos comandos uno por uno:

```cmd
py -m pip install --upgrade pip
```

```cmd
py -m pip install -r requirements.txt
```

```cmd
py -m streamlit run app.py
```

4. Se abrira automaticamente en el navegador (http://localhost:8501).

---

## Si aparece error "No module named streamlit"

Significa que no se instalaron las librerias. Vuelve a ejecutar:

```cmd
py -m pip install -r requirements.txt
```

---

## Detener la aplicacion

En la ventana de CMD presiona:

```
Ctrl + C
```

---

## Si usas Streamlit Cloud

1. Sube solo estos archivos:
   - app.py
   - requirements.txt

2. No subas el archivo viejo de Gradio.

3. Haz Redeploy.

---

## Notas

- La primera carga puede tardar varios minutos (descarga datos de datos.gov.co).
- TensorFlow y SHAP son opcionales. Si fallan al instalar, la app sigue funcionando.
