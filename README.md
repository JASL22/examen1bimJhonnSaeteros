# Recuperación de Información — NFCorpus

Jhonn Saeteros · ICCD753 · Escuela Politécnica Nacional

## Ejecución local

Python 3.13. Desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe retrieval.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Abrir `examen1bim_Jhonn_Saeteros.ipynb`, seleccionar `.venv` como kernel y ejecutar todas las celdas.

## API

- Disponibilidad: `GET /health`
- Búsqueda: `POST /search`
- Documentación local: `http://127.0.0.1:8000/docs`

```json
{"query":"vitamin B12 deficiency","model":"bm25","top_k":10}
```

Modelos disponibles: `bm25` y `tfidf`.

## Render

Crear un **Web Service** desde el repositorio, con **Python 3** y plan **Free**.

| Campo | Valor |
|---|---|
| Build Command | `pip install -r requirements.txt && python retrieval.py` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |
| Variable de entorno | `PYTHON_VERSION=3.13.14` |

La misma configuración está en `render.yaml`. Incluir `nfcorpus/` en el repositorio.
Después de publicar, completar la URL base en la sección 12 del notebook y comprobar
`GET /health` y `POST /search` en el servicio publicado antes de entregar.
El plan gratuito puede requerir una espera de arranque después de un período sin actividad.

## Datos

[NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) ·
[Qrels](https://huggingface.co/datasets/BeIR/nfcorpus-qrels) · CC BY-SA 4.0.
