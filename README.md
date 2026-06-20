# RAG Naive con ChromaDB

Pipeline de RAG (Retrieval-Augmented Generation) básico en Python que indexa un PDF en una base vectorial local con [ChromaDB] y responde preguntas sobre su contenido usando los modelos de embeddings y generación de Gemini.

## ¿Cómo funciona?

1. **Carga del PDF** — extrae el texto de cada página y construye un mapa de offsets para poder rastrear de qué página viene cada fragmento.
2. **Chunking** — divide el texto en fragmentos de tamaño fijo con solapamiento, para preservar contexto entre cortes.
3. **Embeddings** — genera el vector de cada chunk con el modelo `gemini-embedding-001`.
4. **Indexación** — guarda los chunks (texto + embedding + metadata de página) en una colección persistente de ChromaDB. Si el archivo ya fue indexado, se reutiliza sin volver a gastar llamadas a la API (salvo que se use el comando `--reindexar` en la consola).
5. **Retrieval** — ante una pregunta, genera su embedding y recupera los `TOP_K` chunks más similares.
6. **Augmented prompt** — construye un prompt que incluye los fragmentos recuperados (con página y similitud) como contexto.
7. **Generación** — el modelo `gemini-2.5-flash` responde la pregunta basándose únicamente en ese contexto.

## Requisitos

- Python 3.10+
- Una API key de Google AI Studio (Gemini)

## Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Crea un archivo `.env` en la raíz del proyecto con tu API key:

```
GEMINI_API_KEY=tu_api_key_aqui
```

## Uso

Indexar un PDF y entrar al modo de preguntas interactivo:

```bash
python rag_pipeline.py ruta/al/documento.pdf
```

Forzar reindexación (por ejemplo, si el PDF cambió):

```bash
python rag_pipeline.py ruta/al/documento.pdf --reindexar
```

Una vez indexado el documento, escribe tus preguntas en el prompt `>`. Para salir, escribe `salir`, `exit` o `q`.

## Configuración

Los parámetros principales están al inicio de [rag_pipeline.py](rag_pipeline.py):

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `MODELO_EMBEDDING` | Modelo usado para generar embeddings | `gemini-embedding-001` |
| `MODELO_GENERACION` | Modelo usado para generar respuestas | `gemini-2.5-flash` |
| `CHUNK_SIZE` | Tamaño de cada chunk (caracteres) | `500` |
| `CHUNK_OVERLAP` | Solapamiento entre chunks (caracteres) | `100` |
| `TOP_K` | Cantidad de chunks recuperados por consulta | `5` |
| `DIRECTORIO_CHROMA` | Carpeta donde persiste la base vectorial | `./chroma_db` |
| `NOMBRE_COLECCION` | Nombre de la colección en ChromaDB | `documentos_rag` |

## Estructura del proyecto

```
rag_naive_chromadb/
├── rag_pipeline.py     # Pipeline completo: carga, chunking, embeddings, indexación, retrieval y generación
├── requirements.txt    # Dependencias del proyecto
├── chroma_db/          # Base vectorial persistente (generada automáticamente)
└── .env                # Variables de entorno (no versionado)
```
