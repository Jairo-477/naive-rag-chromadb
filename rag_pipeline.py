import os
import time
import argparse
from datetime import date

import chromadb
import pypdf
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
cliente_genai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configuración
#----------------------------------------------------------

MODELO_EMBEDDING = "gemini-embedding-001"
MODELO_GENERACION = "gemini-2.5-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
DIRECTORIO_CHROMA = "./chroma_db"
NOMBRE_COLECCION = "documentos_rag"


# Paso 1: Cargar PDF + mapa de páginas
#----------------------------------------------------------

#Este método extrae el texto de un PDF y genera un mapa de páginas con offsets de inicio/fin.
def cargar_pdf_con_mapa(ruta_pdf: str) -> tuple[str, list[dict]]:
  texto_completo = []
  mapa_paginas = []
  cursor = 0

  with open(ruta_pdf, "rb") as f:
    lector = pypdf.PdfReader(f)
    print(f"📄 PDF: {len(lector.pages)} páginas")
    for i, pagina in enumerate(lector.pages):
      texto = pagina.extract_text()
      if texto:
        bloque = f"[Página {i + 1}]\n{texto}\n\n"
        inicio = cursor
        cursor += len(bloque)
        mapa_paginas.append({"pagina": i + 1, "inicio": inicio, "fin": cursor})
        texto_completo.append(bloque)

  return "".join(texto_completo), mapa_paginas

def pagina_para_offset(offset: int, mapa_paginas: list[dict]) -> int:
  for entrada in mapa_paginas:
    if entrada["inicio"] <= offset < entrada["fin"]:
      return entrada["pagina"]
  return mapa_paginas[-1]["pagina"] if mapa_paginas else 1


# Paso 2: Chunking
#----------------------------------------------------------
def hacer_chunks(texto: str) -> list[dict]:
  chunks = []
  inicio = 0
  idx = 0
  while inicio < len(texto):
    fin = inicio + CHUNK_SIZE
    fragmento = texto[inicio:fin].strip()
    if len(fragmento) > 50:
      chunks.append({
        "indice": idx,
        "texto": fragmento,
        "inicio_char": inicio,
        "fin_char": fin,
      })
      idx += 1
    inicio += CHUNK_SIZE - CHUNK_OVERLAP
  print(f"-Chunks generados: {len(chunks)}")
  return chunks


#Paso 3: Embeddings
#----------------------------------------------------------

#Genera el embedding del documento que se ingresa al RAG
def embedding_documento(texto: str) -> list[float]:
  for intento in range(5):
    try:
      r = cliente_genai.models.embed_content(
        model=MODELO_EMBEDDING,
        contents=texto,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
      )
      return r.embeddings[0].values
    except Exception as e:
      if intento == 4:
        raise
      espera = 2 ** intento
      print(f"   ⚠️  Error (intento {intento + 1}/5): {e}. Reintentando en {espera}s...")
      time.sleep(espera)

#Genera el embedding de la pregunta que hace el usuario al RAG
def embedding_query(texto: str) -> list[float]:
  r = cliente_genai.models.embed_content(
    model=MODELO_EMBEDDING,
    contents=texto,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
  )
  return r.embeddings[0].values


#Paso 4: ChromaDB
#----------------------------------------------------------

#Crea o recupera la colección de ChromaDB 
def obtener_coleccion():
  cliente = chromadb.PersistentClient(path=DIRECTORIO_CHROMA)
  return cliente.get_or_create_collection(name=NOMBRE_COLECCION)

#Verifica si un documento ya fue indexado para evitar llamadas innecesarias a la API de embeddings
def documento_ya_indexado(coleccion, nombre_archivo: str) -> bool:
  existentes = coleccion.get(where={"archivo": nombre_archivo}, limit=1)
  return len(existentes["ids"]) > 0


#Paso 5: Indexación
#----------------------------------------------------------

def indexar_pdf(ruta_pdf: str, forzar_reindexado: bool = False):
  coleccion = obtener_coleccion()
  nombre_archivo = os.path.basename(ruta_pdf)

  #En caso de que ya se haya indexado el documento, se reutilizan los chunks existentes sin volver a generar embeddings (a menos que se indique lo contrario con el comando --reindexar)
  if not forzar_reindexado and documento_ya_indexado(coleccion, nombre_archivo):
    print(f"\n✓ '{nombre_archivo}' ya estaba indexado en ChromaDB. Se reusa (sin gastar embeddings).")
    print(f"  Total de chunks en la colección: {coleccion.count()}")
    return coleccion

  #Se fuerza el reindexado si se indica con el comando --reindexar, eliminando los chunks anteriores.
  if forzar_reindexado:
    coleccion.delete(where={"archivo": nombre_archivo})
    print(f"🗑️  Chunks anteriores de '{nombre_archivo}' eliminados, reindexando...")

  print(f"\n=== INDEXANDO '{nombre_archivo}' EN CHROMADB ===")
  #Construye el mapa de páginas donde se define el numero de página, el offset de inicio y el offset de fin de cada página
  texto, mapa_paginas = cargar_pdf_con_mapa(ruta_pdf)
  #Se divide el texto en chunks de tamaño CHUNK_SIZE con un solapamiento de CHUNK_OVERLAP
  chunks = hacer_chunks(texto)

  ids, documentos, embeddings, metadatas = [], [], [], []
  print(f"🔢 Generando embeddings para {len(chunks)} chunks...")
  for i, chunk in enumerate(chunks):
    pagina_inicio = pagina_para_offset(chunk["inicio_char"], mapa_paginas)
    pagina_fin = pagina_para_offset(chunk["fin_char"], mapa_paginas)

    ids.append(f"{nombre_archivo}_chunk_{chunk['indice']}")
    documentos.append(chunk["texto"])
    embeddings.append(embedding_documento(chunk["texto"]))
    metadatas.append({
      "archivo": nombre_archivo,
      "pagina_inicio": pagina_inicio,
      "pagina_fin": pagina_fin,
      "fecha_procesamiento": str(date.today()),
    })

    if (i + 1) % 10 == 0:
      print(f"   {i + 1}/{len(chunks)}")
    time.sleep(0.4)  # rate limit tier gratuito

  coleccion.add(ids=ids, documents=documentos, embeddings=embeddings, metadatas=metadatas)
  print(f"✓ Indexación completa: {len(ids)} chunks añadidos a ChromaDB ({DIRECTORIO_CHROMA})")
  print("=== FIN INDEXACIÓN ===\n")
  return coleccion


#Paso 6: Retrieval
#----------------------------------------------------------

#Este método recupera los chunks más relevantes para la pregunta del usuario utilizando el embedding de la pregunta y la función de consulta de ChromaDB
def recuperar_chunks(pregunta: str, coleccion, top_k: int = TOP_K) -> list[dict]:
  q_emb = embedding_query(pregunta)
  resultado = coleccion.query(
    query_embeddings=[q_emb],
    n_results=top_k,
  )

  chunks = []
  for doc, meta, dist in zip(
    resultado["documents"][0], resultado["metadatas"][0], resultado["distances"][0]
  ):
    chunks.append({
      "texto": doc,
      "metadata": meta,
      # Chroma devuelve distancia coseno (0 = idéntico). La convertimos
      # a similitud para que se lea igual que en la Fase 01.
      "similitud": 1 - dist,
    })
  return chunks


#Paso 7: Augmented prompt
#----------------------------------------------------------

#Este método construye el prompt enriquecido 
def construir_prompt(pregunta: str, chunks: list[dict]) -> str:
  contexto = "\n\n".join(
    f"[Fragmento {i + 1} — pág. {c['metadata']['pagina_inicio']}-{c['metadata']['pagina_fin']} "
    f"— similitud {c['similitud']:.3f}]:\n{c['texto']}"
    for i, c in enumerate(chunks)
  )
  return f"""Eres un asistente que responde basándose ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{contexto}

PREGUNTA: {pregunta}

Responde usando solo el contexto. Si no hay información suficiente, dilo claramente.
Indica qué fragmento y página usaste.

RESPUESTA:"""


#Paso 8: Generación
#----------------------------------------------------------

#Invoca al LLM para generar la respuesta a partir del prompt enriquecido
def generar_respuesta(prompt: str) -> str:
  return cliente_genai.models.generate_content(model=MODELO_GENERACION, contents=prompt).text


#Consulta completa
#----------------------------------------------------------

#Permite interactuar con el RAG: recibe la pregunta, recupera los chunks relevantes, construye el prompt enriquecido y genera la respuesta.
def consultar(pregunta: str, coleccion) -> str:
  print(f"\n🔍 Pregunta: {pregunta}")

  chunks_relevantes = recuperar_chunks(pregunta, coleccion)

  if not chunks_relevantes:
    print("⚠️  No se encontraron chunks relevantes.")
    return ""

  print("📎 Chunks recuperados:")
  for i, c in enumerate(chunks_relevantes):
    m = c["metadata"]
    print(f"   [{i + 1}] pág {m['pagina_inicio']}-{m['pagina_fin']} | "
          f"sim={c['similitud']:.4f} | {c['texto'][:80]}...")

  prompt = construir_prompt(pregunta, chunks_relevantes)
  respuesta = generar_respuesta(prompt)
  print(f"\n💬 Respuesta:\n{respuesta}")
  return respuesta


#Main
#----------------------------------------------------------

#Método de ejecución principal
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="RAG con ChromaDB — Fase 02")
  parser.add_argument("pdf", help="Ruta al PDF a indexar/consultar")
  parser.add_argument("--reindexar", action="store_true", help="Fuerza reindexación aunque ya exista")
  args = parser.parse_args()

  coleccion = indexar_pdf(args.pdf, forzar_reindexado=args.reindexar)

  print("\n¿Qué quieres saber sobre el documento? (escribe 'salir' para terminar)\n")

  while True:
    entrada = input("> ").strip()

    if entrada.lower() in ("salir", "exit", "q"):
      break
    if not entrada:
      continue

    consultar(entrada, coleccion)
