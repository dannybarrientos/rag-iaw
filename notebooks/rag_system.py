# Contenido del archivo: rag_system_current.py

"""
Sistema RAG - Configuración Actual

Este script implementa un sistema RAG (Retrieval Augmented Generation)
utilizando un modelo LLM pequeño (Flan-T5-Small) y FAISS para la
recuperación de información de un documento local.
"""

# --- 0. Pasos Previos y Configuración del Entorno ---
# Para ejecutar este script, se recomienda usar un entorno Conda.
# 1. Crear y activar un entorno Conda (si no tienes uno):
#    conda create -n rag-env python=3.9 -y
#    conda activate rag-env
# 2. Instalar las dependencias:
#    pip install -U langchain==0.0.354 langchain-community transformers sentence-transformers faiss-cpu torch
# 3. Asegúrate de que 'bre-b.txt' esté en la ruta correcta:
#    Si el script está en 'tu_proyecto/scripts/', 'bre-b.txt' debe estar en 'tu_proyecto/data/raw/'.
#    La ruta relativa utilizada es '../data/raw/bre-b.txt'.

# --- 1. Importar Librerías ---
import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

print("Librerías importadas correctamente.")

# --- 2. Carga del Documento ---
print("\n--- 2. Carga del Documento ---")
file_path = "../data/raw/bre-b.txt"

try:
    loader = TextLoader(file_path, encoding='latin-1')
    documents = loader.load()
    print(f"Documento '{file_path}' cargado exitosamente. Número de páginas/documentos: {len(documents)}")
except FileNotFoundError:
    print(f"Error: El archivo '{file_path}' no se encontró. Asegúrate de que esté en la ubicación correcta.")
    exit() # Salir si el archivo no se encuentra.
except UnicodeDecodeError as e:
    print(f"Error de codificación al leer el archivo: {e}")
    print("Intenta cambiar la codificación (por ejemplo, a 'latin-1' o 'cp1252').")
    exit()
except Exception as e:
    print(f"Ocurrió un error inesperado durante la carga del documento: {e}")
    exit()

# --- 3. Segmentación del Texto (Chunking) ---
print("\n--- 3. Segmentación del Texto (Chunking) ---")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
    add_start_index=True
)
chunks = text_splitter.split_documents(documents)
print(f"Documento segmentado en {len(chunks)} chunks.")

# --- 4. Generación de Embeddings y Creación del Vector Store ---
print("\n--- 4. Generación de Embeddings y Creación del Vector Store ---")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)
print("Embeddings generados y FAISS vector store creado exitosamente.")

# --- 5. Configuración del Large Language Model (LLM) ---
print("\n--- 5. Configuración del Large Language Model (LLM) ---")
model_id = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)

device = 0 if torch.cuda.is_available() else -1
print(f"Usando dispositivo: {'GPU' if device == 0 else 'CPU'}")

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    device=device
)

llm = HuggingFacePipeline(pipeline=pipe)
print("LLM configurado exitosamente.")

# --- 6. Sección para Realizar Preguntas al Documento ---
print("\n--- 6. Sistema RAG listo para recibir preguntas! ---")
print("Escribe 'salir' en cualquier momento para terminar.\n")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    return_source_documents=True
)

while True:
    query = input("Tu pregunta: ")
    if query.lower() == 'salir':
        print("Saliendo del sistema RAG. ¡Adiós!")
        break

    print("Buscando y generando respuesta...")
    try:
        result = qa_chain({"query": query})

        print("\n**Respuesta:**")
        print(result["result"])

        print("\n**--- Fuentes Utilizadas del Documento ---**")
        if result["source_documents"]:
            for i, doc in enumerate(result["source_documents"]):
                print(f"Chunk {i+1}:")
                print(f"  Contenido inicial: \"{doc.page_content[:200]}...\"")
                print("-" * 30)
        else:
            print("No se encontraron fuentes específicas para esta respuesta.")
        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        print(f"Ocurrió un error al procesar la pregunta: {e}")
        print("Por favor, intenta de nuevo.")