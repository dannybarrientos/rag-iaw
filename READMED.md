# Sistema RAG para Documento Bre-B

## Descripción del Proyecto

Este proyecto implementa un sistema de **Generación Aumentada por Recuperación (RAG - Retrieval-Augmented Generation)** diseñado para responder preguntas sobre el documento `bre-b.txt`. El objetivo principal es construir un sistema de Preguntas y Respuestas (QA) que no solo genere respuestas coherentes, sino que también **cite las fuentes específicas del documento original** que fueron utilizadas para formular dichas respuestas, garantizando trazabilidad y reduciendo "alucinaciones".

El sistema está construido en un entorno de Jupyter Notebook, lo que facilita la exploración paso a paso de cada componente del pipeline RAG. Esta configuración es la **versión base**, optimizada para ser ligera y compatible con la mayoría de los sistemas, incluyendo aquellos que solo disponen de CPU.

---

## Estructura del Proyecto

Para una mejor organización, escalabilidad y reproducibilidad, el proyecto sigue una estructura de directorios inspirada en las mejores prácticas:

```

.
├── data/
│   └── raw/
│       └── bre-b.txt             \# Documento original para el sistema RAG
├── notebooks/
│   └── rag\_system\_current.ipynb  \# Jupyter Notebook principal con la implementación del RAG
├── envs/
│   └── environment\_current.yml   \# Archivo de entorno Conda para la configuración actual
├── src/
│   ├── **init**.py               \# Archivo de inicialización del módulo Python
│   └── utils.py                  \# (Opcional) Funciones de utilidad si el proyecto crece
├── .gitignore                    \# Archivo para ignorar archivos y directorios en Git
├── README.md                     \# Este archivo de documentación del proyecto
└── requirements\_current.txt      \# Listado de todas las dependencias del proyecto

````

* **`data/raw/`**: Contiene el documento `bre-b.txt`, que es la fuente de conocimiento para el sistema RAG.
* **`notebooks/`**: Alberga el Jupyter Notebook principal para esta configuración.
* **`envs/`**: Directorio dedicado a los archivos de configuración de entornos (`.yml` para Conda).
* **`src/`**: Para código Python modularizado y reutilizable.
* **`.gitignore`**: Gestiona los archivos y directorios a excluir del control de versiones (ej. entornos virtuales, cachés, modelos descargados).
* **`README.md`**: Este archivo, que ofrece una visión general y guía del proyecto.
* **`requirements_current.txt`**: Listado de dependencias Pip específicas para esta configuración.

---

## Stack Tecnológico

El proyecto utiliza un robusto conjunto de herramientas para la construcción del sistema RAG:

* **Python:** Lenguaje de programación principal (se recomienda Python 3.9 o superior).
* **Jupyter Notebook:** Entorno interactivo para desarrollo, experimentación y demostración.
* **LangChain (v0.0.354):** Framework esencial para la orquestación de componentes RAG (cargadores, segmentadores, embeddings, vector stores y LLMs).
* **Hugging Face Transformers (v4.30.2):** Librería fundamental para la interacción con modelos de lenguaje pre-entrenados.
    * **Hugging Face Embeddings:** Para generar representaciones vectoriales (embeddings) de los fragmentos de texto.
    * **Hugging Face Pipeline:** Para una integración y ejecución simplificada del Large Language Model (LLM).
* **FAISS (Facebook AI Similarity Search - faiss-cpu v1.7.4):** Librería para la búsqueda eficiente de similitud en espacios vectoriales, utilizada como el *vector store* local.
* **PyTorch (v2.0.1):** Framework de aprendizaje profundo, subyacente a la mayoría de los modelos de Hugging Face utilizados.
* **Conda (Opcional):** Sistema de gestión de paquetes y entornos, utilizado para crear y gestionar entornos Python aislados y reproducibles.

---

## Estrategias Implementadas

El sistema RAG se construye siguiendo una serie de estrategias clave, detallando el **qué**, el **cómo** y el **para qué** de cada paso, teniendo presentes los entregables y la integración con Hugging Face.

1.  **Carga del Documento:**
    * **Qué se hace:** Se lee el contenido del archivo `bre-b.txt`, que contiene información sobre el sistema de pagos inmediatos en Colombia.
    * **Cómo se hace:** Se utiliza `TextLoader` de la librería LangChain, especificando la codificación `'latin-1'` para manejar caracteres especiales.
    * **Para qué sirve:** Convierte el contenido del archivo en un objeto `Document` de LangChain, el formato estándar para el procesamiento posterior, y asegura que el texto esté disponible para los siguientes pasos del pipeline RAG.

2.  **Segmentación del Texto (Chunking):**
    * **Qué se hace:** El documento cargado se divide en fragmentos de texto más pequeños, denominados "chunks".
    * **Cómo se hace:** Se emplea la estrategia `RecursiveCharacterTextSplitter` de LangChain. Este segmentador divide el texto recursivamente en función de delimitadores y mantiene una `chunk_overlap` de `100` caracteres entre los chunks. El `chunk_size` se establece en `1000` caracteres.
    * **Para qué sirve:** Es crucial porque los LLMs tienen límites en la cantidad de texto que pueden procesar a la vez, y los embeddings son más efectivos en fragmentos de tamaño manejable. La superposición ayuda a preservar el contexto que podría perderse en los límites de los fragmentos, mejorando la calidad de la recuperación.

3.  **Generación de Embeddings y Almacenamiento en un Vector Store:**
    * **Qué se hace:** Cada "chunk" de texto se transforma en un vector numérico de alta dimensión (*embedding*), que captura el significado semántico del texto. Posteriormente, estos embeddings se almacenan en una base de datos especializada para búsquedas rápidas.
    * **Cómo se hace:**
        * **Generación de Embeddings:** Se utiliza `HuggingFaceEmbeddings` con el modelo **`sentence-transformers/all-MiniLM-L6-v2`**.
        * **Almacenamiento en Vector Store:** Los embeddings generados, junto con sus "chunks" de texto correspondientes, se almacenan en `FAISS` (Facebook AI Similarity Search) como una base de datos en memoria (`faiss-cpu`).
    * **Para qué sirve:** Los embeddings permiten que textos con significados similares tengan vectores cercanos, lo que es fundamental para la búsqueda de información relevante. El vector store (FAISS) permite realizar búsquedas de similitud en grandes colecciones de vectores de manera muy eficiente, siendo la base de la fase de "Retrieval" del RAG.

4.  **Uso de un LLM:**
    * **Qué se hace:** Se integra un Large Language Model (LLM) para generar respuestas a las preguntas del usuario. Este LLM es el componente encargado de la "generación" en RAG.
    * **Cómo se hace:** Se configura un `HuggingFacePipeline` de LangChain, que a su vez utiliza un modelo de Hugging Face específico: **`google/flan-t5-small`**. Este pipeline permite cargar el modelo pre-entrenado y el tokenizador, y luego usarlo para tareas de generación de texto. Se configuran parámetros como `max_new_tokens` (`512`) y `temperature` (`0.1`, para respuestas más enfocadas). Se incluye una detección automática de GPU para aprovechar el hardware si está disponible, o usar la CPU como alternativa.
    * **Para qué sirve:** El LLM es el cerebro del sistema que, al recibir el contexto relevante, formula una respuesta coherente y natural al usuario. La elección de un LLM de Hugging Face permite flexibilidad y adaptación a los recursos disponibles, siendo esta versión ideal para entornos con CPU.

5.  **Prompt de Contexto:**
    * **Qué se hace:** Se construye un "prompt" dinámico que combina la pregunta del usuario con la información relevante recuperada del documento. Este prompt es lo que el LLM recibe como entrada para generar su respuesta.
    * **Cómo se hace:** Aunque no se define explícitamente una variable de prompt en el código principal, la cadena `RetrievalQA` de LangChain se encarga internamente de formatear este prompt.
    * **Para qué sirve:** Guía al LLM para que base su respuesta únicamente en el contexto proporcionado por los chunks recuperados, minimizando las "alucinaciones" y asegurando que la respuesta sea fiel al documento original.

6.  **Interfaz Interactiva con Referencias (Citas):**
    * **Qué se hace:** Se crea una interfaz interactiva donde el usuario puede ingresar preguntas y recibir respuestas generadas por el LLM, acompañadas de las referencias a los fragmentos del documento que se utilizaron.
    * **Cómo se hace:** Se utiliza la cadena `RetrievalQA.from_chain_type` de LangChain, configurada con el LLM y el `retriever` del vector store. La opción `return_source_documents=True` es fundamental para obtener los chunks fuente. En la interfaz de usuario (un bucle `while True` en el notebook), la respuesta se imprime, y luego se itera sobre los `source_documents` para mostrar una porción del contenido de cada chunk utilizado.
    * **Para qué sirve:** Permite al usuario interactuar directamente con el sistema de QA, validar la información proporcionada por el LLM y verificar la procedencia de las respuestas, cumpliendo con el requisito de citar las referencias.

---

## Modelos Utilizados

* **Modelo de Embeddings:**
    * **Nombre:** `sentence-transformers/all-MiniLM-L6-v2`
    * **Descripción:** Este modelo de Sentence Transformers es eficiente y adecuado para la mayoría de tareas de similitud, garantizando una buena representación de los textos. Es un modelo ligero, ideal para entornos locales.

* **Large Language Model (LLM):**
    * **Nombre:** `google/flan-t5-small`
    * **Descripción:** Un modelo de la familia Flan-T5 de Google, optimizado para tareas de texto a texto como Question-Answering. Se eligió la versión "small" para facilitar la ejecución en entornos con recursos limitados (incluyendo CPU).

---

## Buenas Prácticas y Estrategias Adicionales

* **Código Documentado:** Cada sección del Jupyter Notebook y las partes clave del código están extensamente comentadas para explicar su propósito, lógica y cualquier consideración importante.
* **Estructura Clara del Notebook:** El notebook está organizado con encabezados Markdown y celdas de código separadas, siguiendo un flujo lógico que facilita la comprensión del pipeline RAG.
* **Manejo de Errores:** Se incluye un bloque `try-except` para la carga del archivo, lo que mejora la robustez del script.
* **Interactividad:** La interfaz de usuario en la consola del notebook permite una interacción continua para realizar múltiples preguntas sin reiniciar el proceso.
* **Optimización de Recursos:** Se incluye la detección de GPU para aprovecharla si está disponible, o usar CPU como fallback, haciendo el notebook más adaptable a diferentes entornos.
* **Recomendaciones de Instalación:** Se proporcionan los comandos `!pip install` para asegurar que el entorno tenga todas las dependencias necesarias.
* **`.gitignore`:** Se recomienda el uso de un archivo `.gitignore` para excluir archivos temporales, cachés (`.ipynb_checkpoints`, `__pycache__`), y directorios de modelos o índices de vector store si son muy grandes, manteniendo el repositorio limpio y enfocado en el código fuente.
* **`README.md` Completo:** Este mismo archivo `README.md` sirve como la documentación principal del proyecto, proporcionando una visión general, detalles técnicos y guía de ejecución.

---

## Cómo Ejecutar el Proyecto

1.  **Clonar el Repositorio:**
    ```bash
    git clone [https://github.com/tu_usuario/tu_repositorio.git](https://github.com/tu_usuario/tu_repositorio.git)
    cd tu_repositorio
    ```
    (Reemplaza `tu_usuario` y `tu_repositorio` con los datos de tu repositorio en GitHub).

2.  **Preparar el Archivo del Documento:**
    * Asegúrate de que el archivo `bre-b.txt` esté ubicado en la ruta `data/raw/` dentro de la estructura del proyecto.

3.  **Crear y Activar un Entorno Virtual (Recomendado):**
    Puedes elegir entre `venv` (estándar de Python) o `conda` para gestionar tu entorno:
    * **Con `venv` (Python estándar):**
        ```bash
        python -m venv venv_current
        # En Windows:
        .\venv_current\Scripts\activate
        # En macOS/Linux:
        source venv_current/bin/activate
        ```
    * **Con `conda` (Anaconda/Miniconda):**
        ```bash
        conda env create -f envs/environment_current.yml
        conda activate rag-current-env
        ```

4.  **Instalar las Dependencias:**
    * Una vez que tu entorno esté activo, instala las dependencias usando el archivo de requisitos:
        ```bash
        pip install -r requirements_current.txt
        ```

5.  **Ejecutar el Notebook:**
    * Inicia Jupyter Lab o Jupyter Notebook desde tu entorno activado:
        ```bash
        jupyter lab # o jupyter notebook
        ```
    * Navega a `notebooks/rag_system_current.ipynb` y ábrelo.
    * **Ejecuta todas las celdas del notebook en orden, de arriba a abajo.**

6.  **Interactuar con el Sistema:**
    Una vez que el sistema RAG esté listo (verás el mensaje "--- ¡Sistema RAG listo para recibir preguntas! ---"), podrás ingresar tus consultas en la consola.

    * Cuando se te pida, ingresa tus preguntas relacionadas con el contenido de `bre-b.txt`.
    * Escribe `salir` para terminar la interacción.

    **Ejemplos de Interacción:**

    1.  **Pregunta:** `¿Qué es Bre-B?`
        * **Respuesta esperada:** El sistema proporcionará una definición de Bre-B basada en el documento, indicando que es el nuevo sistema de pagos inmediatos interoperado de Colombia.
        * **Fuentes:** Se mostrarán los fragmentos de texto del documento que definen Bre-B.

    2.  **Pregunta:** `¿Cuándo estará disponible Bre-B?`
        * **Respuesta esperada:** El sistema indicará la fecha de disponibilidad de Bre-B según el documento (segundo semestre de 2025).
        * **Fuentes:** Se mostrarán los fragmentos de texto que mencionan la fecha de lanzamiento.

    3.  **Pregunta:** `¿Qué tipos de pagos se pueden hacer con las llaves de Bre-B?`
        * **Respuesta esperada:** El sistema detallará los tipos de pagos (persona a persona o persona a comercio) que se pueden realizar con las llaves.
        * **Fuentes:** Se mostrarán los fragmentos de texto relevantes sobre los tipos de pagos y el uso de las llaves.

---

## Entregables

El entregable principal de este proyecto es el archivo Jupyter Notebook (`rag_system_current.ipynb`) funcional y bien documentado, que contiene todo el código y las explicaciones necesarias para replicar y entender el sistema RAG en su configuración actual.
````