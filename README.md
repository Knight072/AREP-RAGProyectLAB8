# Build a Retrieval Augmented Generation (RAG) App

## Resumen

El objetivo de este taller es construir un servidor RAG, el cual es un enfoque en el procesamiento de lenguaje natural que combina técnicas de recuperación de información con 
generación de texto. Su objetivo es mejorar la capacidad de los modelos de lenguaje, como GPT, para responder a preguntas complejas o generar textos más informados y precisos, 
utilizando información externa. En este contexto, nos basamos en URL de paginas relacionadas con finanzas y ciberseguridad.

### Arquitectura

La arquitectura de la aplicación está basado en FastAPI y LangChain que implementa un modelo de Retrieval-Augmented Generation (RAG). FastAPI actúa como servidor web, exponiendo un único 
endpoint /rag que recibe una pregunta del usuario y devuelve una respuesta generada por el sistema. La API permite que el usuario consulte sobre información específica y obtenga respuestas contextuales y relevantes.

En el corazón de la aplicación está LangChain, una librería que facilita la integración de modelos de lenguaje con flujos de trabajo complejos de procesamiento de texto. Utiliza Chroma, una base de datos vectorial, para 
almacenar y recuperar documentos relacionados con la pregunta del usuario. Estos documentos se cargan desde URLs específicas utilizando WebBaseLoader y se dividen en fragmentos más pequeños mediante 
RecursiveCharacterTextSplitter para su indexación. Los fragmentos de texto se representan mediante OpenAIEmbeddings, que convierte los documentos en vectores que pueden ser recuperados eficientemente.

Cuando el usuario realiza una consulta, la pregunta se procesa junto con los documentos recuperados, utilizando GPT-4 (a través de ChatOpenAI) para generar una respuesta. LangChain conecta todos 
estos componentes en una cadena de procesamiento, que incluye la recuperación de documentos, la generación de texto y la post-procesamiento de la respuesta utilizando un StrOutputParser. Todo esto se orquesta 
mediante una cadena RAG que permite que el modelo aproveche la información recuperada para generar respuestas más precisas.

```mermaid

graph TD
    A[Cliente] -->|Envía solicitud HTTP| B[FastAPI]
    B -->|Pasa solicitud| C[RAG Pipeline]
    C -->|Invoca| D[WebBaseLoader]
    D -->|Cargar Documentos| E[URLs]
    E -->|Procesa| F[Text Splitter]
    F -->|Fragmentos de Texto| G[Chroma Vector Store]
    G -->|Recupera Documentos| H[Retrieval]
    H -->|Contexto| I[Prompt]
    I -->|Pregunta procesada| J[GPT-4 OpenAI]
    J -->|Respuesta Generada| K[StrOutputParser]
    K -->|Texto procesado| B
    B -->|Devuelve respuesta| A
    B -->|Ruta /rag| L[LangChain]
    L -->|Despliega API REST| M[Uvicorn]
    M -->|Ejecuta FastAPI| B
    A -->|Interacción UI| L
    A -->|Interacción programática| N[RemoteRunnable]
    N -->|Invoca API REST| L

    classDef fastAPI fill:#F4A300,stroke:#333,stroke-width:2px;
    class B fastAPI;
    
    classDef langChain fill:#6F9B6A,stroke:#333,stroke-width:2px;
    class C langChain;
    
    classDef model fill:#1E77B9,stroke:#333,stroke-width:2px;
    class J model;
    
    classDef parser fill:#FFC107,stroke:#333,stroke-width:2px;
    class K parser;
    
    classDef langServe fill:#7F8C8D,stroke:#333,stroke-width:2px;
    class L langServe;
    
    classDef uvicorn fill:#16A085,stroke:#333,stroke-width:2px;
    class M uvicorn;
    
    classDef remoteRunnable fill:#8E44AD,stroke:#333,stroke-width:2px;
    class N remoteRunnable;
```

### Instalación y Ejecución
Para ejecutar la aplicación es necesario instalar Python 3.12.7 o versiones similares y git. El primer paso es clonar el repositorio e ingresar a la carpeta resultante

```
git clone https://github.com/Knight072/AREP-RAGProyectLAB8.git
````

A continuación editamos el archivo ragproyect.py en la linea 12 para ingresar la clave API de nuestro modelo LLM, en este caso chatgpt.

```
os.environ["OPENAI_API_KEY"] = "(clave api)"
```

Luego ingrese a la terminal y escriba los siguientes comandos:

```
python -m venv .venv
```

```
.venv\Scripts\activate
```

Una vez terminada la instalación de las dependencias puede ejecutar el servidor con:

```
python ragserver.py
```

Finalmente pruebe el servidor enviando la query a la siguiente dirección con el formato indicado http://localhost:8000/rag?question=(Pregunta relacionada con Forex o ciberseguridad)

![image](https://github.com/user-attachments/assets/0c72b798-4e6d-4df0-a91c-806d3ca46a6b)

Video del programa en funcionamiento: https://youtu.be/FM8VqoLLL0Q

## Autor

- [Daniel Felipe Rojas Hernandez](https://github.com/Knight072)

## Licencia

Este proyecto está bajo la Licencia (MIT) - ver el archivo [LICENSE](LICENSE.md) para ver más detalles.
