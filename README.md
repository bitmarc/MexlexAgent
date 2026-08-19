# mexlex-agent

Agente conversacional RAG para consultar leyes mexicanas y la Constitución
Política de los Estados Unidos Mexicanos (CPEUM). Proyecto de aprendizaje
de LangChain: ingesta, retrieval híbrido, agentes con tools, memoria y UI.

Ver la sección **Roadmap** abajo para el plan completo por fases. Este
scaffold implementa la **Fase 1** (ingesta + índice + chain RAG simple)
y deja listas las carpetas para las fases siguientes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate

pip install -e .
pip install -r requirements.txt

cp .env.example .env
# Edita .env con tus credenciales de Azure OpenAI y Azure AI Search
```

Necesitas tener aprovisionados:
- Un recurso de **Azure OpenAI** con dos deployments: uno de chat (ej.
  `gpt-4o-mini`) y uno de embeddings (ej. `text-embedding-3-large`).
- Un recurso de **Azure AI Search** (el tier gratuito F0 funciona para
  hybrid search; si quieres semantic ranking necesitas Basic o superior).
- *(Opcional, Fase 7)* Una cuenta de **Azure Cosmos DB for NoSQL** para
  persistir el historial. Sin ella la app funciona igual, pero las
  conversaciones se pierden al reiniciar. El contenedor debe tener la
  partition key `/partition_key`; el código lo crea solo si tiene
  permisos. Ver
  [documentation/08-fase7-persistencia-cosmosdb.md](documentation/08-fase7-persistencia-cosmosdb.md).

## Uso

1. Coloca uno o más PDFs de leyes en `data/raw_pdfs/` (para empezar,
   descarga la CPEUM desde el sitio de la Cámara de Diputados).

2. Corre la ingesta (esto crea el índice en Azure AI Search si no existe,
   y sube los chunks):

   ```bash
   python scripts/run_ingestion.py
   ```

3. Prueba el retrieval + la chain desde terminal:

   ```bash
   python scripts/query_test.py
   ```

   Ejemplo de pregunta: `¿qué garantiza el artículo 4 constitucional?`

4. Prueba la memoria conversacional desde terminal (Fase 3):

   ```bash
   python scripts/chat_test.py
   ```

   Haz una pregunta y luego una de seguimiento (`¿y el siguiente
   artículo?`) para ver el historial en acción.

5. Prueba el agente con tools (Fase 4), viendo qué decide usar:

   ```bash
   python scripts/agent_test.py
   ```

6. Corre la suite de evaluación del agente (Fase 6) — úsala como
   regresión después de tocar prompts, tools o chunking:

   ```bash
   python scripts/run_eval.py          # agrega --verbose para ver respuestas
   ```

7. Comprueba que el historial se persiste (Fase 7). Ambos scripts de
   chat aceptan un `thread_id`: conversa, sal, y vuelve con el mismo id.

   ```bash
   python scripts/agent_test.py 11111111-1111-1111-1111-111111111111
   # ...pregunta algo, escribe 'salir', y en un proceso nuevo:
   python scripts/cosmos_memory_test.py 11111111-1111-1111-1111-111111111111
   ```

   Si `AZURE_COSMOS_ENDPOINT` no está configurado, el checkpointer cae a
   memoria y el script te lo dice.

8. Levanta la interfaz de chat (agente + memoria + streaming):

   ```bash
   chainlit run app/chainlit_app.py -w
   ```

   Abre http://localhost:8000. El flag `-w` recarga la app al guardar
   cambios en el archivo.

   Para tener **barra lateral con las conversaciones anteriores**
   (Fase 8) hacen falta, además de Cosmos, un usuario y un secreto de
   firma en el `.env`:

   ```bash
   chainlit create-secret        # copia el valor a CHAINLIT_AUTH_SECRET
   ```

   ```bash
   MEXLEX_AUTH_USER=tu-usuario
   MEXLEX_AUTH_PASSWORD=tu-contraseña
   CHAINLIT_AUTH_SECRET=<el-valor-generado>
   ```

   Chainlit solo persiste y reanuda hilos cuando sabe de quién son, por
   eso el login es obligatorio para esta funcionalidad. Sin estas
   variables la app corre igual, sin login y sin historial.

Si la respuesta no cita bien las fuentes o parece cortar artículos a la
mitad, ajusta `chunk_size` / `chunk_overlap` en `src/mexlex/config.py` y
vuelve a correr la ingesta.

> **Nota sobre el índice:** el esquema (los campos filtrables que hacen
> posible el lookup exacto por artículo) solo se aplica al **crear** el
> índice. Si cambias la estructura de la metadata, apunta
> `AZURE_SEARCH_INDEX_NAME` a un nombre nuevo y reindexa; apuntar al
> índice existente no lo migra.

## Estructura

```
src/mexlex/
├── config.py            # settings centralizados (.env)
├── ingestion/            # Fase 1: loaders, splitters, index_builder
├── retrieval/             # Fase 1: Azure AI Search (hybrid retriever) + formatting
├── chains/                # Fase 1-2: chain LCEL | Fase 3: grafo conversacional
├── tools/                 # Fase 4: legal_search_tool.py, web_search_tool.py
├── agent/                 # Fase 3-7: memory.py | Fase 4: graph.py, prompts.py
└── persistence/           # Fase 7-8: cosmos.py, data_layer.py, auth.py

app/chainlit_app.py         # Fase 2: entrypoint de Chainlit (streaming)
chainlit.md                 # Fase 2: pantalla de bienvenida de la UI
scripts/                    # CLIs de ingesta y prueba
documentation/              # Explicación detallada de cada flujo
```

## Roadmap

- [x] **Fase 1** — Ingesta (loaders, splitters) + índice en Azure AI
      Search con hybrid search + chain RAG simple (LCEL), probada por
      terminal.
- [x] **Fase 2** — Envolver la chain en una interfaz de Chainlit con
      streaming (`app/chainlit_app.py`), sin memoria todavía.
- [x] **Fase 3** — Memoria conversacional: grafo de LangGraph con
      checkpointer (`MemorySaver`) ligado al `thread_id` de la sesión de
      Chainlit, más reescritura de preguntas de seguimiento.
- [x] **Fase 4** — Agente ReAct (`create_react_agent`) con dos tools:
      búsqueda híbrida sobre los documentos legales y búsqueda web
      (Tavily) para preguntas fuera del corpus. La web es **opcional**:
      sin `TAVILY_API_KEY` el agente corre solo con el corpus.
- [x] **Fase 5** — Retrieval estructurado: chunking por frontera de
      artículo, metadata filtrable en el índice (ley, artículos, página),
      tools de lookup exacto (`obtener_articulo`, `expandir_contexto`) y
      panel de fuentes en la UI.
- [x] **Fase 6** — Trazabilidad con LangSmith (opcional), suite de
      evaluación del agente (`scripts/run_eval.py`), disclaimers basados
      en la vigencia real de cada ley y citas legibles con distinción
      corpus/web.
- [x] **Fase 7** — Persistencia del historial en **Azure Cosmos DB
      NoSQL**: el checkpointer pasa de `MemorySaver` a `CosmosDBSaver`,
      con conexión de vida de proceso y `thread_id` estable por
      conversación. Es **opcional**: sin `AZURE_COSMOS_ENDPOINT` la app
      cae a memoria y avisa en el log.
- [x] **Fase 8** — Historial visible en la UI: implementación de
      `BaseDataLayer` de Chainlit sobre Cosmos DB (contenedor aparte,
      `conversations`) más el login que Chainlit exige para saber de
      quién es cada hilo, y `@cl.on_chat_resume`. También es **opcional**:
      sin `MEXLEX_AUTH_USER` la app corre sin login, como en la Fase 7.

## Nota

Este asistente no sustituye asesoría legal profesional. Las respuestas se
basan únicamente en los documentos indexados y pueden no reflejar
reformas recientes si el corpus no está actualizado.
