# Project: Agentic Knowledge Graph Management

This project implements a system for building, managing, and visualizing knowledge graphs from text documents, with a focus on using intelligent agents to organize and traverse the graph. It is based on the concepts outlined in the `AgenticGraphManagement.pdf` document in this repository.

## Core Concepts

The system is designed to address the limitations of traditional Retrieval-Augmented Generation (RAG) by creating a more dynamic and context-aware knowledge management system. The key ideas are:

- **Agent-driven Graph Management**: Instead of static similarity search, this system uses intelligent agents to manage and traverse the knowledge graph.
- **Specialized Agents**: Each agent becomes an expert on a specific topic cluster within the graph.
- **Dynamic Graph Formation**: The graph is built by connecting related text chunks using configurable similarity thresholds.
- **Adaptive Learning**: Agents can decide what information to ingest and when to collaborate with other agents.

## Core Components

The project is structured into several Python modules, each with a specific responsibility:

- **`main.py`**: The main entry point of the application. It instantiates and runs the `Runner`.

- **`src/runner.py`**: Orchestrates the entire knowledge graph construction and visualization process. It loads the configuration, runs the graph builder, processes the graphs with the parent model, and finally, visualizes the results.

- **`src/config_loader.py`**: A simple utility to load the project's configuration from the `config.yaml` file.

- **`src/graph_builder.py`**: The heart of the project. This module is responsible for:
    - Reading text files from the `data/input` directory.
    - Splitting the text into smaller chunks.
    - Generating vector embeddings for each chunk using a pre-trained transformer model (`bert-base-uncased`).
    - Constructing a **semantic similarity graph** by connecting chunks with high cosine similarity between their embeddings.
    - Using a triplet extraction model (`Babelscape/rebel-large`) to identify knowledge triplets (subject, predicate, object) within each chunk.
    - Constructing a **triplet-based knowledge graph** by representing chunks and their extracted triplets as nodes and edges.

- **`src/parent_model.py`**: This module implements the "Parent LLM" concept from the presentation. It uses a locally-run Large Language Model (LLM), accessed via an LM Studio endpoint, to perform higher-level processing on the generated graphs. Its main task is to:
    - Identify overarching themes or topics across all the text chunks.
    - Assign each chunk to a specific topic, effectively partitioning the graph.
    - Add "topic" nodes to the graph to represent these thematic groups.

- **`src/graph_visualizer.py`**: This module uses the `pyvis` library to create interactive HTML visualizations of the knowledge graphs. It color-codes nodes based on their assigned topics and provides a legend for easy interpretation.

- **`config.yaml`**: The main configuration file for the project. It allows users to customize various parameters, such as:
    - Input data directories.
    - Text chunking settings (size, overlap).
    - Embedding and triplet extraction models.
    - The parent model to be used.
    - A placeholder for the `agent_model`, which will likely be used to implement the "Specialized Agents" concept.
    - Database connection details (with a TODO to switch to Pinecone).

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install Flask
    ```

2.  **Set up LM Studio**:
    - Download and install LM Studio.
    - Download a compatible model (e.g., a GGUF version of a model like Mistral or Llama).
    - Start the local server in LM Studio.

3.  **Run the application**:
    ```bash
    python main.py
    ```

4.  **View the output**:
    - The application will start a web server at `http://127.0.0.1:5000`.
    - Open this URL in your browser to see the graph visualization and interact with the agents.

## Future Work

The project is still under development, and the following features are planned for the future:

- **Live Document Ingestion and Agent-led Classification**:
    - Implement an endpoint or a watched directory to allow new documents to be added to the system while it's running.
    - Instead of a global classification model, leverage the existing specialist agents to determine if a new document chunk belongs to their topic.
- **Inter-Agent Collaboration Protocol**:
    - Design and implement a mechanism for agents to communicate with each other.
    - If multiple agents claim a new document chunk, they will engage in a "conversation" (e.g., using a shared prompt with a higher-level LLM, or a predefined negotiation protocol) to resolve the conflict and decide on the best placement for the new information.
- **Dynamic Graph Updates**:
    - Once a new chunk is assigned to an agent, the system needs to dynamically update the main knowledge graph and the agent's subgraph with the new node and its connections.
- **Agent Implementation**: The `agent_model` needs to be implemented to create the "Specialized Agents" that will manage and traverse the subgraphs.
- **Fine-tuning Agents**: The presentation mentions fine-tuning the agents on their respective subgraphs to improve their expertise.
- **Model/Graph Drift**: A mechanism to detect and handle model or graph drift needs to be implemented to keep the knowledge graph up-to-date and relevant.
- **Vector Database Integration**: The current database setup is planned to be replaced with [Pinecone](https://www.pinecone.io/) for more efficient similarity search at scale.
- **Visualize Iterative Search Process**: Explore ways to visualize the agent's iterative search process within the knowledge graph to better understand its traversal and context building.

---

## Development Log

**8/19/25**

- **Iterative Topic Consolidation:** Implemented iterative consolidation in `src/parent_model.py` to merge topics until no further consolidations are possible.
- **Configurable Consolidation:** Added `enable_consolidation` flag to `config.yaml` and implemented conditional execution in `src/runner.py` to control topic consolidation.
- **Robust LLM Response Parsing:** Refined prompts and improved parsing logic in `src/parent_model.py` to better handle LLM output, including malformed responses, for consolidation instructions.
- **Improved Logging for LLM Responses:** Modified `src/parent_model.py` to log concise information about unexpected LLM response formats without exposing full raw output.
- **Process Timing:** Added timing information to major processes in `src/runner.py` (graph building, parent model processing, consolidation, agent creation, visualization, Flask app startup) for performance monitoring.
- **Source Display in UI:**
    - Modified `src/agent.py` to return source contents along with the LLM response.
    - Modified `src/app.py` to pass source contents to the UI.
    - Modified `templates/index.html` to display sources in a separate, toggleable pane.
- **Enhanced Debugging for Source Search:**
    - Added detailed debug logs in `src/agent.py` for similarity search, including initial random nodes, nodes popped, and reasons for skipping nodes.
    - Changed initial random node logging to `info` level and included full content for better visibility.
- **Configurable Search Similarity:** Added `search_similarity` parameter to `config.yaml` and integrated it into `src/runner.py` and `src/agent.py` for agent context search.
- **Bug Fixes & Stability:**
    - Fixed a logic error in `_similarity_search_context` in `src/agent.py` related to `node_embedding` scope.
    - Implemented a more granular state-loading mechanism to intelligently load graphs and agents, preventing unnecessary rebuilding.
    - Resolved a `TypeError` caused by an incorrect number of arguments passed to the `Agent` constructor.
    - Fixed a `TypeError` that occurred when saving the agent state by implementing a new method to only pickle serializable data.
    - Corrected a "Not Found" error in the UI by ensuring the graph visualization files were served correctly.
- **UI Enhancements:**
    - Added a dropdown menu to the UI to allow toggling between the cosine and triplet graph visualizations.
    - Implemented a slider in the UI to dynamically adjust the `search_similarity` threshold for agent context search.

**8/18/25**

- **Agent System:** Implemented a multi-agent system where specialized agents are created for each topic cluster in the knowledge graph. Each agent can be queried individually about its specific domain.
- **Web UI:** Created a web-based user interface using Flask to visualize the knowledge graph and interact with the agents. The UI allows users to select an agent and ask questions.
- **State Persistence:** Implemented a state persistence mechanism using `pickle` to save and load the processed graphs and agents. This allows the application to resume from where it left off without rebuilding the graphs.
- **Logging:** Added a centralized logging system to provide detailed visibility into the system's behavior.
- **Similarity Search:** Implemented a new similarity search method for agents based on a DFS-like graph traversal. This allows agents to dynamically build context based on the question's similarity to nodes in the graph.
- **Configuration:** Made the search sample ratio for the similarity search configurable in the `config.yaml` file.

**Note:** All new code should consistently use the centralized logging system (`src/logger.py`) for better traceability and debugging.

---

**8/27/25**

- **Strands Framework Integration:**
    - Integrated the Strands AI framework for agent management and orchestration.
    - Added `strands-ai` to `requirements.txt`.
    - Updated `config.yaml` to include `strands` configuration, specifying `parent_agent_model_id` (DeepSeek) and `child_agent_model_id` (Claude Sonnet).
    - Modified `main.py` to load `STRANDS_MODEL_ID` from `config.yaml` and set it as an environment variable for Strands.
- **Refactored Agent Classes:**
    - **`src/agent.py`**: Refactored the `Agent` class to use `strands.Agent` and `strands.models.BedrockModel`. The `_similarity_search_context` logic was moved into a `@tool`-decorated method (`similarity_search_tool`) within the `Agent` class.
    - **`src/parent_agent.py`**: Created a new `ParentAgent` class to act as a routing agent. This agent uses its LLM to determine the topic of a question and then forwards the query to the appropriate child agent.
- **Updated Orchestration:**
    - **`src/runner.py`**: Modified to create instances of the new `ParentAgent` and `Agent` (child agents). The `_create_parent_agent` method now instantiates the `ParentAgent` and passes the child agents to it.
    - **`src/app.py`**: Updated to interact with the `ParentAgent` for query handling and to pass the `child_agents` for dynamic `search_similarity` updates.
- **Multi-Model Support:** Configured the system to use different Bedrock models for the parent agent (DeepSeek R1) and child agents (Claude 3.5 Sonnet), leveraging their respective strengths.
- **Testing and Debugging:**
    - Created `test_agent_communication.py` to verify parent-child agent communication and routing.
    - Addressed and resolved several issues during implementation and testing:
        - `AttributeError: 'DecoratedFunctionTool' object has no attribute 'with_parameters'` (fixed by making the tool a method of the `Agent` class).
        - `TypeError: Agent.__init__() got an unexpected keyword argument 'llm'` (fixed by configuring the model via `STRANDS_MODEL_ID` environment variable and passing `model=` to `StrandsAgent`).
        - `KeyError: 'bedrock'` (fixed by removing explicit `BedrockModel` instantiation in `Agent` and `ParentAgent` as Strands handles it via environment variables).
        - `AttributeError: 'Agent' object has no attribute 'run'` (fixed by calling `StrandsAgent` instances directly).
        - `ValidationException` on tool names (fixed by sanitizing topic names to remove spaces).
        - `TypeError: unhashable type: 'AgentResult'` (fixed by converting `AgentResult` to string and stripping whitespace).
        - `EOFError` in test script (fixed by hardcoding the test question).
        - `ValidationException: The provided model identifier is invalid.` (fixed by using the correct Bedrock model ID for DeepSeek R1).
- **UI Query Fixes:**
    - Added a dropdown (`topic-select`) to `templates/index.html` for selecting the target agent topic.
    - Corrected the JavaScript fetch endpoint in `templates/index.html` from `/query` to `/query_agent`.
    - Ensured the selected `agent_topic` is included in the JSON payload sent from the UI to the backend.

**8/28/25**

- **ParentModel Deprecation & ParentAgent Enhancement:**
    - Migrated core graph processing functionalities from `src/parent_model.py` to `src/parent_agent.py`:
        - Implemented `identify_topics_from_graph` for LLM-driven topic identification.
        - Implemented `classify_and_assign_chunks` for assigning text chunks to identified topics and modifying the graph.
        - Implemented `consolidate_topics` for iterative topic consolidation.
        - Created `process_graph` in `src/parent_agent.py` to orchestrate these new graph processing methods.
    - Updated `src/runner.py` to exclusively use `ParentAgent.process_graph` for all graph processing steps, effectively deprecating `ParentModel`.
- **UI/Agent Communication & Context Retrieval Improvements:**
    - **JSON Serialization Fix:** Modified `src/agent.py` to correctly extract `response_text` and `source_contents` from Strands `AgentResult` objects, resolving `TypeError: Object of type Trace is not JSON serializable` during Flask's `jsonify` process.
    - **Redundant Topic Removal:**
        - Removed "Select Agent Topic" dropdown and associated JavaScript payload (`agent_topic`) from `templates/index.html`.
        - Updated `src/app.py` to no longer expect or use the `topics` argument in `create_app` and its `render_template` calls.
        - Updated `src/runner.py` to reflect the change in `create_app` signature.
    - **Context Search Parameter Tuning:**
        - Lowered `search_similarity` to `0.4` in `config.yaml` to make context retrieval less strict.
        - Increased `search_sample_ratio` to `0.8` in `config.yaml` to expand the initial search scope for relevant chunks.

---

**9/04/25**

- **Bedrock Model Compatibility:**
    - Fixed a bug in `src/parent_agent.py` where the API request payload was hardcoded for Anthropic models. The payload is now correctly formatted for DeepSeek models.
    - Updated the response parsing logic in `src/parent_agent.py` to correctly handle the output from DeepSeek models.
- **Strands Integration Stability:**
    - Resolved a `ValidationException` during topic consolidation by ensuring the `StrandsAgent` has a clean history for each API call. This prevents the model's previous reasoning from being sent back to the API.
    - Suppressed verbose, unformatted logging from the `strands` library by setting its logger level to `CRITICAL` in `main.py`.
- **Conditional LLM Logging:**
    - Introduced a `debug` section in `config.yaml` with a `log_llm_responses` flag.
    - Implemented logic throughout the application to log detailed LLM responses only when this flag is enabled.
    - Refined the debug logging to only output the final, cleaned LLM response, removing the noisy reasoning text from the console output.
- **Graph Visualization Enhancements:**
    - Assigned a unique, uniform color to all "Central Topic" nodes in the graph visualization to make them easily distinguishable.
    - Updated the legend to include an entry for the new topic node color.
    - Added detailed logging to `src/graph_visualizer.py` to help diagnose an issue with inconsistent node colors after topic consolidation.
- **Flask Auto-Reload:**
    - Disabled the automatic restart feature of the Flask development server by setting `use_reloader=False` in `src/runner.py`.
