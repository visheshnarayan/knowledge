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

- **Agent Implementation**: The `agent_model` needs to be implemented to create the "Specialized Agents" that will manage and traverse the subgraphs.
- **Fine-tuning Agents**: The presentation mentions fine-tuning the agents on their respective subgraphs to improve their expertise.
- **Inter-Agent Communication**: A protocol for communication between agents needs to be established to enable collaboration.
- **Dynamic Document Ingestion**: The system should be able to dynamically ingest new documents and have the agents decide whether to add them to their subgraphs.
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