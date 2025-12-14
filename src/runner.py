import os
import pickle
import time
import threading
import requests
import sys
from src.config_loader import ConfigLoader
from src.graph_builder import GraphBuilder
from src.graph_visualizer import GraphVisualizer
from src.parent_model import ParentModel, LMStudioQueryRouter
from src.agent import Agent
from src.ollama_agent import OllamaAgent
from src.parent_agent import ParentAgent
from src.app import create_app
from src.cli import AgentCLI
from src.logger import get_logger
import networkx as nx
from datetime import datetime

logger = get_logger(__name__)


class Runner:
    def _check_ollama_availability(self):
        """Checks if the Ollama server is running and the specified model is available."""
        logger.info("Checking Ollama server availability and model...")

        # Prioritize environment variable for Docker Compose, fallback to config
        default_url = self.ollama_config.get("base_url", "http://localhost:11434/v1")
        base_url = os.environ.get("OLLAMA_BASE_URL", default_url).replace("/v1", "")

        model_name = self.ollama_config.get("model")

        if not model_name:
            logger.error("Ollama model name is not specified in config.yaml.")
            sys.exit(1)

        try:
            logger.info(f"Attempting to connect to Ollama at {base_url}...")
            response = requests.get(f"{base_url}/api/tags")
            response.raise_for_status()
            available_models = [
                m["name"].split(":")[0] for m in response.json().get("models", [])
            ]

            # Handle model names with or without tags
            model_base_name = model_name.split(":")[0]

            if model_base_name not in available_models:
                logger.error(f"Model '{model_name}' not found in Ollama.")
                logger.error(f"Available models are: {available_models}")
                logger.error(f"Please pull the model using: ollama pull {model_name}")
                sys.exit(1)

            logger.info(
                f"Ollama server is running and model '{model_name}' is available."
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect to Ollama server at {base_url}.")
            logger.error("Please ensure the Ollama server is running.")
            logger.error(f"Error details: {e}")
            sys.exit(1)

    def __init__(self):
        logger.info("Initializing runner...")
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config()
        self.graph_builder = GraphBuilder(
            self.config.get("data"), self.config.get("graph")
        )

        graph_config = self.config.get("graph", {})
        self.enable_consolidation = graph_config.get("enable_consolidation", True)
        self.search_similarity = graph_config.get("search_similarity", 0.6)
        self.search_sample_ratio = graph_config.get("search_sample_ratio", 0.8)

        self.llm_infra_config = self.config.get("llm_infra", {})
        # Allow overriding infra type with an environment variable for Docker flexibility
        self.infra_type = os.environ.get(
            "LLM_INFRA_TYPE", self.llm_infra_config.get("type", "strands")
        )
        self.debug_config = self.config.get("debug", {})

        if self.infra_type == "strands":
            self.strands_config = self.llm_infra_config.get("strands", {})
            self.child_agent_model_id = self.strands_config.get(
                "child_agent_model_id"
            )
        elif self.infra_type == "lm_studio":
            self.lm_studio_config = self.llm_infra_config.get("lm_studio", {})
        elif self.infra_type == "ollama":
            self.ollama_config = self.llm_infra_config.get("ollama", {})

        output_config = self.config.get("output", {})
        base_output_dir = output_config.get("base_dir", "data/builds")
        build_version = output_config.get("version", "timestamp")
        self.serve_only_mode = output_config.get("serve_only_mode", False)

        if build_version == "timestamp" and not self.serve_only_mode:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.output_dir = os.path.join(base_output_dir, timestamp)
        else:
            self.output_dir = os.path.join(base_output_dir, build_version)

        logger.info(f"Output directory set to: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.processed_cosine_graph_path = os.path.join(
            self.output_dir, "processed_cosine_graph.pkl"
        )
        self.processed_triplets_graph_path = os.path.join(
            self.output_dir, "processed_triplets_graph.pkl"
        )
        self.agents_path = os.path.join(self.output_dir, "agents.pkl")

        # Centralized OrgTable configuration
        self.org_table_config = self.config.get("org_table", {})
        self.org_table_path = os.path.join(self.output_dir, "org_table.json")
        logger.info(f"Org table path set to: {self.org_table_path}")

        logger.info("Runner initialized.")

    def _handle_org_table_loading(self, parent_agent):
        """Loads the organizational table from file if configured."""
        if self.org_table_config.get("load_from_file", False):
            if os.path.exists(self.org_table_path):
                parent_agent.org_table.load_from_json(self.org_table_path)
                logger.info(f"Org table loaded from {self.org_table_path}")
                return True
            else:
                logger.warning(
                    f"Org table file not found at {self.org_table_path}, but 'load_from_file' is true. A new one will be created if a full build is run."
                )
        return False

    def get_status(self):
        """Gathers and returns a dictionary of the current system status."""
        status = {
            "build_directory": self.output_dir,
            "run_mode": self.config.get("output", {}).get("run_mode", ["web"]),
            "infra_type": self.infra_type,
            "graphs_loaded": hasattr(self, "processed_triplets_graph")
            and self.processed_triplets_graph is not None,
            "agents_loaded": hasattr(self, "parent_agent")
            and self.parent_agent is not None
            and bool(self.parent_agent.child_agents),
            "topic_count": 0,
        }
        if (
            hasattr(self, "parent_agent")
            and self.parent_agent
            and self.parent_agent.org_table
        ):
            status["topic_count"] = len(self.parent_agent.org_table.get_topics())
        return status

    def run(self):
        if self.serve_only_mode:
            self.run_serve_only()
            return

        logger.info(f"Starting runner with infra type: {self.infra_type}...")

        if self.infra_type == "strands":
            self.run_strands()
        elif self.infra_type == "lm_studio":
            self.run_lm_studio()
        elif self.infra_type == "ollama":
            self.run_ollama()
        else:
            # Defaulting to lm_studio for now if config is malformed
            logger.warning(
                f"Unknown infra type '{self.infra_type}'. Defaulting to 'lm_studio'."
            )
            self.run_lm_studio()

    def run_serve_only(self):
        logger.info(
            f"Starting in serve-only mode from build directory: {self.output_dir}"
        )

        processed_cosine_graph, processed_triplets_graph = self._load_graphs()
        if not all([processed_cosine_graph, processed_triplets_graph]):
            logger.error("Could not load graphs. Cannot start in serve-only mode.")
            return

        logger.info("Creating parent agent...")
        parent_agent = self._create_parent_agent()
        logger.info("Parent agent created.")

        # Attempt to load the org table from JSON
        org_table_loaded = self._handle_org_table_loading(parent_agent)

        if not org_table_loaded:
            logger.info(
                "Org table not loaded from file. Populating from loaded graph for this session..."
            )
            parent_agent._populate_org_table(processed_triplets_graph)

        # Agent loading/creation depends on the infra type
        if self.infra_type == "strands":
            child_agents = self._load_agents(parent_agent, parent_agent.org_table)
        elif self.infra_type == "ollama":
            child_agents = self._create_ollama_child_agents(processed_triplets_graph)
        else:
            logger.error(
                f"Serve-only mode is not supported for infra type '{self.infra_type}'."
            )
            return

        if not child_agents:
            logger.error("Could not load or create agents. Cannot start in serve-only mode.")
            return

        parent_agent.set_child_agents(child_agents)
        logger.info("Parent agent has been updated with child agent references.")

        vis_cosine_path = os.path.abspath(
            os.path.join(self.output_dir, "output_cosine.html")
        )
        vis_triplets_path = os.path.abspath(
            os.path.join(self.output_dir, "output_triplets.html")
        )

        if not os.path.exists(vis_cosine_path) or not os.path.exists(vis_triplets_path):
            logger.warning(
                "Visualization files not found. They will not be available in the UI."
            )

        run_modes = self.config.get("output", {}).get("run_mode", ["web"])
        if not isinstance(run_modes, list):
            run_modes = [run_modes]

        def run_flask():
            logger.info("Starting Flask app in a background thread...")
            app = create_app(
                parent_agent,
                child_agents,
                {"cosine": vis_cosine_path, "triplets": vis_triplets_path},
                self.search_similarity,
            )
            app.run(host="0.0.0.0", debug=False, use_reloader=False)

        if "web" in run_modes:
            web_thread = threading.Thread(target=run_flask, daemon=True)
            web_thread.start()
            logger.info("Web server thread started.")

        if "cli" in run_modes:
            logger.info("Starting interactive CLI in serve-only mode...")
            cli = AgentCLI(self, parent_agent, parent_agent.org_table)
            cli.cmdloop()
        elif "web" in run_modes:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down web server.")

        logger.info("Runner finished.")

    def run_lm_studio(self):
        # LM Studio-based implementation
        logger.info("Running with LM Studio infrastructure...")

        output_config = self.config.get("output", {})
        build_version = output_config.get("version", "timestamp")

        processed_cosine_graph = None
        processed_triplets_graph = None
        loaded_existing_build = False

        if build_version != "timestamp" and os.path.exists(self.output_dir):
            logger.info(f"Attempting to load existing build '{build_version}'...")
            graphs = self._load_graphs()
            if graphs:
                processed_cosine_graph, processed_triplets_graph = graphs
                loaded_existing_build = True
                logger.info(f"Successfully loaded existing build '{build_version}'.")
            else:
                logger.warning(
                    f"Could not load graphs from existing build '{build_version}'. Proceeding with full build."
                )

        parent_model = ParentModel(self.lm_studio_config)

        if not loaded_existing_build:
            start_time = time.time()
            logger.info("Building graphs from scratch...")
            cosine_graph, triplets_graph = self.graph_builder.build()
            end_time = time.time()
            logger.info(
                f"Graph building finished in {end_time - start_time:.2f} seconds."
            )

            start_time = time.time()
            logger.info("Parent model sorting graph into subgroups...")
            processed_cosine_graph = parent_model.sort_into_subgroups(cosine_graph)
            processed_triplets_graph = parent_model.sort_into_subgroups(triplets_graph)
            end_time = time.time()
            logger.info(
                f"Parent model finished sorting in {end_time - start_time:.2f} seconds."
            )

            if self.enable_consolidation:
                start_time = time.time()
                logger.info("Consolidating topics...")
                processed_cosine_graph = parent_model.consolidate_topics(
                    processed_cosine_graph
                )
                processed_triplets_graph = parent_model.consolidate_topics(
                    processed_triplets_graph
                )
                end_time = time.time()
                logger.info(
                    f"Topic consolidation finished in {end_time - start_time:.2f} seconds."
                )

            self._save_graphs(processed_cosine_graph, processed_triplets_graph)

        child_agent_data = self._create_lm_studio_child_agent_data(
            processed_triplets_graph
        )

        query_router = self._create_lm_studio_query_router(
            parent_model, child_agent_data
        )

        start_time = time.time()
        logger.info("Visualizing graphs...")
        vis_cosine_path = os.path.abspath(
            os.path.join(self.output_dir, "output_cosine.html")
        )
        visualizer_cosine = GraphVisualizer(processed_cosine_graph)
        visualizer_cosine.visualize(output_path=vis_cosine_path)
        logger.info(f"Cosine graph visualization saved to {vis_cosine_path}")

        vis_triplets_path = os.path.abspath(
            os.path.join(self.output_dir, "output_triplets.html")
        )
        visualizer_triplets = GraphVisualizer(processed_triplets_graph)
        visualizer_triplets.visualize(output_path=vis_triplets_path)
        logger.info(f"Triplets graph visualization saved to {vis_triplets_path}")
        end_time = time.time()
        logger.info(
            f"Graph visualization finished in {end_time - start_time:.2f} seconds."
        )

        start_time = time.time()
        logger.info("Starting Flask app...")
        app = create_app(
            query_router,
            child_agent_data,
            {"cosine": vis_cosine_path, "triplets": vis_triplets_path},
            self.search_similarity,
        )
        app.run(host="0.0.0.0", debug=True, use_reloader=False)
        end_time = time.time()
        logger.info(
            f"Flask app startup call finished in {end_time - start_time:.2f} seconds."
        )

        logger.info("Runner finished.")

    def run_ollama(self):
        self._check_ollama_availability()
        logger.info("Running with Ollama infrastructure...")

        output_config = self.config.get("output", {})
        build_version = output_config.get("version", "timestamp")

        self.processed_cosine_graph, self.processed_triplets_graph = None, None
        loaded_existing_build = False

        self.parent_agent = self._create_parent_agent()
        org_table_loaded_from_file = self._handle_org_table_loading(self.parent_agent)

        if build_version != "timestamp" and os.path.exists(self.output_dir):
            logger.info(f"Attempting to load existing build '{build_version}'...")
            (
                self.processed_cosine_graph,
                self.processed_triplets_graph,
            ) = self._load_graphs()
            if self.processed_cosine_graph and self.processed_triplets_graph:
                loaded_existing_build = True
                logger.info(f"Successfully loaded existing build '{build_version}'.")
            else:
                logger.warning(
                    f"Could not load graphs from existing build '{build_version}'. Proceeding with full build."
                )

        if not loaded_existing_build:
            start_time = time.time()
            logger.info("Building graphs from scratch...")
            cosine_graph, triplets_graph = self.graph_builder.build()
            logger.info(
                f"Graph building finished in {time.time() - start_time:.2f} seconds."
            )

            start_time = time.time()
            logger.info("Parent agent processing graphs...")
            self.processed_triplets_graph, _ = self.parent_agent.process_graph(
                triplets_graph, self.config.get("graph")
            )
            self.processed_cosine_graph = self.parent_agent.apply_topics_to_graph(
                cosine_graph, self.processed_triplets_graph
            )
            logger.info(
                f"Parent agent finished processing graphs in {time.time() - start_time:.2f} seconds."
            )

            self._save_graphs(
                self.processed_cosine_graph, self.processed_triplets_graph
            )
            # Org table is saved within process_graph, so no need to save again

        elif loaded_existing_build and not org_table_loaded_from_file:
            logger.info(
                "Graphs loaded, but org table not loaded from file. Generating org table now..."
            )
            self.parent_agent._populate_org_table(self.processed_triplets_graph)
            self.parent_agent.org_table.save_to_json(self.org_table_path)

        start_time = time.time()
        logger.info("Creating Ollama child agents from processed graph...")
        child_agents = self._create_ollama_child_agents(self.processed_triplets_graph)
        logger.info(
            f"Child agent creation finished in {time.time() - start_time:.2f} seconds."
        )

        self.parent_agent.set_child_agents(child_agents)
        logger.info("Parent agent has been updated with child agent references.")

        start_time = time.time()
        logger.info("Visualizing graphs...")
        vis_cosine_path = os.path.abspath(
            os.path.join(self.output_dir, "output_cosine.html")
        )
        visualizer_cosine = GraphVisualizer(self.processed_cosine_graph)
        visualizer_cosine.visualize(output_path=vis_cosine_path)

        vis_triplets_path = os.path.abspath(
            os.path.join(self.output_dir, "output_triplets.html")
        )
        visualizer_triplets = GraphVisualizer(self.processed_triplets_graph)
        visualizer_triplets.visualize(output_path=vis_triplets_path)
        logger.info(
            f"Graph visualization finished in {time.time() - start_time:.2f} seconds."
        )

        run_modes = self.config.get("output", {}).get("run_mode", ["web"])
        if not isinstance(run_modes, list):
            run_modes = [run_modes]

        def run_flask():
            logger.info("Starting Flask app in a background thread...")
            app = create_app(
                self.parent_agent,
                child_agents,
                {"cosine": vis_cosine_path, "triplets": vis_triplets_path},
                self.search_similarity,
            )
            app.run(host="0.0.0.0", debug=False, use_reloader=False)

        if "web" in run_modes:
            web_thread = threading.Thread(target=run_flask, daemon=True)
            web_thread.start()
            logger.info("Web server thread started.")

        if "cli" in run_modes:
            logger.info("Starting interactive CLI...")
            cli = AgentCLI(self, self.parent_agent, self.parent_agent.org_table)
            cli.cmdloop()
        elif "web" in run_modes:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down web server.")

        logger.info("Runner finished.")

    def _create_ollama_child_agents(self, graph):
        logger.info("Creating Ollama child agents...")
        agents = {}
        if not graph:
            logger.error("Cannot create child agents from an empty graph.")
            return agents

        for topic_node, topic_data in graph.nodes(data=True):
            if topic_data.get("type") == "topic":
                topic = topic_data.get("content")
                subgraph = nx.Graph()
                nodes_in_topic = [topic_node]
                for neighbor in graph.neighbors(topic_node):
                    nodes_in_topic.append(neighbor)
                    subgraph.add_node(neighbor, **graph.nodes[neighbor])

                subgraph.add_node(topic_node, **topic_data)
                subgraph.add_edges_from(graph.edges(nodes_in_topic, data=True))

                agents[topic] = OllamaAgent(
                    subgraph,
                    topic,
                    self.graph_builder.embedding_model,
                    self.graph_builder.embedding_tokenizer,
                    self.search_similarity,
                    self.search_sample_ratio,
                    self.ollama_config,
                )
        logger.info(f"{len(agents)} Ollama child agents created.")
        return agents

    def _create_lm_studio_child_agent_data(self, graph):
        logger.info("Creating LM Studio child agent data...")
        agent_data = {}
        for topic_node, topic_data in graph.nodes(data=True):
            if topic_data.get("type") == "topic":
                topic = topic_data.get("content")
                subgraph = nx.Graph()
                nodes_in_topic = [topic_node]
                for neighbor in graph.neighbors(topic_node):
                    nodes_in_topic.append(neighbor)
                    subgraph.add_node(neighbor, **graph.nodes[neighbor])

                subgraph.add_node(topic_node, **topic_data)
                subgraph.add_edges_from(graph.edges(nodes_in_topic, data=True))

                agent_data[topic] = {"subgraph": subgraph}
        logger.info(f"{len(agent_data)} child agent data created.")
        return agent_data

    def _create_lm_studio_query_router(self, parent_model, child_agent_data):
        return LMStudioQueryRouter(
            parent_model,
            child_agent_data,
            self.graph_builder.embedding_model,
            self.graph_builder.embedding_tokenizer,
            self.search_similarity,
            self.search_sample_ratio,
        )

    def reconsolidate_and_reload(self):
        logger.info("Force reconsolidating topics and reloading agents...")
        if self.infra_type != "strands":
            logger.error(
                f"Reconsolidation is not supported for infra_type '{self.infra_type}'."
            )
            print(
                "Error: This command is currently only supported for the 'strands' infrastructure."
            )
            return None

        if not self.parent_agent or not self.processed_triplets_graph:
            logger.error(
                "Cannot reconsolidate without a loaded parent agent and graph."
            )
            print(
                "Error: This command can only be run after a build is loaded or created."
            )
            return

        # 1. Consolidate topics
        start_time = time.time()
        logger.info("Consolidating topics...")
        self.processed_triplets_graph = self.parent_agent.consolidate_topics(
            self.processed_triplets_graph
        )
        # Apply the new topic structure to the other graph as well
        if self.processed_cosine_graph:
            self.processed_cosine_graph = self.parent_agent.apply_topics_to_graph(
                self.processed_cosine_graph, self.processed_triplets_graph
            )
        end_time = time.time()
        logger.info(
            f"Topic consolidation finished in {end_time - start_time:.2f} seconds."
        )

        # 2. Re-create agents
        start_time = time.time()
        logger.info("Re-creating child agents from new topic structure...")
        child_agents = self._create_child_agents(
            self.processed_triplets_graph,
            self.parent_agent,
            self.parent_agent.org_table,
            self.debug_config,
        )
        self.parent_agent.set_child_agents(child_agents)
        self._save_agents(child_agents)  # Save updated agents
        end_time = time.time()
        logger.info(
            f"Child agent re-creation finished in {end_time - start_time:.2f} seconds."
        )

        # 3. Re-visualize graphs
        start_time = time.time()
        logger.info("Re-visualizing graphs...")
        vis_cosine_path = os.path.abspath(
            os.path.join(self.output_dir, "output_cosine.html")
        )
        visualizer_cosine = GraphVisualizer(self.processed_cosine_graph)
        visualizer_cosine.visualize(output_path=vis_cosine_path)

        vis_triplets_path = os.path.abspath(
            os.path.join(self.output_dir, "output_triplets.html")
        )
        visualizer_triplets = GraphVisualizer(self.processed_triplets_graph)
        visualizer_triplets.visualize(output_path=vis_triplets_path)
        end_time = time.time()
        logger.info(
            f"Graph re-visualization finished in {end_time - start_time:.2f} seconds."
        )

        logger.info("Reconsolidation and reload complete. CLI and agents are updated.")
        return child_agents

    def run_strands(self):
        logger.info("Starting runner...")

        output_config = self.config.get("output", {})
        build_version = output_config.get("version", "timestamp")

        processed_cosine_graph, processed_triplets_graph = None, None
        loaded_existing_build = False

        # Create the parent agent first
        parent_agent = self._create_parent_agent()
        org_table_loaded_from_file = self._handle_org_table_loading(parent_agent)

        # Attempt to load the graphs
        if build_version != "timestamp" and os.path.exists(self.output_dir):
            logger.info(f"Attempting to load existing build '{build_version}'...")
            processed_cosine_graph, processed_triplets_graph = self._load_graphs()
            if processed_cosine_graph and processed_triplets_graph:
                loaded_existing_build = True
                logger.info(f"Successfully loaded existing build '{build_version}'.")
            else:
                logger.warning(
                    f"Could not load graphs from existing build '{build_version}'. Proceeding with full build."
                )

        # If full build is needed, run graph processing which creates the org table
        if not loaded_existing_build:
            start_time = time.time()
            logger.info("Building graphs from scratch...")
            cosine_graph, triplets_graph = self.graph_builder.build()
            end_time = time.time()
            logger.info(
                f"Graph building finished in {end_time - start_time:.2f} seconds."
            )

            start_time = time.time()
            logger.info("Parent agent processing graphs...")
            # process_graph populates and saves the org table
            processed_triplets_graph, _ = parent_agent.process_graph(
                triplets_graph, self.config.get("graph")
            )
            processed_cosine_graph = parent_agent.apply_topics_to_graph(
                cosine_graph, processed_triplets_graph
            )
            end_time = time.time()
            logger.info(
                f"Parent agent finished processing graphs in {end_time - start_time:.2f} seconds."
            )

            self._save_graphs(processed_cosine_graph, processed_triplets_graph)

        # If graphs were loaded but org_table wasn't, we must generate it.
        elif loaded_existing_build and not org_table_loaded_from_file:
            logger.info(
                "Graphs loaded, but org table not loaded from file. Generating org table now..."
            )
            parent_agent._populate_org_table(processed_triplets_graph)
            parent_agent.org_table.save_to_json(self.org_table_path)

        start_time = time.time()
        logger.info("Creating/loading child agents from processed graph...")
        child_agents = self._create_child_agents(
            processed_triplets_graph,
            parent_agent,
            parent_agent.org_table,
            self.debug_config,
        )
        self._save_agents(child_agents)
        end_time = time.time()
        logger.info(
            f"Child agent loading/creation finished in {end_time - start_time:.2f} seconds."
        )

        parent_agent.set_child_agents(child_agents)
        logger.info("Parent agent has been updated with child agent references.")

        start_time = time.time()
        logger.info("Visualizing graphs...")
        vis_cosine_path = os.path.abspath(
            os.path.join(self.output_dir, "output_cosine.html")
        )
        visualizer_cosine = GraphVisualizer(processed_cosine_graph)
        visualizer_cosine.visualize(output_path=vis_cosine_path)
        logger.info(f"Cosine graph visualization saved to {vis_cosine_path}")

        vis_triplets_path = os.path.abspath(
            os.path.join(self.output_dir, "output_triplets.html")
        )
        visualizer_triplets = GraphVisualizer(processed_triplets_graph)
        visualizer_triplets.visualize(output_path=vis_triplets_path)
        logger.info(f"Triplets graph visualization saved to {vis_triplets_path}")
        end_time = time.time()
        logger.info(
            f"Graph visualization finished in {end_time - start_time:.2f} seconds."
        )

        run_modes = self.config.get("output", {}).get("run_mode", ["web"])
        if not isinstance(run_modes, list):
            run_modes = [run_modes]

        # Target for the web app thread
        def run_flask():
            logger.info("Starting Flask app in a background thread...")
            app = create_app(
                parent_agent,
                child_agents,
                {"cosine": vis_cosine_path, "triplets": vis_triplets_path},
                self.search_similarity,
            )
            app.run(
                host="0.0.0.0", debug=False, use_reloader=False
            )  # Debug mode is not thread-safe with cmdloop

        # Start the web app in a thread if requested
        if "web" in run_modes:
            web_thread = threading.Thread(target=run_flask, daemon=True)
            web_thread.start()
            logger.info("Web server thread started.")

        # Start the CLI in the main thread if requested
        if "cli" in run_modes:
            logger.info("Starting interactive CLI...")
            cli = AgentCLI(self, parent_agent, parent_agent.org_table)
            cli.cmdloop()
        elif "web" in run_modes:
            # If only web is running, the main thread needs to stay alive.
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down web server.")

        logger.info("Runner finished.")

    def _create_child_agents(self, graph, parent_agent, org_table, debug_config):
        if self.infra_type != "strands":
            logger.warning(
                f"'_create_child_agents' is for 'strands' infra. You are using '{self.infra_type}'. Returning empty agents."
            )
            return {}

        logger.info("Creating child agents...")
        agents = {}
        for topic_node, topic_data in graph.nodes(data=True):
            if topic_data.get("type") == "topic":
                topic = topic_data.get("content")
                subgraph = nx.Graph()
                nodes_in_topic = [topic_node]
                for neighbor in graph.neighbors(topic_node):
                    nodes_in_topic.append(neighbor)
                    subgraph.add_node(neighbor, **graph.nodes[neighbor])

                subgraph.add_node(topic_node, **topic_data)
                subgraph.add_edges_from(graph.edges(nodes_in_topic, data=True))

                agents[topic] = Agent(
                    subgraph,
                    topic,
                    self.graph_builder.embedding_model,
                    self.graph_builder.embedding_tokenizer,
                    self.search_similarity,
                    self.search_sample_ratio,
                    self.strands_config,
                    self.child_agent_model_id,
                    parent_agent,
                    org_table,
                    debug_config,
                )
        logger.info(f"{len(agents)} child agents created.")
        return agents

    def _create_parent_agent(self):
        if self.infra_type == "strands":
            return ParentAgent(
                infra_type="strands",
                llm_config=self.strands_config,
                debug_config=self.debug_config,
            )
        elif self.infra_type == "ollama":
            return ParentAgent(
                infra_type="ollama",
                llm_config=self.ollama_config,
                debug_config=self.debug_config,
            )
        else:
            raise ValueError(
                f"Unsupported infra type for ParentAgent: {self.infra_type}"
            )

    def _save_graphs(self, processed_cosine_graph, processed_triplets_graph):
        logger.info("Saving graphs to disk...")
        with open(self.processed_cosine_graph_path, "wb") as f:
            pickle.dump(processed_cosine_graph, f)
        with open(self.processed_triplets_graph_path, "wb") as f:
            pickle.dump(processed_triplets_graph, f)
        logger.info("Graphs saved successfully.")

    def _load_graphs(self):
        logger.info("Attempting to load graphs from previous run...")
        if os.path.exists(self.processed_cosine_graph_path) and os.path.exists(
            self.processed_triplets_graph_path
        ):
            try:
                with open(self.processed_cosine_graph_path, "rb") as f:
                    processed_cosine_graph = pickle.load(f)
                with open(self.processed_triplets_graph_path, "rb") as f:
                    processed_triplets_graph = pickle.load(f)
                logger.info("Graphs loaded successfully.")
                return processed_cosine_graph, processed_triplets_graph
            except Exception as e:
                logger.error(f"Error loading graphs from files: {e}")
                return None, None
        logger.info("No saved graphs found.")
        return None, None

    def _save_agents(self, agents):
        if self.infra_type != "strands":
            logger.warning(
                f"Agent saving is not supported for infra_type '{self.infra_type}'."
            )
            return
        logger.info("Saving agents to disk...")
        agent_states = {topic: agent.get_state() for topic, agent in agents.items()}
        with open(self.agents_path, "wb") as f:
            pickle.dump(agent_states, f)
        logger.info("Agents saved successfully.")

    def _load_agents(self, parent_agent, org_table):
        if self.infra_type != "strands":
            logger.warning(
                f"Agent loading is not supported for infra_type '{self.infra_type}'. Agents will be recreated."
            )
            return None
        logger.info("Attempting to load agents from previous run...")
        if os.path.exists(self.agents_path):
            try:
                with open(self.agents_path, "rb") as f:
                    agent_states = pickle.load(f)

                agents = {}
                child_agent_model_id = self.strands_config.get("child_agent_model_id")
                for topic, state in agent_states.items():
                    agents[topic] = Agent.from_state(
                        state,
                        self.graph_builder.embedding_model,
                        self.graph_builder.embedding_tokenizer,
                        child_agent_model_id,
                        parent_agent,
                        org_table,
                        self.debug_config,
                    )

                logger.info("Agents loaded successfully.")
                return agents
            except Exception as e:
                logger.error(f"Error loading agents from file: {e}")
                return None
        logger.info("No saved agents found.")
        return None
