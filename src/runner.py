import os
import pickle
import time
from src.config_loader import ConfigLoader
from src.graph_builder import GraphBuilder
from src.graph_visualizer import GraphVisualizer
from src.parent_model import ParentModel, LMStudioQueryRouter
from src.agent import Agent
from src.parent_agent import ParentAgent
from src.app import create_app
from src.logger import get_logger
import networkx as nx
from datetime import datetime

logger = get_logger(__name__)

class Runner:
    def __init__(self):
        logger.info("Initializing runner...")
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config()
        self.graph_builder = GraphBuilder(self.config.get('data'), self.config.get('graph'))
        
        self.agent_model_config = self.config.get('agent_model', {})
        self.search_sample_ratio = self.agent_model_config.get('search_sample_ratio', 0.4)
        self.enable_consolidation = self.config.get('graph', {}).get('enable_consolidation', True)
        self.search_similarity = self.config.get('graph', {}).get('search_similarity', 0.6)
        self.llm_infra_config = self.config.get('llm_infra', {})
        self.infra_type = self.llm_infra_config.get('type', 'strands')
        self.debug_config = self.config.get('debug', {})

        if self.infra_type == 'strands':
            self.strands_config = self.llm_infra_config.get('strands', {})
            self.parent_agent_model_id = self.strands_config.get('parent_agent_model_id')
            self.child_agent_model_id = self.strands_config.get('child_agent_model_id')
        else:
            self.lm_studio_config = self.llm_infra_config.get('lm_studio', {})
        
        output_config = self.config.get('output', {})
        base_output_dir = output_config.get('base_dir', 'data/builds')
        build_version = output_config.get('version', 'timestamp')
        self.serve_only_mode = output_config.get('serve_only_mode', False)

        if build_version == 'timestamp' and not self.serve_only_mode:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.output_dir = os.path.join(base_output_dir, timestamp)
        else:
            self.output_dir = os.path.join(base_output_dir, build_version)

        logger.info(f"Output directory set to: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.processed_cosine_graph_path = os.path.join(self.output_dir, "processed_cosine_graph.pkl")
        self.processed_triplets_graph_path = os.path.join(self.output_dir, "processed_triplets_graph.pkl")
        self.agents_path = os.path.join(self.output_dir, "agents.pkl")
        logger.info("Runner initialized.")

    def run(self):
        if self.serve_only_mode:
            self.run_serve_only()
            return

        logger.info(f"Starting runner with infra type: {self.infra_type}...")

        if self.infra_type == 'strands':
            self.run_strands()
        else:
            self.run_lm_studio()

    def run_serve_only(self):
        logger.info(f"Starting in serve-only mode from build directory: {self.output_dir}")

        graphs = self._load_graphs()
        if not graphs:
            logger.error("Could not load graphs. Cannot start in serve-only mode.")
            return
        
        processed_cosine_graph, processed_triplets_graph = graphs

        child_agents = self._load_agents()
        if not child_agents:
            logger.error("Could not load agents. Cannot start in serve-only mode.")
            return

        logger.info("Creating parent agent...")
        parent_agent = self._create_parent_agent(child_agents, self.debug_config)
        logger.info("Parent agent created.")

        vis_cosine_path = os.path.abspath(os.path.join(self.output_dir, "output_cosine.html"))
        vis_triplets_path = os.path.abspath(os.path.join(self.output_dir, "output_triplets.html"))

        if not os.path.exists(vis_cosine_path) or not os.path.exists(vis_triplets_path):
            logger.warning("Visualization files not found. They will not be available in the UI.")

        logger.info("Starting Flask app...")
        app = create_app(parent_agent, child_agents, {"cosine": vis_cosine_path, "triplets": vis_triplets_path}, self.search_similarity)
        app.run(debug=True, use_reloader=False)
        logger.info("Runner finished.")


    def run_lm_studio(self):
        # LM Studio-based implementation
        logger.info("Running with LM Studio infrastructure...")

        output_config = self.config.get('output', {})
        build_version = output_config.get('version', 'timestamp')
        
        processed_cosine_graph = None
        processed_triplets_graph = None
        loaded_existing_build = False

        if build_version != 'timestamp' and os.path.exists(self.output_dir):
            logger.info(f"Attempting to load existing build '{build_version}'...")
            graphs = self._load_graphs()
            if graphs:
                processed_cosine_graph, processed_triplets_graph = graphs
                loaded_existing_build = True
                logger.info(f"Successfully loaded existing build '{build_version}'.")
            else:
                logger.warning(f"Could not load graphs from existing build '{build_version}'. Proceeding with full build.")
        
        parent_model = ParentModel(self.lm_studio_config)

        if not loaded_existing_build:
            start_time = time.time()
            logger.info("Building graphs from scratch...")
            cosine_graph, triplets_graph = self.graph_builder.build()
            end_time = time.time()
            logger.info(f"Graph building finished in {end_time - start_time:.2f} seconds.")
            
            start_time = time.time()
            logger.info("Parent model sorting graph into subgroups...")
            processed_cosine_graph = parent_model.sort_into_subgroups(cosine_graph)
            processed_triplets_graph = parent_model.sort_into_subgroups(triplets_graph)
            end_time = time.time()
            logger.info(f"Parent model finished sorting in {end_time - start_time:.2f} seconds.")

            if self.enable_consolidation:
                start_time = time.time()
                logger.info("Consolidating topics...")
                processed_cosine_graph = parent_model.consolidate_topics(processed_cosine_graph)
                processed_triplets_graph = parent_model.consolidate_topics(processed_triplets_graph)
                end_time = time.time()
                logger.info(f"Topic consolidation finished in {end_time - start_time:.2f} seconds.")

            self._save_graphs(processed_cosine_graph, processed_triplets_graph)

        child_agent_data = self._create_lm_studio_child_agent_data(processed_triplets_graph)

        query_router = self._create_lm_studio_query_router(parent_model, child_agent_data)

        start_time = time.time()
        logger.info("Visualizing graphs...")
        vis_cosine_path = os.path.abspath(os.path.join(self.output_dir, "output_cosine.html"))
        visualizer_cosine = GraphVisualizer(processed_cosine_graph)
        visualizer_cosine.visualize(output_path=vis_cosine_path)
        logger.info(f"Cosine graph visualization saved to {vis_cosine_path}")

        vis_triplets_path = os.path.abspath(os.path.join(self.output_dir, "output_triplets.html"))
        visualizer_triplets = GraphVisualizer(processed_triplets_graph)
        visualizer_triplets.visualize(output_path=vis_triplets_path)
        logger.info(f"Triplets graph visualization saved to {vis_triplets_path}")
        end_time = time.time()
        logger.info(f"Graph visualization finished in {end_time - start_time:.2f} seconds.")

        start_time = time.time()
        logger.info("Starting Flask app...")
        app = create_app(query_router, child_agent_data, {"cosine": vis_cosine_path, "triplets": vis_triplets_path}, self.search_similarity)
        app.run(debug=True, use_reloader=False)
        end_time = time.time()
        logger.info(f"Flask app startup call finished in {end_time - start_time:.2f} seconds.")

        logger.info("Runner finished.")

    def _create_lm_studio_child_agent_data(self, graph):
        logger.info("Creating LM Studio child agent data...")
        agent_data = {}
        for topic_node, topic_data in graph.nodes(data=True):
            if topic_data.get('type') == 'topic':
                topic = topic_data.get('content')
                subgraph = nx.Graph()
                nodes_in_topic = [topic_node]
                for neighbor in graph.neighbors(topic_node):
                    nodes_in_topic.append(neighbor)
                    subgraph.add_node(neighbor, **graph.nodes[neighbor])
                
                subgraph.add_node(topic_node, **topic_data)
                subgraph.add_edges_from(graph.edges(nodes_in_topic, data=True))

                agent_data[topic] = {'subgraph': subgraph}
        logger.info(f"{len(agent_data)} child agent data created.")
        return agent_data

    def _create_lm_studio_query_router(self, parent_model, child_agent_data):
        return LMStudioQueryRouter(
            parent_model,
            child_agent_data,
            self.graph_builder.embedding_model,
            self.graph_builder.embedding_tokenizer,
            self.search_similarity,
            self.search_sample_ratio
        )

    def run_strands(self):
        logger.info("Starting runner...")

        output_config = self.config.get('output', {})
        build_version = output_config.get('version', 'timestamp')
        
        processed_cosine_graph = None
        processed_triplets_graph = None
        loaded_existing_build = False

        if build_version != 'timestamp' and os.path.exists(self.output_dir):
            logger.info(f"Attempting to load existing build '{build_version}' for re-consolidation...")
            graphs = self._load_graphs()
            if graphs:
                processed_cosine_graph, processed_triplets_graph = graphs
                loaded_existing_build = True
                logger.info(f"Successfully loaded existing build '{build_version}'.")
            else:
                logger.warning(f"Could not load graphs from existing build '{build_version}'. Proceeding with full build.")
        
        if not loaded_existing_build:
            start_time = time.time()
            logger.info("Building graphs from scratch...")
            cosine_graph, triplets_graph = self.graph_builder.build()
            end_time = time.time()
            logger.info(f"Graph building finished in {end_time - start_time:.2f} seconds.")
            
            start_time = time.time()
            logger.info("Parent agent processing graphs...")
            temp_parent_agent = ParentAgent({}, self.strands_config, self.parent_agent_model_id, self.debug_config)

            processed_cosine_graph = temp_parent_agent.process_graph(cosine_graph, self.config.get('graph'))
            processed_triplets_graph = temp_parent_agent.process_graph(triplets_graph, self.config.get('graph'))
            end_time = time.time()
            logger.info(f"Parent agent finished processing graphs in {end_time - start_time:.2f} seconds.")

            self._save_graphs(processed_cosine_graph, processed_triplets_graph)
        else:
            temp_parent_agent = ParentAgent({}, self.strands_config, self.parent_agent_model_id, self.debug_config)

            if self.enable_consolidation:
                start_time = time.time()
                logger.info("Re-consolidating topics on loaded graphs...")
                processed_cosine_graph = temp_parent_agent.process_graph(processed_cosine_graph, self.config.get('graph'))
                processed_triplets_graph = temp_parent_agent.process_graph(processed_triplets_graph, self.config.get('graph'))
                end_time = time.time()
                logger.info(f"Topic re-consolidation finished in {end_time - start_time:.2f} seconds.")
            else:
                logger.info("Topic re-consolidation skipped as per configuration.")
            self._save_graphs(processed_cosine_graph, processed_triplets_graph)

        start_time = time.time()
        logger.info("Creating child agents from processed graph...")
        child_agents = self._create_child_agents(processed_triplets_graph, self.debug_config)
        self._save_agents(child_agents)
        end_time = time.time()
        logger.info(f"Child agent loading/creation finished in {end_time - start_time:.2f} seconds.")

        logger.info("Creating parent agent...")
        parent_agent = self._create_parent_agent(child_agents, self.debug_config)
        logger.info("Parent agent created.")

        start_time = time.time()
        logger.info("Visualizing graphs...")
        vis_cosine_path = os.path.abspath(os.path.join(self.output_dir, "output_cosine.html"))
        visualizer_cosine = GraphVisualizer(processed_cosine_graph)
        visualizer_cosine.visualize(output_path=vis_cosine_path)
        logger.info(f"Cosine graph visualization saved to {vis_cosine_path}")

        vis_triplets_path = os.path.abspath(os.path.join(self.output_dir, "output_triplets.html"))
        visualizer_triplets = GraphVisualizer(processed_triplets_graph)
        visualizer_triplets.visualize(output_path=vis_triplets_path)
        logger.info(f"Triplets graph visualization saved to {vis_triplets_path}")
        end_time = time.time()
        logger.info(f"Graph visualization finished in {end_time - start_time:.2f} seconds.")

        start_time = time.time()
        logger.info("Starting Flask app...")
        app = create_app(parent_agent, child_agents, {"cosine": vis_cosine_path, "triplets": vis_triplets_path}, self.search_similarity)
        app.run(debug=True, use_reloader=False)
        end_time = time.time()
        logger.info(f"Flask app startup call finished in {end_time - start_time:.2f} seconds.")

        logger.info("Runner finished.")

    def _create_child_agents(self, graph, debug_config):
        logger.info("Creating child agents...")
        agents = {}
        for topic_node, topic_data in graph.nodes(data=True):
            if topic_data.get('type') == 'topic':
                topic = topic_data.get('content')
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
                    debug_config
                )
        logger.info(f"{len(agents)} child agents created.")
        return agents

    def _create_parent_agent(self, child_agents, debug_config):
        return ParentAgent(child_agents, self.strands_config, self.parent_agent_model_id, debug_config)

    def _save_graphs(self, processed_cosine_graph, processed_triplets_graph):
        logger.info("Saving graphs to disk...")
        with open(self.processed_cosine_graph_path, 'wb') as f:
            pickle.dump(processed_cosine_graph, f)
        with open(self.processed_triplets_graph_path, 'wb') as f:
            pickle.dump(processed_triplets_graph, f)
        logger.info("Graphs saved successfully.")

    def _load_graphs(self):
        logger.info("Attempting to load graphs from previous run...")
        if os.path.exists(self.processed_cosine_graph_path) and os.path.exists(self.processed_triplets_graph_path):
            try:
                with open(self.processed_cosine_graph_path, 'rb') as f:
                    processed_cosine_graph = pickle.load(f)
                with open(self.processed_triplets_graph_path, 'rb') as f:
                    processed_triplets_graph = pickle.load(f)
                logger.info("Graphs loaded successfully.")
                return processed_cosine_graph, processed_triplets_graph
            except Exception as e:
                logger.error(f"Error loading graphs from files: {e}")
                return None
        logger.info("No saved graphs found.")
        return None

    def _save_agents(self, agents):
        logger.info("Saving agents to disk...")
        agent_states = {topic: agent.get_state() for topic, agent in agents.items()}
        with open(self.agents_path, 'wb') as f:
            pickle.dump(agent_states, f)
        logger.info("Agents saved successfully.")

    def _load_agents(self):
        logger.info("Attempting to load agents from previous run...")
        if os.path.exists(self.agents_path):
            try:
                with open(self.agents_path, 'rb') as f:
                    agent_states = pickle.load(f)
                
                agents = {}
                for topic, state in agent_states.items():
                    agents[topic] = Agent.from_state(state, self.graph_builder.embedding_model, self.graph_builder.embedding_tokenizer, self.child_agent_model_id, self.debug_config)

                logger.info("Agents loaded successfully.")
                return agents
            except Exception as e:
                logger.error(f"Error loading agents from file: {e}")
                return None
        logger.info("No saved agents found.")
        return None