import os
import pickle
import time # New line
from src.config_loader import ConfigLoader
from src.graph_builder import GraphBuilder
from src.graph_visualizer import GraphVisualizer
from src.parent_model import ParentModel
from src.agent import Agent
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
        self.parent_model = ParentModel(self.config.get('parent_model'))
        self.agent_model_config = self.config.get('agent_model', {})
        self.agent_model_name = self.agent_model_config.get('model')
        self.search_sample_ratio = self.agent_model_config.get('search_sample_ratio', 0.4)
        self.enable_consolidation = self.config.get('graph', {}).get('enable_consolidation', True) # New line
        
        output_config = self.config.get('output', {})
        base_output_dir = output_config.get('base_dir', 'data/builds')
        build_version = output_config.get('version', 'timestamp')

        if build_version == 'timestamp':
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.output_dir = os.path.join(base_output_dir, timestamp)
        else:
            self.output_dir = os.path.join(base_output_dir, build_version)

        logger.info(f"Output directory set to: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Define file paths for saving state
        self.processed_cosine_graph_path = os.path.join(self.output_dir, "processed_cosine_graph.pkl")
        self.processed_triplets_graph_path = os.path.join(self.output_dir, "processed_triplets_graph.pkl")
        self.agents_path = os.path.join(self.output_dir, "agents.pkl")
        logger.info("Runner initialized.")

    def run(self):
        logger.info("Starting runner...")

        # Determine if we should load an existing build for re-consolidation
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
            start_time = time.time() # New line
            logger.info("Building graphs from scratch...")
            cosine_graph, triplets_graph = self.graph_builder.build()
            end_time = time.time() # New line
            logger.info(f"Graph building finished in {end_time - start_time:.2f} seconds.") # New line
            
            start_time = time.time() # New line
            logger.info("Parent model processing graphs...")
            processed_cosine_graph = self.parent_model.sort_into_subgroups(cosine_graph)
            processed_triplets_graph = self.parent_model.sort_into_subgroups(triplets_graph)
            end_time = time.time() # New line
            logger.info(f"Parent model finished processing graphs in {end_time - start_time:.2f} seconds.") # New line

            if self.enable_consolidation: # New line
                start_time = time.time() # New line
                logger.info("Consolidating topics...")
                processed_cosine_graph = self.parent_model.consolidate_topics(processed_cosine_graph)
                processed_triplets_graph = self.parent_model.consolidate_topics(processed_triplets_graph)
                end_time = time.time() # New line
                logger.info(f"Topic consolidation finished in {end_time - start_time:.2f} seconds.") # New line
            else: # New line
                logger.info("Topic consolidation skipped as per configuration.") # New line

            self._save_graphs(processed_cosine_graph, processed_triplets_graph)
        else:
            if self.enable_consolidation: # New line
                start_time = time.time() # New line
                logger.info("Re-consolidating topics on loaded graphs...")
                processed_cosine_graph = self.parent_model.consolidate_topics(processed_cosine_graph)
                processed_triplets_graph = self.parent_model.consolidate_topics(processed_triplets_graph)
                end_time = time.time() # New line
                logger.info(f"Topic re-consolidation finished in {end_time - start_time:.2f} seconds.") # New line
            else: # New line
                logger.info("Topic re-consolidation skipped as per configuration.") # New line
            self._save_graphs(processed_cosine_graph, processed_triplets_graph) # Save after re-consolidation

        # Load or create agents
        start_time = time.time() # New line
        agents = self._load_agents()
        if not agents:
            logger.info("Creating agents from loaded graphs...")
            agents = self._create_agents(processed_triplets_graph)
            self._save_agents(agents)
        end_time = time.time() # New line
        logger.info(f"Agent loading/creation finished in {end_time - start_time:.2f} seconds.") # New line

        # Visualize the graphs
        start_time = time.time() # New line
        logger.info("Visualizing graphs...")
        vis_cosine_path = os.path.abspath(os.path.join(self.output_dir, "output_cosine.html"))
        visualizer_cosine = GraphVisualizer(processed_cosine_graph)
        visualizer_cosine.visualize(output_path=vis_cosine_path)
        logger.info(f"Cosine graph visualization saved to {vis_cosine_path}")

        vis_triplets_path = os.path.abspath(os.path.join(self.output_dir, "output_triplets.html"))
        visualizer_triplets = GraphVisualizer(processed_triplets_graph)
        visualizer_triplets.visualize(output_path=vis_triplets_path)
        logger.info(f"Triplets graph visualization saved to {vis_triplets_path}")
        end_time = time.time() # New line
        logger.info(f"Graph visualization finished in {end_time - start_time:.2f} seconds.") # New line

        # Start the Flask app
        start_time = time.time() # New line
        logger.info("Starting Flask app...")
        app = create_app(agents, {"cosine": vis_cosine_path, "triplets": vis_triplets_path})
        app.run(debug=True)
        end_time = time.time() # New line
        logger.info(f"Flask app startup call finished in {end_time - start_time:.2f} seconds.") # New line

        logger.info("Runner finished.")

    def _create_agents(self, graph):
        logger.info("Creating agents...")
        agents = {}
        # Extract topic subgraphs
        for topic_node, topic_data in graph.nodes(data=True):
            if topic_data.get('type') == 'topic':
                topic = topic_data.get('content')
                # Create a subgraph for each topic
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
                    self.agent_model_name,
                    self.graph_builder.embedding_model,
                    self.graph_builder.embedding_tokenizer,
                    self.graph_builder.similarity_threshold,
                    self.search_sample_ratio
                )
        logger.info(f"{len(agents)} agents created.")
        return agents

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
                    agents[topic] = Agent.from_state(state, self.graph_builder.embedding_model, self.graph_builder.embedding_tokenizer)

                logger.info("Agents loaded successfully.")
                return agents
            except Exception as e:
                logger.error(f"Error loading agents from file: {e}")
                return None
        logger.info("No saved agents found.")
        return None
