import os
import pickle
from src.config_loader import ConfigLoader
from src.graph_builder import GraphBuilder
from src.graph_visualizer import GraphVisualizer
from src.parent_model import ParentModel
from src.agent import Agent
from src.app import create_app
from src.logger import get_logger
import networkx as nx

logger = get_logger(__name__)

class Runner:
    def __init__(self):
        logger.info("Initializing runner...")
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config()
        self.graph_builder = GraphBuilder(self.config.get('data'), self.config.get('graph'))
        self.parent_model = ParentModel(self.config.get('parent_model'))
        self.agent_model_name = self.config.get('agent_model', {}).get('model')
        self.output_dir = self.config.get('data', {}).get('output_dir', 'data/output')
        os.makedirs(self.output_dir, exist_ok=True)

        # Define file paths for saving state
        self.cosine_graph_path = os.path.join(self.output_dir, "cosine_graph.pkl")
        self.triplets_graph_path = os.path.join(self.output_dir, "triplets_graph.pkl")
        self.processed_cosine_graph_path = os.path.join(self.output_dir, "processed_cosine_graph.pkl")
        self.processed_triplets_graph_path = os.path.join(self.output_dir, "processed_triplets_graph.pkl")
        self.agents_path = os.path.join(self.output_dir, "agents.pkl")
        logger.info("Runner initialized.")

    def run(self):
        logger.info("Starting runner...")

        loaded_state = self._load_state()
        if loaded_state:
            logger.info("Loaded graphs and agents from previous run.")
            processed_cosine_graph, processed_triplets_graph, agents = loaded_state
        else:
            logger.info("No previous run found. Building graphs from scratch.")
            cosine_graph, triplets_graph = self.graph_builder.build()
            
            logger.info("Parent model processing graphs...")
            processed_cosine_graph = self.parent_model.sort_into_subgroups(cosine_graph)
            processed_triplets_graph = self.parent_model.sort_into_subgroups(triplets_graph)
            logger.info("Parent model finished processing graphs.")

            agents = self._create_agents(processed_triplets_graph)

            self._save_state(processed_cosine_graph, processed_triplets_graph, agents)

        # Visualize the graph
        logger.info("Visualizing graph...")
        visualizer = GraphVisualizer(processed_triplets_graph)
        vis_output_path = os.path.join(self.output_dir, "output_triplets.html")
        visualizer.visualize(output_path=vis_output_path)
        logger.info(f"Graph visualization saved to {vis_output_path}")

        # Start the Flask app
        logger.info("Starting Flask app...")
        app = create_app(agents, vis_output_path)
        app.run(debug=True)

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
                    self.graph_builder.embedding_tokenizer,
                    self.graph_builder.similarity_threshold,
                    self.search_sample_ratio
                )
        logger.info(f"{len(agents)} agents created.")
        return agents

    def _save_state(self, processed_cosine_graph, processed_triplets_graph, agents):
        logger.info("Saving graphs and agents to disk...")
        with open(self.processed_cosine_graph_path, 'wb') as f:
            pickle.dump(processed_cosine_graph, f)
        with open(self.processed_triplets_graph_path, 'wb') as f:
            pickle.dump(processed_triplets_graph, f)
        with open(self.agents_path, 'wb') as f:
            pickle.dump(agents, f)
        logger.info("Graphs and agents saved successfully.")

    def _load_state(self):
        logger.info("Attempting to load state from previous run...")
        if (
            os.path.exists(self.processed_cosine_graph_path)
            and os.path.exists(self.processed_triplets_graph_path)
            and os.path.exists(self.agents_path)
        ):
            with open(self.processed_cosine_graph_path, 'rb') as f:
                processed_cosine_graph = pickle.load(f)
            with open(self.processed_triplets_graph_path, 'rb') as f:
                processed_triplets_graph = pickle.load(f)
            with open(self.agents_path, 'rb') as f:
                agents = pickle.load(f)
            logger.info("State loaded successfully.")
            return processed_cosine_graph, processed_triplets_graph, agents
        logger.info("No saved state found.")
        return None
