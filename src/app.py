from flask import Flask, render_template, request, jsonify, send_from_directory
from src.logger import get_logger
import os

logger = get_logger(__name__)

def create_app(parent_agent, child_agents, graph_paths, initial_search_similarity):
    logger.info("Creating Flask app...")
    app = Flask(__name__, template_folder='../templates')

    @app.route('/')
    def index():
        logger.info("Request received for index page.")
        return render_template('index.html', initial_search_similarity=initial_search_similarity)

    @app.route('/update_search_similarity', methods=['POST'])
    def update_search_similarity():
        data = request.get_json()
        new_similarity = data.get('similarity')
        
        if new_similarity is None:
            logger.error("Missing 'similarity' in update_search_similarity request.")
            return jsonify({'error': 'Missing similarity value'}), 400
        
        try:
            new_similarity = float(new_similarity)
            if not (0.0 <= new_similarity <= 1.0):
                raise ValueError("Similarity value must be between 0.0 and 1.0.")
        except ValueError as e:
            logger.error(f"Invalid similarity value: {new_similarity}. Error: {e}")
            return jsonify({'error': f'Invalid similarity value: {e}'}), 400

        for agent_name, agent_instance in child_agents.items():
            agent_instance.search_similarity = new_similarity
            logger.info(f"Updated search_similarity for agent '{agent_name}' to {new_similarity}.")
        
        return jsonify({'status': 'success', 'new_similarity': new_similarity})

    @app.route('/graph/<graph_type>')
    def graph(graph_type):
        logger.info(f"Request received for {graph_type} graph visualization.")
        graph_path = graph_paths.get(graph_type)
        if not graph_path or not os.path.exists(graph_path):
            logger.error(f"{graph_type} graph not found at path: {graph_path}")
            return "Graph not found", 404
        
        graph_dir = os.path.dirname(graph_path)
        graph_filename = os.path.basename(graph_path)
        return send_from_directory(graph_dir, graph_filename)

    @app.route('/query_agent', methods=['POST'])
    def query():
        logger.info("Request received for query.")
        data = request.get_json()
        question = data.get('question')
        logger.info(f"Querying parent agent with question: '{question}'")

        if not question:
            logger.error("Missing question in query.")
            return jsonify({'error': 'Missing question'}), 400

        response, sources, routed_topic = parent_agent.query(question)
        logger.info(f"Returning response from parent agent. Routed topic: {routed_topic}")
        return jsonify({'response': response, 'sources': sources, 'routed_topic': routed_topic})

    logger.info("Flask app created.")
    return app
