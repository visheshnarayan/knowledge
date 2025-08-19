from flask import Flask, render_template, request, jsonify
from src.logger import get_logger

logger = get_logger(__name__)

def create_app(agents, graph_path):
    logger.info("Creating Flask app...")
    app = Flask(__name__, template_folder='../templates')

    @app.route('/')
    def index():
        logger.info("Request received for index page.")
        return render_template('index.html', topics=list(agents.keys()), graph_path=graph_path)

    @app.route('/query', methods=['POST'])
    def query():
        logger.info("Request received for query.")
        data = request.get_json()
        topic = data.get('topic')
        question = data.get('question')
        logger.info(f"Querying agent for topic: '{topic}' with question: '{question}'")

        if not topic or not question:
            logger.error("Missing topic or question in query.")
            return jsonify({'error': 'Missing topic or question'}), 400

        agent = agents.get(topic)
        if not agent:
            logger.error(f"Agent for topic '{topic}' not found.")
            return jsonify({'error': 'Agent for the selected topic not found'}), 404

        response = agent.query(question)
        logger.info(f"Returning response from agent for topic: '{topic}'")
        return jsonify({'response': response})

    logger.info("Flask app created.")
    return app