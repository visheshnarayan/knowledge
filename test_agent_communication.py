import os
from src.runner import Runner
from src.logger import get_logger
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('botocore').setLevel(logging.DEBUG)
logging.getLogger('strands').setLevel(logging.DEBUG)

logger = get_logger(__name__)

def test_parent_agent_communication():
    logger.info("Starting parent-child agent communication test...")

    # 1. Initialize the Runner
    runner = Runner()

    # 2. Load graphs
    graphs = runner._load_graphs()
    if not graphs:
        logger.error("Could not load graphs. Please run main.py once to build the graphs.")
        return
    processed_cosine_graph, processed_triplets_graph = graphs

    # 3. Load or create child agents
    child_agents = runner._load_agents()
    if not child_agents:
        logger.info("Creating child agents from loaded graphs...")
        child_agents = runner._create_child_agents(processed_triplets_graph)
    
    if not child_agents:
        logger.error("Could not create child agents.")
        return

    # 4. Create the parent agent
    logger.info("Creating parent agent...")
    parent_agent = runner._create_parent_agent(child_agents)
    logger.info("Parent agent created.")

    # Define test questions
    test_questions = [
        {"question": "Who are Mr. and Mrs. Dursley and where do they live?", "expected_topic": "harry potter"},
        {"question": "Who is Professor McGonagall and what did she do at Privet Drive?", "expected_topic": "harry potter"},
        {"question": "What is the main purpose of an Ingestion Agent in Agentic Graph Management?", "expected_topic": "agentic graph management"},
        {"question": "What is the capital of France?", "expected_topic": "unrelated"}
    ]

    # Create a log directory and file for test results
    log_dir = "test_results"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"agent_communication_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def log_conversation(log_file, question, topic, response, sources):
        log_file.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Question: {question}\n")
        log_file.write(f"Routed Topic: {topic}\n")
        log_file.write(f"Agent Response: {response}\n")
        if sources:
            log_file.write("Sources:\n")
            for source in sources:
                log_file.write(f"- {source}\n")
        log_file.write("-" * 50 + "\n\n")

    with open(log_file_path, "w") as f:
        for i, test_case in enumerate(test_questions):
            question = test_case["question"]
            logger.info(f"--- Test Case {i+1}: Querying parent agent with question: '{question}' ---")

            response, sources, routed_topic = parent_agent.query(question)

            logger.info(f"Response from parent agent: {response}")
            logger.info(f"Routed Topic: {routed_topic}")
            if sources:
                logger.info(f"Sources: {sources}")

            log_conversation(f, question, routed_topic, response, sources)

if __name__ == "__main__":
    test_parent_agent_communication()
