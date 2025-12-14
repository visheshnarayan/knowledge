import os
import unittest
from src.runner import Runner
from src.logger import get_logger
from datetime import datetime
import logging

# Configure logging for the test
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class TestAgentCommunication(unittest.TestCase):

    def setUp(self):
        """Set up the runner and agents for the test."""
        logger.info("--- Setting up for Agent Communication Test ---")
        self.runner = Runner()

        # Ensure the build exists, run main.py if not
        if not os.path.exists(self.runner.output_dir):
            raise RuntimeError(
                f"Build directory not found at {self.runner.output_dir}. Please run main.py once to create a build."
            )

        logger.info("Loading graphs for test...")
        processed_cosine_graph, processed_triplets_graph = self.runner._load_graphs()
        if not processed_cosine_graph or not processed_triplets_graph:
            raise RuntimeError("Could not load graphs.")

        logger.info("Creating parent agent...")
        self.parent_agent = self.runner._create_parent_agent()

        # Populate org table from the loaded graph
        self.parent_agent._populate_org_table(processed_triplets_graph)

        logger.info("Creating child agents...")
        if self.runner.infra_type == 'strands':
            child_agents = self.runner._create_child_agents(
                processed_triplets_graph,
                self.parent_agent,
                self.parent_agent.org_table,
                self.runner.debug_config,
            )
        elif self.runner.infra_type == 'ollama':
            child_agents = self.runner._create_ollama_child_agents(processed_triplets_graph)
        else:
            raise ValueError(f"Unsupported infra type for child agent creation in test: {self.runner.infra_type}")
        self.parent_agent.set_child_agents(child_agents)
        logger.info("--- Setup complete ---")

    def test_query_routing_and_response(self):
        """
        Tests the full query lifecycle: decomposition, routing, and response generation
        by asking a series of questions and logging the results.
        """
        logger.info("--- Running test: test_query_routing_and_response ---")

        test_questions = [
            {
                "question": "Who are Mr. and Mrs. Dursley and where do they live?",
                "expected_topic": "Harry Potter series",
            },
            {
                "question": "Describe the architecture of an agentic graph management system.",
                "expected_topic": "Agentic Graph Management",
            },
            {
                "question": "How is natural language processing used in autonomous agents?",
                "expected_topic": "natural language processing, autonomous agents",
            },
            {
                "question": "What is the capital of France?",
                "expected_topic": "unclassified",
            },  # Assuming no agent for this
        ]

        log_dir = "tests/results"
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(
            log_dir,
            f"agent_communication_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        logger.info(f"Test results will be logged to: {log_file_path}")

        with open(log_file_path, "w") as f:
            for i, test_case in enumerate(test_questions):
                question = test_case["question"]
                expected_topic = test_case["expected_topic"]

                f.write(f"--- Test Case {i+1} ---\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Question: {question}\n")
                f.write(f"Expected Topic(s): {expected_topic}\n")

                logger.info(f"Querying parent agent with question: '{question}'")
                response, sources, routed_topics = self.parent_agent.query(question)

                f.write(
                    f"Routed Topic(s): {', '.join(routed_topics) if isinstance(routed_topics, list) else routed_topics}\n"
                )
                f.write(f"Agent Response: {response}\n")
                if sources:
                    f.write("Sources:\n")
                    for source in sources:
                        f.write(f"- {source}\n")
                f.write("-" * 50 + "\n\n")

                logger.info(f"Response received. Routed to: {routed_topics}")
                # Simple assertion to make it a real test
                self.assertIsNotNone(response, "Response should not be None")


if __name__ == "__main__":
    unittest.main()
