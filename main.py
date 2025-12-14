import os
import logging
from src.runner import Runner
from dotenv import load_dotenv
from src.config_loader import ConfigLoader

if __name__ == "__main__":
    # Configure root logger to capture all logs and output to console
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Configure logging for the 'strands' library to redirect to a file
    strands_logger = logging.getLogger("strands")
    strands_logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Create a file handler
    file_handler = logging.FileHandler("logs/strands.log", mode="w")
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    strands_logger.addHandler(file_handler)

    # Prevent the logger from propagating to the root logger
    strands_logger.propagate = False

    load_dotenv()

    # Load the config to get the model ID
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    strands_config = config.get("strands", {})
    model_id = strands_config.get("model_id")

    if model_id:
        os.environ["STRANDS_MODEL_ID"] = model_id

    runner = Runner()
    runner.run()
