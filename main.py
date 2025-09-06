import os
import logging
from src.runner import Runner
from dotenv import load_dotenv
from src.config_loader import ConfigLoader

if __name__ == "__main__":
    # Suppress verbose logging from the strands library
    logging.getLogger("strands").setLevel(logging.CRITICAL)

    load_dotenv()

    # Load the config to get the model ID
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    strands_config = config.get('strands', {})
    model_id = strands_config.get('model_id')

    if model_id:
        os.environ['STRANDS_MODEL_ID'] = model_id

    runner = Runner()
    runner.run()