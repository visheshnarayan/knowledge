import yaml


class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = None  # Don't load on init

    def load_config(self):
        # Always re-read the file when called
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        return self.config

    def get(self, key, default=None):
        if self.config is None:
            self.load_config()
        return self.config.get(key, default)
