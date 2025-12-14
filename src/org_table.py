import logging
import json
import os

logger = logging.getLogger(__name__)


class OrgTable:
    """
    A directory of all specialized agents in the knowledge system.
    This table provides information about each agent's area of expertise
    and is accessible to all agents to facilitate inter-agent communication.
    """

    def __init__(self):
        self.agents = {}

    def add_agent(self, agent_id: str, description: str, keywords: list):
        """
        Adds a new agent to the table.

        Args:
            agent_id (str): The unique identifier for the agent (e.g., its topic).
            description (str): A summary of the agent's knowledge domain.
            keywords (list): A list of keywords representing the core concepts.
        """
        if agent_id in self.agents:
            logger.warning(
                f"Agent ID {agent_id} already exists in the org table. Overwriting."
            )
        self.agents[agent_id] = {
            "description": description,
            "keywords": keywords,
        }
        logger.info(f"Added agent '{agent_id}' to the org table.")

    def get_agent_info(self, agent_id: str):
        """
        Retrieves information about a specific agent.

        Args:
            agent_id (str): The ID of the agent to retrieve.

        Returns:
            dict: The agent's information, or None if not found.
        """
        return self.agents.get(agent_id)

    def get_full_table(self):
        """
        Returns the entire agent directory.

        Returns:
            dict: The complete dictionary of agents.
        """
        return self.agents

    def get_topics(self):
        """
        Returns a list of all agent topics (IDs).

        Returns:
            list: A list of agent topic strings.
        """
        return list(self.agents.keys())

    def save_to_json(self, file_path: str):
        """
        Saves the organizational table to a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        try:
            # Ensure the directory exists
            dir_name = os.path.dirname(file_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            with open(file_path, "w") as f:
                json.dump(self.agents, f, indent=4)
            logger.info(f"Organizational table saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving organizational table to {file_path}: {e}")

    def load_from_json(self, file_path: str):
        """
        Loads the organizational table from a JSON file.

        Args:
            file_path (str): The path to the JSON file.
        """
        try:
            with open(file_path, "r") as f:
                self.agents = json.load(f)
            logger.info(f"Organizational table loaded from {file_path}")
        except FileNotFoundError:
            logger.warning(
                f"Organizational table file not found at {file_path}. A new one will be created."
            )
        except Exception as e:
            logger.error(f"Error loading organizational table from {file_path}: {e}")

    def __str__(self):
        """
        Returns a string representation of the org table, formatted for easy reading
        and for inclusion in an LLM prompt.
        """
        if not self.agents:
            return "The organizational table is currently empty."

        table_str = "Organizational Table of Specialized Agents:\n"
        table_str += "-" * 50 + "\n"
        for agent_id, info in self.agents.items():
            table_str += f"Agent Topic: {agent_id}\n"
            table_str += f"  Description: {info['description']}\n"
            table_str += f"  Keywords: {', '.join(info['keywords'])}\n"
            table_str += "-" * 50 + "\n"
        return table_str
