import cmd
import json
from src.parent_agent import ParentAgent
from src.org_table import OrgTable


class AgentCLI(cmd.Cmd):
    intro = "Welcome to the Agentic Knowledge System CLI. Type help or ? to list commands.\n"
    prompt = "(agent-cli) "

    def __init__(self, runner, parent_agent: ParentAgent, org_table: OrgTable):
        super().__init__()
        self.runner = runner
        self.parent_agent = parent_agent
        self.child_agents = parent_agent.child_agents
        self.org_table = org_table

    def do_list_agents(self, arg):
        """Lists all available specialist agents."""
        print("\nAvailable Agents:")
        print("-" * 20)
        if not self.child_agents:
            print("No child agents are currently active.")
        else:
            for agent_name in sorted(self.child_agents.keys()):
                print(f"- {agent_name}")
        print("-" * 20)

    def do_org_table(self, arg):
        """Displays the full organizational table of all agents."""
        print("\n" + str(self.org_table))

    def do_agent_info(self, arg):
        """
        Displays detailed information about a specific agent from the org table.
        Usage: agent_info <agent_name>
        """
        agent_name = arg.strip()
        if not agent_name:
            print("Error: Please provide an agent name.")
            print("Usage: agent_info <agent_name>")
            return

        agent_info = self.org_table.get_agent_info(agent_name)
        if not agent_info:
            print(f"Error: Agent '{agent_name}' not found in the organizational table.")
        else:
            print(f"\n--- Agent Info: {agent_name} ---")
            print(json.dumps(agent_info, indent=2))
            print("------------------------" + "-" * len(agent_name))

    def do_ping_agent(self, arg):
        """
        Sends a simple test query to a specific agent to check its responsiveness.
        Usage: ping_agent <agent_name>
        """
        agent_name = arg.strip()
        if not agent_name:
            print("Error: Please provide an agent name.")
            print("Usage: ping_agent <agent_name>")
            return

        child_agent = self.child_agents.get(agent_name)
        if not child_agent:
            print(f"Error: Agent '{agent_name}' not found.")
            return

        print(f"Pinging agent '{agent_name}'...")
        try:
            # For OllamaAgent, context is a required argument. For a simple ping, it can be empty.
            # This might need adjustment if other agent types have different signatures.
            if "ollama" in str(type(child_agent)).lower():
                response, _ = child_agent.query(
                    "Hello, are you there? Please respond with a simple confirmation.",
                    context="",
                )
            else:
                response, _ = child_agent.query(
                    "Hello, are you there? Please respond with a simple confirmation."
                )
            print(f"Response from '{agent_name}':\n---")
            print(response)
            print("---")
        except Exception as e:
            print(f"An error occurred while pinging agent '{agent_name}': {e}")

    def do_query(self, arg):
        """
        Asks a question to the ParentAgent for decomposition and answering.
        Usage: query <your question>
        """
        question = arg.strip()
        if not question:
            print("Error: Please provide a question.")
            print("Usage: query <your question>")
            return

        print(f"Sending query to ParentAgent: '{question}'")
        print("Please wait...")
        try:
            response, sources, _ = self.parent_agent.query(question)
            print("\n--- Final Answer ---")
            print(response)
            print("\n--- Sources ---")
            if sources:
                for i, source in enumerate(sources):
                    print(f"[{i+1}]: {source[:200]}...")  # Print snippet
            else:
                print("No sources provided.")
            print("-" * 20)
        except Exception as e:
            print(f"An error occurred during the query: {e}")

    def do_status(self, arg):
        """Displays the current status of the agentic system."""
        print("\n--- System Status ---")
        status = self.runner.get_status()
        for key, value in status.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
        print("-" * 23)

    def do_show_config(self, arg):
        """Displays the current system configuration from config.yaml."""
        print("\n--- Current Configuration ---")
        config_str = json.dumps(self.runner.config, indent=2)
        print(config_str)
        print("-" * 29)

    def do_reconsolidate(self, arg):
        """
        Forces topic consolidation, re-creates agents, and re-builds visualizations.
        This will update the current session with the new topic structure.
        """
        print("Starting forced reconsolidation process... This may take a moment.")
        try:
            new_child_agents = self.runner.reconsolidate_and_reload()
            if new_child_agents is not None:
                self.child_agents = new_child_agents
                # The org_table is updated within the parent_agent, so we just need to reference it
                self.org_table = self.runner.parent_agent.org_table
                print("Reconsolidation complete. The agent structure has been updated.")
                print("Run 'list_agents' or 'org_table' to see the new structure.")
            else:
                print("Reconsolidation process could not complete.")
        except Exception as e:
            print(f"An error occurred during reconsolidation: {e}")

    def do_exit(self, arg):
        """Exits the CLI."""
        print("Exiting Agent CLI. Goodbye!")
        return True

    def do_EOF(self, arg):
        """Exit the CLI with Ctrl-D."""
        return self.do_exit(arg)
