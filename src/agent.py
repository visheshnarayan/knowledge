from src.logger import get_logger
import torch
from sklearn.metrics.pairwise import cosine_similarity
import random
from strands import Agent as StrandsAgent, tool
from strands.models import BedrockModel

logger = get_logger(__name__)


class Agent:
    def __init__(
        self,
        subgraph,
        topic,
        embedding_model,
        embedding_tokenizer,
        search_similarity,
        search_sample_ratio,
        strands_config,
        child_agent_model_id,
        parent_agent,
        org_table,
        debug_config=None,
    ):
        self.subgraph = subgraph
        self.topic = topic
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.search_similarity = search_similarity
        self.search_sample_ratio = search_sample_ratio
        self.strands_config = strands_config
        self.parent_agent = parent_agent
        self.org_table = org_table
        self.debug_config = debug_config or {}

        logger.info(
            f"Connecting to AWS Bedrock for Strands API for child agent '{topic}'..."
        )
        llm = BedrockModel(model_id=child_agent_model_id)

        system_prompt = (
            f"{self.strands_config['agent']['system_prompt']}\n\n"
            "You are a specialist agent with deep knowledge of your assigned topic. "
            "However, you are also part of a larger team of agents. "
            "If a question requires knowledge outside your expertise, you MUST delegate it to the appropriate expert using the 'consult_expert' tool. "
            "Consult the organizational table below to understand which topics are handled by other agents.\n\n"
            f"{self.org_table}"
        )

        self.agent = StrandsAgent(
            system_prompt=system_prompt,
            model=llm,
            tools=[self.similarity_search_tool, self.consult_expert_tool],
        )

        logger.info(
            f"Strands agent for topic '{self.topic}' initialized with model {child_agent_model_id}."
        )

    @tool
    def similarity_search_tool(self, question: str) -> str:
        """
        Performs a similarity search on a subgraph to find contextually relevant information to answer a question.
        """
        logger.info(f"Performing similarity search for question: '{question}'")

        def get_embedding(text):
            inputs = self.embedding_tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            )
            with torch.no_grad():
                embedding = (
                    self.embedding_model(**inputs)
                    .last_hidden_state.mean(dim=1)
                    .squeeze()
                    .tolist()
                )
            return embedding

        question_embedding = get_embedding(question)

        topic_node_id = None
        for node_id, node_data in self.subgraph.nodes(data=True):
            if (
                node_data.get("type") == "topic"
                and node_data.get("content") == self.topic
            ):
                topic_node_id = node_id
                break

        if not topic_node_id:
            logger.error(f"Topic node for topic '{self.topic}' not found in subgraph.")
            return ""

        neighbors = list(self.subgraph.neighbors(topic_node_id))
        sample_size = int(len(neighbors) * self.search_sample_ratio)
        start_nodes = random.sample(neighbors, sample_size)
        logger.info(f"Starting similarity search from {len(start_nodes)} random nodes.")

        stack = start_nodes.copy()
        visited = set(start_nodes)
        context = []

        while stack:
            node_id = stack.pop()
            node_data = self.subgraph.nodes[node_id]

            if node_data.get("type") == "chunk":
                node_embedding = node_data.get("embedding")
                if node_embedding:
                    similarity = cosine_similarity(
                        [question_embedding], [node_embedding]
                    )[0][0]
                    if similarity > self.search_similarity:
                        context.append(node_data.get("content", ""))
                        for neighbor_id in self.subgraph.neighbors(node_id):
                            if neighbor_id not in visited:
                                stack.append(neighbor_id)
                                visited.add(neighbor_id)

        return "\n\n".join(context)

    @tool
    def consult_expert_tool(self, question: str, topic: str) -> str:
        """
        Consults another specialized agent to get information on a topic outside of this agent's expertise.
        Use this tool when the user's question cannot be answered using the internal similarity search.
        Look at the organizational table to see which topics are available.

        Args:
            question (str): The specific question to ask the other agent.
            topic (str): The topic of the agent you want to consult.

        Returns:
            str: The answer received from the expert agent.
        """
        if not self.parent_agent:
            return "Error: Parent agent not available for consultation."

        response, _ = self.parent_agent.consult_expert(question, topic)
        return response

    def query(self, question):
        logger.info(f"Querying Strands agent for topic '{self.topic}'...")
        response = self.agent(question)

        if self.debug_config.get("log_llm_responses", False):
            logger.info(f"LLM response for topic '{self.topic}': {response}")

        logger.info(f"Received response from Strands agent for topic '{self.topic}'.")

        response_text = (
            response.response if hasattr(response, "response") else str(response)
        )

        source_contents = []
        if hasattr(response, "source_content") and response.source_content:
            if isinstance(response.source_content, str):
                source_contents = [response.source_content]
            elif isinstance(response.source_content, list):
                source_contents = response.source_content
        elif hasattr(response, "tool_outputs") and response.tool_outputs:
            for output in response.tool_outputs:
                if isinstance(output, str):
                    source_contents.append(output)
                elif hasattr(output, "result") and isinstance(output.result, str):
                    source_contents.append(output.result)

        return response_text, source_contents

    def get_state(self):
        return {
            "subgraph": self.subgraph,
            "topic": self.topic,
            "search_similarity": self.search_similarity,
            "search_sample_ratio": self.search_sample_ratio,
            "strands_config": self.strands_config,
            "debug_config": self.debug_config,
        }

    @classmethod
    def from_state(
        cls,
        state,
        embedding_model,
        embedding_tokenizer,
        child_agent_model_id,
        parent_agent,
        org_table,
        debug_config=None,
    ):
        return cls(
            state["subgraph"],
            state["topic"],
            embedding_model,
            embedding_tokenizer,
            state["search_similarity"],
            state["search_sample_ratio"],
            state["strands_config"],
            child_agent_model_id,
            parent_agent,
            org_table,
            debug_config,
        )
