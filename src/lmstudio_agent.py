from openai import OpenAI
from src.logger import get_logger
import torch
from sklearn.metrics.pairwise import cosine_similarity
import random

logger = get_logger(__name__)

class LMStudioAgent:
    def __init__(self, subgraph, topic, embedding_model, embedding_tokenizer, search_similarity, search_sample_ratio, llm_config):
        self.subgraph = subgraph
        self.topic = topic
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.search_similarity = search_similarity
        self.search_sample_ratio = search_sample_ratio
        self.llm_config = llm_config
        self.client = OpenAI(base_url=self.llm_config.get('base_url', 'http://localhost:1234/v1'), api_key=self.llm_config.get('api_key', 'not-needed'))
        self.model = self.llm_config.get('model', 'local-model')
        logger.info(f"LMStudio agent for topic '{self.topic}' initialized with model {self.model}.")

    def similarity_search(self, question: str) -> str:
        logger.info(f"Performing similarity search for question: '{question}'")

        def get_embedding(text):
            inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                embedding = self.embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
            return embedding

        question_embedding = get_embedding(question)

        topic_node_id = None
        for node_id, node_data in self.subgraph.nodes(data=True):
            if node_data.get('type') == 'topic' and node_data.get('content') == self.topic:
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

            if node_data.get('type') == 'chunk':
                node_embedding = node_data.get('embedding')
                if node_embedding:
                    similarity = cosine_similarity([question_embedding], [node_embedding])[0][0]
                    if similarity > self.search_similarity:
                        context.append(node_data.get('content', ''))
                        for neighbor_id in self.subgraph.neighbors(node_id):
                            if neighbor_id not in visited:
                                stack.append(neighbor_id)
                                visited.add(neighbor_id)

        return "\n\n".join(context)

    def query(self, question, context):
        logger.info(f"Querying LMStudio agent for topic '{self.topic}'...")

        system_prompt = f"""You are a helpful assistant for the topic '{self.topic}'. 
        Answer the question based on the provided context.
        Context: {context}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            completion = self.client.chat.completions.create(
              model=self.model,
              messages=messages,
              temperature=0.7,
            )
            response = completion.choices[0].message.content
            logger.info(f"Received response from LMStudio agent for topic '{self.topic}'.")
            return response, context
        except Exception as e:
            logger.error(f"Error during text generation with LMStudio: {e}")
            return "Error: Could not generate text.", ""