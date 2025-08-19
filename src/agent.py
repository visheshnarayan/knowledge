from openai import OpenAI
from src.logger import get_logger
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

logger = get_logger(__name__)

class Agent:
    def __init__(self, subgraph, topic, model_name, embedding_model, embedding_tokenizer, similarity_threshold, search_sample_ratio, search_similarity): # Added search_similarity
        self.subgraph = subgraph
        self.topic = topic
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.embedding_tokenizer = embedding_tokenizer
        self.similarity_threshold = similarity_threshold # This is for graph building, not agent search
        self.search_sample_ratio = search_sample_ratio
        self.search_similarity = search_similarity # New attribute
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        logger.info(f"Agent for topic '{self.topic}' initialized.")

    def query(self, question):
        logger.info(f"Querying agent for topic '{self.topic}'...")
        source_contents = self._similarity_search_context(question) # Get list of source contents
        logger.info(f"Found {len(source_contents)} sources for topic '{self.topic}'.") # New line
        context = "\n\n".join(source_contents) # Join for LLM context
        logger.debug(f"Built context for topic '{self.topic}':\n{context}")

        system_prompt = f"You are an expert on the topic of '{self.topic}'. You will be given a context and a question. Your task is to answer the question based on the provided context."
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        logger.debug(f"Prompt for topic '{self.topic}':\n{user_prompt}")

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
            )
            response = completion.choices[0].message.content
            logger.info(f"Received response from LLM for topic '{self.topic}'.")
            logger.debug(f"Response for topic '{self.topic}':\n{response}")
            return response, source_contents # Return both response and source_contents
        except Exception as e:
            logger.error(f"Error during text generation for topic '{self.topic}': {e}")
            return "Error: Could not generate text.", [] # Return empty list for sources on error

    def _get_embedding(self, text):
        inputs = self.embedding_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            embedding = self.embedding_model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
        return embedding

    def _similarity_search_context(self, question):
        logger.info(f"Performing similarity search for question: '{question}'")
        question_embedding = self._get_embedding(question)

        # Find the topic node
        topic_node_id = None
        for node_id, node_data in self.subgraph.nodes(data=True):
            if node_data.get('type') == 'topic' and node_data.get('content') == self.topic:
                topic_node_id = node_id
                break
        
        if not topic_node_id:
            logger.error(f"Topic node for topic '{self.topic}' not found in subgraph.")
            return ""

        # Get neighbors of the topic node (all nodes in the cluster)
        neighbors = list(self.subgraph.neighbors(topic_node_id))
        
        # Take a random sample of the neighbors
        sample_size = int(len(neighbors) * self.search_sample_ratio)
        start_nodes = random.sample(neighbors, sample_size)
        logger.info(f"Starting similarity search from {len(start_nodes)} random nodes.")
        for node_id in start_nodes:
            node_data = self.subgraph.nodes[node_id]
            logger.info(f"  Initial node: {node_id}, Type: {node_data.get('type')}, Content: {node_data.get('content', '')}") # Changed to info and full content

        stack = start_nodes.copy()
        visited = set(start_nodes)
        context = []

        while stack:
            node_id = stack.pop()
            node_data = self.subgraph.nodes[node_id]
            logger.debug(f"  Popped node: {node_id}, Type: {node_data.get('type')}") # New line

            if node_data.get('type') == 'chunk':
                node_embedding = node_data.get('embedding')
                if node_embedding:
                    similarity = cosine_similarity([question_embedding], [node_embedding])[0][0]
                    logger.debug(f"Similarity between question and node {node_id}: {similarity:.4f}")

                    if similarity > self.search_similarity: # Changed to search_similarity
                        logger.info(f"Node {node_id} passed similarity threshold. Adding to context.")
                        context.append(node_data.get('content', ''))
                        
                        # Add neighbors to the stack
                        for neighbor_id in self.subgraph.neighbors(node_id):
                            if neighbor_id not in visited:
                                stack.append(neighbor_id)
                                visited.add(neighbor_id)
                    else:
                        logger.debug(f"  Skipping node {node_id}: Similarity {similarity:.4f} below threshold {self.search_similarity:.4f}.") # Changed to search_similarity
                else:
                    logger.debug(f"  Skipping node {node_id}: No embedding found.")
            else:
                logger.debug(f"  Skipping node {node_id}: Not a chunk type ({node_data.get('type')}).")

        return context # Return the list of chunk contents

    def get_state(self):
        return {
            'subgraph': self.subgraph,
            'topic': self.topic,
            'model_name': self.model_name,
            'similarity_threshold': self.similarity_threshold,
            'search_sample_ratio': self.search_sample_ratio,
            'search_similarity': self.search_similarity # New attribute
        }

    @classmethod
    def from_state(cls, state, embedding_model, embedding_tokenizer):
        return cls(
            state['subgraph'],
            state['topic'],
            state['model_name'],
            embedding_model,
            embedding_tokenizer,
            state['similarity_threshold'],
            state['search_sample_ratio'],
            state['search_similarity'] # New parameter
        )