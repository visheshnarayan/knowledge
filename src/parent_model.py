# from transformers import pipeline
from openai import OpenAI
import os

class ParentModel:
    def __init__(self, config):
        self.config = config
        # The model name from config is now just for reference, as the model is loaded in LM Studio.
        self.model_name = self.config.get('model', 'local-model') 
        print(f"Connecting to Parent Model via LM Studio endpoint...")

        try:
            # Point to the local server started by LM Studio
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
            print("Successfully connected to LM Studio endpoint.")
        except Exception as e:
            print(f"Could not connect to LM Studio endpoint: {e}")
            self.client = None

    def _generate_text(self, user_prompt, system_prompt="You are a helpful assistant.", temperature=0.7):
        """Helper function to send a prompt to the LM Studio endpoint."""
        if not self.client:
            return "Error: LM Studio client not initialized."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            completion = self.client.chat.completions.create(
              model=self.model_name,
              messages=messages,
              temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error during text generation: {e}")
            return "Error: Could not generate text."

    def sort_into_subgroups(self, graph):
        print(f"Parent Model ({self.model_name}) sorting graph into subgroups via LM Studio...")
        if not self.client:
            print("LM Studio client not initialized, skipping subgrouping.")
            return graph

        # Collect all text content from chunk nodes
        all_chunk_texts = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data['type'] == "chunk":
                all_chunk_texts.append(node_data['content'])
        combined_text = "\n\n".join(all_chunk_texts)

        # 1. Identify overarching topics with a robust prompt
        topic_system_prompt = "You are an expert text analyst. Your task is to identify overarching topics from a collection of texts. Respond ONLY with a comma-separated list of topics. Do not add any conversational text, numbering, or explanations."
        topic_identification_prompt = f"Given the following collection of texts, identify a minimal set of overarching topics or categories. Consider these that are predefined for the initial data ingestion: {', '.join(self.config.get('predefined_topics', []))}. If a topic from the predefined list doesn't match, create a new one. List all identified topics as a single, comma-separated list.\n\nTexts:\n{combined_text}\n\nTopics:"
        
        try:
            generated_topics_str = self._generate_text(topic_identification_prompt, system_prompt=topic_system_prompt)
            
            # Clean the output just in case the model didn't follow instructions perfectly
            cleaned_topics_str = generated_topics_str.strip().split('\n')[-1]
            if ':' in cleaned_topics_str:
                cleaned_topics_str = cleaned_topics_str.split(':')[-1]

            overarching_topics = [t.strip() for t in cleaned_topics_str.split(',') if t.strip()]
            if not overarching_topics:
                raise ValueError("Model returned no topics.")

            print(f"Identified Overarching Topics: {overarching_topics}")
        except Exception as e:
            print(f"Error identifying overarching topics: {e}")
            overarching_topics = ["unclassified"]

        # 2. Assign each chunk to one of the identified topics
        classification_system_prompt = "You are an expert text classifier. Your task is to classify a given text into one of the provided categories. Respond ONLY with the single, most appropriate category name. Do not add any conversational text or explanations."
        topics_to_chunk_ids = {}
        for node_id, node_data in graph.nodes(data=True):
            if node_data['type'] == "chunk":
                text = node_data['content']
                classification_prompt = f"Given the following text, classify it into one of these categories: {', '.join(overarching_topics)}. If none of these categories fit, suggest a new, single-word category. Respond with only the category name.\n\nText: {text}\nCategory:"
                
                try:
                    generated_category = self._generate_text(classification_prompt, system_prompt=classification_system_prompt)
                    
                    # Robustly parse the output to get the category
                    # Take the last line, then the part after a colon if it exists, and clean it up.
                    cleaned_category = generated_category.strip().split('\n')[-1]
                    if ':' in cleaned_category:
                        cleaned_category = cleaned_category.split(':')[-1].strip()
                    cleaned_category = cleaned_category.strip('." ')

                    # If a new category is generated, add it to the list
                    if cleaned_category and cleaned_category not in overarching_topics and cleaned_category != "unclassified":
                        overarching_topics.append(cleaned_category)
                        print(f"New topic generated and added: {cleaned_category}")

                    node_data['topic'] = cleaned_category
                    topics_to_chunk_ids.setdefault(cleaned_category, []).append(node_id)
                    print(f"Chunk {node_id} assigned to topic: {cleaned_category}")
                except Exception as e:
                    print(f"Error classifying chunk {node_id}: {e}")
                    node_data['topic'] = "unclassified"
                    topics_to_chunk_ids.setdefault("unclassified", []).append(node_id)

        # 3. Add topic nodes and connect them to chunk nodes
        for topic, chunk_ids in topics_to_chunk_ids.items():
            if not topic: continue # Skip if the topic is an empty string
            topic_node_id = f"topic_{topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            graph.add_node(topic_node_id, content=topic, type="topic", label=topic)
            for chunk_id in chunk_ids:
                graph.add_edge(topic_node_id, chunk_id, type="has_topic")

        return graph