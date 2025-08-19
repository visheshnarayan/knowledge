from openai import OpenAI
import os
import itertools
from src.logger import get_logger

logger = get_logger(__name__)

class ParentModel:
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.get('model', 'local-model') 
        logger.info(f"Connecting to Parent Model via LM Studio endpoint...")

        try:
            self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
            logger.info("Successfully connected to LM Studio endpoint.")
        except Exception as e:
            logger.error(f"Could not connect to LM Studio endpoint: {e}")
            self.client = None

    def _generate_text(self, user_prompt, system_prompt="You are a helpful assistant.", temperature=0.7):
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
            logger.error(f"Error during text generation: {e}")
            return "Error: Could not generate text."

    def sort_into_subgroups(self, graph):
        logger.info(f"Parent Model ({self.model_name}) sorting graph into subgroups...")
        if not self.client:
            logger.error("LM Studio client not initialized, skipping subgrouping.")
            return graph

        all_chunk_texts = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') == "chunk":
                all_chunk_texts.append(node_data['content'])
        combined_text = "\n\n".join(all_chunk_texts)

        topic_system_prompt = "You are an expert text analyst. Your task is to identify overarching topics from a collection of texts. Respond ONLY with a comma-separated list of topics. Do not add any conversational text, numbering, or explanations."
        topic_identification_prompt = f"Given the following collection of texts, identify a minimal set of overarching topics or categories. List all identified topics as a single, comma-separated list.\n\nTexts:\n{combined_text}\n\nTopics:"
        
        try:
            generated_topics_str = self._generate_text(topic_identification_prompt, system_prompt=topic_system_prompt)
            cleaned_topics_str = generated_topics_str.strip().split('\n')[-1]
            if ':' in cleaned_topics_str:
                cleaned_topics_str = cleaned_topics_str.split(':')[-1]

            overarching_topics = [t.strip() for t in cleaned_topics_str.split(',') if t.strip()]
            if not overarching_topics:
                raise ValueError("Model returned no topics.")
            logger.info(f"Identified Overarching Topics: {overarching_topics}")
        except Exception as e:
            logger.error(f"Error identifying overarching topics: {e}")
            overarching_topics = ["unclassified"]

        classification_system_prompt = "You are an expert text classifier. Your task is to classify a given text into one of the provided categories. Respond ONLY with the single, most appropriate category name. Do not add any conversational text or explanations."
        topics_to_chunk_ids = {}
        for node_id, node_data in graph.nodes(data=True):
            if node_data['type'] == "chunk":
                text = node_data['content']
                classification_prompt = f"Given the following text, classify it into one of these categories: {', '.join(overarching_topics)}. Respond with only the category name.\n\nText: {text}\nCategory:"
                
                try:
                    generated_category = self._generate_text(classification_prompt, system_prompt=classification_system_prompt)
                    cleaned_category = generated_category.strip().split('\n')[-1]
                    if ':' in cleaned_category:
                        cleaned_category = cleaned_category.split(':')[-1].strip()
                    cleaned_category = cleaned_category.strip('." ')

                    if cleaned_category and cleaned_category not in overarching_topics and cleaned_category != "unclassified":
                        overarching_topics.append(cleaned_category)
                        logger.info(f"New topic generated and added: {cleaned_category}")

                    node_data['topic'] = cleaned_category
                    topics_to_chunk_ids.setdefault(cleaned_category, []).append(node_id)
                    logger.info(f"Chunk {node_id} assigned to topic: {cleaned_category}")
                except Exception as e:
                    logger.error(f"Error classifying chunk {node_id}: {e}")
                    node_data['topic'] = "unclassified"
                    topics_to_chunk_ids.setdefault("unclassified", []).append(node_id)

        for topic, chunk_ids in topics_to_chunk_ids.items():
            if not topic: continue
            topic_node_id = f"topic_{topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            graph.add_node(topic_node_id, content=topic, type="topic", label=topic)
            for chunk_id in chunk_ids:
                graph.add_edge(topic_node_id, chunk_id, type="has_topic")

        return graph

    def consolidate_topics(self, graph):
        logger.info("Starting iterative topic consolidation...")
        
        system_prompt = "You are an expert in knowledge organization and ontology. Your task is to determine if two topics can be consolidated. If they can, respond ONLY and STRICTLY with 'PARENT: <parent_topic_name>'. If they cannot, respond ONLY and STRICTLY with 'NO_CONSOLIDATION'. Do NOT include any conversational text, explanations, or any other characters. Your response must be one of these two formats, and nothing else."

        # Keep track of changes to iterate until no more consolidations occur
        consolidated_something_in_this_pass = True
        pass_count = 0

        while consolidated_something_in_this_pass:
            pass_count += 1
            logger.info(f"Consolidation Pass {pass_count}...")
            consolidated_something_in_this_pass = False
            current_topics = [data['content'] for _, data in graph.nodes(data=True) if data.get('type') == 'topic']
            
            if len(current_topics) < 2:
                logger.info(f"Less than 2 topics remaining. Ending consolidation after {pass_count} passes.")
                break

            consolidation_map_this_pass = {}
            topic_pairs = list(itertools.combinations(current_topics, 2))

            for topic_a, topic_b in topic_pairs:
                # Skip if either topic has already been marked for consolidation in this pass
                if topic_a in consolidation_map_this_pass or topic_b in consolidation_map_this_pass:
                    continue

                user_prompt = f"Given Topic A: \"{topic_a}\" and Topic B: \"{topic_b}\". Can these two topics be consolidated? If so, which topic is the more general parent topic? Respond ONLY with 'PARENT: <parent_topic_name>' or 'NO_CONSOLIDATION'. No other text, no explanations, no conversational filler."
                raw_response = self._generate_text(user_prompt, system_prompt=system_prompt).strip()

                # Process the raw_response to find the actual instruction
                processed_response = ""
                lines = raw_response.split('\n')
                for line in reversed(lines):
                    stripped_line = line.strip()
                    if stripped_line.startswith("PARENT:") or stripped_line == "NO_CONSOLIDATION":
                        processed_response = stripped_line
                        break
                    # Handle cases where </think> might be on a separate line before the instruction
                    if "</think>" in stripped_line:
                        continue # Skip this line and look for the instruction in previous lines
                
                if not processed_response and lines: # If no clear instruction found, try the last line as a fallback
                    processed_response = lines[-1].strip()

                parent_topic = None
                if processed_response.startswith("PARENT:"):
                    try:
                        parent_topic = processed_response.split("PARENT:", 1)[1].strip()
                    except IndexError:
                        logger.warning(f"Could not parse PARENT: from processed response: {processed_response}. Raw response: {raw_response}")
                        parent_topic = None
                
                if parent_topic:
                    child_topic = None
                    if parent_topic == topic_a:
                        child_topic = topic_b
                    elif parent_topic == topic_b:
                        child_topic = topic_a
                    else:
                        # If LLM returned a new parent topic, we need to decide which is the child.
                        # For now, assume the shorter or less specific one is the child.
                        # A more robust solution might involve another LLM call or semantic comparison.
                        if len(topic_a) < len(topic_b):
                            child_topic = topic_a
                        else:
                            child_topic = topic_b
                        
                        # Add the new parent topic to the list of topics if it's truly new
                        if parent_topic not in current_topics and parent_topic not in [v for k,v in consolidation_map_this_pass.items()]:
                            current_topics.append(parent_topic) # Add to current_topics for subsequent pair generation in this pass
                            logger.info(f"New parent topic '{parent_topic}' identified by LLM.")

                    if child_topic and child_topic in current_topics:
                        consolidation_map_this_pass[child_topic] = parent_topic
                        consolidated_something_in_this_pass = True
                        logger.info(f"Consolidating '{child_topic}' into '{parent_topic}'.")
                elif processed_response == "NO_CONSOLIDATION": # Exact match now
                    logger.info(f"No consolidation for '{topic_a}' and '{topic_b}'.")
                else:
                    # Attempt to extract the intended response even if format is unexpected
                    detected_type = "UNKNOWN"
                    if "PARENT:" in processed_response:
                        detected_type = "PARENT: (malformed)"
                    elif "NO_CONSOLIDATION" in processed_response:
                        detected_type = "NO_CONSOLIDATION (malformed)"
                    
                    logger.warning(f"Unexpected LLM response format. Detected: '{detected_type}'. Processed response: '{processed_response}'. Skipping consolidation for '{topic_a}' and '{topic_b}'.")

            if not consolidated_something_in_this_pass:
                logger.info(f"No further consolidations found in Pass {pass_count}. Ending iterative consolidation.")
                break

            # Apply consolidations from this pass to the graph
            nodes_to_update = []
            for node_id, node_data in graph.nodes(data=True):
                if node_data.get('type') == 'chunk' and node_data.get('topic') in consolidation_map_this_pass:
                    nodes_to_update.append(node_id)

            for node_id in nodes_to_update:
                old_topic = graph.nodes[node_id]['topic']
                new_topic = consolidation_map_this_pass[old_topic]
                graph.nodes[node_id]['topic'] = new_topic

                old_topic_node_id = f"topic_{old_topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"
                new_topic_node_id = f"topic_{new_topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"

                if graph.has_edge(old_topic_node_id, node_id):
                    graph.remove_edge(old_topic_node_id, node_id)
                
                if not graph.has_node(new_topic_node_id):
                    graph.add_node(new_topic_node_id, content=new_topic, type="topic", label=new_topic)
                graph.add_edge(new_topic_node_id, node_id, type="has_topic")

            # Remove old topic nodes that are no longer connected
            # Create a set of topics that are still active (either parent or not consolidated)
            active_topics_after_pass = set(current_topics) - set(consolidation_map_this_pass.keys())
            
            # Remove any topic nodes that are no longer active and have no connections
            topics_to_remove_node_ids = []
            for topic_name in consolidation_map_this_pass.keys():
                topic_node_id = f"topic_{topic_name.replace(' ', '_').replace('/', '_').replace('-', '_')}"
                if graph.has_node(topic_node_id) and graph.degree(topic_node_id) == 0:
                    topics_to_remove_node_ids.append(topic_node_id)
            
            for node_id_to_remove in topics_to_remove_node_ids:
                graph.remove_node(node_id_to_remove)
                logger.info(f"Removed old topic node: {node_id_to_remove}")

        logger.info("Iterative topic consolidation finished.")
        return graph