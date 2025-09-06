from strands import Agent as StrandsAgent
from src.logger import get_logger
from strands.models import BedrockModel
import itertools
import boto3
import json

logger = get_logger(__name__)

class ParentAgent:
    def __init__(self, child_agents, strands_config, parent_agent_model_id, debug_config=None):
        self.child_agents = child_agents
        self.strands_config = strands_config
        self.parent_agent_model_id = parent_agent_model_id
        self.debug_config = debug_config or {}

        system_prompt = (
            "You are a routing agent. Your job is to determine the topic of a question and select the appropriate specialist agent."
            f"The available topics are: {list(self.child_agents.keys())}. "
            "Respond with only the name of the topic, and nothing else. Do NOT include any reasoning or conversational filler."
        )

        logger.info(f"Connecting to AWS Bedrock for Strands API for parent agent...")
        self.llm_client = BedrockModel(model_id=parent_agent_model_id)

        self.router_agent = StrandsAgent(
            system_prompt=system_prompt,
            model=self.llm_client
        )

        self.bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")

        logger.info(f"Parent agent initialized with model {parent_agent_model_id}.")

    def query(self, question):
        logger.info(f"Parent agent received query: '{question}'")

        # 1. Use the router agent to determine the topic
        topic_agent_result = self.router_agent(question)
        
        topic = str(topic_agent_result).strip()

        if self.debug_config.get('log_llm_responses', False):
            logger.info(f"LLM routing response (cleaned): {topic}")

        logger.debug(f"Raw AgentResult from router_agent: {topic_agent_result}")
        logger.debug(f"Type of AgentResult: {type(topic_agent_result)}")
        logger.debug(f"Dir of AgentResult: {dir(topic_agent_result)}")
        logger.info(f"Router agent selected topic: '{topic}'")

        # 2. Find the corresponding child agent
        child_agent = self.child_agents.get(topic)

        if not child_agent:
            logger.error(f"Could not find a child agent for topic: '{topic}'")
            return "I am sorry, but I cannot find an appropriate specialist for your question.", [], topic

        # 3. Forward the query to the child agent
        logger.info(f"Forwarding query to child agent for topic: '{topic}'")
        response, sources = child_agent.query(question)
        return response, sources, topic

    def identify_topics_from_graph(self, graph):
        logger.info("Parent Agent identifying topics from graph...")
        all_chunk_texts = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get('type') == "chunk":
                all_chunk_texts.append(node_data['content'])
        combined_text = "\n\n".join(all_chunk_texts)

        topic_system_prompt = (
            "You are an expert text analyst. Your task is to identify overarching topics from a collection of texts. "
            "Respond ONLY with a comma-separated list of topics. Do not add any conversational text, numbering, or explanations."
        )
        topic_identification_prompt = (
            f"Given the following collection of texts, identify a minimal set of overarching topics or categories. "
            f"List all identified topics as a single, comma-separated list.\n\nTexts:\n{combined_text}\n\nTopics:"
        )

        try:
            topic_agent = StrandsAgent(
                system_prompt=topic_system_prompt,
                model=self.llm_client
            )
            agent_result = topic_agent(topic_identification_prompt)
            
            generated_topics_str = agent_result.response if hasattr(agent_result, 'response') else str(agent_result)

            # Robust parsing to find the topic list, ignoring conversational filler
            lines = generated_topics_str.strip().split('\n')
            cleaned_topics_str = ""
            for line in reversed(lines):
                stripped_line = line.strip()
                # A simple heuristic: the topic list is likely the last line with commas
                if ',' in stripped_line and not any(phrase in stripped_line for phrase in ["Okay", "Let me think", "I will determine", "Based on the information"]):
                    cleaned_topics_str = stripped_line
                    break
            
            # Fallback to the last line if no better candidate is found
            if not cleaned_topics_str:
                for line in reversed(lines):
                    if line.strip():
                        cleaned_topics_str = line.strip()
                        break

            if self.debug_config.get('log_llm_responses', False):
                logger.info(f"LLM topic identification response (cleaned): {cleaned_topics_str}")

            if ':' in cleaned_topics_str:
                cleaned_topics_str = cleaned_topics_str.split(':')[-1]

            overarching_topics = [t.strip() for t in cleaned_topics_str.split(',') if t.strip()]
            if not overarching_topics:
                raise ValueError("Model returned no topics.")
            logger.info(f"Identified Overarching Topics: {overarching_topics}")
            return overarching_topics
        except Exception as e:
            logger.error(f"Error identifying overarching topics: {e}")
            return ["unclassified"]

    def classify_and_assign_chunks(self, graph, overarching_topics):
        logger.info("Parent Agent classifying and assigning chunks...")

        topics_to_chunk_ids = {}
        current_overarching_topics = list(overarching_topics) 

        for node_id, node_data in graph.nodes(data=True):
            if node_data['type'] == "chunk":
                text = node_data['content']
                classification_prompt = (
                    f"Given the following text, classify it into one of these categories: {', '.join(current_overarching_topics)}. "
                    f"Respond with only the category name.\n\nText: {text}\nCategory:"
                )
                
                try:
                    body = json.dumps({
                        "prompt": classification_prompt,
                        "max_tokens": 256
                    })

                    response = self.bedrock_client.invoke_model(
                        modelId=self.parent_agent_model_id,
                        contentType='application/json',
                        accept='application/json',
                        body=body
                    )
                    
                    result = json.loads(response['body'].read().decode("utf-8"))
                    
                    if self.debug_config.get('log_llm_responses', False):
                        logger.info(f"LLM classification response for chunk {node_id}: {result}")

                    # The DeepSeek model can return a long string, but the category is expected to be the first line.
                    generated_category = result['choices'][0]['text']

                    # Extract the first non-empty line as the category
                    cleaned_category = ""
                    for line in generated_category.strip().split('\n'):
                        if line.strip():
                            cleaned_category = line.strip()
                            break

                    if ':' in cleaned_category:
                        cleaned_category = cleaned_category.split(':')[-1].strip()
                    cleaned_category = cleaned_category.strip('." ')

                    if cleaned_category and cleaned_category not in current_overarching_topics and cleaned_category != "unclassified":
                        current_overarching_topics.append(cleaned_category)
                        logger.info(f"New topic generated and added: {cleaned_category}")

                    node_data['topic'] = cleaned_category
                    topics_to_chunk_ids.setdefault(cleaned_category, []).append(node_id)
                    logger.info(f"Chunk {node_id} assigned to topic: {cleaned_category}")
                except Exception as e:
                    logger.error(f"Error classifying chunk {node_id}: {e}")
                    if hasattr(e, 'response'):
                        logger.error(f"Error response: {e.response}")
                    node_data['topic'] = "unclassified"
                    topics_to_chunk_ids.setdefault("unclassified", []).append(node_id)

        for topic, chunk_ids in topics_to_chunk_ids.items():
            if not topic: continue
            topic_node_id = f"topic_{topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            if not graph.has_node(topic_node_id): 
                graph.add_node(topic_node_id, content=topic, type="topic", label=topic)
            for chunk_id in chunk_ids:
                graph.add_edge(topic_node_id, chunk_id, type="has_topic")

        return graph

    def consolidate_topics(self, graph):
        logger.info("Starting iterative topic consolidation...")
        
        system_prompt = (
            "You are an expert in knowledge organization and ontology. Your task is to determine if two topics can be consolidated. "
            "If they can, respond ONLY and STRICTLY with 'PARENT: <parent_topic_name>'. If they cannot, respond ONLY and STRICTLY with 'NO_CONSOLIDATION'. "
            "Do NOT include any conversational text, explanations, or any other characters. Your response must be one of these two formats, and nothing else."
        )

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

                # Create a new agent for each pair to avoid history contamination
                consolidation_agent = StrandsAgent(
                    system_prompt=system_prompt,
                    model=self.llm_client
                )

                user_prompt = (
                    f"Given Topic A: \"{topic_a}\" and Topic B: \"{topic_b}\". Can these two topics be consolidated? "
                    f"If so, which topic is the more general parent topic? Respond ONLY with 'PARENT: <parent_topic_name>' or 'NO_CONSOLIDATION'. "
                    f"No other text, no explanations, no conversational filler."
                )
                agent_result = consolidation_agent(user_prompt)
                
                raw_response = agent_result.response if hasattr(agent_result, 'response') else str(agent_result)
                raw_response = raw_response.strip()

                # Robust parsing to find the actual instruction
                processed_response = ""
                lines = raw_response.split('\n')
                for line in reversed(lines):
                    stripped_line = line.strip()
                    if stripped_line.startswith("PARENT:") or stripped_line == "NO_CONSOLIDATION":
                        processed_response = stripped_line
                        break
                    # If the line contains common LLM reasoning phrases, skip it
                    if any(phrase in stripped_line for phrase in ["Okay", "Let me think", "I will determine", "Based on the information"]):
                        continue
                
                # Fallback if no clear instruction found, but try to be strict
                if not processed_response:
                    # If the response is short and doesn't contain newlines, it might be the direct answer
                    if '\n' not in raw_response and (raw_response.startswith("PARENT:") or raw_response == "NO_CONSOLIDATION"):
                        processed_response = raw_response
                    else:
                        # As a last resort, try to extract from the last non-empty line
                        for line in reversed(lines):
                            stripped_line = line.strip()
                            if stripped_line:
                                processed_response = stripped_line
                                break
                
                if self.debug_config.get('log_llm_responses', False):
                    logger.info(f"LLM consolidation response for topics '{topic_a}' and '{topic_b}' (cleaned): {processed_response}")

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
                        if len(topic_a) < len(topic_b):
                            child_topic = topic_a
                        else:
                            child_topic = topic_b
                        
                        if parent_topic not in current_topics and parent_topic not in [v for k,v in consolidation_map_this_pass.items()]:
                            current_topics.append(parent_topic)
                            logger.info(f"New parent topic '{parent_topic}' identified by LLM.")

                    if child_topic and child_topic in current_topics:
                        consolidation_map_this_pass[child_topic] = parent_topic
                        consolidated_something_in_this_pass = True
                        logger.info(f"Consolidating '{child_topic}' into '{parent_topic}'.")
                elif processed_response == "NO_CONSOLIDATION":
                    logger.info(f"No consolidation for '{topic_a}' and '{topic_b}'.")
                else:
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
            active_topics_after_pass = set(current_topics) - set(consolidation_map_this_pass.keys())
            
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

    def process_graph(self, graph, config):
        logger.info("Parent Agent processing graph...")
        
        # 1. Identify topics
        overarching_topics = self.identify_topics_from_graph(graph)
        
        # 2. Classify and assign chunks
        processed_graph = self.classify_and_assign_chunks(graph, overarching_topics)
        
        # 3. Conditionally consolidate topics
        if config.get('enable_consolidation', False):
            processed_graph = self.consolidate_topics(processed_graph)
        else:
            logger.info("Topic consolidation is disabled in config.")
        
        return processed_graph