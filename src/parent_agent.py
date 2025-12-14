from openai import OpenAI
import os
from src.logger import get_logger
from src.org_table import OrgTable
import itertools
import json
from concurrent.futures import ThreadPoolExecutor

# Conditional imports based on infra type
try:
    from strands import Agent as StrandsAgent
    from strands.models import BedrockModel
    import boto3

    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

logger = get_logger(__name__)


class OllamaModel:
    """A wrapper for Ollama to mimic the behavior of a callable agent like StrandsAgent."""

    def __init__(self, config, system_prompt=""):
        default_url = config.get("base_url", "http://localhost:11434/v1")
        base_url = os.environ.get("OLLAMA_BASE_URL", default_url)
        self.client = OpenAI(
            base_url=base_url, api_key=config.get("api_key", "not-needed")
        )
        self.model = config.get("model", "mistral")
        self.system_prompt = system_prompt

    def __call__(self, prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            response = completion.choices[0].message.content

            class SimpleResponse:
                def __init__(self, text):
                    self.response = text

                def __str__(self):
                    return self.response

            return SimpleResponse(response)
        except Exception as e:
            logger.error(f"Error during Ollama model call: {e}")
            return ""


class ParentAgent:
    def __init__(self, infra_type, llm_config, debug_config=None):
        self.infra_type = infra_type
        self.llm_config = llm_config
        self.debug_config = debug_config or {}
        self.child_agents = {}
        self.org_table = OrgTable()
        self.llm_client = None
        self.bedrock_client = None

        if self.infra_type == "strands":
            if not STRANDS_AVAILABLE:
                raise ImportError(
                    "Strands AI library not found. Please install it to use 'strands' infrastructure."
                )
            parent_agent_model_id = self.llm_config.get("parent_agent_model_id")
            logger.info(
                "Connecting to AWS Bedrock for Strands API for parent agent..."
            )
            self.llm_client = BedrockModel(model_id=parent_agent_model_id)
            self.bedrock_client = boto3.client(
                "bedrock-runtime", region_name="us-east-1"
            )
            logger.info(
                f"Parent agent initialized with Strands and model {parent_agent_model_id}."
            )
        elif self.infra_type == "ollama":
            logger.info("Initializing Parent agent for Ollama...")
            self.llm_client = OllamaModel(
                self.llm_config
            )  # The client is the callable agent itself
            logger.info(
                f"Parent agent initialized with Ollama model {self.llm_config.get('model')}."
            )
        else:
            raise ValueError(f"Unsupported infrastructure type: {self.infra_type}")

    def _get_agent_for_prompt(self, system_prompt):
        if self.infra_type == "strands":
            if not STRANDS_AVAILABLE:
                raise ImportError("Strands library not found.")
            return StrandsAgent(system_prompt=system_prompt, model=self.llm_client)
        elif self.infra_type == "ollama":
            return OllamaModel(self.llm_config, system_prompt=system_prompt)

    def set_child_agents(self, child_agents):
        self.child_agents = child_agents
        self._initialize_routing_agents()

    def _initialize_routing_agents(self):
        # System prompt for question decomposition
        system_prompt = (
            "You are a master agent responsible for decomposing a complex question into sub-questions and routing them to the correct specialist agent."
            "The available topics are: "
            f"{list(self.child_agents.keys())}"
            "Your task is to: "
            "1. Analyze the user's question."
            "2. Decompose it into one or more sub-questions."
            "3. For each sub-question, you MUST select the most relevant topic from the provided list of available topics."
            "4. Respond with a JSON object containing a list of these sub-questions and their assigned topics. The format should be: "
            '{"sub_tasks": [{"question": "<sub_question_1>", "topic": "<assigned_topic_1>"}, {"question": "<sub_question_2>", "topic": "<assigned_topic_2>"}]}'
            "Do NOT include any reasoning or conversational filler in your response. It must be only the JSON object."
        )
        self.router_agent = self._get_agent_for_prompt(system_prompt)

        synthesis_system_prompt = (
            "You are a master synthesis agent. Your purpose is to craft a single, comprehensive, and well-structured answer to a user's original question. "
            "You will be provided with the original question and a set of answers to sub-questions that have been gathered from specialist agents. "
            "Your task is to synthesize this information into a final response. If the sub-answers are contradictory or insufficient, acknowledge this limitation. "
            "Structure your response clearly. Do not simply list the sub-answers; integrate them into a holistic response."
        )
        self.synthesis_agent = self._get_agent_for_prompt(synthesis_system_prompt)
        logger.info("Parent agent routing and synthesis agents initialized.")

    def _query_child_agent(self, sub_task):
        question = sub_task["question"]
        topic = sub_task["topic"]
        child_agent = self.child_agents.get(topic)
        if not child_agent:
            logger.error(f"Could not find a child agent for topic: '{topic}'")
            return topic, question, "No agent found for this topic.", []

        logger.info(f"Forwarding sub-query to child agent for topic: '{topic}'")
        response, sources = child_agent.query(question)
        logger.info(f"Received response from child agent for topic: '{topic}'")
        return topic, question, response, sources

    def query(self, question):
        logger.info(f"Parent agent received query: '{question}'")

        # Re-initialize routing agents to ensure a clean state for each query
        self._initialize_routing_agents()

        # 1. Decompose the question and route sub-questions
        logger.info("Consulting organizational table to route query...")
        decomposition_result = self.router_agent(question)
        decomposition_response = str(decomposition_result).strip()

        if self.debug_config.get("log_llm_responses", False):
            logger.info(f"LLM decomposition response: {decomposition_response}")

        try:
            # Find the start of the JSON object
            json_start = decomposition_response.find("{")
            if json_start == -1:
                raise json.JSONDecodeError(
                    "No JSON object found in the response.", decomposition_response, 0
                )

            # Extract the JSON part of the string
            json_str = decomposition_response[json_start:]
            sub_tasks_data = json.loads(json_str)
            sub_tasks = sub_tasks_data.get("sub_tasks", [])
        except (json.JSONDecodeError, AttributeError) as e:
            logger.error(f"Failed to parse JSON from decomposition response: {e}")
            logger.error(f"Raw response was: {decomposition_response}")
            # Fallback to single-agent routing if decomposition fails
            logger.info("Falling back to single-agent routing.")
            return self._fallback_query(question)

        if not sub_tasks:
            logger.warning(
                "Decomposition resulted in no sub-tasks. Falling back to single-agent routing."
            )
            return self._fallback_query(question)

        # 2. Concurrently query child agents
        reports = []
        all_sources = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._query_child_agent, task) for task in sub_tasks
            ]
            for future in futures:
                topic, sub_question, response, sources = future.result()
                reports.append(
                    f"--- Start of Report from Specialist Agent: {topic} ---\n"
                    f"Specialist's Question: {sub_question}\n"
                    f"Specialist's Answer: {response}\n"
                    f"--- End of Report from Specialist Agent: {topic} ---"
                )
                all_sources.extend(sources)

        # 3. Synthesize the final answer
        synthesis_prompt = (
            f"Please provide a comprehensive answer to the following original question: '{question}'\n\n"
            "To construct your answer, synthesize the reports I received from my specialist agents, which are provided below:\n\n"
            + "\n\n".join(reports)
        )

        final_response_result = self.synthesis_agent(synthesis_prompt)
        final_response = str(final_response_result).strip()

        if self.debug_config.get("log_llm_responses", False):
            logger.info(f"LLM synthesis response: {final_response}")

        # Use a set to get unique sources
        unique_sources = list(set(all_sources))

        return final_response, unique_sources, [task["topic"] for task in sub_tasks]

    def _fallback_query(self, question):
        # This is the original single-agent query logic
        logger.info("Consulting organizational table for fallback query...")
        fallback_router_prompt = (
            "You are a routing agent. Your job is to determine the most relevant topic for a question from the list of available topics."
            "The available topics are: "
            f"{list(self.child_agents.keys())}"
            "Respond with only the name of the topic, and nothing else. Do NOT include any reasoning or conversational filler."
        )
        fallback_router = self._get_agent_for_prompt(fallback_router_prompt)

        topic_agent_result = fallback_router(question)
        topic = str(topic_agent_result).strip()

        if self.debug_config.get("log_llm_responses", False):
            logger.info(f"LLM fallback routing response (cleaned): {topic}")

        child_agent = self.child_agents.get(topic)
        if not child_agent:
            logger.error(f"Could not find a child agent for topic: '{topic}'")
            return (
                "I am sorry, but I cannot find an appropriate specialist for your question.",
                [],
                topic,
            )

        response, sources = child_agent.query(question)
        return response, sources, topic

    def consult_expert(self, question: str, topic: str):
        """
        Allows one child agent to consult another through the parent.
        """
        logger.info(
            f"Agent is consulting expert on topic '{topic}' with question: '{question}'"
        )
        child_agent = self.child_agents.get(topic)
        if not child_agent:
            logger.error(
                f"Could not find a child agent for topic: '{topic}' during consultation."
            )
            return "No agent found for this topic.", []

        if self.infra_type == "strands":
            if not STRANDS_AVAILABLE:
                return "Strands library not available for consultation.", []
            # Create a new, clean StrandsAgent for the consultation to avoid history contamination
            consultation_agent = StrandsAgent(
                system_prompt=child_agent.agent.system_prompt,
                model=child_agent.agent.model,
                tools=child_agent.agent.tools,
            )
            try:
                response_result = consultation_agent(question)
                response_text = (
                    response_result.response
                    if hasattr(response_result, "response")
                    else str(response_result)
                )
                # Source extraction for strands is more complex and might need to be improved
                return response_text, []
            except Exception as e:
                logger.error(f"Error during Strands expert consultation: {e}")
                return f"Error consulting expert on {topic}: {e}", []
        else:
            # For Ollama and other types, we do a direct query
            try:
                response, sources = child_agent.query(question)
                return response, sources
            except Exception as e:
                logger.error(f"Error during expert consultation: {e}")
                return f"Error consulting expert on {topic}: {e}", []

    def _populate_org_table(self, graph):
        """
        Uses the LLM to generate descriptions and keywords for each topic
        and populates the org_table.
        """
        logger.info("Populating organizational table...")
        topics = {
            data["content"]
            for _, data in graph.nodes(data=True)
            if data.get("type") == "topic"
        }

        for topic in topics:
            # 1. Generate Description
            desc_prompt = (
                f'Provide a concise, one-sentence description for the topic: "{topic}".'
            )
            description_agent = self._get_agent_for_prompt(
                "You are a helpful assistant."
            )
            desc_result = description_agent(desc_prompt)
            description = str(desc_result).strip()

            # 2. Generate Keywords
            kw_prompt = f'List 3-5 relevant keywords for the topic: "{topic}". Respond with a comma-separated list.'
            keywords_agent = self._get_agent_for_prompt("You are a helpful assistant.")
            kw_result = keywords_agent(kw_prompt)
            keywords = [kw.strip() for kw in str(kw_result).split(",")]

            self.org_table.add_agent(topic, description, keywords)

        logger.info("Organizational table populated.")
        logger.info(f"\n{self.org_table}")

    def identify_topics_from_graph(self, graph):
        logger.info("Parent Agent identifying topics from graph...")
        all_chunk_texts = []
        for node_id, node_data in graph.nodes(data=True):
            if node_data.get("type") == "chunk":
                all_chunk_texts.append(node_data["content"])
        combined_text = "\n\n".join(all_chunk_texts)

        topic_system_prompt = (
            "You are an expert text analyst. Your task is to identify a set of distinct, representative, and high-level topics from a collection of texts. "
            "Ensure that conceptually different subjects (e.g., fictional stories vs. technical documents) are assigned to completely separate topics. "
            "Respond ONLY with a comma-separated list of topics. Do not add any conversational text, numbering, or explanations."
        )
        topic_identification_prompt = (
            "From the following texts, extract a minimal set of high-level topics that are clearly distinct and representative of all the information. "
            "For example, if the texts contain information about both 'Harry Potter' and 'Natural Language Processing', these should be two separate topics. "
            f"List all identified topics as a single, comma-separated list.\n\nTexts:\n{combined_text}\n\nTopics:"
        )

        try:
            topic_agent = self._get_agent_for_prompt(topic_system_prompt)
            agent_result = topic_agent(topic_identification_prompt)

            generated_topics_str = str(agent_result)

            # Robust parsing to find the topic list, ignoring conversational filler
            lines = generated_topics_str.strip().split("\n")
            cleaned_topics_str = ""
            for line in reversed(lines):
                stripped_line = line.strip()
                # A simple heuristic: the topic list is likely the last line with commas
                if "," in stripped_line and not any(
                    phrase in stripped_line
                    for phrase in [
                        "Okay",
                        "Let me think",
                        "I will determine",
                        "Based on the information",
                    ]
                ):
                    cleaned_topics_str = stripped_line
                    break

            # Fallback to the last line if no better candidate is found
            if not cleaned_topics_str:
                for line in reversed(lines):
                    if line.strip():
                        cleaned_topics_str = line.strip()
                        break

            if self.debug_config.get("log_llm_responses", False):
                logger.info(
                    f"LLM topic identification response (cleaned): {cleaned_topics_str}"
                )

            if ":" in cleaned_topics_str:
                cleaned_topics_str = cleaned_topics_str.split(":")[-1]

            overarching_topics = [
                t.strip() for t in cleaned_topics_str.split(",") if t.strip()
            ]
            if not overarching_topics:
                raise ValueError("Model returned no topics.")
            logger.info(f"Identified Overarching Topics: {overarching_topics}")
            return overarching_topics
        except Exception as e:
            logger.error(f"Error identifying overarching topics: {e}")
            return ["unclassified"]

    def classify_and_assign_chunks(self, graph, overarching_topics):
        logger.info("Parent Agent classifying and assigning chunks...")

        classification_agent = self._get_agent_for_prompt(
            "You are a classification agent. Respond with only the category name."
        )

        topics_to_chunk_ids = {}

        for node_id, node_data in graph.nodes(data=True):
            if node_data.get("type") == "chunk":
                text = node_data["content"]
                classification_prompt = (
                    f"Given the following text, classify it into one of these categories: {', '.join(overarching_topics)}. "
                    f"Respond with only the category name.\n\nText: {text}\nCategory:"
                )

                try:
                    # This unified approach uses the generic agent for classification
                    cleaned_category = str(
                        classification_agent(classification_prompt)
                    ).strip()

                    if self.debug_config.get("log_llm_responses", False):
                        logger.info(
                            f"LLM classification response for chunk {node_id}: {cleaned_category}"
                        )

                    # Basic cleanup
                    if ":" in cleaned_category:
                        cleaned_category = cleaned_category.split(":")[-1].strip()
                    cleaned_category = cleaned_category.strip('." ')

                    # Dynamically add new topics if the model generates one
                    if (
                        cleaned_category
                        and cleaned_category not in overarching_topics
                        and cleaned_category != "unclassified"
                    ):
                        overarching_topics.append(cleaned_category)
                        logger.info(
                            f"New topic generated and added: {cleaned_category}"
                        )

                    node_data["topic"] = cleaned_category
                    topics_to_chunk_ids.setdefault(cleaned_category, []).append(node_id)
                    logger.info(
                        f"Chunk {node_id} assigned to topic: {cleaned_category}"
                    )

                except Exception as e:
                    logger.error(f"Error classifying chunk {node_id}: {e}")
                    node_data["topic"] = "unclassified"
                    topics_to_chunk_ids.setdefault("unclassified", []).append(node_id)

        for topic, chunk_ids in topics_to_chunk_ids.items():
            if not topic:
                continue
            topic_node_id = (
                f"topic_{topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            )
            if not graph.has_node(topic_node_id):
                graph.add_node(topic_node_id, content=topic, type="topic", label=topic)
            for chunk_id in chunk_ids:
                graph.add_edge(topic_node_id, chunk_id, type="has_topic")

        return graph

    def consolidate_topics(self, graph):
        logger.info("Starting iterative topic consolidation...")

        system_prompt = (
            "You are an expert in knowledge organization and ontology. Your task is to determine if two topics are so closely related that they should be merged under a single, more general parent topic. "
            "Be very strict: only consolidate if the topics are highly similar or one is a direct sub-topic of the other. "
            "If they can be consolidated, respond ONLY and STRICTLY with 'PARENT: <parent_topic_name>'. "
            "If they are distinct enough to remain separate, respond ONLY and STRICTLY with 'NO_CONSOLIDATION'. "
            "Your response must be one of these two formats, and nothing else."
        )

        # Keep track of changes to iterate until no more consolidations occur
        consolidated_something_in_this_pass = True
        pass_count = 0

        while consolidated_something_in_this_pass:
            pass_count += 1
            logger.info(f"Consolidation Pass {pass_count}...")
            consolidated_something_in_this_pass = False
            current_topics = [
                data["content"]
                for _, data in graph.nodes(data=True)
                if data.get("type") == "topic"
            ]

            if len(current_topics) < 2:
                logger.info(
                    f"Less than 2 topics remaining. Ending consolidation after {pass_count} passes."
                )
                break

            consolidation_map_this_pass = {}
            topic_pairs = list(itertools.combinations(current_topics, 2))

            for topic_a, topic_b in topic_pairs:
                # Skip if either topic has already been marked for consolidation in this pass
                if (
                    topic_a in consolidation_map_this_pass
                    or topic_b in consolidation_map_this_pass
                ):
                    continue

                # Create a new agent for each pair to avoid history contamination
                consolidation_agent = self._get_agent_for_prompt(system_prompt)

                user_prompt = (
                    f'Given Topic A: "{topic_a}" and Topic B: "{topic_b}". Are these two topics so similar that they should be merged? '
                    "If yes, which topic is the more general parent? Respond ONLY with 'PARENT: <parent_topic_name>' or 'NO_CONSOLIDATION'. "
                    "Do not consolidate loosely related topics. For example, 'Machine Learning' and 'Deep Learning' could be consolidated, but 'Harry Potter' and 'Fantasy Novels' should remain separate if the goal is specificity. "
                    "No other text, no explanations, no conversational filler."
                )
                agent_result = consolidation_agent(user_prompt)

                raw_response = str(agent_result).strip()

                # Robust parsing to find the actual instruction
                lines = raw_response.split("\n")
                processed_response = ""
                for line in reversed(lines):
                    stripped_line = line.strip()
                    if (
                        stripped_line.startswith("PARENT:")
                        or stripped_line == "NO_CONSOLIDATION"
                    ):
                        processed_response = stripped_line
                        break
                    if any(
                        phrase in stripped_line
                        for phrase in [
                            "Okay",
                            "Let me think",
                            "I will determine",
                            "Based on the information",
                        ]
                    ):
                        continue

                if not processed_response:
                    if "\n" not in raw_response and (
                        raw_response.startswith("PARENT:")
                        or raw_response == "NO_CONSOLIDATION"
                    ):
                        processed_response = raw_response
                    else:
                        for line in reversed(lines):
                            stripped_line = line.strip()
                            if stripped_line:
                                processed_response = stripped_line
                                break

                if self.debug_config.get("log_llm_responses", False):
                    logger.info(
                        f"LLM consolidation response for topics '{topic_a}' and '{topic_b}' (cleaned): {processed_response}"
                    )

                parent_topic = None
                if processed_response.startswith("PARENT:"):
                    try:
                        parent_topic = processed_response.split("PARENT:", 1)[1].strip()
                    except IndexError:
                        logger.warning(
                            f"Could not parse PARENT: from processed response: {processed_response}. Raw response: {raw_response}"
                        )
                        parent_topic = None

                if parent_topic:
                    child_topic = topic_b if parent_topic == topic_a else topic_a

                    if (
                        parent_topic not in current_topics
                        and parent_topic not in consolidation_map_this_pass.values()
                    ):
                        current_topics.append(parent_topic)
                        logger.info(
                            f"New parent topic '{parent_topic}' identified by LLM."
                        )

                    if child_topic in current_topics:
                        consolidation_map_this_pass[child_topic] = parent_topic
                        consolidated_something_in_this_pass = True
                        logger.info(
                            f"Marking '{child_topic}' for consolidation into '{parent_topic}'."
                        )
                elif processed_response == "NO_CONSOLIDATION":
                    logger.info(f"No consolidation for '{topic_a}' and '{topic_b}'.")
                else:
                    logger.warning(
                        f"Unexpected LLM response format. Skipping consolidation for '{topic_a}' and '{topic_b}'. Response: '{processed_response}'"
                    )

            if not consolidated_something_in_this_pass:
                logger.info(
                    f"No further consolidations found in Pass {pass_count}. Ending iterative consolidation."
                )
                break

            # Apply consolidations from this pass to the graph
            for child_topic, parent_topic in consolidation_map_this_pass.items():
                # Find all nodes that need their topic updated
                nodes_to_update = [
                    n
                    for n, d in graph.nodes(data=True)
                    if d.get("topic") == child_topic
                ]

                for node_id in nodes_to_update:
                    graph.nodes[node_id]["topic"] = parent_topic

                # Re-wire edges from chunks to the new parent topic node
                child_topic_node_id = f"topic_{child_topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"
                parent_topic_node_id = f"topic_{parent_topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"

                if not graph.has_node(parent_topic_node_id):
                    graph.add_node(
                        parent_topic_node_id,
                        content=parent_topic,
                        type="topic",
                        label=parent_topic,
                    )

                if graph.has_node(child_topic_node_id):
                    for u, v, data in list(graph.edges(child_topic_node_id, data=True)):
                        if graph.nodes[v].get("type") == "chunk":
                            graph.add_edge(parent_topic_node_id, v, **data)
                    graph.remove_node(child_topic_node_id)
                    logger.info(f"Removed old topic node: {child_topic_node_id}")

        logger.info("Iterative topic consolidation finished.")
        return graph

    def process_graph(self, graph, config):
        logger.info("Parent Agent processing graph...")

        # 1. Identify topics
        overarching_topics = self.identify_topics_from_graph(graph)

        # 2. Classify and assign chunks
        processed_graph = self.classify_and_assign_chunks(graph, overarching_topics)

        # 3. Conditionally consolidate topics
        if config.get("enable_consolidation", False):
            processed_graph = self.consolidate_topics(processed_graph)
        else:
            logger.info("Topic consolidation is disabled in config.")

        # 4. Populate the organizational table
        self._populate_org_table(processed_graph)

        # 5. Save the organizational table if a path is provided
        org_table_config = config.get("org_table", {})
        org_table_path = org_table_config.get("path")
        if org_table_path:
            self.org_table.save_to_json(org_table_path)

        return processed_graph, self.org_table

    def apply_topics_to_graph(self, target_graph, reference_graph):
        """
        Applies the topic assignments from a reference graph to a target graph.
        This is used to ensure consistency between different graph types (e.g., cosine and triplets)
        without re-running expensive topic modeling.
        """
        logger.info("Applying topic structure from reference graph to target graph...")
        for node_id, node_data in reference_graph.nodes(data=True):
            if node_data.get("type") == "chunk" and node_id in target_graph:
                target_graph.nodes[node_id]["topic"] = node_data.get("topic")

        # Add topic nodes and edges to the target graph
        topics = {
            data["content"]
            for _, data in reference_graph.nodes(data=True)
            if data.get("type") == "topic"
        }
        for topic in topics:
            topic_node_id = (
                f"topic_{topic.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            )
            if not target_graph.has_node(topic_node_id):
                target_graph.add_node(
                    topic_node_id, content=topic, type="topic", label=topic
                )

            # Find all chunks with this topic and add edges
            for node_id, node_data in target_graph.nodes(data=True):
                if node_data.get("type") == "chunk" and node_data.get("topic") == topic:
                    if not target_graph.has_edge(topic_node_id, node_id):
                        target_graph.add_edge(topic_node_id, node_id, type="has_topic")

        logger.info("Finished applying topic structure.")
        return target_graph

    def update_search_similarity(self, new_similarity_value):
        for agent in self.child_agents.values():
            agent.search_similarity = new_similarity_value
        logger.info(
            f"Updated search similarity for all child agents to {new_similarity_value}"
        )
        return {"status": "success", "new_value": new_similarity_value}
