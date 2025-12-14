# Changes for Question Decomposition

This document outlines the changes made to the `ParentAgent` to support question decomposition. The goal of these changes is to allow the system to break down a complex user query into smaller, more specific sub-questions that can be answered by specialized child agents.

## 1. `ParentAgent` Refactoring

The `ParentAgent` in `src/parent_agent.py` was refactored to include two new `StrandsAgent` instances:

-   **`router_agent`**: This agent is responsible for the initial decomposition of the user's question. It uses a detailed system prompt that instructs the underlying LLM to analyze the question, break it into sub-questions, and assign each sub-question to the most relevant topic. The agent is prompted to return a JSON object containing a list of these sub-tasks.

    ```python
    # src/parent_agent.py

    # New system prompt for question decomposition
    system_prompt = (
        "You are a master agent responsible for decomposing a complex question into sub-questions and routing them to the correct specialist agent."
        f"The available topics and their corresponding agents are: {list(self.child_agents.keys())}. "
        "Your task is to: "
        "1. Analyze the user's question."
        "2. Decompose it into one or more sub-questions."
        "3. For each sub-question, identify the most relevant topic from the available list."
        "4. Respond with a JSON object containing a list of these sub-questions and their assigned topics. The format should be: "
        '{'sub_tasks': [{'question': '<sub_question_1>', 'topic': '<assigned_topic_1>'}, {'question': '<sub_question_2>', 'topic': '<assigned_topic_2>'}]}'
        "Do NOT include any reasoning or conversational filler in your response. It must be only the JSON object."
    )

    self.router_agent = StrandsAgent(
        system_prompt=system_prompt,
        model=self.llm_client
    )
    ```

-   **`synthesis_agent`**: This agent is responsible for synthesizing the answers from the child agents into a single, comprehensive response. It receives the original question and the reports from the child agents and is prompted to generate a holistic answer.

    ```python
    # src/parent_agent.py

    synthesis_system_prompt = (
        "You are a master synthesis agent. Your purpose is to craft a single, comprehensive, and well-structured answer to a user's original question. "
        "You will be provided with the original question and a set of answers to sub-questions that have been gathered from specialist agents. "
        "Your task is to synthesize this information into a final response. If the sub-answers are contradictory or insufficient, acknowledge this limitation. "
        "Structure your response clearly. Do not simply list the sub-answers; integrate them into a holistic response."
    )

    self.synthesis_agent = StrandsAgent(
        system_prompt=synthesis_system_prompt,
        model=self.llm_client
    )
    ```

## 2. Modified `query` Method

The `query` method in `ParentAgent` was updated to orchestrate the new decomposition and synthesis workflow:

1.  **Decomposition**: The method now first calls the `router_agent` with the user's question to get the decomposed sub-tasks.
2.  **Concurrent Execution**: It then uses a `ThreadPoolExecutor` to concurrently query the appropriate child agents for each sub-task.
3.  **Synthesis**: After all child agents have responded, the `query` method calls the `synthesis_agent` to generate a single, coherent answer.

    ```python
    # src/parent_agent.py

    def query(self, question):
        logger.info(f"Parent agent received query: '{question}'")

        # 1. Decompose the question and route sub-questions
        decomposition_result = self.router_agent(question)
        # ... (JSON parsing logic)

        # 2. Concurrently query child agents
        reports = []
        all_sources = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._query_child_agent, task) for task in sub_tasks]
            # ... (result processing)

        # 3. Synthesize the final answer
        synthesis_prompt = (
            f"Please provide a comprehensive answer to the following original question: '{question}'\n\n"
            "To construct your answer, synthesize the reports I received from my specialist agents, which are provided below:\n\n"
            + "\n\n".join(reports)
        )
        
        final_response_result = self.synthesis_agent(synthesis_prompt)
        # ... (return final response)
    ```

## 3. Fallback Mechanism

A `_fallback_query` method was introduced to handle cases where the `router_agent` fails to return a valid JSON object or returns no sub-tasks. This method implements the original, simpler logic of routing the entire question to a single agent.

    ```python
    # src/parent_agent.py

    def _fallback_query(self, question):
        # This is the original single-agent query logic
        fallback_router_prompt = (
            "You are a routing agent. Your job is to determine the topic of a question and select the appropriate specialist agent."
            f"The available topics are: {list(self.child_agents.keys())}. "
            "Respond with only the name of the topic, and nothing else. Do NOT include any reasoning or conversational filler."
        )
        fallback_router = StrandsAgent(system_prompt=fallback_router_prompt, model=self.llm_client)
        
        topic_agent_result = fallback_router(question)
        topic = str(topic_agent_result).strip()

        child_agent = self.child_agents.get(topic)
        # ... (query child agent and return response)
    ```