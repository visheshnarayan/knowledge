# System Behavior Testing Questions

This file contains a set of questions designed to test and observe various behaviors of the agentic system.

---

### 1. Simple, Single-Agent Query

**Goal:** To confirm that a straightforward question is correctly routed to and answered by a single specialist agent without unnecessary decomposition.

*   **Question:** `What is a hypergraph?`
    *   **Expected Behavior:** The `ParentAgent` should route this question directly to the `hypergraph structures` agent. The agent should use its `similarity_search_tool` to find relevant information within its subgraph and provide a definition.

*   **Question:** `Who is the main character in the Harry Potter series?`
    *   **Expected Behavior:** The question should be routed to the `Harry Potter series` agent, which will provide the answer.

---

### 2. Decomposition Query

**Goal:** To test the `router_agent`'s ability to decompose a complex question into multiple sub-questions and route them to the correct specialist agents.

*   **Question:** `What is the connection between hypergraph structures and natural language processing?`
    *   **Expected Behavior:** The `router_agent` should create two sub-questions:
        1.  A question about "hypergraph structures" for the `hypergraph structures` agent.
        2.  A question about "natural language processing" for the `natural language processing` agent.
        The `synthesis_agent` will then combine the answers into a single response.

*   **Question:** `How can autonomous agents be used to manage knowledge graphs about the Harry Potter series?`
    *   **Expected Behavior:** The query should be decomposed and sent to the `autonomous agents` and `Harry Potter series` agents respectively. The final answer should be a synthesis of their responses.

---

### 3. Inter-Agent Consultation Query

**Goal:** To force one agent to use the `consult_expert_tool` to ask another agent for information, testing the full inter-agent communication loop.

*   **Question:** `Describe the architecture of an agentic graph management system. How would such a system process and understand the relationships between characters in the Harry Potter series?`
    *   **Expected Behavior:**
        1.  The `ParentAgent` routes the entire question to the `Agentic Graph Management` agent.
        2.  This agent answers the first part about its architecture.
        3.  For the second part, it recognizes "Harry Potter" is outside its expertise, looks up the `Harry Potter series` agent in its `OrgTable`, and uses the `consult_expert_tool` to ask about character relationships.
        4.  The response from the `Harry Potter series` agent is returned, and the `Agentic Graph Management` agent incorporates it into its final answer.

---

### 4. Fallback Query

**Goal:** To test the system's robustness by providing a vague or poorly structured question that may be difficult to decompose.

*   **Question:** `Tell me about graphs and agents.`
    *   **Expected Behavior:** The `router_agent` might fail to create distinct sub-tasks from this ambiguous prompt. This should trigger the `_fallback_query` mechanism, where the system attempts to route the *entire* question to the single most likely agent (probably `Agentic Graph Management` or `autonomous agents`).

---

### 5. Out-of-Scope Query

**Goal:** To observe how the system responds when asked a question for which no specialist agent exists.

*   **Question:** `What are the principles of quantum mechanics?`
    *   **Expected Behavior:** The `router_agent` will fail to find a relevant topic in its list of agents. The system should gracefully handle this, likely returning a message indicating that it cannot find an appropriate specialist for the question. This tests the system's ability to recognize the limits of its knowledge.

---
