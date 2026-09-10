# 🔀 Study Notes: Routing in AI Agents

---

## 1. What is Routing in AI Agents? (Detailed Explanation)

**Routing** in AI agents is the architectural pattern of dynamically directing a user request, query, or sub-task to the most appropriate **model, specialized sub-agent, tool, prompt, or database**.

Instead of treating an LLM as a single monolithic "jack-of-all-trades" that handles everything through one massive prompt, a **Router** acts as a traffic controller at the front door. It analyzes incoming queries, determines their intent, and forwards them to specialized handlers optimized for that specific domain.

---

### Why Routing is Essential

1. **Accuracy & Specialization:**
   * A prompt stuffed with 30 tools and instructions for accounting, coding, and general conversation causes the model to hallucinate or pick the wrong tool.
   * Routing lets each destination have a laser-focused system prompt and only the 2–3 tools it actually needs.

2. **Cost & Latency Optimization:**
   * Simple queries (e.g., *"Hello", "What time is it in Tokyo?"*) can be routed to a small, ultra-fast model (e.g., `llama-3.1-8b-instant`).
   * Complex mathematical reasoning or code refactoring can be routed to a large model (e.g., `llama-3.3-70b-versatile` or `gpt-4o`).

3. **Data Security & Privacy:**
   * Queries containing sensitive customer records or financial data can be routed to an on-premise private database / local model, while public knowledge queries route to web search tools.

---

### How Routing Works (The Architectural Flow)

```
                            ┌─────────────────────┐
                            │   User Input Query  │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │    ROUTER ENGINE    │
                            │  (Classifier/LLM)   │
                            └──────────┬──────────┘
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             │                         │                         │
             ▼                         ▼                         ▼
   [Path A: Math/Billing]    [Path B: Tech Support]    [Path C: General Chat]
   - Uses Calculator Tool    - Uses Docs & RAG         - Fast Cheap LLM
   - Formats invoice total   - Code Sandbox            - Casual Persona
```

---

### 4 Common Types of Routing

| Routing Strategy | Mechanism | Best Use Case |
| :--- | :--- | :--- |
| **1. Semantic / Intent Routing** | Uses an embedding model or fast LLM to classify user intent into discrete categories (e.g., `sales`, `support`, `refund`). | Directing customer questions to specialized RAG knowledge bases. |
| **2. Model / Tier Routing** | Evaluates complexity of the question and routes to an 8B vs. 70B model. | Reducing API cost and improving response speeds. |
| **3. Multi-Agent (Supervisor) Routing** | A "Boss" or "Supervisor" Agent plans tasks and delegates sub-tasks to specialist worker agents. | Complex workflows requiring multiple steps across different domains. |
| **4. Fallback Routing** | Attempts Primary Path $\rightarrow$ If tool/API errors out or confidence is low, falls back to a backup route or human agent. | High-reliability production systems. |

---

## 2. Two Concrete Code Examples

### Example 1: Intent-Based Router using LangChain Runnables
This example uses a lightweight classification chain that routes queries to different specialized prompts and handlers.

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

# 1. INITIALIZE MODEL
# Using a fast, accurate model for routing
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 2. DEFINE THE ROUTER CLASSIFICATION CHAIN
router_prompt = ChatPromptTemplate.from_template("""
Given the user input, classify its intent into exactly one of these categories:
- MATH (calculations, numbers, budgeting, financial totals)
- CODING (programming, bugs, code explanations, syntax)
- GENERAL (general conversation, greetings, definitions, other)

Respond ONLY with the category name (MATH, CODING, or GENERAL).

User input: {query}
Category:
""")

router_chain = router_prompt | llm | StrOutputParser()

# 3. DEFINE SPECIALIZED HANDLERS (Branches)
math_prompt = ChatPromptTemplate.from_template(
    "You are a Senior Mathematician. Solve this step-by-step with formulas:\n\n{query}"
)
coding_prompt = ChatPromptTemplate.from_template(
    "You are an expert Software Engineer. Provide clean, commented Python code for:\n\n{query}"
)
general_prompt = ChatPromptTemplate.from_template(
    "You are a friendly general assistant. Answer conversationally:\n\n{query}"
)

math_branch = math_prompt | llm | StrOutputParser()
coding_branch = coding_prompt | llm | StrOutputParser()
general_branch = general_prompt | llm | StrOutputParser()

# 4. BUILD THE CONDITIONAL ROUTING LOGIC
def route_query(info: dict):
    category = info["category"].strip().upper()
    print(f"🔀 [Router Decision]: Detected Category -> {category}")
    
    if "MATH" in category:
        return math_branch
    elif "CODING" in category:
        return coding_branch
    else:
        return general_branch

# 5. ASSEMBLE FULL ROUTED CHAIN
# Step A: Classify intent -> Step B: Dynamically invoke selected branch
full_routing_pipeline = (
    {"category": router_chain, "query": lambda x: x["query"]}
    | RunnableLambda(route_query)
)

# 6. TEST WITH DIFFERENT QUERIES
print("\n--- Test 1: Coding Route ---")
ans1 = full_routing_pipeline.invoke({"query": "How do I reverse a singly linked list in Python?"})
print(ans1[:200] + "...\n")

print("--- Test 2: Math Route ---")
ans2 = full_routing_pipeline.invoke({"query": "If I invest $5000 at 7% compound interest for 10 years, what is the balance?"})
print(ans2[:200] + "...\n")
```

---

### Example 2: Supervisor Agent Routing to Specialized Tools
This example demonstrates a **Supervisor Router** that dynamically chooses between specialized tools (like a live SQL database tool vs. a Web Search tool) using native tool-calling schemas.

```python
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. SPECIALIZED DOMAIN TOOLS
@tool
def customer_database_router(customer_id: str) -> str:
    """Routes query to internal private customer DB. Use ONLY when looking up customer accounts, billing, or orders."""
    db = {
        "CUST-101": "Customer: Alice, Plan: Enterprise, Status: Active, Past Invoices: Paid",
        "CUST-102": "Customer: Bob, Plan: Pro, Status: Past Due ($150 balance)"
    }
    return db.get(customer_id.upper().strip(), f"Customer record '{customer_id}' not found in internal DB.")

@tool
def public_web_search_router(search_term: str) -> str:
    """Routes query to public search engine. Use ONLY for external real-world news, public tech specs, and market trends."""
    return f"Simulated Web Search Results for '{search_term}': LangChain 1.3 released with improved multi-agent routing capabilities."

tools = [customer_database_router, public_web_search_router]

# 2. SETUP ROUTER AGENT
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an Intelligent Gateway Router for a company.\n"
        "Carefully decide which system to route queries to:\n"
        "- For internal customer accounts or billing: route to customer_database_router.\n"
        "- For general public web info or company external news: route to public_web_search_router.\n"
        "Never invent customer data without consulting the customer database."
    )),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
supervisor_router = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 3. TEST QUERIES
print("\n--- Test Query 1: Internal Account Query ---")
res1 = supervisor_router.invoke({"input": "What is the account status for customer CUST-102?"})
print("Result:", res1["output"])

print("\n--- Test Query 2: Public External Query ---")
res2 = supervisor_router.invoke({"input": "What is new in the latest LangChain updates?"})
print("Result:", res2["output"])
```

---

## 3. Two Hands-On Practice Tasks

### Task 1: Build a Cost & Latency Tier-Router
* **Objective:** Optimize token costs by using a fast model for simple questions and a heavy model for complex ones.
* **Requirements:**
  1. Instantiate two models:
     * `fast_model = ChatGroq(model="llama-3.1-8b-instant")` (Cheap & fast)
     * `smart_model = ChatGroq(model="llama-3.3-70b-versatile")` (Deep reasoning)
  2. Create a router classification step:
     * If query is casual conversation, greeting, or simple factual definition $\rightarrow$ Route to `fast_model`.
     * If query contains code, mathematical deduction, or multi-step logic $\rightarrow$ Route to `smart_model`.
  3. **Verify:** Print which model handled the query and measure the difference in response time.

---

### Task 2: Build a Safe Fallback Router
* **Objective:** Gracefully handle tool or API failures without breaking the application.
* **Requirements:**
  1. Build a primary route that queries a simulated external weather API tool `fetch_live_weather(city: str)`.
  2. Intentionally make the primary tool throw an exception (e.g. `ConnectionError("Weather API service unavailable")`).
  3. Implement a fallback branch using `with_fallbacks([fallback_chain])`:
     * When the primary API route fails, the fallback route catches the error, generates an estimate based on seasonal averages, and includes a disclaimer: *"Note: Live weather service is currently offline. Estimated average provided."*
  4. **Verify:** Confirm that when the primary route fails, the system executes the fallback without crashing.
