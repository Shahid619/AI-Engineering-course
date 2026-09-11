# 🔍 Study Notes: Introduction to LangSmith

---

## 1. What is LangSmith? (Detailed Explanation)

**LangSmith** is an enterprise-grade LLM observability, debugging, testing, and evaluation platform developed by LangChain. 

In standard software engineering, developers rely on logging (e.g., Datadog, CloudWatch, Prometheus) and debuggers to inspect stack traces. However, LLM-powered applications introduce non-deterministic behavior, multi-step agent reasoning loops, tool calls, and variable token costs that traditional APMs (Application Performance Monitoring) cannot adequately capture.

LangSmith provides **complete glass-box observability** into compound AI systems—allowing engineers to visualize execution paths, isolate hallucinations or failed tool calls, measure per-step latency, organize multi-turn conversations, and monitor token expenses in real time.

---

### Why LLM Observability is Essential

1. **Non-Deterministic "Black Box" Problem:**
   * In traditional APIs, $A + B = C$ deterministically.
   * In LLM applications, a model might correctly answer a query 9 times out of 10, but fail on edge cases. Without detailed step-by-step telemetry, identifying *why* the model failed is guesswork.
2. **Compound Multi-Step Complexity:**
   * Modern applications chain together prompt formatting, retrieval from vector databases (RAG), multiple tool calls, output parsers, and guardrail validations. If the final answer is wrong, LangSmith reveals whether the culprit was bad retrieval, a malformed tool argument, or model hallucination.
3. **Financial & Latency Governance:**
   * Agent loops can easily enter runaway cycles (e.g., calling tools repeatedly until context windows fill up), costing hundreds of dollars in minutes. LangSmith enforces token accounting and latency tracing for every invocation.

---

### The 4 Core Observability Pillars in LangSmith

```
                        ┌───────────────────────────────────────────────────────────────┐
                        │                     THREAD (Session / User)                   │
                        │       e.g., thread_id = "user-session-429"                    │
                        │                                                               │
                        │  ┌─────────────────────────────────────────────────────────┐  │
                        │  │               TRACE 1 (Turn 1: User Query)              │  │
                        │  │  ┌───────────────────────────────────────────────────┐  │  │
                        │  │  │ Root Run (Chain): "Customer Billing Inquiry"      │  │  │
                        │  │  │   ├── Run (Retriever): Vector DB Query (120ms)    │  │  │
                        │  │  │   ├── Run (LLM): Llama-3.3-70b (450 tokens, $0.001)│  │
                        │  │  │   └── Run (Tool): check_invoice_status(id=101)    │  │  │
                        │  │  └───────────────────────────────────────────────────┘  │  │
                        │  └─────────────────────────────────────────────────────────┘  │
                        │                                                               │
                        │  ┌─────────────────────────────────────────────────────────┐  │
                        │  │               TRACE 2 (Turn 2: Follow-up Query)         │  │
                        │  │  ┌───────────────────────────────────────────────────┐  │  │
                        │  │  │ Root Run (Chain): "Apply 10% Discount"            │  │  │
                        │  │  │   ├── Run (LLM): Llama-3.3-70b (320 tokens)       │  │  │
                        │  │  │   └── Run (Tool): apply_coupon_code()             │  │  │
                        │  │  └───────────────────────────────────────────────────┘  │  │
                        │  └─────────────────────────────────────────────────────────┘  │
                        └───────────────────────────────────────────────────────────────┘
```

---

### 1. Traces
* **Definition:** A **Trace** is a tree-like Directed Acyclic Graph (DAG) that represents the complete end-to-end execution path of a single interaction or request, from initial user input to final output.
* **Structure:** A trace is anchored by a single **Root Run** (such as an AgentExecutor or an overarching workflow function), which spawns child runs for sub-steps.
* **Key Features:**
  * **Latency Waterfall:** Displays how much time each sub-component took (e.g., 200ms vector search vs. 1800ms model generation).
  * **Error Bubbling:** If an intermediate tool throws an exception, the trace highlights the exact node where execution halted.
  * **Tags & Metadata:** Allows annotating traces with custom tags (e.g., `environment: "production"`, `release: "v2.1"`, `user_tier: "enterprise"`).

---

### 2. Runs
* **Definition:** A **Run** is an individual execution unit—a single node inside a Trace tree. Every step that ingests input and produces output is a Run.
* **Run Types in LangSmith:**
  * `llm`: A raw call to a language model (captures prompts, completions, temperature, model name).
  * `chain`: A logical sequence connecting multiple components (e.g., LCEL chain, router, agent loop).
  * `tool`: An external function or API call executed by an agent.
  * `retriever`: A query to an index or vector store returning context documents.
  * `parser`: A serialization or extraction step converting text into structured schemas.
* **Attributes Captured Per Run:**
  * Unique Run ID (UUID)
  * Exact Input and Output payloads
  * Start Time, End Time, and Total Latency
  * Execution Status (`SUCCESS` or `ERROR` with full stack trace)
  * Token Usage Breakdown (Prompt tokens, Completion tokens, Total tokens)

---

### 3. Threads
* **Definition:** A **Thread** (or Session) groups multiple independent traces chronologically to represent an ongoing multi-turn conversation or interaction between a user and an assistant.
* **Why Threads are Crucial:**
  * Individual traces represent a single request-response cycle. In conversational agents, errors often manifest due to **context window accumulation** or **memory drift** over several turns.
  * Assigning a `thread_id` or `session_id` allows developers to inspect the entire conversation timeline, assess user retention, and review user-reported feedback on specific conversation turns.

---

### 4. Cost Tracking
* **How It Works:** LangSmith automatically matches the model identifier reported in the run (e.g., `llama-3.3-70b-versatile`, `gpt-4o`, `claude-3-5-sonnet`) with its updated token pricing index.
* **Token Metric Dimensions:**
  * **Prompt (Input) Tokens:** Cost of instructions, few-shot examples, retrieved context, and conversation history.
  * **Completion (Output) Tokens:** Cost of model responses and generated tool arguments (typically 3–5x more expensive per token than input tokens).
  * **Cached Tokens:** Discounted rates for prompt caching where supported.
* **Governance & Safeguards:**
  * Real-time dashboard breakdowns by Project, Model, User, or Tag.
  * Identifying high-expense queries and detecting infinite tool-calling loops before they drain API budgets.

---

### Summary Comparison Table

| Concept | Scope | Analogy in Traditional APM | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Run** | Single step / function call | Span / Log Event | Inspecting inputs, outputs, parameters, and status of an isolated action. |
| **Trace** | Entire end-to-end request | Distributed Trace (HTTP transaction) | Analyzing the full execution tree, latency bottlenecks, and error propagation. |
| **Thread** | Multi-turn user session | Session / User Journey | Tracking memory, context retention, and multi-turn conversational flow. |
| **Cost Tracking** | Aggregate & granular billing | Resource / Cloud Spend Meter | Monitoring token consumption, preventing budget overruns, and financial auditing. |

---

### Essential Setup & Environment Variables

To enable LangSmith in any Python environment or framework, set these standard variables:

```bash
# Enable LangChain tracing v2 engine
export LANGCHAIN_TRACING_V2="true"

# Your personal or organization LangSmith API Key (from smith.langchain.com)
export LANGCHAIN_API_KEY="lsv2_pt_xxxxxxxxxxxxxxxxxxxx"

# The target project name in your LangSmith workspace
export LANGCHAIN_PROJECT="genai-week-4-practice"

# Optional: LangSmith API Endpoint (default is public SaaS)
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
```

---

## 2. Two Concrete Code Examples

### Example 1: Full Observability Pipeline with `@traceable`, Tools, and LLM Chains
This example demonstrates how to trace custom pure-Python functions alongside LangChain components using the `@traceable` decorator, adding custom tags, metadata, and error tracking.

```python
import os
from langsmith import traceable
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. CONFIGURE LANGSMITH TELEMETRY
# In production, set these via your environment or .env file
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "week4-langsmith-intro"
# os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
# os.environ["GROQ_API_KEY"] = "your-groq-api-key"

# 2. TRACE PURE PYTHON FUNCTIONS WITH @traceable
# The @traceable decorator automatically creates a Run in LangSmith
@traceable(
    run_type="tool",
    name="query_internal_pricing_db",
    tags=["database", "pricing", "internal"]
)
def query_internal_pricing_db(product_id: str) -> dict:
    """Mock database lookup with latency and metadata."""
    catalog = {
        "PROD-001": {"name": "AI Developer Pro Seat", "base_price": 49.00, "tier": "enterprise"},
        "PROD-002": {"name": "Cloud GPU Cluster Hours", "base_price": 120.00, "tier": "standard"},
    }
    product = catalog.get(product_id.upper())
    if not product:
        raise ValueError(f"Product '{product_id}' not found in internal catalog.")
    return product

# 3. CONSTRUCT THE LLM CHAIN
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Pricing Consultant. Generate a concise quote explanation given product details."),
    ("human", "Product Data: {product_info}\nCustomer Discount Request: {discount}%\nGenerate quote summary:")
])

quote_chain = prompt | llm | StrOutputParser()

# 4. TRACE AN END-TO-END WORKFLOW (Root Run)
@traceable(
    run_type="chain",
    name="generate_customer_quote_flow",
    metadata={"service": "billing-service", "version": "1.0.4"}
)
def generate_customer_quote_flow(product_id: str, discount_pct: float) -> str:
    print(f"\n🚀 Starting quote workflow for {product_id}...")
    
    # Step 1: Tool execution run (child run)
    product_data = query_internal_pricing_db(product_id)
    
    # Step 2: LLM Chain execution run (child run)
    # Tags passed via config merge into the LangSmith trace
    result = quote_chain.invoke(
        {"product_info": str(product_data), "discount": discount_pct},
        config={"tags": ["llm-quote-generation"], "metadata": {"product_tier": product_data["tier"]}}
    )
    
    return result

# 5. EXECUTION & VERIFICATION
if __name__ == "__main__":
    try:
        # Successful Trace (generates Root Run with 2 child runs: tool + chain)
        quote = generate_customer_quote_flow(product_id="PROD-001", discount_pct=15.0)
        print("\n--- Quote Generated Successfully ---")
        print(quote)
        
        # Demonstrating Error Bubbling in LangSmith:
        # Uncommenting below creates an ERROR run in LangSmith with full stack trace:
        # generate_customer_quote_flow(product_id="INVALID-ID", discount_pct=10.0)
    except Exception as e:
        print(f"\n❌ Error captured in LangSmith: {e}")
```

---

### Example 2: Multi-Turn Conversation Threading and Token/Cost Inspection
This example demonstrates how to group multiple conversation turns under a shared **Thread (Session ID)** and monitor token consumption across turns.

```python
import os
import uuid
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. SETUP ENVIRONMENT
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "week4-threading-and-costs"

# 2. INITIALIZE MODEL WITH STREAMING USAGE TRACKING
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Support Specialist. Maintain conversation continuity and refer to past details."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chat_chain = prompt | llm

# 3. MULTI-TURN CONVERSATION ENGINE WITH THREAD ID
class ConversationalThreadSession:
    def __init__(self, thread_id: str, user_id: str):
        self.thread_id = thread_id
        self.user_id = user_id
        self.history = []

    def send_message(self, user_text: str) -> str:
        # Configure metadata and thread grouping for LangSmith
        # By setting "thread_id" or "session_id", LangSmith associates all traces under one thread
        tracing_config = {
            "metadata": {
                "thread_id": self.thread_id,
                "session_id": self.thread_id,
                "user_id": self.user_id,
                "turn_count": len(self.history) // 2 + 1
            },
            "tags": ["chat-session", f"user:{self.user_id}"]
        }
        
        # Invoke chain with explicit tracing metadata
        ai_response = chat_chain.invoke(
            {"history": self.history, "input": user_text},
            config=tracing_config
        )
        
        # Extract response content and token usage data
        content = ai_response.content
        token_usage = getattr(ai_response, "response_metadata", {}).get("token_usage", {})
        
        # Update conversation history
        self.history.append(HumanMessage(content=user_text))
        self.history.append(AIMessage(content=content))
        
        return content, token_usage

# 4. RUN MULTI-TURN SESSION
if __name__ == "__main__":
    # Create a unique thread representing a customer session
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    client_thread = ConversationalThreadSession(thread_id=session_id, user_id="user_john_doe")
    
    print(f"🧵 [Starting Conversation Thread]: {session_id}\n")
    
    # Turn 1
    t1_input = "Hello! My company name is Acme Corp and we manage 50 Kubernetes clusters."
    resp1, usage1 = client_thread.send_message(t1_input)
    print(f"User (Turn 1): {t1_input}")
    print(f"AI: {resp1}")
    print(f"📊 [Token Usage]: {usage1}\n")
    
    # Turn 2: Tests context retention and cumulative token growth
    t2_input = "What was my company name and how many clusters did I say we manage?"
    resp2, usage2 = client_thread.send_message(t2_input)
    print(f"User (Turn 2): {t2_input}")
    print(f"AI: {resp2}")
    print(f"📊 [Token Usage]: {usage2}\n")
    
    print("✅ Open your LangSmith Project dashboard to view:")
    print(f"   1. Both Traces grouped under Thread ID: {session_id}")
    print("   2. Token increment between Turn 1 and Turn 2 (Prompt tokens grow with history)")
    print("   3. Exact dollar cost calculated for each turn based on model rates")
```

---

## 3. Two Hands-On Practice Tasks

### Task 1: Instrument a Multi-Step RAG / Agent Pipeline with `@traceable` & Metadata Tagging
* **Objective:** Create an end-to-end custom RAG or tool pipeline where every step is logged into LangSmith with detailed tags, latency tracking, and custom metadata.
* **Requirements:**
  1. Define a mock retriever function `fetch_knowledge_docs(topic: str)` decorated with `@traceable(run_type="retriever")` that returns 2 relevant text snippets.
  2. Define a data sanitization function `sanitize_output(text: str)` decorated with `@traceable(run_type="parser")`.
  3. Create a parent workflow `@traceable(run_type="chain")` that ties the retriever, an LLM call (`ChatGroq` or equivalent), and the sanitizer together.
  4. Pass custom `metadata={"customer_tier": "gold", "query_type": "technical"}` in the execution configuration.
  5. **Verify:**
     * Open your LangSmith dashboard under the active project.
     * Confirm that the Trace contains a 3-level Run tree (`chain` $\rightarrow$ `retriever`, `llm`, `parser`).
     * Verify that custom metadata tags are searchable in the filter bar.

---

### Task 2: Build a Cost & Latency Budget Alert Monitor
* **Objective:** Track token costs and latency across multiple agent calls and trigger warnings if limits are exceeded.
* **Requirements:**
  1. Initialize a conversational session using a `thread_id` metadata tag.
  2. Simulate 3 sequential turns with progressively larger prompts (e.g., summarizing expanding documents).
  3. For each turn, extract token metrics (`prompt_tokens`, `completion_tokens`, `total_tokens`) from the LLM response metadata.
  4. Write a budget checker function:
     * Compute approximate cost ($ \text{Prompt Cost} = \text{tokens} \times \frac{\text{Price}}{1\text{M}}, \text{Completion Cost} = \text{tokens} \times \frac{\text{Price}}{1\text{M}} $).
     * If any single turn exceeds **1,000 tokens** or cost exceeds **\$0.005**, print a `⚠️ [BUDGET WARNING]` alert with the Thread ID.
  5. **Verify:** Confirm from both Python console output and the LangSmith Runs table that token counts match the calculated cost threshold warnings.

