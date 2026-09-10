# 🦜🔗 Study Notes: Building AI Agents in LangChain

---

## 1. What is an AI Agent in LangChain? (Detailed Explanation)

In LangChain, an **AI Agent** is an architecture where an LLM acts as a reasoning engine to decide **which actions (tools) to take**, in **what order**, and with **what inputs**. Unlike a static chain (which follows hardcoded sequential steps like `prompt | model | parser`), an agent dynamically determines its execution path at runtime based on the user’s input and intermediate feedback.

---

### The 4 Core Building Blocks in LangChain

To build any agent in LangChain, you assemble four fundamental components:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          LangChain Agent Stack                         │
│                                                                        │
│  1. TOOLS (@tool)          ──► Python functions exposed to the LLM     │
│  2. MODEL (ChatGroq/etc.)  ──► The reasoning and decision engine       │
│  3. PROMPT & AGENT         ──► Combines system instructions + scratchpad│
│  4. AGENT EXECUTOR         ──► Runtime loop that executes tools & state│
└────────────────────────────────────────────────────────────────────────┘
```

#### 1. Tools (`@tool` decorator)
Tools are functions the LLM can execute. 
* The **function name**, **type hints**, and **docstring** are crucial: LangChain converts them into JSON schema descriptions that tell the LLM *when* and *how* to use the tool.
* Without a clear docstring, the agent won’t know what the tool does!

#### 2. Model (The Brain)
* A Chat Model (e.g., `ChatGroq`, `ChatOpenAI`, `ChatAnthropic`).
* Modern models support **native tool calling** (the model directly returns structured tool invocation objects instead of plain text).

#### 3. The Prompt & The Scratchpad
* Every agent prompt must provide a place for the agent's work-in-progress thoughts and tool results.
* In LangChain, this is represented by:
  * `MessagesPlaceholder(variable_name="agent_scratchpad")` (in tool-calling chat agents)
  * `{agent_scratchpad}` (in classic text-based ReAct prompts)

#### 4. The Agent Executor (`AgentExecutor`)
* The **runtime environment** that runs the loop:
  1. Calls the agent/model.
  2. If the agent returns an **Action** (tool call), `AgentExecutor` runs that Python function.
  3. Feeds the tool output (**Observation**) back into the prompt.
  4. Repeats until the agent returns an **AgentFinish** (Final Answer).
* Key parameters:
  * `verbose=True`: Prints every Thought, Action, and Observation in real-time.
  * `handle_parsing_errors=True`: Prevents the script from crashing if the model outputs malformed tool arguments; instructs the model to retry instead.
  * `max_iterations=5`: Caps how many steps the agent can take to prevent infinite loops.

---

### Two Major Agent Types in LangChain

| Agent Type | How It Operates | Best Used For |
| :--- | :--- | :--- |
| **Tool-Calling Agent** (`create_tool_calling_agent`) | Uses the model's native API function-calling capabilities. Fast, robust, and handles JSON arguments cleanly. | Modern models (Llama 3.1/3.3, GPT-4o, Claude 3.5). **(Recommended)** |
| **ReAct Agent** (`create_react_agent`) | Relies on raw string prompting with `Thought:`, `Action:`, `Observation:` text parsing. | Older models or models without native function-calling APIs. |

---

## 2. Two Concrete Code Examples

### Example 1: Modern Tool-Calling Agent with ChatGroq
This example demonstrates how to build an agent with multiple tools, custom parameter types, and `AgentExecutor`.

```python
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

# ----------------------------------------------------
# 1. DEFINE TOOLS (Docstrings are instructions to the LLM)
# ----------------------------------------------------
@tool
def calculate_emi(principal: float, annual_rate: float, tenure_years: int) -> str:
    """Calculates monthly loan EMI given principal amount, annual interest rate (e.g. 8.5), and tenure in years."""
    monthly_rate = (annual_rate / 100) / 12
    months = tenure_years * 12
    emi = (principal * monthly_rate * ((1 + monthly_rate) ** months)) / (((1 + monthly_rate) ** months) - 1)
    total_payment = emi * months
    return f"Monthly EMI: ${emi:.2f} | Total Payment: ${total_payment:.2f}"

@tool
def get_credit_score_bracket(score: int) -> str:
    """Checks interest rate qualification based on credit score."""
    if score >= 750:
        return "Excellent credit score. Eligible for prime rate: 6.5%."
    elif score >= 650:
        return "Good credit score. Eligible for standard rate: 8.5%."
    else:
        return "Subprime credit score. Eligible rate: 12.0%."

tools = [calculate_emi, get_credit_score_bracket]

# ----------------------------------------------------
# 2. SETUP MODEL & PROMPT
# ----------------------------------------------------
# Initialize ChatGroq (Ensure GROQ_API_KEY environment variable is set)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert loan advisor assistant. Use the tools provided to answer questions accurately."),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Stores intermediate tool calls & results
])

# ----------------------------------------------------
# 3. CREATE AGENT AND EXECUTOR
# ----------------------------------------------------
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,               # Real-time console logs of tool calls
    handle_parsing_errors=True, # Recovers from formatting errors
    max_iterations=5            # Guardrail against infinite loops
)

# ----------------------------------------------------
# 4. RUN THE AGENT
# ----------------------------------------------------
user_query = "My credit score is 780. If I take a $50,000 loan for 5 years at the rate I qualify for, what will my monthly EMI be?"
response = agent_executor.invoke({"input": user_query})

print("\n--- Final Agent Response ---")
print(response["output"])
```

---

### Example 2: Conversational Agent with Memory (Chat History)
An agent that can maintain conversational context across multiple turns while still invoking tools when needed.

```python
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage

# ----------------------------------------------------
# 1. DEFINE TOOLS
# ----------------------------------------------------
@tool
def search_product_specs(product_name: str) -> str:
    """Retrieves technical specifications for gadgets."""
    specs = {
        "macbook air m3": "8-core CPU, 10-core GPU, up to 24GB RAM, 18 hours battery life.",
        "dell xps 13": "Intel Core Ultra 7, 16GB RAM, 512GB SSD, 14 hours battery life."
    }
    return specs.get(product_name.lower().strip(), "Product specifications not found.")

tools = [search_product_specs]

# ----------------------------------------------------
# 2. PROMPT WITH CHAT HISTORY & SCRATCHPAD
# ----------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tech retail assistant. Answer queries using specifications from your tools."),
    MessagesPlaceholder(variable_name="chat_history"),     # Maintains multi-turn conversation memory
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # Stores tool calls
])

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ----------------------------------------------------
# 3. MULTI-TURN CONVERSATION LOOP
# ----------------------------------------------------
chat_history = []

def chat_with_agent(user_message: str):
    print(f"\nUser: {user_message}")
    result = agent_executor.invoke({
        "input": user_message,
        "chat_history": chat_history
    })
    
    # Update conversational memory with the user message and AI response
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=result["output"]))
    
    print(f"Agent: {result['output']}")

# Turn 1: Requires Tool Call
chat_with_agent("What are the specs for the MacBook Air M3?")

# Turn 2: Uses Conversation Memory (Referring to "it" without repeating product name)
chat_with_agent("How long does its battery last?")
```

---

## 3. Two Hands-On Practice Tasks

### Task 1: Fix Common Syntax Bugs in an Agent Script
* **Objective:** Review common setup errors when wiring `AgentExecutor` and tools.
* **Requirements:**
  1. Inspect the following common mistakes in LangChain agent scripts:
     * Using `handle_parsing_error=True` instead of the plural `handle_parsing_errors=True`.
     * Using `max_iteration=5` instead of the plural `max_iterations=5`.
     * Calling `agent_executor({"input": query})` directly instead of `agent_executor.invoke({"input": query})`.
     * Missing `MessagesPlaceholder(variable_name="agent_scratchpad")` in the prompt template.
  2. Write a clean, working script using `create_tool_calling_agent` that includes at least two financial tools and executes without runtime warnings.

---

### Task 2: Build a Multi-Step "Order Tracking & Refund" Agent
* **Objective:** Build an agent that combines multiple tools and makes conditional decisions.
* **Requirements:**
  1. Create two tools:
     * `check_order(order_id: str) -> dict`: Returns order details like `{"status": "delivered" / "in_transit", "amount": 45.0}`.
     * `process_refund(order_id: str, reason: str) -> str`: Issues a refund if the policy allows.
  2. Set a business rule in the agent's system prompt: *"Refunds can only be processed if the order status is 'delivered' and the reason is damaged or incorrect item. Never refund orders that are still in_transit."*
  3. Test with:
     * Query A: Order in transit asking for a refund $\rightarrow$ Agent must decline based on the tool's status output.
     * Query B: Delivered order asking for a refund due to damage $\rightarrow$ Agent must call `process_refund`.
