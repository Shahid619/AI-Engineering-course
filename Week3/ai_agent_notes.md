# 🤖 Study Notes: What is an AI Agent?

---

## 1. What is an AI Agent? (Detailed Explanation)

An **AI Agent** is an autonomous system powered by an LLM (Large Language Model) that doesn't just reply to prompts with static text, but can **perceive** its environment, **reason** through a multi-step problem, **choose and call tools**, inspect the results, and **repeat the process** until it completes a goal.

### Traditional LLM vs. AI Agent
* **Standard LLM Pipeline:**
  $$\text{User Prompt} \longrightarrow \text{Model Output}$$
  *Single-turn, passive, cannot query live data, cannot execute commands or fix its mistakes.*
* **AI Agent (ReAct Loop: Reason + Act):**
  $$\text{Goal} \longrightarrow \text{Thought} \longrightarrow \text{Action (Tool Call)} \longrightarrow \text{Observation (Result)} \longrightarrow \text{Repeat / Final Answer}$$
  *Iterative, active, interacts with outside tools (APIs, Python code, databases), and self-corrects.*

### Key Components of an Agent
1. **Model (The Brain):** Decides *what* to do and *which* tool to call.
2. **Tools (The Hands):** Functions the model can trigger (e.g., calculator, database lookup, web search).
3. **Memory:** Maintains history of user messages and past tool outputs.
4. **Execution Loop (The Controller):** Keeps feeding tool results back into the model until the model decides it has the final answer.

---

## 2. Two Concrete Code Examples

### Example 1: Building a Minimal Agent from Scratch (Pure Python Logic)
This example shows how an agent's **Reasoning & Tool Execution Loop** works under the hood without heavy frameworks.

```python
import json

# 1. DEFINE TOOLS (The "Hands" of the Agent)
def get_weather(city: str) -> str:
    """Mock weather service tool."""
    data = {"tokyo": "15°C, Rainy", "new york": "22°C, Sunny", "paris": "18°C, Cloudy"}
    return data.get(city.lower(), "Weather data not found for this city.")

def calculate_budget(daily_cost: float, days: int) -> str:
    """Budget calculator tool."""
    return f"Total estimated cost: ${daily_cost * days:.2f}"

TOOLS = {
    "get_weather": get_weather,
    "calculate_budget": calculate_budget
}

# 2. SIMULATE THE AGENT LOOP (Thought -> Action -> Observation -> Final Answer)
def run_simple_agent(user_query: str):
    print(f"\n[User Query]: {user_query}")
    
    # Step 1: Agent perceives and decides to call Tool 1
    thought_1 = "I need to check the weather in Tokyo first to see if outdoor activities work."
    action_1 = {"tool": "get_weather", "params": {"city": "Tokyo"}}
    print(f"Thought: {thought_1}")
    print(f"Action: Calling {action_1['tool']}({action_1['params']})")
    
    # Step 2: System executes tool and returns Observation
    observation_1 = TOOLS[action_1["tool"]](**action_1["params"])
    print(f"Observation: {observation_1}")
    
    # Step 3: Agent reasons about observation and decides next step
    thought_2 = "Weather is rainy. Now calculate a 3-day indoor trip budget ($120/day)."
    action_2 = {"tool": "calculate_budget", "params": {"daily_cost": 120.0, "days": 3}}
    print(f"Thought: {thought_2}")
    print(f"Action: Calling {action_2['tool']}({action_2['params']})")
    
    # Step 4: System executes Tool 2
    observation_2 = TOOLS[action_2["tool"]](**action_2["params"])
    print(f"Observation: {observation_2}")
    
    # Step 5: Final Response synthesized by Agent
    final_response = (
        f"In Tokyo, the weather is currently {observation_1}. "
        f"For a 3-day stay with indoor plans, your {observation_2}."
    )
    print(f"\n[Final Agent Answer]:\n{final_response}")

run_simple_agent("Plan a 3-day budget for Tokyo and check the weather.")
```

---

### Example 2: LangChain Tool-Calling Agent
This example uses **LangChain** and a chat model (like `ChatGroq` or `ChatOpenAI`) to dynamically decide when to call tools.

```python
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. DEFINE TOOLS WITH @tool DECORATOR
@tool
def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculates the discounted price given original price and percentage discount."""
    return round(price * (1 - discount_percent / 100), 2)

@tool
def check_item_inventory(item_name: str) -> str:
    """Checks stock status for a given product."""
    inventory = {"laptop": 5, "wireless mouse": 0, "keyboard": 14}
    stock = inventory.get(item_name.lower(), "Item not found in catalog")
    return f"Stock count for '{item_name}': {stock}"

tools = [calculate_discount, check_item_inventory]

# 2. INITIALIZE THE LLM
# (Ensure GROQ_API_KEY is set in environment or pass directly)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 3. CONSTRUCT THE PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant capable of looking up inventory and calculating prices using tools."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # Scratchpad holds thoughts and intermediate tool outputs
])

# 4. CREATE THE AGENT & EXECUTOR
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. EXECUTE THE QUERY
query = "Can I buy 2 wireless mice? Also, what would a $1200 laptop cost with a 15% discount?"
response = agent_executor.invoke({"input": query})

print("\n--- Final Result ---")
print(response["output"])
```

---

## 3. Two Hands-On Practice Tasks

### Task 1: Build a "Currency Converter & Expense Splitter" Agent
* **Objective:** Implement a 2-tool agent using LangChain or native Python.
* **Requirements:**
  1. Create Tool 1: `convert_currency(amount: float, from_curr: str, to_curr: str) -> float` (e.g. 1 USD = 0.92 EUR).
  2. Create Tool 2: `split_bill(total_amount: float, people: int) -> float`.
  3. Prompt the agent: *"We spent 450 EUR on dinner in Paris with 5 people. How much does each person owe in USD?"*
  4. **Verify:** Confirm from the verbose logs that the agent invokes `split_bill` (or calculates individual share) and `convert_currency` in the correct logical sequence.

---

### Task 2: Implement Guardrails & Error Handling in an Agent
* **Objective:** Test how an agent handles invalid queries and tool failures.
* **Requirements:**
  1. Create a tool `search_user_database(user_id: str)` that raises a `ValueError("User ID not found")` if the ID does not start with `"USR-"`.
  2. Pass this tool to an agent executor with `handle_parsing_errors=True`.
  3. Prompt the agent with an invalid ID: *"Look up details for user 9999"*.
  4. **Verify:** Observe how the agent perceives the error feedback, self-corrects, and politely explains the format requirement to the user without crashing.
