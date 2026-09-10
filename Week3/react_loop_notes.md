# 🔄 Study Notes: ReAct (Reasoning + Action) Loop

---

## 1. What is the ReAct Loop? (Detailed Explanation)

**ReAct** stands for **Reasoning + Acting**. Introduced by researchers from Princeton and Google in 2022 (*Yao et al.*), it is a foundational paradigm that allows Large Language Models (LLMs) to solve complex, multi-step problems by interleaving **verbal reasoning traces** ("Thoughts") with **domain-specific actions** ("Tool Calls"), and observing their results.

---

### Why ReAct was Created: The Core Problem

Prior to ReAct, models were used in one of two ways, both of which had critical flaws:

| Approach | How it Works | The Big Flaw |
| :--- | :--- | :--- |
| **Reasoning Only** *(e.g., Chain-of-Thought)* | Model talks to itself step-by-step in natural language. | **Hallucination & Stale Data:** Has no external tools; cannot query live data, verify facts, or calculate accurately. |
| **Acting Only** *(e.g., Direct Tool Calling without Thinking)* | Model directly executes tools one after another. | **No Planning or Reflection:** Lacks strategic planning; cannot adapt when tool output is unexpected or incomplete. |
| **ReAct (Synergy)** | Interleaves **Thought** $\rightarrow$ **Action** $\rightarrow$ **Observation**. | **Best of Both Worlds:** Uses thoughts to plan and adjust; uses actions to gather factual grounded data. |

---

### The Anatomy of a ReAct Loop

A standard ReAct agent runs through an iterative cycle until it reaches the goal:

```
                  ┌────────────────────────────────────────┐
                  │              User Query                │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
           ┌─────────────────► 1. THOUGHT (Reasoning)
           │                   Model plans what to do next.
           │                          │
           │                          ▼
           │                   2. ACTION (Execution)
           │                   Model specifies tool & inputs.
           │                          │
           │                          ▼
           │                   3. OBSERVATION (Perception)
           │                   Environment executes tool &
           │                   returns output into context.
           │                          │
           └──────────────────────────┘
                      (Loop repeats until done)
                                      │
                                      ▼
                                FINAL ANSWER
```

### Standard Text Format (The ReAct Prompt Pattern)
In ReAct, the model follows a specific structured pattern:
* **Question:** The initial user task.
* **Thought:** What the model intends to do and why.
* **Action:** The name of the tool to invoke.
* **Action Input:** The arguments passed to that tool.
* **Observation:** The actual output returned from the tool.
* *(Repeats Thought $\rightarrow$ Action $\rightarrow$ Observation)*
* **Thought:** "I now know the final answer."
* **Final Answer:** The completed response delivered to the user.

---

## 2. Two Concrete Code Examples

### Example 1: Full ReAct Loop from Scratch (Pure Python)
This example shows how an agent engine parses **Thought**, **Action**, executes the function, injects the **Observation**, and stops when a **Final Answer** is found.

```python
import re

# ----------------------------------------------------
# 1. TOOL DEFINITIONS (Action Space)
# ----------------------------------------------------
def search_database(query: str) -> str:
    """Mock database lookup tool."""
    db = {
        "iphone 15 price": "$799",
        "macbook air m3 price": "$1099",
        "tax rate": "0.08"
    }
    return db.get(query.lower().strip(), "No record found.")

def calculate(expression: str) -> str:
    """Safe evaluation calculator tool."""
    try:
        # Evaluates simple mathematical expressions
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(eval(expression))
        return "Invalid math expression."
    except Exception as e:
        return f"Calculation error: {e}"

TOOLS = {
    "search_database": search_database,
    "calculate": calculate
}

# ----------------------------------------------------
# 2. SIMULATED LLM REASONING ENGINE
# (In production, this is replaced by an API call to Groq/OpenAI)
# ----------------------------------------------------
def mock_llm_react_step(prompt_context: str) -> str:
    """Simulates LLM generating Thought + Action or Final Answer."""
    if "Observation: $799" in prompt_context and "Observation: 0.08" not in prompt_context:
        return (
            "Thought: I have the iPhone 15 price ($799). Now I need to check the tax rate.\n"
            "Action: search_database\n"
            "Action Input: tax rate"
        )
    elif "Observation: 0.08" in prompt_context and "Observation: 862.92" not in prompt_context:
        return (
            "Thought: Tax rate is 0.08 (8%). Total price is 799 + (799 * 0.08).\n"
            "Action: calculate\n"
            "Action Input: 799 * 1.08"
        )
    elif "Observation: 862.92" in prompt_context:
        return (
            "Thought: I have the total price with tax calculated.\n"
            "Final Answer: An iPhone 15 costs $799 before tax. With 8% tax, the total cost is $862.92."
        )
    else:
        return (
            "Thought: I need to look up the base price of the iPhone 15 first.\n"
            "Action: search_database\n"
            "Action Input: iphone 15 price"
        )

# ----------------------------------------------------
# 3. THE REACT CONTROLLER LOOP
# ----------------------------------------------------
def run_react_agent(question: str, max_iterations: int = 5):
    context = f"Question: {question}\n"
    print("=" * 60)
    print(f"User Question: {question}")
    print("=" * 60)

    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1} ---")
        
        # 1. Model outputs Thought and next Action (or Final Answer)
        llm_response = mock_llm_react_step(context)
        print(llm_response)
        context += llm_response + "\n"

        # Check if the agent reached the Final Answer
        if "Final Answer:" in llm_response:
            final_answer = llm_response.split("Final Answer:")[-1].strip()
            print("\n🎯 Goal Reached!")
            print(f"Result: {final_answer}")
            return final_answer

        # 2. Parse Action and Action Input
        action_match = re.search(r"Action:\s*(\w+)", llm_response)
        input_match = re.search(r"Action Input:\s*(.+)", llm_response)

        if not action_match or not input_match:
            print("Error: Could not parse action. Halting.")
            break

        tool_name = action_match.group(1).strip()
        tool_input = input_match.group(1).strip()

        # 3. Execute Tool & Feed Observation Back into Context
        if tool_name in TOOLS:
            observation = TOOLS[tool_name](tool_input)
        else:
            observation = f"Error: Tool '{tool_name}' not available."

        print(f"Observation: {observation}")
        context += f"Observation: {observation}\n"

# Run the agent
run_react_agent("How much is an iPhone 15 including tax?")
```

---

### Example 2: ReAct Agent using LangChain & ChatGroq
This example shows how to use LangChain's official `create_react_agent` implementation with `ChatGroq` (using the `llama-3.3-70b-versatile` model).

```python
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# ----------------------------------------------------
# 1. DEFINE EXECUTABLE TOOLS
# ----------------------------------------------------
@tool
def get_crypto_price(symbol: str) -> str:
    """Returns current live mock price for a cryptocurrency (e.g. BTC, ETH, SOL)."""
    prices = {"BTC": "95000", "ETH": "3400", "SOL": "220"}
    sym = symbol.upper().strip()
    return prices.get(sym, f"Price unavailable for {sym}")

@tool
def calculate_investment(expression: str) -> str:
    """Evaluates mathematical operations like multiplication, division, and addition."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except Exception as err:
        return f"Error: {err}"

tools = [get_crypto_price, calculate_investment]

# ----------------------------------------------------
# 2. DEFINE THE CLASSIC REACT PROMPT TEMPLATE
# ----------------------------------------------------
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best as you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

# ----------------------------------------------------
# 3. INITIALIZE LLM & BUILD REACT AGENT
# ----------------------------------------------------
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Construct the ReAct agent
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# AgentExecutor handles the Reason + Act loop automatically
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,            # Shows Thought, Action, and Observation in real-time
    handle_parsing_errors=True,
    max_iterations=5
)

# ----------------------------------------------------
# 4. EXECUTE QUERY
# ----------------------------------------------------
query = "If I invest in 0.5 BTC and 2 ETH, what is my total portfolio value?"
response = agent_executor.invoke({"input": query})

print("\n--- Final Answer ---")
print(response["output"])
```

---

## 3. Two Hands-On Practice Tasks

### Task 1: Trace Analysis & Manual ReAct Simulation
* **Objective:** Understand how ReAct handles missing information and multi-step dependencies without code.
* **Problem:** *"Who is older: Elon Musk or Jeff Bezos, and by how many years?"*
* **Requirements:**
  1. Assume you have two tools: `wikipedia_search(query: str)` and `calculator(expression: str)`.
  2. Write down the complete text trace containing:
     * **Thought 1** $\rightarrow$ **Action 1** $\rightarrow$ **Observation 1** (Find Elon Musk's birth date)
     * **Thought 2** $\rightarrow$ **Action 2** $\rightarrow$ **Observation 2** (Find Jeff Bezos's birth date)
     * **Thought 3** $\rightarrow$ **Action 3** $\rightarrow$ **Observation 3** (Calculate difference using the calculator)
     * **Thought 4** $\rightarrow$ **Final Answer** (Synthesized comparison).
  3. Notice how the agent does not calculate in its head; it routes all math to the calculator tool.

---

### Task 2: Implement a Self-Correcting ReAct Agent with Tool Retries
* **Objective:** Observe how ReAct self-corrects when a tool call returns an error or invalid result.
* **Requirements:**
  1. In Python, write a tool `fetch_country_capital(country_name: str)`:
     * If the user inputs an abbreviation (e.g., `"USA"`, `"UK"`), have the tool return: `"Error: Country names must be full names (e.g., 'United States', 'United Kingdom')."`
     * If given a valid full name, return the correct capital.
  2. Build a ReAct agent using LangChain or Python and give it the tool.
  3. Prompt the agent: *"What is the capital of the USA?"*
  4. **Verify:** Check the logs to see how the agent reads the error in the `Observation`, creates a new `Thought` to correct its input to `"United States"`, and successfully recovers.
