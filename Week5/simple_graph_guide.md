# Topic: Simple Graph in LangGraph

---

## 1. Deep Dive: What is a Simple Graph?

### Why Graphs? (The Motivation)
In traditional programming or standard LangChain chains (LCEL), execution is often strictly **linear**:
$$\text{Input} \longrightarrow \text{Step 1} \longrightarrow \text{Step 2} \longrightarrow \text{Output}$$

However, real-world AI workflows and agentic architectures are rarely purely linear. They involve:
- **Cycles & Loops**: (e.g., iterative reflection, tool-calling loops, retry until an answer is verified)
- **Branching & Decisions**: (e.g., conditional routing based on intent)
- **Multi-Agent Coordination**: (multiple specialized agents passing tasks to each other)
- **State Persistence**: (saving checkpoints, resuming sessions, human-in-the-loop approvals)

**LangGraph** models your workflow as a **Directed Graph**. Even the most basic "Simple Graph" establishes the architectural foundation needed for complex agentic systems.

---

### The 4 Pillars of Every LangGraph Application

Think of LangGraph like an assembly line in a workshop:

```
           +---------------------------------------------------+
           |                   SHARED STATE                    |
           |   (A shared digital clipboard passed between us)  |
           +---------------------------------------------------+
                                    |
                                    v
 [START]  ───────>  [ Worker / Node 1 ]  ───────>  [ Worker / Node 2 ]  ───────>  [END]
 (Input in)            (Modifies State)               (Modifies State)           (Final Out)
```

1. **State (`TypedDict` or Pydantic model)**:
   - The shared "clipboard". It defines what data exists throughout the entire process.
   - Every node receives the current state and returns a dictionary with updates to that state.

2. **Nodes (Python functions)**:
   - The "workers" on the assembly line.
   - A node is just a normal Python function:
     ```python
     def my_node(state: MyState) -> dict:
         # Do some processing
         return {"key_to_update": new_value}
     ```
   - **Key Rule**: A node only needs to return the keys it *updates*, not the entire state dictionary. LangGraph automatically merges the update.

3. **Edges (Transitions & Flow)**:
   - The conveyor belt connecting workers.
   - `START`: A special entry point where initial input enters the graph.
   - `END`: The terminal node where execution finishes.
   - Fixed edges: `builder.add_edge("node_a", "node_b")` guarantees that after `node_a` finishes, `node_b` runs.

4. **Compilation & Invocation**:
   - `builder.compile()`: Validates that there are no orphan nodes or disconnected edges. It returns a runnable object.
   - `app.invoke({"key": "value"})`: Feeds data into `START` and executes through to `END`.

---

## 2. Two Simple Examples

---

### Example 1: The Number Transformation Pipeline (Math Graph)

Let's trace numbers moving through two sequential nodes:
1. `double_node`: multiplies `number` by 2.
2. `add_ten_node`: adds 10 to the doubled number and records step history.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Step 1: Define the State
class MathState(TypedDict):
    number: int
    step_history: list[str]

# Step 2: Define the Nodes
def double_node(state: MathState) -> dict:
    doubled = state["number"] * 2
    history = state["step_history"] + [f"Doubled to {doubled}"]
    return {"number": doubled, "step_history": history}

def add_ten_node(state: MathState) -> dict:
    result = state["number"] + 10
    history = state["step_history"] + [f"Added 10 to get {result}"]
    return {"number": result, "step_history": history}

# Step 3: Build the Graph
builder = StateGraph(MathState)
builder.add_node("doubler", double_node)
builder.add_node("adder", add_ten_node)

# Flow: START -> doubler -> adder -> END
builder.add_edge(START, "doubler")
builder.add_edge("doubler", "adder")
builder.add_edge("adder", END)

# Step 4: Compile
math_app = builder.compile()

# Step 5: Run
if __name__ == "__main__":
    result = math_app.invoke({"number": 5, "step_history": ["Started with 5"]})
    print("Final Result:", result["number"])
    print("History:", result["step_history"])
```

**Output:**
```text
Final Result: 20
History: ['Started with 5', 'Doubled to 10', 'Added 10 to get 20']
```

---

### Example 2: The User Profile Formatter (Text Graph)

In this example:
1. `format_name_node`: Cleans up extra spaces and capitalizes a user's name.
2. `create_bio_node`: Generates a short bio sentence using the cleaned name and their role.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Step 1: Define State
class ProfileState(TypedDict):
    raw_name: str
    role: str
    cleaned_name: str
    bio: str

# Step 2: Define Nodes
def format_name_node(state: ProfileState) -> dict:
    cleaned = state["raw_name"].strip().title()
    return {"cleaned_name": cleaned}

def create_bio_node(state: ProfileState) -> dict:
    name = state["cleaned_name"]
    role = state["role"]
    bio = f"{name} works as an exceptional {role}."
    return {"bio": bio}

# Step 3: Connect Graph
builder = StateGraph(ProfileState)
builder.add_node("clean_name", format_name_node)
builder.add_node("make_bio", create_bio_node)

builder.add_edge(START, "clean_name")
builder.add_edge("clean_name", "make_bio")
builder.add_edge("make_bio", END)

# Step 4: Compile & Run
profile_app = builder.compile()

if __name__ == "__main__":
    user_input = {"raw_name": "  aLex joHNson  ", "role": "AI Engineer"}
    output = profile_app.invoke(user_input)
    print(output["bio"])
```

**Output:**
```text
Alex Johnson works as an exceptional AI Engineer.
```

---

## 3. Two Practice Tasks For You

### 🎯 Task 1: Temperature Converter & Messenger
Create a simple graph with two sequential nodes:
- **State Fields**:
  - `celsius: float`
  - `fahrenheit: float`
  - `summary: str`
- **Node 1 (`convert_temp`)**: Converts `celsius` to `fahrenheit` using the formula:
  $$F = (C \times 9/5) + 32$$
- **Node 2 (`create_summary`)**: Creates a string: `"It is <celsius>°C which equals <fahrenheit>°F."`
- **Goal**: Invoke it with `{"celsius": 25.0}` and verify that `fahrenheit` equals `77.0`.

---

### 🎯 Task 2: E-Commerce Bill / Discount Calculator
Create a 3-node graph representing a checkout pipeline:
- **State Fields**:
  - `item_price: float`
  - `discount_percent: float` *(e.g. 20.0 for 20%)*
  - `discounted_price: float`
  - `tax_amount: float`
  - `final_total: float`
- **Node 1 (`apply_discount`)**: Calculates `discounted_price = item_price * (1 - discount_percent / 100)`.
- **Node 2 (`calculate_tax`)**: Calculates a 10% tax on the `discounted_price` (`tax_amount = discounted_price * 0.10`).
- **Node 3 (`calculate_total`)**: Calculates `final_total = discounted_price + tax_amount`.
- **Flow**: `START -> apply_discount -> calculate_tax -> calculate_total -> END`
- **Goal**: Invoke it with `{"item_price": 100.0, "discount_percent": 20.0}` and verify the final state calculations.

