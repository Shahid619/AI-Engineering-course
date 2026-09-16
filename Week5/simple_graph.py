"""
Simple LangGraph Example
=========================
A minimal LangGraph workflow demonstrating:
1. State definition (TypedDict)
2. Nodes (Python functions that receive and update state)
3. Edges (Connecting START -> Node A -> Node B -> END)
4. Compilation and Invocation
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# -------------------------------------------------------------
# 1. Define the State
# -------------------------------------------------------------
# State is the central data structure passed between nodes.
# Every node receives this state and returns a dictionary with updates.
class GraphState(TypedDict):
    original_text: str
    processed_text: str
    word_count: int


# -------------------------------------------------------------
# 2. Define the Nodes
# -------------------------------------------------------------
# A node is simply a Python function that takes the current state
# and returns an updated dictionary for the state keys it modifies.

def uppercase_node(state: GraphState) -> dict:
    print("\n--- [Node 1: uppercase_node] ---")
    raw_text = state["original_text"]
    transformed = raw_text.upper()
    print(f"Transformed: '{raw_text}' -> '{transformed}'")
    return {"processed_text": transformed}


def count_words_node(state: GraphState) -> dict:
    print("\n--- [Node 2: count_words_node] ---")
    text = state["processed_text"]
    words = text.split()
    count = len(words)
    print(f"Counted {count} word(s) in: '{text}'")
    return {"word_count": count}


# -------------------------------------------------------------
# 3. Build the Graph
# -------------------------------------------------------------
# Initialize StateGraph with the state schema
builder = StateGraph(GraphState)

# Add nodes to the graph
builder.add_node("uppercase_step", uppercase_node)
builder.add_node("count_step", count_words_node)

# Connect the nodes using edges:
# START -> uppercase_step -> count_step -> END
builder.add_edge(START, "uppercase_step")
builder.add_edge("uppercase_step", "count_step")
builder.add_edge("count_step", END)

# -------------------------------------------------------------
# 4. Compile the Graph
# -------------------------------------------------------------
# Compiling validates the graph structure and returns a runnable
app = builder.compile()


# -------------------------------------------------------------
# 5. Run / Invoke the Graph
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=== Invoking Simple Graph ===")
    initial_input = {"original_text": "hello world from langgraph"}
    
    # Run the graph
    final_output = app.invoke(initial_input)
    
    print("\n=== Final State Output ===")
    print(final_output)

