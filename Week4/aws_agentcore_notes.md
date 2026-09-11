# ☁️ Study Notes: AWS AgentCore (Amazon Bedrock)

---

## 1. What is AWS AgentCore? (Detailed Explanation)

**AWS AgentCore** (part of the Amazon Bedrock ecosystem) is AWS's modular, serverless foundation designed to help developers build, deploy, orchestrate, and operate enterprise-grade AI agents at production scale.

In standard agent development, developers often struggle with the "undifferentiated heavy lifting": provisioning serverless runtimes, managing conversational state across distributed containers, sandboxing code execution, configuring IAM security roles, and converting legacy enterprise APIs into LLM-compatible tool schemas.

AgentCore solves this by abstracting the operational complexity into a **framework-agnostic, model-agnostic, serverless platform**. Whether you build agents using **LangGraph, CrewAI, LlamaIndex, Strands, or raw LangChain**, AgentCore provides the underlying runtime, memory, gateway, and identity layer.

---

### Why AWS AgentCore? (DIY Hosting vs. AgentCore)

* **The DIY Problem:** Running agents on standard Kubernetes or EC2 requires custom state checkpointing (Postgres/Redis), manual container sandboxing for tool execution, DIY token metering, complex auth token brokering, and brittle HTTP connection management for multi-minute reasoning loops.
* **The AgentCore Solution:** A fully managed, serverless control plane where each agent execution runs inside secure, isolated microVMs, tools are managed via the open **Model Context Protocol (MCP)** standard, and identity is directly anchored in AWS IAM.

---

### Architectural Blueprint of AWS AgentCore

```
                               ┌─────────────────────────────┐
                               │  Client Application / User  │
                               └──────────────┬──────────────┘
                                              │ AWS SigV4 / HTTPS
                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   AWS AGENTCORE                                       │
 │                                                                                        │
 │   ┌────────────────────────┐  Invokes   ┌──────────────────────────────────────────┐   │
 │   │   AgentCore Gateway    │ ─────────> │            AgentCore Runtime             │   │
 │   │  (MCP Tool Management) │            │     (Isolated Serverless MicroVMs)       │   │
 │   └───────────┬────────────┘            │                                          │   │
 │               │                         │  • Framework-Agnostic (LangGraph/CrewAI) │   │
 │               ▼                         │  • Model-Agnostic (Claude, Nova, Llama)  │   │
 │   ┌────────────────────────┐            └─────────────────────┬────────────────────┘   │
 │   │ Enterprise Connectors  │                                  │                        │
 │   │ (Lambdas, APIs, OpenAPI│                                  │                        │
 │   └────────────────────────┘                                  │                        │
 │                                                               │                        │
 │               ┌───────────────────────────────────────────────┴────────────────────┐   │
 │               │                               │                                    │   │
 │               ▼                               ▼                                    ▼   │
 │   ┌──────────────────────┐        ┌──────────────────────┐         ┌──────────────────┐│
 │   │   AgentCore Memory   │        │  AgentCore Identity  │         │  Observability   ││
 │   │  • Session State     │        │  • Fine-grained IAM  │         │  • CloudWatch    ││
 │   │  • Episodic Memory   │        │  • Credential Broker │         │  • Traces & Logs ││
 │   └──────────────────────┘        └──────────────────────┘         └──────────────────┘│
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### The 5 Modular Pillars of AgentCore

#### 1. AgentCore Runtime
* **Definition:** A managed, serverless execution environment that runs agent loops and orchestration logic inside secure, ephemeral microVMs (powered by AWS Firecracker technology).
* **Key Features:**
  * **Sub-Second Spin-up:** Eliminates heavy container boot times.
  * **Scale-to-Zero:** Zero compute cost when agents are idle.
  * **Framework Agnostic:** Native support for LangGraph, CrewAI, LlamaIndex, AutoGen, and native Python scripts.

#### 2. AgentCore Gateway (MCP-Native)
* **Definition:** A unified tool integration layer that converts enterprise databases, internal REST APIs, and AWS Lambda functions into tools following the **Model Context Protocol (MCP)**.
* **Key Features:**
  * Declarative tool registration (OpenAPI 3.0 / MCP schemas).
  * Automatically maps agent tool-call JSON payloads into downstream AWS Lambda invocations or REST API requests without writing custom glue code.

#### 3. AgentCore Memory
* **Definition:** A persistent, managed state and memory service that bridges short-term conversational context with long-term episodic knowledge.
* **Key Features:**
  * **Session Memory:** Retains prompt scratchpads, message histories, and active tool call states across multi-turn sessions.
  * **Semantic/Episodic Memory:** Automatically embeds and indexes key user preferences, historical decisions, and facts across conversations so agents "remember" users over months.

#### 4. AgentCore Identity & Governance
* **Definition:** Enterprise-grade security mechanism ensuring agents act under strict least-privilege boundaries.
* **Key Features:**
  * Scoped IAM execution roles (e.g., the agent can read from `S3:AnalyticsBucket` but cannot delete objects).
  * On-behalf-of user token brokering: Ensures agents only access data that the authenticated end-user has rights to view.

#### 5. AgentCore Observability
* **Definition:** Full telemetry integration providing trace-level visibility into agent thoughts, tool latencies, and token expenditures.
* **Key Features:**
  * Automated Amazon CloudWatch metrics and log group streams.
  * Distributed trace waterfalls capturing prompt inputs, model sampling configs, and tool durations.

---

### Comparison: Traditional DIY Agent vs. AWS AgentCore

| Dimension | Traditional DIY Agent Hosting (EC2/K8s) | AWS Bedrock AgentCore |
| :--- | :--- | :--- |
| **Compute / Runtime** | Long-running containers, custom auto-scalers. | Managed serverless microVMs, auto-scales down to zero. |
| **Tool Integration** | Custom Python wrapper code for each API/Lambda. | Declarative MCP Gateway; instant Lambda & OpenAPI binding. |
| **Memory & State** | Self-managed Redis/PostgreSQL clusters. | Managed Session & Long-Term Episodic Memory store. |
| **Tool Sandboxing** | Risky on host container; complex gVisor/Docker setup. | Hardware-isolated microVM sandboxes out of the box. |
| **Security & IAM** | Custom JWTs + manual AWS SDK credential passing. | Native AWS IAM scoped role delegation and token brokering. |

---

## 2. Two Concrete Code Examples

### Example 1: Building an AgentCore-Compatible Agent with Tools via Bedrock and Boto3
This example demonstrates how to configure and invoke an agent using AWS Bedrock's agent runtime, configuring tools (Action Groups) and passing session-based execution parameters.

```python
import os
import json
import boto3
from botocore.exceptions import ClientError

# 1. INITIALIZE BEDROCK AGENT RUNTIME CLIENT
# Requires AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
bedrock_agent_runtime = boto3.client(
    service_name="bedrock-agent-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# 2. INVOKE AGENTCORE RUNTIME
def invoke_agentcore_runtime(
    agent_id: str,
    agent_alias_id: str,
    session_id: str,
    user_prompt: str
) -> str:
    """
    Invokes the Bedrock AgentCore runtime with persistent session tracking.
    Streams execution events and aggregates the final response.
    """
    print(f"\n🚀 [AgentCore Invocation] Session: {session_id}")
    print(f"👤 User: {user_prompt}\n")
    
    try:
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=user_prompt,
            enableTrace=True  # Enables trace collection for CloudWatch & debugging
        )
        
        event_stream = response.get("completion")
        full_agent_response = ""
        
        # Process streaming chunks from the AgentCore runtime
        for event in event_stream:
            # 1. Process intermediate trace events (thoughts, tool calls)
            if "trace" in event:
                trace_data = event["trace"].get("trace", {})
                
                # Check for rationale / reasoning steps
                rationale = trace_data.get("orchestrationTrace", {}).get("rationale", {})
                if rationale:
                    print(f"🧠 [AgentCore Thought]: {rationale.get('text', '').strip()}")
                
                # Check for tool / action group invocations
                invoc = trace_data.get("orchestrationTrace", {}).get("invocationInput", {})
                if invoc:
                    action = invoc.get("actionGroupInvocationInput", {})
                    print(f"🛠️ [AgentCore Tool Call]: {action.get('actionGroupName')} -> {action.get('verb')}")

            # 2. Process final text tokens
            if "chunk" in event:
                text_chunk = event["chunk"]["bytes"].decode("utf-8")
                full_agent_response += text_chunk

        return full_agent_response

    except ClientError as e:
        print(f"❌ AWS ClientError: {e.response['Error']['Message']}")
        return "An error occurred while executing the AgentCore runtime."

# 3. RUN SIMULATION
if __name__ == "__main__":
    # In production, replace with your active Agent ID and Alias ID
    SAMPLE_AGENT_ID = os.getenv("BEDROCK_AGENT_ID", "AGENT12345")
    SAMPLE_ALIAS_ID = os.getenv("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID")
    SESSION_ID = "session-enterprise-user-881"
    
    query = "Check our Q3 revenue from the ERP database and calculate tax at 18%."
    print("Simulated execution with Bedrock AgentCore Runtime:")
    print("=" * 60)
    # response = invoke_agentcore_runtime(SAMPLE_AGENT_ID, SAMPLE_ALIAS_ID, SESSION_ID, query)
    print("Ready for deployment to AWS Bedrock AgentCore environment.")
```

---

### Example 2: Implementing an MCP-Compatible Tool Lambda for AgentCore Gateway
This example shows how to write a Python AWS Lambda function that acts as an **Action Group Tool** behind the **AgentCore Gateway**, parsing the standardized invocation schema and returning formatted responses.

```python
import json

def lambda_handler(event, context):
    """
    Standard AWS Lambda handler configured as an AgentCore Gateway Tool.
    Receives structured invocation payloads from the AgentCore reasoning engine.
    """
    print(f"📥 Received AgentCore Gateway Event:\n{json.dumps(event)}")
    
    # 1. EXTRACT AGENT METADATA & PARAMETERS
    action_group = event.get("actionGroup", "unknown")
    function_name = event.get("function", "")
    parameters = {p["name"]: p["value"] for p in event.get("parameters", [])}
    
    response_body = {}
    
    # 2. ROUTE TO SPECIFIC TOOL LOGIC
    if function_name == "calculate_mortgage_payment":
        principal = float(parameters.get("principal", 0))
        annual_rate = float(parameters.get("interest_rate", 0)) / 100 / 12
        years = int(parameters.get("years", 30))
        num_payments = years * 12
        
        if annual_rate > 0 and num_payments > 0:
            monthly_payment = (principal * annual_rate * ((1 + annual_rate) ** num_payments)) / (((1 + annual_rate) ** num_payments) - 1)
            response_body = {
                "principal": principal,
                "interest_rate": annual_rate * 12 * 100,
                "years": years,
                "monthly_payment": round(monthly_payment, 2),
                "total_repaid": round(monthly_payment * num_payments, 2)
            }
        else:
            response_body = {"error": "Invalid interest rate or loan term."}
            
    elif function_name == "lookup_property_tax":
        city = parameters.get("city", "").lower()
        tax_rates = {"seattle": 0.01025, "austin": 0.0195, "boston": 0.0121}
        rate = tax_rates.get(city, 0.015)
        response_body = {"city": city.title(), "estimated_annual_tax_rate": rate}
        
    else:
        response_body = {"error": f"Tool function '{function_name}' not implemented."}

    # 3. CONSTRUCT STANDARDIZED AGENTCORE RESPONSE SCHEMA
    response_payload = {
        "response": {
            "actionGroup": action_group,
            "function": function_name,
            "functionResponse": {
                "responseBody": {
                    "TEXT": {
                        "body": json.dumps(response_body)
                    }
                }
            }
        }
    }
    
    print(f"📤 Returning response to AgentCore:\n{json.dumps(response_payload)}")
    return response_payload
```

---

## 3. Two Hands-On Practice Tasks

### Task 1: Define an MCP-Compatible OpenAPI Tool Spec for AgentCore Gateway
* **Objective:** Design the contract that allows the AgentCore Gateway to discover and invoke enterprise tools without custom boilerplate.
* **Requirements:**
  1. Write a JSON/YAML OpenAPI 3.0 specification defining two tools:
     * `get_inventory_status(item_sku: string) -> {sku, in_stock: boolean, quantity: integer}`
     * `generate_restock_order(item_sku: string, quantity: integer) -> {order_id, status: string}`
  2. Include explicit parameter descriptions so the agent's LLM knows when and why to select each tool.
  3. Map the OpenAPI specification to a simulated AWS Lambda target ARN.
  4. **Verify:** Validate the OpenAPI schema with Swagger/OpenAPI linting tools and verify parameter types conform to Bedrock Agent action group standards.

---

### Task 2: Implement Multi-Turn Session Memory with Bedrock AgentCore & IAM Scoping
* **Objective:** Build a Python client script that tests session persistence and verifies that conversation state is preserved across multiple turns.
* **Requirements:**
  1. Generate a random `sessionId` (e.g., `session-client-9912`).
  2. Perform Turn 1: *"We have a project named Apollo with a budget of $50,000."*
  3. Perform Turn 2 with the **same** `sessionId`: *"What is the project name and how much budget is allocated?"*
  4. Perform Turn 3 with a **different** `sessionId` to verify isolation: confirm that Turn 3 does *not* know about Project Apollo.
  5. Inspect the IAM permissions required for `bedrock:InvokeAgent` and specify the minimal policy JSON granting least-privilege access.
  6. **Verify:** Confirm that Turn 2 successfully retrieves state stored in AgentCore Memory, while the separate session returns a polite missing-context response.

