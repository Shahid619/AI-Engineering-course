# 🛡️📊 Study Notes: Guardrails & Evaluation in AI Systems

---

## 1. What are Guardrails & Evaluation? (The Core Distinction)

While both concepts ensure an AI system is reliable and safe, they operate at completely different stages of the AI lifecycle:

```
                            ┌──────────────────────────────────────────────┐
                            │               USER REQUEST                   │
                            └──────────────────────┬───────────────────────┘
                                                   │
                                                   ▼
                                     [1. INPUT GUARDRAILS]
                                     • Topic whitelisting
                                     • Jailbreak & injection screening
                                     • PII masking
                                                   │
                                                   ▼
                                      [2. LLM / AGENT REASONING]
                                     • Tool calling & Generation
                                                   │
                                                   ▼
                                     [3. OUTPUT GUARDRAILS]
                                     • Grounding / Hallucination gate
                                     • Format & Schema validation
                                     • Toxic/harmful speech check
                                                   │
                                                   ▼
                            ┌──────────────────────────────────────────────┐
                            │               USER RECEIVES ANSWER           │
                            └──────────────────────────────────────────────┘
                                                   │
                           (Logged to Dataset for Offline Testing)
                                                   │
                                                   ▼
                                    [4. EVALUATION (EVALS)]
                               • Automated CI/CD Benchmarks
                               • LLM-as-a-Judge (Faithfulness & Relevance)
                               • Prompt regression testing
```

| Dimension | Guardrails (Runtime Protection) | Evaluation / Evals (Offline / Benchmarking) |
| :--- | :--- | :--- |
| **When it runs** | **Real-time**, inline with every user query. | **Offline / Batch / CI-CD pipeline** before release or on sampled production logs. |
| **Primary Goal** | Intercept and block harmful inputs, hallucinations, or schema violations **before the user sees them**. | Quantify model accuracy, regression, cost, and reliability across hundreds of test cases. |
| **Latency Budget** | Extremely strict (5ms – 150ms). Cannot add massive delays. | Flexible (can take minutes or hours to judge a test suite). |
| **Action Taken** | Reject query, sanitize input, regenerate answer, or trigger human fallback. | Generates scorecards, dashboards, and passes/fails pull requests. |

---

## 2. Guardrails Deep Dive

### A. Input Guardrails
Executed **before** the request reaches the core model.
1. **Topic Whitelisting:** Rejects questions outside the agent's designated scope (e.g., preventing a banking bot from discussing politics or creative writing).
2. **Jailbreak & Injection Screening:** Detects phrases like *"Ignore previous instructions"*, *"System override"*, or known jailbreak prefixes.
3. **PII Anonymization:** Redacts social security numbers, credit cards, or personal emails before the data enters model context.

### B. Output Guardrails
Executed **after** generation, before returning text to the user.
1. **Grounding & Hallucination Filter:** Compares the generated claims against retrieved context. If claims are invented, it blocks or regenerates the answer.
2. **Schema & Deterministic Parsing:** Enforces structured output (e.g., Pydantic models). If the JSON is malformed, it triggers a repair loop.
3. **Safety & Brand Voice Filter:** Blocks toxic outputs, competitors' promotions, or off-brand advice.

---

## 3. Evaluation (Evals) Deep Dive

In non-deterministic systems, traditional unit tests like `assert response == "Hello"` fail. Instead, we use **AI Evals** (often using an **LLM-as-a-Judge**).

### The Core Evaluation Triad (RAG & Agent Metrics)

```
                       ┌───────────────────────────────┐
                       │     Retrieved Context         │
                       └───────┬───────────────┬───────┘
                               │               │
                 Context       │               │ Faithfulness / Groundedness
                Relevance      │               │ (Did the model make things up?)
                               │               │
                               ▼               ▼
┌──────────────────┐ Answer Relevance ┌───────────────────────────┐
│   User Query     ├─────────────────►│     Generated Answer      │
└──────────────────┘ (Did it answer?  └───────────────────────────┘
```

1. **Faithfulness / Groundedness:**
   * *Question:* Is every factual claim in the response directly supported by the context?
   * *Target:* $\ge 95\%$ grounded. Prevents dangerous hallucinations.
2. **Answer Relevance:**
   * *Question:* Does the response directly address the user's specific prompt without unnecessary waffle or evasion?
3. **Context Relevance / Precision:**
   * *Question:* Did the retrieval system fetch relevant context chunks, or is it polluting the prompt with noisy irrelevant text?

---

## 4. Popular Guardrails & Eval Frameworks

| Tool | Category | Key Strength |
| :--- | :--- | :--- |
| **Guardrails AI** | Guardrails | Pydantic-native validators for output structures and safety. |
| **NeMo Guardrails (NVIDIA)** | Guardrails | Colang-based programmable rails for conversational flow and boundaries. |
| **Llama Guard (Meta)** | Guardrails | Fine-tuned safety classifier model for input/output risk taxonomy. |
| **Ragas** | Evaluation | Industry standard for RAG triad evaluation (faithfulness, relevance). |
| **DeepEval** | Evaluation | Pytest-like unit testing framework for LLMs in CI/CD pipelines. |
| **TruLens** | Evaluation | Observability and evaluation engine tracking groundedness feedback loops. |

---

## 5. Concrete Code Implementation

Here is the clean, production-ready architecture from [`guardrails_and_eval.py`](file:///c:/Users/PMLS/Desktop/Practice/GenAI/practicing/guardrails_and_eval.py):

```python
"""
Guardrails & Evaluation in AI Systems
File: guardrails_and_eval.py
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# 1. INPUT GUARDRAIL: Topic Whitelist & Injection Screener
# ----------------------------------------------------------------------
class GuardrailDecision(BaseModel):
    is_safe: bool
    blocked_reason: Optional[str] = None
    sanitized_input: Optional[str] = None


class InputGuardrail:
    DISALLOWED_KEYWORDS = [
        "ignore previous instructions",
        "system override",
        "bypass security",
        "reveal api key",
    ]
    ALLOWED_DOMAINS = ["billing", "account_settings", "technical_support"]

    @classmethod
    def validate(cls, user_prompt: str, domain: str = "technical_support") -> GuardrailDecision:
        lowered = user_prompt.lower()

        # Check for injection attacks
        for bad_kw in cls.DISALLOWED_KEYWORDS:
            if bad_kw in lowered:
                return GuardrailDecision(
                    is_safe=False,
                    blocked_reason=f"Injection detected: matched forbidden phrase '{bad_kw}'"
                )

        # Check domain boundaries
        if domain not in cls.ALLOWED_DOMAINS:
            return GuardrailDecision(
                is_safe=False,
                blocked_reason=f"Domain '{domain}' is outside supported boundaries."
            )

        return GuardrailDecision(is_safe=True, sanitized_input=user_prompt.strip())


# ----------------------------------------------------------------------
# 2. OUTPUT GUARDRAIL: Grounding & Hallucination Gate
# ----------------------------------------------------------------------
class GroundingOutputGuard(BaseModel):
    has_hallucination: bool
    groundedness_score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: List[str] = Field(default_factory=list)
    action: str  # "PASS", "WARN", or "REJECT"


def evaluate_grounding(context: str, generated_answer: str) -> GroundingOutputGuard:
    context_lower = context.lower()
    sentences = [s.strip() for s in generated_answer.split(".") if len(s.strip()) > 5]
    unsupported = []

    for sentence in sentences:
        words = [w for w in sentence.lower().split() if len(w) > 4]
        matches = [w for w in words if w in context_lower]
        if len(matches) / max(len(words), 1) < 0.3:
            unsupported.append(sentence)

    score = 1.0 - (len(unsupported) / max(len(sentences), 1))
    action = "PASS" if score >= 0.7 else ("WARN" if score >= 0.4 else "REJECT")

    return GroundingOutputGuard(
        has_hallucination=len(unsupported) > 0,
        groundedness_score=round(score, 2),
        unsupported_claims=unsupported,
        action=action
    )


# ----------------------------------------------------------------------
# 3. EVALUATION (EVALS): Structured LLM-as-a-Judge
# ----------------------------------------------------------------------
class MetricEvalResult(BaseModel):
    metric_name: str
    score: int = Field(ge=1, le=5)
    reasoning: str
    passed: bool


class AgentEvalReport(BaseModel):
    user_query: str
    context: str
    agent_response: str
    faithfulness: MetricEvalResult
    answer_relevance: MetricEvalResult
    overall_passed: bool


class LLMJudgeEvaluator:
    @staticmethod
    def judge(query: str, context: str, response: str) -> AgentEvalReport:
        grounding = evaluate_grounding(context, response)
        faith_score = 5 if grounding.action == "PASS" else (3 if grounding.action == "WARN" else 1)

        faithfulness = MetricEvalResult(
            metric_name="Faithfulness",
            score=faith_score,
            reasoning=f"Groundedness score: {grounding.groundedness_score}. Claims: {grounding.unsupported_claims}",
            passed=faith_score >= 4
        )

        words_in_query = set(query.lower().split())
        words_in_response = set(response.lower().split())
        overlap = words_in_query.intersection(words_in_response)
        rel_score = 5 if len(overlap) >= 2 else 2

        answer_relevance = MetricEvalResult(
            metric_name="Answer Relevance",
            score=rel_score,
            reasoning=f"Response overlaps with query keywords: {list(overlap)}",
            passed=rel_score >= 4
        )

        return AgentEvalReport(
            user_query=query,
            context=context,
            agent_response=response,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            overall_passed=faithfulness.passed and answer_relevance.passed
        )
```

---

## 6. Summary Cheat Sheet: Guardrails vs. Evals

| Question | Guardrails | Evaluations (Evals) |
| :--- | :--- | :--- |
| **Where does it live?** | Inside your application server (runtime). | In your CI/CD test runner (GitHub Actions, Pytest). |
| **What does it protect?** | The live user from seeing bad outputs. | The development team from releasing bad models/prompts. |
| **How fast must it be?** | Real-time (< 100ms). | Asynchronous / Batch. |
| **Typical Failure Handling**| Returns fallback response or safe rejection message. | Fails test suite / blocks deployment PR. |

