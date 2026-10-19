# Calling `/v1/decisions`

The plugin serves OpenAI's Decisions wire format (`POST /v1/decisions`) alongside
`/v1/systemone`, over the same engine and the same decision adapter. Anything written
against OpenAI's Decisions API — the official SDKs, pydantic-ai, plain HTTP — works
against this server by changing `base_url` only.

## curl

```bash
curl http://localhost:8000/v1/decisions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "granite-switch",
      "input": "My payouts have failed three times. The bank says everything is fine.",
      "questions": [
        {
          "type": "choice",
          "name": "queue",
          "instructions": "Which team should handle this ticket?",
          "choices": [
            {"value": "payments", "description": "Payout failures and refunds"},
            {"value": "account", "description": "Login and account access"},
            {"value": "other", "description": "Something else"}
          ]
        },
        {
          "type": "predicate",
          "name": "escalate",
          "instructions": "Does this need urgent attention?"
        },
        {
          "type": "score",
          "name": "severity",
          "instructions": "How urgent is this?",
          "levels": [
            {"label": "can wait", "description": "No rush"},
            {"label": "this week", "description": "Within the week"},
            {"label": "today", "description": "Same day"}
          ]
        }
      ]
    }'
```

Response (all three answer types; answers come back in question order, values
illustrative):

```json
{
  "model": "granite-switch",
  "answers": [
    {
      "type": "choice",
      "name": "queue",
      "choice": "payments",
      "probabilities": [
        {"value": "payments", "probability": 0.94},
        {"value": "account", "probability": 0.03},
        {"value": "other", "probability": 0.03}
      ],
      "confidence": 0.91
    },
    {
      "type": "predicate",
      "name": "escalate",
      "probability": 0.89
    },
    {
      "type": "score",
      "name": "severity",
      "score": 1.87,
      "probabilities": [
        {"value": 0, "label": "can wait", "probability": 0.04},
        {"value": 1, "label": "this week", "probability": 0.19},
        {"value": 2, "label": "today", "probability": 0.77}
      ],
      "confidence": 0.54
    }
  ],
  "usage": {"input_tokens": 132, "output_tokens": 3, "total_tokens": 135}
}
```

## OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

decision = client.decisions.create(
    model="granite-switch",
    input="My payouts have failed three times. The bank says everything is fine.",
    questions=[
        {
            "type": "choice",
            "name": "queue",
            "instructions": "Which team should handle this ticket?",
            "choices": [
                {"value": "payments", "description": "Payout failures and refunds"},
                {"value": "account", "description": "Login and account access"},
            ],
        },
        {"type": "predicate", "name": "escalate",
         "instructions": "Does this need urgent attention?"},
    ],
)

print(decision.answers[0].choice)       # payments
print(decision.answers[1].probability)  # 0.89
```

## The same questions through `/v1/systemone`

Both routes share one engine path: the `/v1/decisions` translation produces exactly the
internal request the systemone route builds for the equivalent body, so the answers
match. The jev-shaped equivalent of the call above:

```bash
curl http://localhost:8000/v1/systemone \
    -H "Content-Type: application/json" \
    -d '{
      "model": "granite-switch",
      "state": "My payouts have failed three times. The bank says everything is fine.",
      "questions": {
        "queue": {
          "type": "choice",
          "instructions": "Which team should handle this ticket?",
          "criteria": {
            "payments": "Payout failures and refunds",
            "account": "Login and account access",
            "other": "Something else"
          }
        },
        "escalate": {"type": "noul", "instructions": "Does this need urgent attention?"},
        "severity": {
          "type": "score",
          "instructions": "How urgent is this?",
          "criteria": ["No rush", "Within the week", "Same day"]
        }
      }
    }'
```

## Mapping and scope

- `input` → `state` (a message list flattens to its text parts, joined with blank
  lines); `predicate` ↔ `noul` (`probability` = P(true)); `choices` ↔ `criteria`;
  `levels` ↔ score `criteria`.
- Choice values typed `string | boolean` are coerced to their text form; mixing boolean
  `true` with the string `"true"` is rejected (the model cannot distinguish them).
- Image `input` parts are rejected with a 400 — text decision adapter.
- Errors use OpenAI's `{"error": {"message", "type", "param", "code"}}` shape.
- `usage` reports input/output/total tokens, without OpenAI's `*_details` breakdowns.
