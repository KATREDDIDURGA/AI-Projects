'''
💡 What This Is:
A clean way for agents to raise fallbacks when they hit unsupported queries, rule violations, or unknown paths.

🧠 Why We Need This:
Agents must say: “I don’t know” or “This needs escalation” cleanly

Helps developers understand why fallbacks are triggered (via logs + graph)

Satisfies the CEO's request for transparent error recovery paths

⏰ When It’s Used:
Any time an agent cannot proceed confidently — e.g., invalid input, policy breach, unsupported logic.

🛠️ How It Works:
Raise FallbackTrigger with message, confidence, and optional suggestion

Captured by ExecutionService and broadcast to WebSocket clients + stored
'''
class FallbackTrigger(Exception):
    """
    Exception to raise when an agent hits a fallback condition.
    Includes reason and optional suggestion for recovery/escalation.
    """

    def __init__(self, reason: str, confidence: float = 0.0, suggestion: str = "Please escalate or rephrase."):
        super().__init__(reason)
        self.reason = reason
        self.confidence = confidence
        self.suggestion = suggestion
