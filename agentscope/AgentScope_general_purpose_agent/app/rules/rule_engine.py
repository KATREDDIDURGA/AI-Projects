'''

💡 What This Is:
A plug-and-play rule engine interface that lets you swap rules easily:

One agent uses refund_rules.json

Another uses fraud_rules.yaml

Another calls an API

🧠 Why We Need This:
CTO asked: “Why are policies hardcoded?”

You’ll be able to support any rule source (file, API, DB)

Enables cross-domain usage (finance, legal, healthcare)

⏰ When It’s Used:
During agent step execution — agents will call engine.evaluate(...) to check if a rule passes.

🛠️ How It Works:
Define a base interface (RuleEngine)

Implement a SimpleRuleEngine with a rules.json file for now

Future: swap with API-backed or LangChain-based rules


'''
import json
from typing import Dict, Optional


class RuleEngine:
    """
    Base class for rule engines.
    Override `evaluate_rule` and `load_rules` for domain-specific behavior.
    """

    def load_rules(self):
        raise NotImplementedError

    def evaluate_rule(self, rule_key: str, input_data: Dict) -> Optional[Dict]:
        raise NotImplementedError


class SimpleRuleEngine(RuleEngine):
    """
    Loads rules from a JSON file and performs basic condition matching.
    Example file: rules/refund_rules.json
    """

    def __init__(self, rule_file: str):
        self.rule_file = rule_file
        self.rules = {}
        self.load_rules()

    def load_rules(self):
        with open(self.rule_file, "r") as f:
            self.rules = json.load(f)

    def evaluate_rule(self, rule_key: str, input_data: Dict) -> Optional[Dict]:
        rule = self.rules.get(rule_key)
        if not rule:
            return None

        # Example: match based on item age
        if "max_days" in rule and "days_since_purchase" in input_data:
            if input_data["days_since_purchase"] > rule["max_days"]:
                return {
                    "result": "fail",
                    "reason": rule.get("fail_reason", "Policy violation"),
                    "rule": rule_key
                }
        return {
            "result": "pass",
            "rule": rule_key
        }

