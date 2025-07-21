"""
Working Refund Agent with Human Review
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd

from app.agents.base_agent import BaseAgent, AgentContext, FallbackTrigger, AgentExecutionError
from app.core.config import settings

# Mock Together client
class MockTogether:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                class MockResponse:
                    def __init__(self):
                        self.choices = [type('obj', (object,), {
                            'message': type('obj', (object,), {
                                'content': 'refund'
                            })()
                        })()]
                return MockResponse()

together_client = MockTogether()

# Load data
try:
    transactions_df = pd.read_csv("data/transactions.csv")
    policy_df = pd.read_csv("data/policy.csv")
except FileNotFoundError as e:
    print(f"Warning: Could not load data files: {e}")
    transactions_df = pd.DataFrame()
    policy_df = pd.DataFrame()


class RefundAgent(BaseAgent):
    """Refund agent with human review capability"""
    
    def __init__(self):
        super().__init__(agent_type="refund_agent")
        self.fallback_confidence_threshold = 0.6
        self.min_confidence_threshold = 0.7
        
        # Policy lookup
        self._policy_lookup = {}
        if not policy_df.empty:
            for _, row in policy_df.iterrows():
                self._policy_lookup[row['item'].lower()] = {
                    'policy_text': row['policy_text'],
                    'allowed': row['refund_allowed'].lower() == 'yes',
                    'policy_days': self._extract_days_from_policy(row['policy_text'])
                }
    
    def get_agent_description(self) -> str:
        return "Refund Agent: Processes refund requests with human review"
    
    async def execute_logic(self, context: AgentContext) -> str:
        return await self.execute(context)
    
    def _extract_days_from_policy(self, policy_text: str) -> int:
        matches = re.findall(r'(\d+)\s*day', policy_text.lower())
        return int(matches[0]) if matches else 0
    
    async def execute(self, context: AgentContext) -> str:
        """Execute refund processing"""
        try:
            # Step 1: Intent
            await self.thought(context, "Analyzing customer query for intent classification")
            intent = await self._classify_intent(context)
            
            if intent != "refund":
                return await self._trigger_human_review(
                    context, 
                    f"Non-refund intent detected: {intent}",
                    "Route to appropriate agent"
                )
            
            # Step 2: Product
            await self.thought(context, "Extracting product information from customer query")
            product_info = await self._extract_product_info(context)
            
            if not product_info.get("item"):
                return await self._trigger_human_review(
                    context,
                    "Could not identify product from query",
                    "Request clarification from customer"
                )
            
            # Step 3: Transaction
            await self.thought(context, f"Looking up transaction for {product_info['item']}")
            transaction = await self._lookup_transaction(context, product_info["item"])
            
            if not transaction:
                return await self._trigger_human_review(
                    context,
                    f"No transaction found for {product_info['item']}",
                    "Verify customer details"
                )
            
            # Step 4: Policy
            await self.thought(context, f"Validating refund policy for {product_info['item']}")
            policy_result = await self._validate_policy(context, product_info["item"])
            
            if not policy_result.get("allowed"):
                decision = f"Refund denied - {policy_result.get('policy_text', 'Policy does not allow refunds')}"
                
                await self.observation(
                    context,
                    "Final decision: Refund denied due to policy",
                    {"reason": "policy_denied", "policy": policy_result.get('policy_text')},
                    confidence=0.9
                )
                return decision
            
            # Step 5: Timeframe
            await self.thought(context, "Analyzing timeframe compliance")
            timeframe_result = await self._analyze_timeframe(context, product_info, policy_result)
            
            # Step 6: Check for contradictions
            await self.thought(context, "Checking for business rule violations")
            rule_violation = await self._check_business_rules(context, product_info, policy_result, timeframe_result)
            
            if rule_violation:
                return await self._trigger_human_review(
                    context,
                    rule_violation,
                    "Human review required for contradiction"
                )
            
            # Step 7: Confidence
            await self.thought(context, "Calculating overall confidence score")
            overall_confidence = await self._calculate_confidence(context, transaction, policy_result, timeframe_result)
            
            # Step 8: Decision
            await self.thought(context, "Making final refund decision")
            decision = await self._make_final_decision(context, transaction, policy_result, timeframe_result, overall_confidence)
            
            return decision
            
        except Exception as e:
            return await self._trigger_human_review(
                context,
                f"System error: {str(e)}",
                "Technical review required"
            )
    
    async def _trigger_human_review(self, context: AgentContext, reason: str, suggestion: str) -> str:
        """Trigger human review"""
        await self.observation(
            context,
            f"🚨 HUMAN REVIEW REQUIRED: {reason}",
            {
                "reason": reason,
                "requires_human_review": True,
                "suggestion": suggestion
            },
            confidence=0.0
        )
        
        return f"🚨 BLOCKED - Human Review Required: {reason}"
    
    async def _check_business_rules(self, context: AgentContext, product_info: Dict, policy_result: Dict, timeframe_result: Dict) -> Optional[str]:
        """Check for business rule violations"""
        query = context.query.lower()
        
        # Check for "after X days" contradiction
        if "after" in query and "days" in query:
            after_matches = re.findall(r'after\s+(\d+)\s+days?', query)
            if after_matches:
                days_mentioned = int(after_matches[0])
                policy_days = policy_result.get("policy_days", 0)
                
                if days_mentioned >= policy_days:
                    return f"Customer says 'after {days_mentioned} days' but policy limit is {policy_days} days"
        
        # Check timeframe compliance
        if timeframe_result.get("compliant") is False:
            return "Request appears to exceed policy timeframe"
        
        return None
    
    async def _classify_intent(self, context: AgentContext) -> str:
        """Classify intent"""
        query_lower = context.query.lower()
        
        # Enhanced intent detection - look for refund-related keywords
        refund_keywords = ["refund", "return", "money back", "cancel", "defect", "broken", "broke", 
                          "stopped working", "not working", "faulty", "issue", "problem", "damaged"]
        
        if any(word in query_lower for word in refund_keywords):
            intent = "refund"
            confidence = 0.95
        else:
            intent = "other" 
            confidence = 0.7
        
        await self.observation(
            context,
            f"Intent successfully classified as {intent} request",
            {"intent": intent},
            confidence=confidence
        )
        
        return intent
    
    async def _extract_product_info(self, context: AgentContext) -> Dict[str, Any]:
        """Extract product info"""
        query_lower = context.query.lower()
        
        detected_item = None
        for _, row in transactions_df.iterrows():
            if row['item'].lower() in query_lower:
                detected_item = row['item']
                break
        
        timeframe_days = None
        timeframe_matches = re.findall(r'(\d+)\s*days?', query_lower)
        if timeframe_matches:
            timeframe_days = int(timeframe_matches[0])
        
        result = {
            "item": detected_item,
            "timeframe_days": timeframe_days,
            "extraction_confidence": 0.8 if detected_item else 0.4
        }
        
        await self.observation(
            context,
            "Product information extracted",
            result,
            confidence=result["extraction_confidence"]
        )
        
        return result
    
    async def _lookup_transaction(self, context: AgentContext, item: str) -> Optional[Dict]:
        """Look up transaction"""
        if transactions_df.empty:
            await self.observation(
                context,
                "No transaction data available",
                {"error": "empty_database"},
                confidence=0.0
            )
            return None
        
        matches = transactions_df[transactions_df['item'].str.lower() == item.lower()]
        
        if matches.empty:
            await self.observation(
                context,
                f"No transaction found for {item}",
                {"item": item, "matches_found": 0},
                confidence=0.1
            )
            return None
        
        transaction = matches.iloc[0].to_dict()
        
        await self.observation(
            context,
            f"Transaction found (exact match)",
            {"transaction_id": transaction["transaction_id"], "item": transaction["item"]},
            confidence=0.9
        )
        
        return transaction
    
    async def _validate_policy(self, context: AgentContext, item: str) -> Dict[str, Any]:
        """Validate policy"""
        item_lower = item.lower()
        
        if item_lower in self._policy_lookup:
            policy_info = self._policy_lookup[item_lower]
            
            await self.observation(
                context,
                f"Policy found (exact match)",
                {
                    "allowed": policy_info["allowed"],
                    "policy_text": policy_info["policy_text"],
                    "policy_days": policy_info["policy_days"],
                    "confidence": 0.9
                },
                confidence=0.9
            )
            
            return policy_info
        
        await self.observation(
            context,
            f"No policy found for {item}",
            {"item": item, "policy_found": False},
            confidence=0.1
        )
        
        return {"allowed": False, "policy_text": "No policy found", "policy_days": 0}
    
    async def _analyze_timeframe(self, context: AgentContext, product_info: Dict, policy_result: Dict) -> Dict[str, Any]:
        """Analyze timeframe"""
        user_timeframe = product_info.get("timeframe_days")
        policy_days = policy_result.get("policy_days", 0)
        query_lower = context.query.lower()
        
        # Check for "after X days" pattern
        if "after" in query_lower and "days" in query_lower:
            after_matches = re.findall(r'after\s+(\d+)\s+days?', query_lower)
            if after_matches:
                user_timeframe = int(after_matches[0])
                actual_timeframe = user_timeframe + 1  # "After X days" means more than X days
                
                result = {
                    "compliant": actual_timeframe <= policy_days,
                    "user_timeframe": user_timeframe,
                    "actual_timeframe": actual_timeframe,
                    "policy_limit": policy_days,
                    "status": "WITHIN_POLICY_LIMIT" if actual_timeframe <= policy_days else "EXCEEDS_POLICY_LIMIT",
                    "interpretation": f"Customer said 'after {user_timeframe} days' - interpreting as {actual_timeframe}+ days"
                }
                
                confidence = 0.9 if actual_timeframe <= policy_days else 0.1
                
                await self.observation(
                    context,
                    f"Enhanced timeframe analysis: {result['status']}",
                    result,
                    confidence=confidence
                )
                
                return result
        
        # Standard analysis
        if user_timeframe is None:
            result = {
                "compliant": None,
                "user_timeframe": None,
                "policy_limit": policy_days,
                "status": "TIMEFRAME_NOT_SPECIFIED"
            }
            await self.observation(
                context,
                "Customer did not specify timeframe",
                result,
                confidence=0.5
            )
            return result
        
        compliant = user_timeframe <= policy_days
        result = {
            "compliant": compliant,
            "user_timeframe": user_timeframe,
            "policy_limit": policy_days,
            "status": "WITHIN_POLICY_LIMIT" if compliant else "EXCEEDS_POLICY_LIMIT"
        }
        
        await self.observation(
            context,
            f"Timeframe validation: {result['status']}",
            result,
            confidence=0.9 if compliant else 0.3
        )
        
        return result
    
    async def _calculate_confidence(self, context: AgentContext, transaction: Dict, policy_result: Dict, timeframe_result: Dict) -> float:
        """Calculate confidence"""
        confidence_factors = {
            "transaction_found": 1.0 if transaction else 0.0,
            "policy_allows_refund": 1.0 if policy_result.get("allowed") else 0.0,
            "timeframe_compliant": 1.0 if timeframe_result.get("compliant") else 0.0 if timeframe_result.get("compliant") is not None else 0.5
        }
        
        weights = {"transaction_found": 0.3, "policy_allows_refund": 0.4, "timeframe_compliant": 0.3}
        
        overall_confidence = sum(
            confidence_factors[factor] * weights[factor] 
            for factor in confidence_factors
        )
        
        await self.observation(
            context,
            "Overall confidence calculated",
            {"confidence": overall_confidence, "factors": confidence_factors},
            confidence=overall_confidence
        )
        
        return overall_confidence
    
    async def _make_final_decision(self, context: AgentContext, transaction: Dict, policy_result: Dict, timeframe_result: Dict, overall_confidence: float) -> str:
        """Make final decision"""
        
        if overall_confidence >= self.min_confidence_threshold:
            decision = "approved"
            reason = "All criteria met for refund approval"
        else:
            decision = "denied"
            
            if not transaction:
                reason = "No valid transaction found"
            elif not policy_result.get("allowed"):
                reason = f"Policy does not allow refunds: {policy_result.get('policy_text')}"
            elif timeframe_result.get("compliant") is False:
                reason = f"Request exceeds {policy_result.get('policy_days', 'policy')} day limit"
            else:
                reason = f"Insufficient confidence in request ({overall_confidence:.2f})"
        
        decision_text = f"Refund {decision} - {reason}"
        
        await self.observation(
            context,
            f"Decision: Refund {decision}",
            {"reason": decision, "confidence": overall_confidence},
            confidence=overall_confidence
        )
        
        return decision_text