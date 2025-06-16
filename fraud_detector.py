from typing import Dict, Any, List
from ai_engine import ClaimsAI
from datetime import datetime, timedelta

class FraudDetector:
    """
    Specialized fraud detection using AI analysis.
    Provides detailed fraud investigation capabilities.
    """
    
    def __init__(self, ai_engine: ClaimsAI):
        """
        Initialize fraud detector with AI engine.
        
        Args:
            ai_engine: Instance of ClaimsAI for AI-powered analysis
        """
        self.ai_engine = ai_engine
        self.fraud_thresholds = {
            'low_risk': 0.3,
            'medium_risk': 0.7,
            'high_risk': 0.9
        }
    
    def detailed_fraud_analysis(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive fraud analysis on a claim.
        
        Args:
            claim_data: Complete claim data including previous analysis
            
        Returns:
            Detailed fraud analysis results
        """
        # Get detailed investigation from AI
        investigation_result = self.ai_engine.detailed_fraud_investigation(claim_data)
        
        # Add additional fraud checks
        behavioral_indicators = self._analyze_behavioral_patterns(claim_data)
        temporal_indicators = self._analyze_temporal_patterns(claim_data)
        amount_indicators = self._analyze_amount_patterns(claim_data)
        
        # Combine all analysis
        detailed_analysis = {
            'indicators': investigation_result.get('indicators', []),
            'legitimacy_factors': investigation_result.get('legitimacy_factors', []),
            'investigation_steps': investigation_result.get('investigation_steps', []),
            'evidence_needed': investigation_result.get('evidence_needed', []),
            'reasoning': investigation_result.get('reasoning', ''),
            'behavioral_flags': behavioral_indicators,
            'temporal_flags': temporal_indicators,
            'amount_flags': amount_indicators,
            'overall_risk_level': self._calculate_overall_risk_level(claim_data),
            'recommended_priority': self._get_investigation_priority(claim_data)
        }
        
        return detailed_analysis
    
    def _analyze_behavioral_patterns(self, claim_data: Dict[str, Any]) -> List[str]:
        """
        Analyze behavioral indicators that might suggest fraud.
        
        Args:
            claim_data: Claim data to analyze
            
        Returns:
            List of behavioral fraud indicators
        """
        indicators = []
        description = claim_data.get('incident_description', '').lower()
        
        # Check for overly detailed descriptions (potential coaching)
        if len(description) > 500:
            indicators.append("Unusually detailed incident description may indicate coaching")
        
        # Check for vague descriptions
        if len(description) < 50:
            indicators.append("Vague incident description lacks necessary details")
        
        # Check for emotional language
        emotional_words = ['devastated', 'traumatized', 'horrible', 'nightmare', 'disaster']
        if any(word in description for word in emotional_words):
            indicators.append("Excessive emotional language may be manipulation attempt")
        
        # Check for technical inconsistencies
        if 'auto accident' in claim_data.get('incident_type', '').lower():
            if 'no injuries' in description and claim_data.get('claim_amount', 0) > 50000:
                indicators.append("High claim amount for no-injury accident needs verification")
        
        return indicators
    
    def _analyze_temporal_patterns(self, claim_data: Dict[str, Any]) -> List[str]:
        """
        Analyze timing-related fraud indicators.
        
        Args:
            claim_data: Claim data to analyze
            
        Returns:
            List of temporal fraud indicators
        """
        indicators = []
        
        try:
            incident_date = datetime.strptime(claim_data.get('incident_date', ''), '%Y-%m-%d')
            current_date = datetime.now()
            
            # Check reporting delay
            days_delayed = (current_date - incident_date).days
            if days_delayed > 30:
                indicators.append(f"Claim reported {days_delayed} days after incident - unusual delay")
            
            # Check for weekend/holiday incidents (higher fraud rates)
            if incident_date.weekday() >= 5:  # Saturday or Sunday
                indicators.append("Weekend incident - statistically higher fraud risk")
            
            # Check for end-of-month incidents
            if incident_date.day >= 28:
                indicators.append("End-of-month incident timing may indicate financial pressure")
                
        except (ValueError, TypeError):
            indicators.append("Invalid or missing incident date")
        
        return indicators
    
    def _analyze_amount_patterns(self, claim_data: Dict[str, Any]) -> List[str]:
        """
        Analyze claim amount patterns for fraud indicators.
        
        Args:
            claim_data: Claim data to analyze
            
        Returns:
            List of amount-related fraud indicators
        """
        indicators = []
        claim_amount = claim_data.get('claim_amount', 0)
        incident_type = claim_data.get('incident_type', '')
        
        # Check for round numbers (potential estimate inflation)
        if claim_amount > 1000 and claim_amount % 1000 == 0:
            indicators.append("Round number claim amount may indicate estimation or inflation")
        
        # Check amounts vs incident type
        type_thresholds = {
            'Auto Accident': 75000,
            'Property Damage': 50000,
            'Theft': 25000,
            'Natural Disaster': 200000
        }
        
        threshold = type_thresholds.get(incident_type, 100000)
        if claim_amount > threshold:
            indicators.append(f"Claim amount (${claim_amount:,.2f}) exceeds typical range for {incident_type}")
        
        # Check for suspiciously low amounts (potential underreporting for quick settlement)
        if claim_amount < 500:
            indicators.append("Unusually low claim amount may indicate quick settlement attempt")
        
        return indicators
    
    def _calculate_overall_risk_level(self, claim_data: Dict[str, Any]) -> str:
        """
        Calculate overall fraud risk level based on all indicators.
        
        Args:
            claim_data: Complete claim data
            
        Returns:
            Risk level string (LOW, MEDIUM, HIGH, CRITICAL)
        """
        fraud_risk = claim_data.get('fraud_risk', 0.5)
        
        if fraud_risk >= self.fraud_thresholds['high_risk']:
            return 'CRITICAL'
        elif fraud_risk >= self.fraud_thresholds['medium_risk']:
            return 'HIGH'
        elif fraud_risk >= self.fraud_thresholds['low_risk']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_investigation_priority(self, claim_data: Dict[str, Any]) -> str:
        """
        Determine investigation priority based on risk and claim characteristics.
        
        Args:
            claim_data: Complete claim data
            
        Returns:
            Priority level string
        """
        risk_level = self._calculate_overall_risk_level(claim_data)
        claim_amount = claim_data.get('claim_amount', 0)
        
        if risk_level == 'CRITICAL' or claim_amount > 100000:
            return 'IMMEDIATE'
        elif risk_level == 'HIGH' or claim_amount > 50000:
            return 'URGENT'
        elif risk_level == 'MEDIUM':
            return 'STANDARD'
        else:
            return 'ROUTINE'
    
    def generate_fraud_report(self, claim_data: Dict[str, Any], detailed_analysis: Dict[str, Any]) -> str:
        """
        Generate a comprehensive fraud investigation report.
        
        Args:
            claim_data: Original claim data
            detailed_analysis: Results from detailed fraud analysis
            
        Returns:
            Formatted fraud report
        """
        report = f"""
FRAUD INVESTIGATION REPORT
==========================

Claim ID: {claim_data.get('claim_id')}
Customer: {claim_data.get('customer_name')}
Policy: {claim_data.get('policy_number')}
Claim Amount: ${claim_data.get('claim_amount', 0):,.2f}
Investigation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK ASSESSMENT
---------------
Overall Risk Level: {detailed_analysis.get('overall_risk_level', 'UNKNOWN')}
Fraud Risk Score: {claim_data.get('fraud_risk', 0):.1%}
Investigation Priority: {detailed_analysis.get('recommended_priority', 'STANDARD')}

FRAUD INDICATORS IDENTIFIED
---------------------------
"""
        
        indicators = detailed_analysis.get('indicators', [])
        if indicators:
            for i, indicator in enumerate(indicators, 1):
                report += f"{i}. {indicator}\n"
        else:
            report += "No significant fraud indicators identified.\n"
        
        report += "\nLEGITIMACY FACTORS\n------------------\n"
        legitimacy_factors = detailed_analysis.get('legitimacy_factors', [])
        if legitimacy_factors:
            for i, factor in enumerate(legitimacy_factors, 1):
                report += f"{i}. {factor}\n"
        else:
            report += "No significant legitimacy factors identified.\n"
        
        report += "\nRECOMMENDED INVESTIGATION STEPS\n------------------------------\n"
        investigation_steps = detailed_analysis.get('investigation_steps', [])
        if investigation_steps:
            for i, step in enumerate(investigation_steps, 1):
                report += f"{i}. {step}\n"
        else:
            report += "Standard claim processing procedures apply.\n"
        
        report += f"\nINVESTIGATOR REASONING\n---------------------\n{detailed_analysis.get('reasoning', 'No detailed reasoning available.')}\n"
        
        return report
    
    def compare_with_fraud_patterns(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare claim against known fraud patterns.
        
        Args:
            claim_data: Claim data to analyze
            
        Returns:
            Pattern matching results
        """
        # This would typically integrate with a fraud pattern database
        # For demo purposes, we'll simulate pattern matching
        
        common_patterns = {
            'staged_accident': {
                'description': 'Staged auto accident with minimal damage but high medical claims',
                'indicators': ['minor damage', 'delayed medical treatment', 'multiple passengers'],
                'risk_multiplier': 1.5
            },
            'inflated_repair': {
                'description': 'Inflated repair costs for property damage',
                'indicators': ['round number amounts', 'single estimate', 'preferred contractor'],
                'risk_multiplier': 1.3
            },
            'false_theft': {
                'description': 'False theft claim for items never owned',
                'indicators': ['no police report', 'high-value items', 'vague descriptions'],
                'risk_multiplier': 2.0
            }
        }
        
        description = claim_data.get('incident_description', '').lower()
        incident_type = claim_data.get('incident_type', '').lower()
        
        matched_patterns = []
        total_risk_multiplier = 1.0
        
        for pattern_name, pattern_data in common_patterns.items():
            indicator_matches = sum(1 for indicator in pattern_data['indicators'] 
                                  if indicator.lower() in description)
            
            if indicator_matches >= 2:  # Threshold for pattern match
                matched_patterns.append({
                    'pattern': pattern_name,
                    'description': pattern_data['description'],
                    'matches': indicator_matches,
                    'indicators': pattern_data['indicators']
                })
                total_risk_multiplier *= pattern_data['risk_multiplier']
        
        return {
            'matched_patterns': matched_patterns,
            'pattern_risk_multiplier': min(total_risk_multiplier, 3.0),  # Cap at 3x
            'pattern_analysis_summary': f"Found {len(matched_patterns)} potential fraud patterns"
        }