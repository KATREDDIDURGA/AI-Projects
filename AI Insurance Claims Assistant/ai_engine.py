import openai
from openai import OpenAI
import json
import re
from typing import Dict, List, Any
import config

class ClaimsAI:
    """
    Core AI engine for insurance claims processing using OpenAI API.
    This class handles all AI-powered functionality in the application.
    """
    
    def __init__(self):
        """Initialize OpenAI client with API key from config."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL
    
    def analyze_claim_description(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze claim description and extract key information using AI.
        
        Args:
            claim_data: Dictionary containing claim information
            
        Returns:
            Dictionary with extracted details and analysis
        """
        prompt = f"""
        You are an expert insurance claims analyst. Analyze the following claim and extract key information.
        
        Claim Details:
        - Claim ID: {claim_data.get('claim_id')}
        - Customer: {claim_data.get('customer_name')}
        - Policy: {claim_data.get('policy_number')}
        - Incident Type: {claim_data.get('incident_type')}
        - Claimed Amount: ${claim_data.get('claim_amount', 0):,.2f}
        - Incident Date: {claim_data.get('incident_date')}
        
        Incident Description:
        {claim_data.get('incident_description', '')}
        
        Please provide:
        1. A professional summary of the incident (2-3 sentences)
        2. Key details extracted from the description
        3. Potential concerns or red flags
        4. Questions that should be asked for clarification
        
        Format your response as JSON with keys: summary, extracted_details, concerns, clarification_questions
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert insurance claims analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the JSON response
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            # Fallback response if AI fails
            return {
                "summary": "Unable to analyze claim description automatically. Manual review required.",
                "extracted_details": {"error": str(e)},
                "concerns": ["AI analysis failed"],
                "clarification_questions": ["Please review claim manually"]
            }
    
    def assess_fraud_risk(self, claim_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess fraud risk using AI analysis of claim patterns and details.
        
        Args:
            claim_data: Original claim data
            analysis_result: Results from claim analysis
            
        Returns:
            Dictionary with fraud risk score and reasoning
        """
        prompt = f"""
        You are a fraud detection specialist for insurance claims. Analyze this claim for fraud risk.
        
        Claim Information:
        - Amount: ${claim_data.get('claim_amount', 0):,.2f}
        - Type: {claim_data.get('incident_type')}
        - Description: {claim_data.get('incident_description', '')}
        
        AI Analysis Results:
        - Summary: {analysis_result.get('summary', '')}
        - Concerns: {', '.join(analysis_result.get('concerns', []))}
        
        Consider these fraud indicators:
        1. Inconsistent details in the story
        2. Unusually high claim amount for incident type
        3. Vague or overly detailed descriptions
        4. Timing of incident (holidays, weekends)
        5. Previous claim history patterns
        6. Geographic risk factors
        
        Provide:
        1. Fraud risk score (0.0 to 1.0, where 1.0 is highest risk)
        2. Primary risk factors identified
        3. Legitimacy indicators (positive factors)
        4. Detailed reasoning for the risk assessment
        
        Format as JSON: {{"risk_score": 0.0, "risk_factors": [], "legitimacy_factors": [], "reasoning": ""}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fraud detection expert. Respond with valid JSON and be thorough in your analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            return {
                "risk_score": 0.5,
                "risk_factors": ["Unable to complete fraud analysis"],
                "legitimacy_factors": [],
                "reasoning": f"Fraud analysis failed: {str(e)}"
            }
    
    def generate_recommended_actions(self, claim_data: Dict[str, Any], fraud_assessment: Dict[str, Any]) -> List[str]:
        """
        Generate recommended actions based on claim analysis and fraud risk.
        
        Args:
            claim_data: Original claim data
            fraud_assessment: Fraud risk assessment results
            
        Returns:
            List of recommended actions
        """
        risk_score = fraud_assessment.get('risk_score', 0.5)
        claim_amount = claim_data.get('claim_amount', 0)
        
        prompt = f"""
        Based on this insurance claim analysis, recommend specific actions:
        
        Claim Amount: ${claim_amount:,.2f}
        Fraud Risk Score: {risk_score:.2f} (0.0 = low risk, 1.0 = high risk)
        Risk Factors: {', '.join(fraud_assessment.get('risk_factors', []))}
        
        Provide 3-5 specific, actionable recommendations for claims processors.
        Consider approval, investigation, documentation, or rejection actions.
        
        Format as a simple list of strings.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an insurance claims manager. Provide clear, actionable recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=400
            )
            
            # Extract recommendations from response
            content = response.choices[0].message.content
            recommendations = [line.strip('- ').strip() for line in content.split('\n') if line.strip()]
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            # Fallback recommendations based on risk score
            if risk_score < 0.3:
                return ["Approve claim for standard processing", "Verify customer identity", "Process payment within 5 business days"]
            elif risk_score < 0.7:
                return ["Request additional documentation", "Verify incident details", "Conduct phone interview with claimant"]
            else:
                return ["Flag for detailed investigation", "Request police report if available", "Assign to senior claims adjuster", "Consider claim rejection"]
    
    def generate_claim_summary(self, claim_data: Dict[str, Any], ai_analysis: Dict[str, Any]) -> str:
        """
        Generate a professional claim summary for reports.
        
        Args:
            claim_data: Original claim data
            ai_analysis: Complete AI analysis results
            
        Returns:
            Professional summary text
        """
        prompt = f"""
        Create a professional executive summary for this insurance claim:
        
        Claim ID: {claim_data.get('claim_id')}
        Customer: {claim_data.get('customer_name')}
        Incident: {claim_data.get('incident_type')}
        Amount: ${claim_data.get('claim_amount', 0):,.2f}
        Date: {claim_data.get('incident_date')}
        
        AI Analysis: {ai_analysis.get('summary', '')}
        Fraud Risk: {ai_analysis.get('fraud_risk', 0):.1%}
        
        Write a concise, professional summary (150-200 words) suitable for executive review.
        Include key facts, risk assessment, and status.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional claims manager writing executive summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Claim {claim_data.get('claim_id')} submitted by {claim_data.get('customer_name')} for ${claim_data.get('claim_amount', 0):,.2f}. AI analysis unavailable due to technical error: {str(e)}"
    
    def detailed_fraud_investigation(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform detailed fraud investigation analysis.
        
        Args:
            claim_data: Complete claim data including previous analysis
            
        Returns:
            Detailed investigation results
        """
        prompt = f"""
        Conduct a comprehensive fraud investigation analysis for this insurance claim:
        
        Claim Details:
        - ID: {claim_data.get('claim_id')}
        - Customer: {claim_data.get('customer_name')}
        - Amount: ${claim_data.get('claim_amount', 0):,.2f}
        - Type: {claim_data.get('incident_type')}
        - Description: {claim_data.get('incident_description', '')}
        
        Previous Analysis:
        - Summary: {claim_data.get('summary', '')}
        - Initial Risk Score: {claim_data.get('fraud_risk', 0):.2f}
        
        Provide a detailed investigation analysis including:
        1. Specific fraud indicators found
        2. Factors supporting claim legitimacy  
        3. Recommended investigation steps
        4. Additional evidence needed
        5. Final risk assessment with reasoning
        
        Format as JSON with keys: indicators, legitimacy_factors, investigation_steps, evidence_needed, reasoning
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior fraud investigator with 20 years of experience. Provide thorough analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            return {
                "indicators": ["Analysis error occurred"],
                "legitimacy_factors": ["Unable to assess"],
                "investigation_steps": ["Manual review required"],
                "evidence_needed": ["Technical review needed"],
                "reasoning": f"Investigation analysis failed: {str(e)}"
            }
    
    def generate_professional_report(self, claim_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive professional report for the claim.
        
        Args:
            claim_data: Complete claim data with analysis results
            
        Returns:
            Professional report content
        """
        prompt = f"""
        Generate a comprehensive professional insurance claim report:
        
        CLAIM INFORMATION:
        - Claim ID: {claim_data.get('claim_id')}
        - Customer: {claim_data.get('customer_name')}
        - Policy: {claim_data.get('policy_number')}
        - Incident Date: {claim_data.get('incident_date')}
        - Claim Amount: ${claim_data.get('claim_amount', 0):,.2f}
        - Incident Type: {claim_data.get('incident_type')}
        
        ANALYSIS RESULTS:
        - AI Summary: {claim_data.get('summary', 'Not available')}
        - Fraud Risk Score: {claim_data.get('fraud_risk', 0):.1%}
        - Key Concerns: {', '.join(claim_data.get('concerns', []))}
        
        Create a professional report with these sections:
        1. EXECUTIVE SUMMARY
        2. CLAIM DETAILS
        3. INCIDENT ANALYSIS
        4. RISK ASSESSMENT
        5. RECOMMENDATIONS
        6. NEXT STEPS
        
        Use professional language suitable for insurance executives and legal review.
        Make it comprehensive but concise (500-800 words).
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior claims manager writing official insurance reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"""
            INSURANCE CLAIM REPORT
            
            Claim ID: {claim_data.get('claim_id')}
            Customer: {claim_data.get('customer_name')}
            Amount: ${claim_data.get('claim_amount', 0):,.2f}
            
            ERROR: Unable to generate detailed report due to AI service error.
            Manual review required.
            
            Error Details: {str(e)}
            """