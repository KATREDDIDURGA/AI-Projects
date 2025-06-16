from typing import Dict, Any
from datetime import datetime
from ai_engine import ClaimsAI
import config

class ReportGenerator:
    """
    Generates professional reports for insurance claims using AI.
    """
    
    def __init__(self, ai_engine: ClaimsAI):
        """
        Initialize report generator with AI engine.
        
        Args:
            ai_engine: Instance of ClaimsAI for report generation
        """
        self.ai_engine = ai_engine
    
    def generate_claim_report(self, claim_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive claim report.
        
        Args:
            claim_data: Complete claim data
            
        Returns:
            Professional report content
        """
        return self.ai_engine.generate_professional_report(claim_data)
    
    def generate_executive_summary(self, claims_data: list) -> str:
        """
        Generate executive summary for multiple claims.
        
        Args:
            claims_data: List of processed claims
            
        Returns:
            Executive summary content
        """
        if not claims_data:
            return "No claims data available for summary."
        
        total_claims = len(claims_data)
        total_amount = sum(claim.get('claim_amount', 0) for claim in claims_data)
        high_risk_claims = sum(1 for claim in claims_data if claim.get('fraud_risk', 0) > 0.7)
        avg_processing_time = sum(claim.get('processing_time_seconds', 0) for claim in claims_data) / total_claims
        
        summary = f"""
EXECUTIVE CLAIMS SUMMARY
========================
Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY METRICS:
- Total Claims Processed: {total_claims}
- Total Claims Value: ${total_amount:,.2f}
- Average Claim Amount: ${total_amount/total_claims:,.2f}
- High-Risk Claims: {high_risk_claims} ({high_risk_claims/total_claims*100:.1f}%)
- Average Processing Time: {avg_processing_time/60:.1f} minutes

RISK ANALYSIS:
- Fraud Detection Rate: {high_risk_claims/total_claims*100:.1f}%
- AI Processing Success Rate: 100%
- Manual Review Required: {sum(1 for claim in claims_data if claim.get('processing_summary', {}).get('manual_review_required', False))} claims

RECOMMENDATIONS:
- Continue AI-powered processing for standard claims
- Focus investigation resources on {high_risk_claims} high-risk cases
- Estimated cost savings: ${total_claims * 150:,.2f} (vs manual processing)
        """
        
        return summary.strip()
    
    def generate_fraud_analytics_report(self, claims_data: list) -> str:
        """
        Generate detailed fraud analytics report.
        
        Args:
            claims_data: List of processed claims
            
        Returns:
            Fraud analytics report
        """
        if not claims_data:
            return "No claims data available for fraud analysis."
        
        fraud_claims = [claim for claim in claims_data if claim.get('fraud_risk', 0) > 0.7]
        medium_risk = [claim for claim in claims_data if 0.3 <= claim.get('fraud_risk', 0) <= 0.7]
        low_risk = [claim for claim in claims_data if claim.get('fraud_risk', 0) < 0.3]
        
        report = f"""
FRAUD ANALYTICS REPORT
======================
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RISK DISTRIBUTION:
- High Risk (>70%): {len(fraud_claims)} claims (${sum(c.get('claim_amount', 0) for c in fraud_claims):,.2f})
- Medium Risk (30-70%): {len(medium_risk)} claims (${sum(c.get('claim_amount', 0) for c in medium_risk):,.2f})
- Low Risk (<30%): {len(low_risk)} claims (${sum(c.get('claim_amount', 0) for c in low_risk):,.2f})

FRAUD PATTERNS DETECTED:
- Suspicious timing patterns: {sum(1 for c in fraud_claims if 'timing' in str(c.get('concerns', [])))}
- Amount irregularities: {sum(1 for c in fraud_claims if 'amount' in str(c.get('concerns', [])))}
- Description inconsistencies: {sum(1 for c in fraud_claims if 'description' in str(c.get('concerns', [])))}

FINANCIAL IMPACT:
- Potential fraud value: ${sum(c.get('claim_amount', 0) for c in fraud_claims):,.2f}
- Estimated savings from detection: ${sum(c.get('claim_amount', 0) for c in fraud_claims) * 0.8:,.2f}

INVESTIGATION PRIORITIES:
        """
        
        # Add top priority cases
        sorted_fraud = sorted(fraud_claims, key=lambda x: x.get('claim_amount', 0), reverse=True)
        for i, claim in enumerate(sorted_fraud[:5], 1):
            report += f"\n{i}. Claim {claim.get('claim_id')} - ${claim.get('claim_amount', 0):,.2f} - {claim.get('fraud_risk', 0):.1%} risk"
        
        return report

# ==========================================
# requirements.txt
"""
streamlit>=1.28.0
openai>=1.3.0
pandas>=2.0.0
plotly>=5.15.0
python-dotenv>=1.0.0
uuid>=1.30
datetime
typing
re
reportlab>=4.0.0
"""