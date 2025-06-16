from typing import Dict, Any
from datetime import datetime
import uuid
from ai_engine import ClaimsAI
from customer_database import CustomerDatabase

class ClaimProcessor:
    """
    Handles the business logic for processing insurance claims.
    Orchestrates AI analysis and manages claim workflow with customer history.
    """
    
    def __init__(self, ai_engine: ClaimsAI):
        """
        Initialize claim processor with AI engine and customer database.
        
        Args:
            ai_engine: Instance of ClaimsAI for AI-powered analysis
        """
        self.ai_engine = ai_engine
        self.customer_db = CustomerDatabase()
        self.processing_start_time = None
    
    def process_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method to process a complete insurance claim using AI and customer history.
        
        Args:
            claim_data: Dictionary containing all claim information
            
        Returns:
            Dictionary with complete analysis results including customer history
        """
        self.processing_start_time = datetime.now()
        
        try:
            # Step 1: Validate claim data
            validation_result = self._validate_claim_data(claim_data)
            if not validation_result['valid']:
                return self._create_error_response(validation_result['errors'])
            
            # Step 2: Customer History Lookup
            customer_history = self._lookup_customer_history(claim_data)
            
            # Step 3: Auto-fill missing information from customer history
            enhanced_claim_data = self._enhance_claim_with_history(claim_data, customer_history)
            
            # Step 4: AI Analysis of claim description (enhanced with history)
            ai_analysis = self.ai_engine.analyze_claim_description(enhanced_claim_data)
            
            # Step 5: Enhanced fraud assessment with customer history
            fraud_assessment = self._enhanced_fraud_assessment(enhanced_claim_data, ai_analysis, customer_history)
            
            # Step 6: Generate recommended actions (considering history)
            recommended_actions = self._generate_enhanced_recommendations(
                enhanced_claim_data, fraud_assessment, customer_history
            )
            
            # Step 7: Create processing summary
            processing_summary = self._create_processing_summary(enhanced_claim_data, customer_history)
            
            # Combine all results
            complete_analysis = {
                'processing_id': str(uuid.uuid4()),
                'processed_at': datetime.now().isoformat(),
                'processing_time_seconds': self._get_processing_time(),
                'status': 'completed',
                'summary': ai_analysis.get('summary', 'No summary available'),
                'extracted_details': ai_analysis.get('extracted_details', {}),
                'concerns': ai_analysis.get('concerns', []),
                'clarification_questions': ai_analysis.get('clarification_questions', []),
                'fraud_risk': fraud_assessment.get('risk_score', 0.5),
                'risk_factors': fraud_assessment.get('risk_factors', []),
                'legitimacy_factors': fraud_assessment.get('legitimacy_factors', []),
                'fraud_reasoning': fraud_assessment.get('reasoning', ''),
                'recommended_actions': recommended_actions,
                'processing_summary': processing_summary,
                'ai_confidence': self._calculate_ai_confidence(ai_analysis, fraud_assessment),
                # New customer history information
                'customer_history': customer_history,
                'enhanced_data_used': enhanced_claim_data != claim_data,
                'historical_risk_analysis': customer_history.get('risk_analysis', {}) if customer_history['found'] else {},
                'base_risk_score': fraud_assessment.get('base_risk_score', fraud_assessment.get('risk_score', 0.5)),
                'historical_adjustment': fraud_assessment.get('historical_adjustment', 1.0)
            }
            
            # Save this claim to customer database
            self.customer_db.add_new_claim({
                **enhanced_claim_data,
                **complete_analysis
            })
            
            return complete_analysis
            
        except Exception as e:
            return self._create_error_response([f"Processing failed: {str(e)}"])
    
    def _lookup_customer_history(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Look up customer history using name, phone, or policy number.
        
        Args:
            claim_data: Current claim data
            
        Returns:
            Customer history information
        """
        customer_name = claim_data.get('customer_name', '')
        phone = claim_data.get('phone', '')
        policy_number = claim_data.get('policy_number', '')
        
        # Try lookup by name first
        if customer_name:
            history = self.customer_db.lookup_customer_by_name(customer_name)
            if history['found']:
                return history
        
        # Try lookup by phone
        if phone:
            history = self.customer_db.lookup_customer_by_phone(phone)
            if history['found']:
                return history
        
        # Try lookup by policy number
        if policy_number:
            history = self.customer_db.lookup_customer_by_policy(policy_number)
            if history['found']:
                return history
        
        # No history found
        return {'found': False, 'is_new_customer': True}
    
    def _enhance_claim_with_history(self, claim_data: Dict[str, Any], customer_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance claim data with information from customer history.
        
        Args:
            claim_data: Original claim data
            customer_history: Customer history lookup result
            
        Returns:
            Enhanced claim data with auto-filled information
        """
        enhanced_data = claim_data.copy()
        
        if customer_history['found']:
            # Auto-fill missing information
            if not enhanced_data.get('phone') and customer_history.get('phone'):
                enhanced_data['phone'] = customer_history['phone']
                enhanced_data['auto_filled_phone'] = True
            
            if not enhanced_data.get('address') and customer_history.get('address'):
                enhanced_data['address'] = customer_history['address']
                enhanced_data['auto_filled_address'] = True
            
            if not enhanced_data.get('email') and customer_history.get('email'):
                enhanced_data['email'] = customer_history['email']
                enhanced_data['auto_filled_email'] = True
            
            # Add customer context for AI
            enhanced_data['customer_context'] = {
                'total_previous_claims': customer_history['total_claims'],
                'total_claimed_amount': customer_history['total_claimed_amount'],
                'average_fraud_score': customer_history['average_fraud_score'],
                'customer_since': customer_history['customer_since'],
                'risk_level': customer_history['risk_analysis']['overall_risk_level']
            }
        
        return enhanced_data
    
    def _enhanced_fraud_assessment(self, claim_data: Dict[str, Any], ai_analysis: Dict[str, Any], customer_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform enhanced fraud assessment using customer history.
        
        Args:
            claim_data: Enhanced claim data
            ai_analysis: AI analysis results
            customer_history: Customer history
            
        Returns:
            Enhanced fraud assessment
        """
        # Get base fraud assessment from AI
        base_assessment = self.ai_engine.assess_fraud_risk(claim_data, ai_analysis)
        
        if not customer_history['found']:
            # New customer - return base assessment
            base_assessment['base_risk_score'] = base_assessment.get('risk_score', 0.5)
            base_assessment['historical_adjustment'] = 1.0
            return base_assessment
        
        # Enhance with historical analysis
        historical_factors = []
        risk_multiplier = 1.0
        
        # Analyze customer history patterns
        risk_analysis = customer_history.get('risk_analysis', {})
        
        # Add historical risk factors
        for factor in risk_analysis.get('risk_factors', []):
            historical_factors.append(f"HISTORY: {factor}")
            risk_multiplier *= 1.2  # Increase risk for each historical factor
        
        # Add historical legitimacy factors
        legitimacy_factors = base_assessment.get('legitimacy_factors', [])
        for factor in risk_analysis.get('legitimacy_factors', []):
            legitimacy_factors.append(f"HISTORY: {factor}")
            risk_multiplier *= 0.9  # Decrease risk for positive factors
        
        # Cap the multiplier to reasonable bounds
        risk_multiplier = max(0.5, min(risk_multiplier, 2.0))
        
        # Adjust risk score based on history
        base_risk = base_assessment.get('risk_score', 0.5)
        adjusted_risk = min(base_risk * risk_multiplier, 1.0)
        
        # Enhanced reasoning
        enhanced_reasoning = base_assessment.get('reasoning', '')
        if historical_factors:
            enhanced_reasoning += f"\n\nCUSTOMER HISTORY ANALYSIS:\n" + "\n".join(historical_factors)
        
        return {
            'risk_score': adjusted_risk,
            'risk_factors': base_assessment.get('risk_factors', []) + historical_factors,
            'legitimacy_factors': legitimacy_factors,
            'reasoning': enhanced_reasoning,
            'historical_adjustment': risk_multiplier,
            'base_risk_score': base_risk
        }
    
    def _generate_enhanced_recommendations(self, claim_data: Dict[str, Any], fraud_assessment: Dict[str, Any], customer_history: Dict[str, Any]) -> list:
        """
        Generate enhanced recommendations considering customer history.
        
        Args:
            claim_data: Enhanced claim data
            fraud_assessment: Enhanced fraud assessment
            customer_history: Customer history
            
        Returns:
            List of enhanced recommendations
        """
        base_recommendations = self.ai_engine.generate_recommended_actions(claim_data, fraud_assessment)
        
        # Add history-based recommendations
        if customer_history['found']:
            risk_level = customer_history['risk_analysis']['overall_risk_level']
            
            if risk_level == 'HIGH':
                base_recommendations.insert(0, "HIGH-RISK CUSTOMER: Escalate to senior adjuster immediately")
                base_recommendations.append("Review all previous claims for patterns")
            
            elif risk_level == 'MEDIUM':
                base_recommendations.append("Monitor customer for emerging patterns")
            
            # Check for denied claims
            denied_claims = [claim for claim in customer_history.get('claims_history', []) if claim['claim_result'] == 'DENIED']
            if denied_claims:
                base_recommendations.insert(0, f"WARNING: Customer has previous denied claim (${denied_claims[0]['claim_amount']:,.2f})")
            
            # Check for frequent claims
            if customer_history['total_claims'] > 3:
                base_recommendations.append(f"Frequent claimer: {customer_history['total_claims']} previous claims totaling ${customer_history['total_claimed_amount']:,.2f}")
        
        return base_recommendations
    
    def _create_processing_summary(self, claim_data: Dict[str, Any], customer_history: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced processing summary including customer history.
        
        Args:
            claim_data: Enhanced claim data
            customer_history: Customer history
            
        Returns:
            Enhanced processing summary
        """
        base_summary = {
            'steps_completed': [
                'Data validation',
                'Customer history lookup',
                'AI description analysis',
                'Enhanced fraud risk assessment',
                'Historical pattern analysis',
                'Action recommendations generated'
            ],
            'processing_method': 'AI-powered analysis with customer history',
            'data_sources': ['Customer input', 'Historical claims database', 'OpenAI GPT analysis'],
            'automation_level': 'Fully automated with AI and historical intelligence',
            'manual_review_required': self._requires_manual_review(claim_data),
            'priority_level': self._determine_priority_level(claim_data)
        }
        
        # Add customer-specific information
        if customer_history['found']:
            base_summary['customer_status'] = f"Existing customer with {customer_history['total_claims']} previous claims"
            base_summary['customer_risk_level'] = customer_history['risk_analysis']['overall_risk_level']
            base_summary['historical_data_used'] = True
        else:
            base_summary['customer_status'] = 'New customer - no previous history'
            base_summary['historical_data_used'] = False
        
        return base_summary
    
    def _validate_claim_data(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate required claim data fields.
        
        Args:
            claim_data: Claim data to validate
            
        Returns:
            Validation result with success status and any errors
        """
        errors = []
        required_fields = ['claim_id', 'customer_name', 'policy_number', 'incident_description']
        
        # Check required fields
        for field in required_fields:
            if not claim_data.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate claim amount
        claim_amount = claim_data.get('claim_amount', 0)
        if not isinstance(claim_amount, (int, float)) or claim_amount < 0:
            errors.append("Claim amount must be a positive number")
        
        # Validate incident description length
        description = claim_data.get('incident_description', '')
        if len(description) < 20:
            errors.append("Incident description must be at least 20 characters")
        
        # Check for extremely high claim amounts (potential data entry error)
        if claim_amount > 1000000:  # $1M threshold
            errors.append("Claim amount exceeds $1M - requires special handling")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _requires_manual_review(self, claim_data: Dict[str, Any]) -> bool:
        """
        Determine if claim requires manual review based on various factors.
        
        Args:
            claim_data: Claim data to evaluate
            
        Returns:
            Boolean indicating if manual review is needed
        """
        # High amount claims need review
        if claim_data.get('claim_amount', 0) > 50000:
            return True
        
        # Complex incident types need review
        complex_types = ['Natural Disaster', 'Other']
        if claim_data.get('incident_type') in complex_types:
            return True
        
        # Multiple uploaded documents might indicate complexity
        if claim_data.get('uploaded_files', 0) > 3:
            return True
        
        # High-risk customers need review
        if claim_data.get('customer_context', {}).get('risk_level') == 'HIGH':
            return True
        
        return False
    
    def _determine_priority_level(self, claim_data: Dict[str, Any]) -> str:
        """
        Determine processing priority based on claim characteristics.
        
        Args:
            claim_data: Claim data to evaluate
            
        Returns:
            Priority level string
        """
        claim_amount = claim_data.get('claim_amount', 0)
        customer_risk = claim_data.get('customer_context', {}).get('risk_level', 'UNKNOWN')
        
        if claim_amount > 100000 or customer_risk == 'HIGH':
            return 'HIGH'
        elif claim_amount > 25000 or customer_risk == 'MEDIUM':
            return 'MEDIUM'
        else:
            return 'STANDARD'
    
    def _calculate_ai_confidence(self, ai_analysis: Dict[str, Any], fraud_assessment: Dict[str, Any]) -> float:
        """
        Calculate confidence score for AI analysis results.
        
        Args:
            ai_analysis: Results from AI description analysis
            fraud_assessment: Results from fraud risk assessment
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence_factors = []
        
        # Check if AI provided detailed analysis
        if ai_analysis.get('extracted_details'):
            confidence_factors.append(0.3)
        
        if ai_analysis.get('summary') and len(ai_analysis['summary']) > 50:
            confidence_factors.append(0.2)
        
        if fraud_assessment.get('reasoning') and len(fraud_assessment['reasoning']) > 100:
            confidence_factors.append(0.3)
        
        if fraud_assessment.get('risk_factors') or fraud_assessment.get('legitimacy_factors'):
            confidence_factors.append(0.2)
        
        return min(sum(confidence_factors), 1.0)
    
    def _get_processing_time(self) -> float:
        """
        Calculate processing time in seconds.
        
        Returns:
            Processing time in seconds
        """
        if self.processing_start_time:
            return (datetime.now() - self.processing_start_time).total_seconds()
        return 0.0
    
    def _create_error_response(self, errors: list) -> Dict[str, Any]:
        """
        Create standardized error response.
        
        Args:
            errors: List of error messages
            
        Returns:
            Error response dictionary
        """
        return {
            'processing_id': str(uuid.uuid4()),
            'processed_at': datetime.now().isoformat(),
            'processing_time_seconds': self._get_processing_time(),
            'status': 'error',
            'errors': errors,
            'summary': 'Claim processing failed due to validation errors',
            'extracted_details': {},
            'concerns': errors,
            'clarification_questions': ['Please correct the errors and resubmit'],
            'fraud_risk': 0.0,
            'risk_factors': [],
            'legitimacy_factors': [],
            'fraud_reasoning': 'Unable to assess due to processing errors',
            'recommended_actions': ['Correct errors and resubmit claim'],
            'processing_summary': {
                'steps_completed': ['Initial validation'],
                'processing_method': 'Validation only',
                'automation_level': 'Failed',
                'manual_review_required': True,
                'priority_level': 'ERROR'
            },
            'ai_confidence': 0.0,
            'customer_history': {'found': False, 'is_new_customer': True},
            'enhanced_data_used': False,
            'historical_risk_analysis': {}
        }
    
    def get_claim_status(self, claim_id: str, processed_claims: list) -> Dict[str, Any]:
        """
        Get current status of a processed claim.
        
        Args:
            claim_id: ID of the claim to check
            processed_claims: List of all processed claims
            
        Returns:
            Claim status information
        """
        for claim in processed_claims:
            if claim.get('claim_id') == claim_id:
                return {
                    'found': True,
                    'status': claim.get('status', 'unknown'),
                    'fraud_risk': claim.get('fraud_risk', 0),
                    'processing_time': claim.get('processing_time_seconds', 0),
                    'recommended_actions': claim.get('recommended_actions', []),
                    'customer_history_used': claim.get('customer_history', {}).get('found', False)
                }
        
        return {
            'found': False,
            'status': 'not_found',
            'message': f'Claim {claim_id} not found in processed claims'
        }
    
    def update_claim_status(self, claim_id: str, new_status: str, processed_claims: list) -> bool:
        """
        Update the status of a processed claim.
        
        Args:
            claim_id: ID of the claim to update
            new_status: New status to set
            processed_claims: List of all processed claims
            
        Returns:
            Boolean indicating if update was successful
        """
        for i, claim in enumerate(processed_claims):
            if claim.get('claim_id') == claim_id:
                processed_claims[i]['status'] = new_status
                processed_claims[i]['last_updated'] = datetime.now().isoformat()
                return True
        
        return False