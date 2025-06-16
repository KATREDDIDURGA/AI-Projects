# customer_database.py - Customer History Management
import pandas as pd
from datetime import datetime, timedelta
import random

class CustomerDatabase:
    """
    Manages customer history and previous claims data.
    Provides fraud detection insights based on claim patterns.
    """
    
    def __init__(self):
        """Initialize with sample customer data."""
        self.customers_df = self._create_sample_data()
    
    def _create_sample_data(self):
        """Create realistic sample customer data with claim history."""
        
        # Sample customer data with multiple claims per person
        customer_data = [
            # Customer 1 - Normal profile
            {
                'customer_id': 'CUST-001',
                'customer_name': 'Michael Chen',
                'phone': '(214) 555-0123',
                'address': '123 Oak Street, Dallas, TX 75201',
                'email': 'mchen@email.com',
                'policy_number': 'POL-2024-3456',
                'customer_since': '2019-03-15',
                'claim_id': 'CLM-2023-1001',
                'claim_date': '2023-08-15',
                'claim_amount': 8750.00,
                'incident_type': 'Auto Accident',
                'claim_result': 'APPROVED',
                'fraud_score': 0.12,
                'processing_notes': 'Standard rear-end collision, clear documentation'
            },
            {
                'customer_id': 'CUST-001',
                'customer_name': 'Michael Chen',
                'phone': '(214) 555-0123',
                'address': '123 Oak Street, Dallas, TX 75201',
                'email': 'mchen@email.com',
                'policy_number': 'POL-2024-3456',
                'customer_since': '2019-03-15',
                'claim_id': 'CLM-2024-0234',
                'claim_date': '2024-02-10',
                'claim_amount': 3200.00,
                'incident_type': 'Property Damage',
                'claim_result': 'APPROVED',
                'fraud_score': 0.08,
                'processing_notes': 'Hail damage to roof, weather confirmed'
            },
            
            # Customer 2 - Suspicious pattern
            {
                'customer_id': 'CUST-002',
                'customer_name': 'Robert Jackson',
                'phone': '(214) 555-9876',
                'address': '456 Elm Avenue, Dallas, TX 75202',
                'email': 'rjackson@email.com',
                'policy_number': 'POL-2024-1122',
                'customer_since': '2022-07-20',
                'claim_id': 'CLM-2023-0445',
                'claim_date': '2023-04-22',
                'claim_amount': 15000.00,
                'incident_type': 'Theft',
                'claim_result': 'APPROVED',
                'fraud_score': 0.45,
                'processing_notes': 'Electronics theft, police report provided late'
            },
            {
                'customer_id': 'CUST-002',
                'customer_name': 'Robert Jackson',
                'phone': '(214) 555-9876',
                'address': '456 Elm Avenue, Dallas, TX 75202',
                'email': 'rjackson@email.com',
                'policy_number': 'POL-2024-1122',
                'customer_since': '2022-07-20',
                'claim_id': 'CLM-2023-0892',
                'claim_date': '2023-11-18',
                'claim_amount': 22500.00,
                'incident_type': 'Auto Accident',
                'claim_result': 'APPROVED',
                'fraud_score': 0.62,
                'processing_notes': 'Single vehicle accident, no witnesses, weekend'
            },
            {
                'customer_id': 'CUST-002',
                'customer_name': 'Robert Jackson',
                'phone': '(214) 555-9876',
                'address': '456 Elm Avenue, Dallas, TX 75202',
                'email': 'rjackson@email.com',
                'policy_number': 'POL-2024-1122',
                'customer_since': '2022-07-20',
                'claim_id': 'CLM-2024-0156',
                'claim_date': '2024-01-28',
                'claim_amount': 8900.00,
                'incident_type': 'Property Damage',
                'claim_result': 'UNDER_REVIEW',
                'fraud_score': 0.71,
                'processing_notes': 'Water damage claim, timeline questionable'
            },
            
            # Customer 3 - High frequency claimer
            {
                'customer_id': 'CUST-003',
                'customer_name': 'Sarah Williams',
                'phone': '(469) 555-4567',
                'address': '789 Pine Road, Plano, TX 75024',
                'email': 'swilliams@email.com',
                'policy_number': 'POL-2024-7890',
                'customer_since': '2020-12-03',
                'claim_id': 'CLM-2023-0234',
                'claim_date': '2023-03-12',
                'claim_amount': 12400.00,
                'incident_type': 'Auto Accident',
                'claim_result': 'APPROVED',
                'fraud_score': 0.25,
                'processing_notes': 'Intersection collision, other party at fault'
            },
            {
                'customer_id': 'CUST-003',
                'customer_name': 'Sarah Williams',
                'phone': '(469) 555-4567',
                'address': '789 Pine Road, Plano, TX 75024',
                'email': 'swilliams@email.com',
                'policy_number': 'POL-2024-7890',
                'customer_since': '2020-12-03',
                'claim_id': 'CLM-2023-0678',
                'claim_date': '2023-09-05',
                'claim_amount': 25500.00,
                'incident_type': 'Property Damage',
                'claim_result': 'APPROVED',
                'fraud_score': 0.18,
                'processing_notes': 'Storm damage, extensive documentation provided'
            },
            
            # Customer 4 - Single previous claim
            {
                'customer_id': 'CUST-004',
                'customer_name': 'David Thompson',
                'phone': '(972) 555-7890',
                'address': '321 Maple Drive, Irving, TX 75061',
                'email': 'dthompson@email.com',
                'policy_number': 'POL-2024-9999',
                'customer_since': '2021-05-18',
                'claim_id': 'CLM-2023-0567',
                'claim_date': '2023-10-14',
                'claim_amount': 4750.00,
                'incident_type': 'Auto Accident',
                'claim_result': 'APPROVED',
                'fraud_score': 0.09,
                'processing_notes': 'Minor parking lot collision, straightforward'
            },
            
            # Customer 5 - Recently denied claim
            {
                'customer_id': 'CUST-005',
                'customer_name': 'Marcus Thompson',
                'phone': '(214) 555-3333',
                'address': '654 Cedar Lane, Dallas, TX 75203',
                'email': 'mthompson@email.com',
                'policy_number': 'POL-2024-8888',
                'customer_since': '2023-01-10',
                'claim_id': 'CLM-2024-0089',
                'claim_date': '2024-03-22',
                'claim_amount': 45000.00,
                'incident_type': 'Auto Accident',
                'claim_result': 'DENIED',
                'fraud_score': 0.89,
                'processing_notes': 'Suspicious circumstances, inconsistent story, investigation revealed fraud'
            },
            
            # Customer 6 - New customer, no history
            # (This will be useful for testing new customer scenarios)
            
            # Customer 7 - Pattern of weekend incidents
            {
                'customer_id': 'CUST-007',
                'customer_name': 'Jennifer Martinez',
                'phone': '(469) 555-1111',
                'address': '987 Sunset Boulevard, Garland, TX 75040',
                'email': 'jmartinez@email.com',
                'policy_number': 'POL-2024-5555',
                'customer_since': '2022-11-08',
                'claim_id': 'CLM-2023-0445',
                'claim_date': '2023-06-17',  # Saturday
                'claim_amount': 18900.00,
                'incident_type': 'Auto Accident',
                'claim_result': 'APPROVED',
                'fraud_score': 0.38,
                'processing_notes': 'Weekend accident, moderate suspicion but approved'
            },
            {
                'customer_id': 'CUST-007',
                'customer_name': 'Jennifer Martinez',
                'phone': '(469) 555-1111',
                'address': '987 Sunset Boulevard, Garland, TX 75040',
                'email': 'jmartinez@email.com',
                'policy_number': 'POL-2024-5555',
                'customer_since': '2022-11-08',
                'claim_id': 'CLM-2024-0234',
                'claim_date': '2024-01-13',  # Saturday
                'claim_amount': 32100.00,
                'incident_type': 'Property Damage',
                'claim_result': 'APPROVED',
                'fraud_score': 0.52,
                'processing_notes': 'Another weekend incident, pattern noted'
            }
        ]
        
        return pd.DataFrame(customer_data)
    
    def lookup_customer_by_name(self, customer_name):
        """
        Look up customer by name and return their information and claim history.
        
        Args:
            customer_name: Name to search for
            
        Returns:
            Dictionary with customer info and claim history
        """
        customer_claims = self.customers_df[
            self.customers_df['customer_name'].str.lower() == customer_name.lower()
        ]
        
        if customer_claims.empty:
            return {
                'found': False,
                'is_new_customer': True,
                'message': 'New customer - no previous claims history'
            }
        
        # Get customer basic info from first record
        customer_info = customer_claims.iloc[0]
        
        # Get all claims for this customer
        claims_history = []
        for _, claim in customer_claims.iterrows():
            claims_history.append({
                'claim_id': claim['claim_id'],
                'claim_date': claim['claim_date'],
                'claim_amount': claim['claim_amount'],
                'incident_type': claim['incident_type'],
                'claim_result': claim['claim_result'],
                'fraud_score': claim['fraud_score'],
                'processing_notes': claim['processing_notes']
            })
        
        # Calculate risk factors
        risk_analysis = self._analyze_customer_risk(customer_claims)
        
        return {
            'found': True,
            'is_new_customer': False,
            'customer_id': customer_info['customer_id'],
            'phone': customer_info['phone'],
            'address': customer_info['address'],
            'email': customer_info['email'],
            'policy_number': customer_info['policy_number'],
            'customer_since': customer_info['customer_since'],
            'claims_history': claims_history,
            'risk_analysis': risk_analysis,
            'total_claims': len(claims_history),
            'total_claimed_amount': customer_claims['claim_amount'].sum(),
            'average_fraud_score': customer_claims['fraud_score'].mean()
        }
    
    def lookup_customer_by_phone(self, phone):
        """Look up customer by phone number."""
        customer_claims = self.customers_df[self.customers_df['phone'] == phone]
        
        if not customer_claims.empty:
            customer_name = customer_claims.iloc[0]['customer_name']
            return self.lookup_customer_by_name(customer_name)
        
        return {'found': False, 'is_new_customer': True}
    
    def lookup_customer_by_policy(self, policy_number):
        """Look up customer by policy number."""
        customer_claims = self.customers_df[self.customers_df['policy_number'] == policy_number]
        
        if not customer_claims.empty:
            customer_name = customer_claims.iloc[0]['customer_name']
            return self.lookup_customer_by_name(customer_name)
        
        return {'found': False, 'is_new_customer': True}
    
    def _analyze_customer_risk(self, customer_claims):
        """
        Analyze customer risk based on claim history patterns.
        
        Args:
            customer_claims: DataFrame of customer's claims
            
        Returns:
            Dictionary with risk analysis
        """
        risk_factors = []
        legitimacy_factors = []
        
        # Claim frequency analysis
        claim_count = len(customer_claims)
        if claim_count > 3:
            risk_factors.append(f"High claim frequency: {claim_count} claims")
        elif claim_count == 1:
            legitimacy_factors.append("Low claim frequency indicates careful driver")
        
        # Average fraud score analysis
        avg_fraud_score = customer_claims['fraud_score'].mean()
        if avg_fraud_score > 0.5:
            risk_factors.append(f"Historical high fraud scores (avg: {avg_fraud_score:.1%})")
        elif avg_fraud_score < 0.3:
            legitimacy_factors.append(f"Historical low fraud scores (avg: {avg_fraud_score:.1%})")
        
        # Claim amount analysis
        total_claimed = customer_claims['claim_amount'].sum()
        if total_claimed > 50000:
            risk_factors.append(f"High total claim amounts: ${total_claimed:,.2f}")
        
        # Denial history
        denied_claims = customer_claims[customer_claims['claim_result'] == 'DENIED']
        if len(denied_claims) > 0:
            risk_factors.append(f"Previous claim denied for fraud (${denied_claims.iloc[0]['claim_amount']:,.2f})")
        
        # Review history
        under_review = customer_claims[customer_claims['claim_result'] == 'UNDER_REVIEW']
        if len(under_review) > 0:
            risk_factors.append("Has claims currently under investigation")
        
        # Pattern analysis (weekend claims, etc.)
        weekend_pattern = self._check_weekend_pattern(customer_claims)
        if weekend_pattern:
            risk_factors.append("Pattern of weekend/holiday incidents")
        
        # Time between claims
        if claim_count > 1:
            claim_dates = pd.to_datetime(customer_claims['claim_date'])
            days_between = (claim_dates.max() - claim_dates.min()).days
            if days_between < 365 and claim_count > 2:
                risk_factors.append("Multiple claims within 12 months")
        
        # Customer tenure
        customer_since = pd.to_datetime(customer_claims.iloc[0]['customer_since'])
        years_customer = (datetime.now() - customer_since).days / 365
        if years_customer > 3 and avg_fraud_score < 0.3:
            legitimacy_factors.append(f"Long-term customer ({years_customer:.1f} years) with good history")
        
        return {
            'risk_factors': risk_factors,
            'legitimacy_factors': legitimacy_factors,
            'overall_risk_level': self._calculate_overall_customer_risk(avg_fraud_score, claim_count, len(risk_factors))
        }
    
    def _check_weekend_pattern(self, customer_claims):
        """Check if customer has pattern of weekend incidents."""
        weekend_claims = 0
        for _, claim in customer_claims.iterrows():
            claim_date = pd.to_datetime(claim['claim_date'])
            if claim_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                weekend_claims += 1
        
        return weekend_claims >= 2 and weekend_claims / len(customer_claims) > 0.5
    
    def _calculate_overall_customer_risk(self, avg_fraud_score, claim_count, risk_factor_count):
        """Calculate overall customer risk level."""
        if avg_fraud_score > 0.7 or risk_factor_count >= 3:
            return 'HIGH'
        elif avg_fraud_score > 0.4 or risk_factor_count >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_missing_info_suggestions(self, customer_info, current_claim_data):
        """
        Suggest missing information based on customer history.
        
        Args:
            customer_info: Customer lookup result
            current_claim_data: Current claim being processed
            
        Returns:
            Dictionary with suggestions for missing information
        """
        suggestions = []
        auto_filled = {}
        
        if customer_info['found']:
            # Auto-fill missing basic information
            if not current_claim_data.get('phone') and customer_info.get('phone'):
                auto_filled['phone'] = customer_info['phone']
                suggestions.append(f"Auto-filled phone: {customer_info['phone']}")
            
            if not current_claim_data.get('address') and customer_info.get('address'):
                auto_filled['address'] = customer_info['address']
                suggestions.append(f"Auto-filled address: {customer_info['address']}")
            
            if not current_claim_data.get('email') and customer_info.get('email'):
                auto_filled['email'] = customer_info['email']
                suggestions.append(f"Auto-filled email: {customer_info['email']}")
            
            # Policy number verification
            if current_claim_data.get('policy_number') != customer_info.get('policy_number'):
                suggestions.append(f"Policy number mismatch! Expected: {customer_info['policy_number']}")
        
        return {
            'suggestions': suggestions,
            'auto_filled': auto_filled,
            'verification_needed': len(suggestions) > 0
        }
    
    def add_new_claim(self, claim_data):
        """Add a new processed claim to the database."""
        # In a real system, this would save to a database
        # For demo purposes, we'll add to our DataFrame
        new_row = {
            'customer_id': claim_data.get('customer_id', f"CUST-NEW-{len(self.customers_df)+1:03d}"),
            'customer_name': claim_data.get('customer_name'),
            'phone': claim_data.get('phone', ''),
            'address': claim_data.get('address', ''),
            'email': claim_data.get('email', ''),
            'policy_number': claim_data.get('policy_number'),
            'customer_since': claim_data.get('customer_since', datetime.now().strftime('%Y-%m-%d')),
            'claim_id': claim_data.get('claim_id'),
            'claim_date': claim_data.get('incident_date'),
            'claim_amount': claim_data.get('claim_amount', 0),
            'incident_type': claim_data.get('incident_type'),
            'claim_result': claim_data.get('status', 'PROCESSED'),
            'fraud_score': claim_data.get('fraud_risk', 0),
            'processing_notes': claim_data.get('summary', '')
        }
        
        # Convert to DataFrame and append
        new_df = pd.DataFrame([new_row])
        self.customers_df = pd.concat([self.customers_df, new_df], ignore_index=True)
        
        return True