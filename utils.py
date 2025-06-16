from datetime import datetime
import pandas as pd
from typing import List, Dict, Any
import re

def format_currency(amount: float) -> str:
    """
    Format currency amount for display.
    
    Args:
        amount: Numeric amount
        
    Returns:
        Formatted currency string
    """
    return f"${amount:,.2f}"

def calculate_processing_time(processed_claims: List[Dict]) -> float:
    """
    Calculate average processing time for claims.
    
    Args:
        processed_claims: List of processed claim dictionaries
        
    Returns:
        Average processing time in minutes
    """
    if not processed_claims:
        return 0.0
    
    total_time = sum(claim.get('processing_time_seconds', 0) for claim in processed_claims)
    return round(total_time / len(processed_claims) / 60, 1)  # Convert to minutes

def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email string to validate
        
    Returns:
        Boolean indicating if email is valid
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_text_input(text: str) -> str:
    """
    Sanitize text input for safety.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    if not text:
        return ""
    
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\']', '', text)
    return sanitized.strip()

def calculate_risk_color(risk_score: float) -> str:
    """
    Get color code for risk score visualization.
    
    Args:
        risk_score: Risk score between 0 and 1
        
    Returns:
        Color name for UI
    """
    if risk_score < 0.3:
        return "green"
    elif risk_score < 0.7:
        return "orange"
    else:
        return "red"

def format_timestamp(timestamp_str: str) -> str:
    """
    Format timestamp for display.
    
    Args:
        timestamp_str: ISO format timestamp string
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str

def export_claims_to_csv(claims_data: List[Dict]) -> pd.DataFrame:
    """
    Export claims data to CSV format.
    
    Args:
        claims_data: List of claim dictionaries
        
    Returns:
        Pandas DataFrame ready for export
    """
    if not claims_data:
        return pd.DataFrame()
    
    # Flatten the nested dictionaries for CSV export
    flattened_data = []
    for claim in claims_data:
        flat_claim = {
            'claim_id': claim.get('claim_id'),
            'customer_name': claim.get('customer_name'),
            'policy_number': claim.get('policy_number'),
            'incident_date': claim.get('incident_date'),
            'claim_amount': claim.get('claim_amount'),
            'incident_type': claim.get('incident_type'),
            'fraud_risk': claim.get('fraud_risk'),
            'status': claim.get('status'),
            'processed_at': claim.get('processed_at'),
            'processing_time_seconds': claim.get('processing_time_seconds'),
            'summary': claim.get('summary', '').replace('\n', ' ')
        }
        flattened_data.append(flat_claim)
    
    return pd.DataFrame(flattened_data)
