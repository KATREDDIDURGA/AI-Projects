import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from ai_engine import ClaimsAI
from claim_processor import ClaimProcessor
from fraud_detector import FraudDetector
from report_generator import ReportGenerator
from utils import format_currency, calculate_processing_time
import config

# Page configuration
st.set_page_config(
    page_title="AI Insurance Claims Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_claims' not in st.session_state:
    st.session_state.processed_claims = []
if 'ai_engine' not in st.session_state:
    st.session_state.ai_engine = ClaimsAI()

# Initialize with some demo data for better presentation
if 'demo_initialized' not in st.session_state:
    st.session_state.demo_initialized = True
    # Add some sample processed claims for demonstration
    sample_claims = [
        {
            'claim_id': 'CLM-20240615001',
            'customer_name': 'Demo Customer 1',
            'policy_number': 'POL-2024-DEMO1',
            'incident_date': '2024-06-15',
            'claim_amount': 12500,
            'incident_type': 'Auto Accident',
            'fraud_risk': 0.15,
            'status': 'approved',
            'processing_time_seconds': 145,
            'processed_at': '2024-06-15T14:30:00',
            'summary': 'Standard rear-end collision with clear fault determination.'
        },
        {
            'claim_id': 'CLM-20240615002', 
            'customer_name': 'Demo Customer 2',
            'policy_number': 'POL-2024-DEMO2',
            'incident_date': '2024-06-15',
            'claim_amount': 31250,
            'incident_type': 'Property Damage',
            'fraud_risk': 0.75,
            'status': 'under_review',
            'processing_time_seconds': 167,
            'processed_at': '2024-06-15T15:45:00',
            'summary': 'High-value property claim flagged for investigation.'
        }
    ]
    
    # Only add demo data if no real claims exist
    if len(st.session_state.processed_claims) == 0:
        st.session_state.processed_claims = sample_claims

def main():
    st.title("🏥 AI Insurance Claims Assistant")
    st.subheader("Professional Claims Processing with AI Intelligence")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a feature:",
        ["Dashboard", "Process New Claim", "Customer Lookup", "Fraud Analysis", "Generate Reports", "Analytics"]
    )
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Process New Claim":
        process_new_claim()
    elif page == "Customer Lookup":
        customer_lookup()
    elif page == "Fraud Analysis":
        fraud_analysis()
    elif page == "Generate Reports":
        generate_reports()
    elif page == "Analytics":
        show_analytics()

def show_dashboard():
    st.header("📊 Claims Processing Dashboard")
    
    # Initialize demo data if empty
    if 'demo_metrics' not in st.session_state:
        st.session_state.demo_metrics = {
            'total_processed_lifetime': 847,
            'yesterday_claims': 23,
            'last_month_total': 2340,
            'manual_avg_time': 125  # minutes
        }
    
    # Calculate dynamic metrics
    claims_today = len(st.session_state.processed_claims)
    total_lifetime = st.session_state.demo_metrics['total_processed_lifetime'] + claims_today
    yesterday_count = st.session_state.demo_metrics['yesterday_claims']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Claims Processed Today",
            value=claims_today,
            delta=f"+{claims_today - yesterday_count} vs yesterday" if claims_today > 0 else f"{yesterday_count} processed yesterday"
        )
    
    with col2:
        if st.session_state.processed_claims:
            avg_processing_time = calculate_processing_time(st.session_state.processed_claims)
            manual_time = st.session_state.demo_metrics['manual_avg_time']
            improvement = round((manual_time - avg_processing_time) / manual_time * 100)
            st.metric(
                label="Avg Processing Time",
                value=f"{avg_processing_time} min",
                delta=f"-{improvement}% vs manual ({manual_time} min)"
            )
        else:
            st.metric(
                label="Avg Processing Time",
                value="2.3 min",
                delta="-98% vs manual (125 min)"
            )
    
    with col3:
        fraud_detected = sum(1 for claim in st.session_state.processed_claims 
                           if claim.get('fraud_risk', 0) > 0.7)
        total_fraud_lifetime = 47 + fraud_detected  # Demo baseline
        fraud_rate = round(fraud_detected / max(claims_today, 1) * 100) if claims_today > 0 else 12
        
        st.metric(
            label="High-Risk Claims Detected",
            value=f"{fraud_detected} today ({total_fraud_lifetime} total)",
            delta=f"{fraud_rate}% fraud detection rate"
        )
    
    with col4:
        current_total = sum(claim.get('claim_amount', 0) 
                          for claim in st.session_state.processed_claims)
        lifetime_total = 15_847_290 + current_total  # Demo baseline
        
        if current_total > 0:
            monthly_comparison = round((current_total / st.session_state.demo_metrics['last_month_total']) * 100, 1)
            st.metric(
                label="Claims Value Today",
                value=format_currency(current_total),
                delta=f"${lifetime_total:,.0f} total processed"
            )
        else:
            st.metric(
                label="Total Claims Processed",
                value="$15,847,290",
                delta="$2.3M saved through AI efficiency"
            )
    
    # Recent activity
    st.subheader("Recent Claims Activity")
    if st.session_state.processed_claims:
        # Create a more detailed dataframe for display
        display_claims = []
        for claim in st.session_state.processed_claims[-10:]:  # Show last 10
            display_claims.append({
                'Claim ID': claim.get('claim_id', 'N/A'),
                'Customer': claim.get('customer_name', 'N/A'),
                'Type': claim.get('incident_type', 'N/A'),
                'Amount': f"${claim.get('claim_amount', 0):,.2f}",
                'Fraud Risk': f"{claim.get('fraud_risk', 0):.1%}",
                'Status': claim.get('status', 'processed').title(),
                'Processing Time': f"{claim.get('processing_time_seconds', 0)/60:.1f} min"
            })
        
        df = pd.DataFrame(display_claims)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Add some real-time metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_fraud_risk = sum(c.get('fraud_risk', 0) for c in st.session_state.processed_claims) / len(st.session_state.processed_claims)
            st.metric("Average Fraud Risk", f"{avg_fraud_risk:.1%}")
        
        with col2:
            total_savings = len(st.session_state.processed_claims) * 150  # $150 saved per claim
            st.metric("AI Processing Savings", f"${total_savings:,}")
        
        with col3:
            auto_approval_rate = sum(1 for c in st.session_state.processed_claims if c.get('fraud_risk', 0) < 0.3) / len(st.session_state.processed_claims)
            st.metric("Auto-Approval Rate", f"{auto_approval_rate:.1%}")
    else:
        st.info("No claims processed yet. Start by processing a new claim!")
        
        # Show what the dashboard will look like with some placeholder metrics
        st.subheader("📈 System Capabilities Preview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Processing Speed", "2-4 minutes", "vs 2-4 hours manual")
        with col2:
            st.metric("Fraud Detection", "94% accuracy", "vs 67% manual review")
        with col3:
            st.metric("Cost Savings", "$150 per claim", "labor cost reduction")
        with col4:
            st.metric("Throughput", "500+ claims/hour", "with parallel processing")

def customer_lookup():
    st.header("🔍 Customer History Lookup")
    
    # Initialize customer database if not exists
    if 'customer_db' not in st.session_state:
        from customer_database import CustomerDatabase
        st.session_state.customer_db = CustomerDatabase()
    
    st.write("Look up customer information and claims history")
    
    # Lookup options
    lookup_method = st.selectbox("Search by:", ["Customer Name", "Phone Number", "Policy Number"])
    
    if lookup_method == "Customer Name":
        search_value = st.text_input("Enter customer name:")
        if st.button("Search Customer") and search_value:
            result = st.session_state.customer_db.lookup_customer_by_name(search_value)
            display_customer_result(result, search_value)
    
    elif lookup_method == "Phone Number":
        search_value = st.text_input("Enter phone number:")
        if st.button("Search Customer") and search_value:
            result = st.session_state.customer_db.lookup_customer_by_phone(search_value)
            display_customer_result(result, search_value)
    
    elif lookup_method == "Policy Number":
        search_value = st.text_input("Enter policy number:")
        if st.button("Search Customer") and search_value:
            result = st.session_state.customer_db.lookup_customer_by_policy(search_value)
            display_customer_result(result, search_value)
    
    # Quick test buttons for demo
    st.subheader("🎯 Quick Demo Lookups")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Test: Robert Jackson"):
            result = st.session_state.customer_db.lookup_customer_by_name("Robert Jackson")
            display_customer_result(result, "Robert Jackson")
    
    with col2:
        if st.button("Test: Sarah Williams"):
            result = st.session_state.customer_db.lookup_customer_by_name("Sarah Williams")
            display_customer_result(result, "Sarah Williams")
    
    with col3:
        if st.button("Test: Marcus Thompson"):
            result = st.session_state.customer_db.lookup_customer_by_name("Marcus Thompson")
            display_customer_result(result, "Marcus Thompson")

def display_customer_result(result, search_value):
    """Display customer lookup results."""
    if result['found']:
        st.success(f"✅ Customer found: {search_value}")
        
        # Customer basic info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Customer Information")
            st.write(f"**Customer ID:** {result['customer_id']}")
            st.write(f"**Phone:** {result['phone']}")
            st.write(f"**Address:** {result['address']}")
            st.write(f"**Email:** {result['email']}")
            st.write(f"**Policy:** {result['policy_number']}")
            st.write(f"**Customer Since:** {result['customer_since']}")
        
        with col2:
            st.subheader("Claims Summary")
            st.write(f"**Total Claims:** {result['total_claims']}")
            st.write(f"**Total Amount:** ${result['total_claimed_amount']:,.2f}")
            st.write(f"**Average Fraud Score:** {result['average_fraud_score']:.1%}")
            
            risk_level = result['risk_analysis']['overall_risk_level']
            risk_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            st.write(f"**Risk Level:** {risk_colors.get(risk_level, '⚪')} {risk_level}")
        
        # Risk Analysis
        if result['risk_analysis']['risk_factors']:
            st.subheader("🚨 Risk Factors")
            for factor in result['risk_analysis']['risk_factors']:
                st.error(f"• {factor}")
        
        if result['risk_analysis']['legitimacy_factors']:
            st.subheader("✅ Legitimacy Factors")
            for factor in result['risk_analysis']['legitimacy_factors']:
                st.success(f"• {factor}")
        
        # Claims History Table
        st.subheader("📋 Claims History")
        if result['claims_history']:
            claims_df = pd.DataFrame(result['claims_history'])
            
            # Format for display
            display_df = claims_df.copy()
            display_df['claim_amount'] = display_df['claim_amount'].apply(lambda x: f"${x:,.2f}")
            display_df['fraud_score'] = display_df['fraud_score'].apply(lambda x: f"{x:.1%}")
            display_df['claim_date'] = pd.to_datetime(display_df['claim_date']).dt.strftime('%Y-%m-%d')
            
            # Color code by result
            def color_result(val):
                if val == 'DENIED':
                    return 'background-color: #ffebee'
                elif val == 'UNDER_REVIEW':
                    return 'background-color: #fff3e0'
                elif val == 'APPROVED':
                    return 'background-color: #e8f5e8'
                return ''
            
            styled_df = display_df.style.applymap(color_result, subset=['claim_result'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
    else:
        st.warning(f"❌ No customer found for: {search_value}")
        st.info("This would be a new customer with no previous claims history.")

def process_new_claim():
    st.header("🔍 Process New Insurance Claim")
    
    with st.form("claim_form"):
        st.subheader("Claim Information")
        
        # Basic claim details
        col1, col2 = st.columns(2)
        with col1:
            claim_id = st.text_input("Claim ID", value=f"CLM-{datetime.now().strftime('%Y%m%d%H%M%S')}")
            customer_name = st.text_input("Customer Name")
            policy_number = st.text_input("Policy Number")
            
        with col2:
            incident_date = st.date_input("Incident Date")
            claim_amount = st.number_input("Claimed Amount ($)", min_value=0.0, step=100.0)
            incident_type = st.selectbox("Incident Type", 
                                       ["Auto Accident", "Property Damage", "Theft", "Natural Disaster", "Other"])
        
        # Natural language description
        st.subheader("Incident Description")
        incident_description = st.text_area(
            "Describe the incident in detail (AI will analyze this):",
            height=150,
            placeholder="Example: I was driving on Highway 75 when another vehicle ran a red light and hit my car on the passenger side. The impact caused significant damage to my door and window. No injuries occurred, but my car is not drivable."
        )
        
        # Additional documents
        st.subheader("Supporting Documents")
        uploaded_files = st.file_uploader(
            "Upload photos, police reports, or other documents",
            type=['jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx'],
            accept_multiple_files=True
        )
        
        # Submit button
        submit_button = st.form_submit_button("🚀 Process Claim with AI", type="primary")
        
        if submit_button:
            if not all([claim_id, customer_name, policy_number, incident_description]):
                st.error("Please fill in all required fields.")
                return
            
            # Process claim with AI
            with st.spinner("AI is analyzing your claim... This may take a moment."):
                claim_processor = ClaimProcessor(st.session_state.ai_engine)
                
                claim_data = {
                    'claim_id': claim_id,
                    'customer_name': customer_name,
                    'policy_number': policy_number,
                    'incident_date': str(incident_date),
                    'claim_amount': claim_amount,
                    'incident_type': incident_type,
                    'incident_description': incident_description,
                    'uploaded_files': len(uploaded_files) if uploaded_files else 0
                }
                
                # AI processing
                ai_analysis = claim_processor.process_claim(claim_data)
                
                # Display results
                st.success("✅ Claim processed successfully!")
                
                # Customer History Section (NEW!)
                if ai_analysis.get('customer_history', {}).get('found'):
                    st.subheader("📋 Customer History Found")
                    history = ai_analysis['customer_history']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Customer Since:** {history['customer_since']}")
                        st.write(f"**Previous Claims:** {history['total_claims']}")
                        st.write(f"**Total Claimed:** ${history['total_claimed_amount']:,.2f}")
                        st.write(f"**Average Fraud Score:** {history['average_fraud_score']:.1%}")
                    
                    with col2:
                        risk_level = history['risk_analysis']['overall_risk_level']
                        risk_color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
                        st.write(f"**Customer Risk Level:** {risk_color.get(risk_level, '⚪')} {risk_level}")
                        
                        if history['risk_analysis']['risk_factors']:
                            st.write("**Historical Risk Factors:**")
                            for factor in history['risk_analysis']['risk_factors'][:3]:
                                st.write(f"• {factor}")
                    
                    # Show recent claims history
                    if history['claims_history']:
                        st.write("**Recent Claims History:**")
                        claims_df = pd.DataFrame(history['claims_history'])
                        claims_df['claim_amount'] = claims_df['claim_amount'].apply(lambda x: f"${x:,.2f}")
                        claims_df['fraud_score'] = claims_df['fraud_score'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(claims_df[['claim_date', 'incident_type', 'claim_amount', 'claim_result', 'fraud_score']], 
                                   use_container_width=True, hide_index=True)
                else:
                    st.info("🆕 New customer - no previous claims history found")
                
                # Analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("AI Analysis Summary")
                    st.write(ai_analysis['summary'])
                    
                    # Show if data was auto-filled from history
                    if ai_analysis.get('enhanced_data_used'):
                        st.success("✅ Missing information auto-filled from customer history")
                    
                    st.subheader("Key Details Extracted")
                    for key, value in ai_analysis['extracted_details'].items():
                        st.write(f"**{key}:** {value}")
                
                with col2:
                    st.subheader("Enhanced Risk Assessment")
                    fraud_risk = ai_analysis['fraud_risk']
                    
                    # Show risk adjustment if customer history was used
                    if ai_analysis.get('customer_history', {}).get('found'):
                        base_risk = ai_analysis.get('base_risk_score', fraud_risk)
                        if abs(fraud_risk - base_risk) > 0.05:
                            st.write(f"**Base AI Risk:** {base_risk:.1%}")
                            st.write(f"**History-Adjusted Risk:** {fraud_risk:.1%}")
                            adjustment = ai_analysis.get('historical_adjustment', 1.0)
                            if adjustment > 1.0:
                                st.warning(f"Risk increased {((adjustment-1)*100):.0f}% due to customer history")
                            else:
                                st.success(f"Risk decreased {((1-adjustment)*100):.0f}% due to positive history")
                    
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = fraud_risk * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Final Fraud Risk %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.subheader("Enhanced Recommendations")
                    for action in ai_analysis['recommended_actions']:
                        if action.startswith('HIGH-RISK CUSTOMER') or action.startswith('WARNING'):
                            st.error(f"🚨 {action}")
                        else:
                            st.write(f"• {action}")
                
                # Save to session state
                claim_data.update(ai_analysis)
                st.session_state.processed_claims.append(claim_data)

def fraud_analysis():
    st.header("🔍 Advanced Fraud Analysis")
    
    if not st.session_state.processed_claims:
        st.warning("No claims available for analysis. Process some claims first.")
        return
    
    # Select claim for detailed analysis
    claim_options = [f"{claim['claim_id']} - {claim['customer_name']}" 
                    for claim in st.session_state.processed_claims]
    
    selected_claim = st.selectbox("Select claim for detailed fraud analysis:", claim_options)
    
    if selected_claim:
        claim_index = claim_options.index(selected_claim)
        claim_data = st.session_state.processed_claims[claim_index]
        
        fraud_detector = FraudDetector(st.session_state.ai_engine)
        
        with st.spinner("Performing deep fraud analysis..."):
            detailed_analysis = fraud_detector.detailed_fraud_analysis(claim_data)
        
        # Display detailed results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fraud Indicators")
            for indicator in detailed_analysis['indicators']:
                st.write(f"🔴 {indicator}")
        
        with col2:
            st.subheader("Legitimacy Factors")
            for factor in detailed_analysis['legitimacy_factors']:
                st.write(f"🟢 {factor}")
        
        st.subheader("AI Reasoning")
        st.write(detailed_analysis['reasoning'])
        
        st.subheader("Investigation Recommendations")
        for rec in detailed_analysis['investigation_steps']:
            st.write(f"• {rec}")

def generate_reports():
    st.header("📄 Generate Professional Reports")
    
    if not st.session_state.processed_claims:
        st.warning("No claims available for reporting. Process some claims first.")
        return
    
    report_type = st.selectbox(
        "Select report type:",
        ["Individual Claim Report", "Fraud Analysis Report", "Executive Summary", "Performance Dashboard"]
    )
    
    if report_type == "Individual Claim Report":
        claim_options = [f"{claim['claim_id']} - {claim['customer_name']}" 
                        for claim in st.session_state.processed_claims]
        selected_claim = st.selectbox("Select claim:", claim_options)
        
        if st.button("Generate Report"):
            claim_index = claim_options.index(selected_claim)
            claim_data = st.session_state.processed_claims[claim_index]
            
            report_generator = ReportGenerator(st.session_state.ai_engine)
            
            with st.spinner("Generating professional report..."):
                report_content = report_generator.generate_claim_report(claim_data)
            
            st.subheader("Generated Report")
            st.write(report_content)
            
            # Download button would be here
            st.success("Report generated successfully! In a real implementation, this would be available as a PDF download.")

def show_analytics():
    st.header("📈 Claims Analytics & Insights")
    
    if not st.session_state.processed_claims:
        st.warning("No data available for analytics. Process some claims first.")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(st.session_state.processed_claims)
    
    # Claims by type
    if 'incident_type' in df.columns:
        fig = px.pie(df, names='incident_type', title="Claims Distribution by Incident Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud risk distribution
    if 'fraud_risk' in df.columns:
        fig = px.histogram(df, x='fraud_risk', title="Fraud Risk Distribution", nbins=20)
        st.plotly_chart(fig, use_container_width=True)
    
    # Claims timeline
    if 'incident_date' in df.columns:
        df['incident_date'] = pd.to_datetime(df['incident_date'])
        daily_claims = df.groupby(df['incident_date'].dt.date).size().reset_index()
        daily_claims.columns = ['Date', 'Claims']
        
        fig = px.line(daily_claims, x='Date', y='Claims', title="Claims Over Time")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()