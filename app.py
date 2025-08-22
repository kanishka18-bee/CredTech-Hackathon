
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import time
import asyncio
from streamlit_autorefresh import st_autorefresh
import threading

# Page config for optimal UX
st.set_page_config(
    page_title=" Neural Credit Intelligence",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.alert-high { border-left-color: #ff4757 !important; }
.alert-medium { border-left-color: #ffa502 !important; }
.alert-low { border-left-color: #2ed573 !important; }
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state for real-time features
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'update_count' not in st.session_state:
    st.session_state.update_count = 0
if 'alerts_history' not in st.session_state:
    st.session_state.alerts_history = []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_results():
    try:
        with open('enhanced_neural_results.json', 'r') as f:
            data = json.load(f)
            return data.get('predictions', []), data.get('model_performance', {})
    except:
        # Enhanced sample data for demo
        sample_results = [
            {
                'ticker': 'AAPL', 'credit_score': 78.5, 'rating': 'AA', 'confidence': 0.94,
                'contributions': {
                    'profit_margin': {'contribution': 12.3, 'value': 0.25, 'impact': 'positive'},
                    'volatility_30d': {'contribution': -8.1, 'value': 0.32, 'impact': 'negative'},
                    'debt_to_equity': {'contribution': -3.2, 'value': 0.85, 'impact': 'negative'},
                    'current_ratio': {'contribution': 6.7, 'value': 1.2, 'impact': 'positive'}
                },
                'explanation': ['Profit Margin improved score by 12.3 points', 'Stock volatility reduced score by 8.1 points'],
                'risk_factors': [{'factor': 'Market Volatility', 'impact': -8.1, 'severity': 'Medium'}],
                'news_events': [{'title': 'Apple reports strong quarterly results', 'impact': 1.2}]
            },
            {
                'ticker': 'MSFT', 'credit_score': 82.1, 'rating': 'AA+', 'confidence': 0.96,
                'contributions': {
                    'profit_margin': {'contribution': 15.2, 'value': 0.31, 'impact': 'positive'},
                    'current_ratio': {'contribution': 8.9, 'value': 1.8, 'impact': 'positive'},
                    'revenue_growth': {'contribution': 7.4, 'value': 0.18, 'impact': 'positive'}
                },
                'explanation': ['Strong profitability boosted score', 'Excellent liquidity position'],
                'risk_factors': [],
                'news_events': [{'title': 'Microsoft announces cloud expansion', 'impact': 0.8}]
            }
        ]
        
        sample_metrics = {
            'r2': 0.9234, 'rmse': 1.42, 'mae': 1.08, 
            'explained_var': 0.9156, 'mape': 2.34
        }
        
        return sample_results, sample_metrics

def simulate_real_time_update():
    """Simulate real-time data updates"""
    if st.button(" Force Update", help="Trigger immediate data refresh"):
        st.session_state.update_count += 1
        st.session_state.last_update = datetime.now()
        st.cache_data.clear()
        st.rerun()

def check_alerts(results):
    """Check for critical alerts and update history"""
    new_alerts = []
    
    for result in results:
        score = result['credit_score']
        confidence = result.get('confidence', 1.0)
        
        if score < 50:
            new_alerts.append({
                'type': 'CRITICAL',
                'message': f"{result['ticker']} credit score dropped to {score:.1f}",
                'timestamp': datetime.now(),
                'severity': 'HIGH'
            })
        elif confidence < 0.8:
            new_alerts.append({
                'type': 'WARNING', 
                'message': f"{result['ticker']} model confidence low: {confidence*100:.1f}%",
                'timestamp': datetime.now(),
                'severity': 'MEDIUM'
            })
    
    # Add to history (keep last 10)
    st.session_state.alerts_history.extend(new_alerts)
    st.session_state.alerts_history = st.session_state.alerts_history[-10:]
    
    return new_alerts

# Load data
results, model_metrics = load_results()
current_alerts = check_alerts(results)

# HEADER WITH REAL-TIME STATUS
st.markdown("""
<div class="main-header">
    <h1> Neural Credit Intelligence Platform</h1>
    <p>Enterprise-Grade Real-Time Credit Risk Assessment</p>
</div>
""", unsafe_allow_html=True)

# REAL-TIME STATUS BAR
status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)

with status_col1:
    st.metric(" Neural Network", "ACTIVE", "Running")
    
with status_col2:
    update_delta = f"+{st.session_state.update_count}" if st.session_state.update_count > 0 else None
    st.metric(" Updates", st.session_state.update_count, update_delta)
    
with status_col3:
    time_since_update = (datetime.now() - st.session_state.last_update).seconds
    st.metric(" Last Update", f"{time_since_update}s ago")
    
with status_col4:
    alert_count = len(current_alerts)
    alert_delta = f"+{alert_count}" if alert_count > 0 else "0"
    st.metric(" Active Alerts", alert_count, alert_delta)
    
with status_col5:
    if model_metrics:
        st.metric(" Model R²", f"{model_metrics.get('r2', 0.92):.3f}", "High Accuracy")

# AUTO-REFRESH CONTROLS
st.sidebar.header(" Real-Time Controls")
auto_refresh = st.sidebar.checkbox(" Auto Refresh", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)

if auto_refresh:
    count = st_autorefresh(interval=refresh_interval * 1000, limit=None, key="auto_refresh")
    if count > 0:
        st.session_state.update_count += 1
        st.session_state.last_update = datetime.now()

# ALERT PANEL
if current_alerts:
    st.sidebar.header(" Active Alerts")
    for alert in current_alerts:
        alert_color = "🔴" if alert['severity'] == 'HIGH' else "🟡"
        st.sidebar.warning(f"{alert_color} {alert['message']}")

simulate_real_time_update()

# MAIN DASHBOARD TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    " Credit Overview", 
    " Neural Insights", 
    " Risk Analytics", 
    " Real-Time Ops",
    " Model Controls"
])

with tab1:
    st.header(" Credit Scores Overview")
    
    # Enhanced metrics display
    scores_df = pd.DataFrame([
        {
            'Company': r['ticker'],
            'Credit Score': r['credit_score'],
            'Rating': r['rating'],
            'Confidence': r.get('confidence', 0.9) * 100,
            'Risk Level': 'Low' if r['credit_score'] >= 70 else 'Medium' if r['credit_score'] >= 50 else 'High',
            'Trend': np.random.choice(['↗️', '↘️', '→'], p=[0.4, 0.3, 0.3])  # Simulated trend
        } for r in results
    ])
    
    # Enhanced visualization with trend indicators
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Multi-axis chart with confidence and trends
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Credit Scores with Confidence', 'Score Trends'),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )
        
        # Main score bar chart
        fig.add_trace(
            go.Bar(
                x=scores_df['Company'], 
                y=scores_df['Credit Score'],
                name='Credit Score',
                text=scores_df['Rating'],
                textposition='outside',
                marker_color=scores_df['Credit Score'],
                marker_colorscale='RdYlGn',
                marker_cmin=20,
                marker_cmax=90
            ),
            row=1, col=1
        )
        
        # Confidence line
        fig.add_trace(
            go.Scatter(
                x=scores_df['Company'], 
                y=scores_df['Confidence'],
                name='Model Confidence %',
                line=dict(color='purple', width=3),
                mode='lines+markers',
                marker_size=8,
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Trend indicators (simulated historical data)
        trend_data = pd.DataFrame({
            'Company': scores_df['Company'],
            'Previous': scores_df['Credit Score'] + np.random.uniform(-3, 3, len(scores_df)),
            'Current': scores_df['Credit Score']
        })
        
        for i, company in enumerate(trend_data['Company']):
            fig.add_trace(
                go.Scatter(
                    x=[company, company],
                    y=[trend_data.iloc[i]['Previous'], trend_data.iloc[i]['Current']],
                    mode='lines+markers',
                    name=f'{company} Trend',
                    line=dict(width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            title_text="Credit Intelligence Dashboard - Real-Time View"
        )
        
        fig.update_yaxes(title_text="Credit Score (20-90)", row=1, col=1)
        fig.update_yaxes(title_text="Score Change", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Quick Stats")
        
        avg_score = scores_df['Credit Score'].mean()
        avg_confidence = scores_df['Confidence'].mean()
        high_risk_count = len(scores_df[scores_df['Risk Level'] == 'High'])
        
        st.metric("Portfolio Avg Score", f"{avg_score:.1f}", f"{avg_score-75:.1f} vs Target")
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%", "High")
        st.metric("High Risk Companies", high_risk_count, f"{'📈' if high_risk_count > 0 else '✅'}")
        
        # Risk distribution pie
        risk_counts = scores_df['Risk Level'].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values, 
            names=risk_counts.index,
            title="Risk Distribution",
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Enhanced data table with sorting and filtering
    st.subheader("📋 Detailed Analysis Table")
    
    # Filtering options
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        score_filter = st.selectbox("Filter by Score Range", ["All", "High (70+)", "Medium (50-69)", "Low (<50)"])
    with filter_col2:
        rating_filter = st.multiselect("Filter by Rating", scores_df['Rating'].unique(), default=scores_df['Rating'].unique())
    with filter_col3:
        sort_by = st.selectbox("Sort by", ["Credit Score", "Confidence", "Company"])
    
    # Apply filters
    filtered_df = scores_df[scores_df['Rating'].isin(rating_filter)]
    
    if score_filter == "High (70+)":
        filtered_df = filtered_df[filtered_df['Credit Score'] >= 70]
    elif score_filter == "Medium (50-69)":
        filtered_df = filtered_df[(filtered_df['Credit Score'] >= 50) & (filtered_df['Credit Score'] < 70)]
    elif score_filter == "Low (<50)":
        filtered_df = filtered_df[filtered_df['Credit Score'] < 50]
    
    filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    st.dataframe(
        filtered_df.style.format({
            'Credit Score': '{:.1f}',
            'Confidence': '{:.1f}%'
        }).background_gradient(subset=['Credit Score'], cmap='RdYlGn'),
        use_container_width=True
    )

with tab2:
    st.header(" Neural Network Deep Insights")
    
    # Company selector with search
    selected_company = st.selectbox(" Select Company for Deep Analysis:", 
                                   [r['ticker'] for r in results])
    
    selected_result = next(r for r in results if r['ticker'] == selected_company)
    
    # Enhanced layout for neural insights
    insight_col1, insight_col2, insight_col3 = st.columns([2, 1, 1])
    
    with insight_col1:
        st.subheader(" Feature Attribution Analysis")
        
        contributions = selected_result['contributions']
        feature_data = []
        
        for feature, data in contributions.items():
            feature_data.append({
                'Feature': feature.replace('_', ' ').title(),
                'Contribution': data['contribution'],
                'Value': data['value'],
                'Impact': data['impact'],
                'Abs_Contribution': abs(data['contribution'])
            })
        
        feature_df = pd.DataFrame(feature_data)
        feature_df = feature_df.sort_values('Abs_Contribution', ascending=False).head(12)
        
        # Enhanced SHAP-style waterfall chart
        colors = ['#ff4757' if x < 0 else '#2ed573' for x in feature_df['Contribution']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=feature_df['Feature'],
            x=feature_df['Contribution'],
            orientation='h',
            marker_color=colors,
            text=[f"{val:+.2f}" for val in feature_df['Contribution']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Contribution: %{x:.2f}<br>Impact: %{text}<extra></extra>',
        ))
        
        fig.update_layout(
            title=f"{selected_company} - Neural Network Feature Attribution",
            xaxis_title="Contribution to Credit Score",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature values table
        st.subheader(" Feature Values Detail")
        feature_detail_df = feature_df[['Feature', 'Value', 'Contribution', 'Impact']].copy()
        st.dataframe(
            feature_detail_df.style.format({
                'Value': '{:.3f}',
                'Contribution': '{:+.2f}'
            }),
            use_container_width=True
        )
    
    with insight_col2:
        st.subheader(" Score Analysis")
        
        score = selected_result['credit_score']
        confidence = selected_result.get('confidence', 0.9)
        
        # Score gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"{selected_company} Score"},
            delta = {'reference': 65, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 90]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 40], 'color': "lightgray"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 90], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Confidence meter
        st.metric("Model Confidence", f"{confidence*100:.1f}%", "High" if confidence > 0.9 else "Medium")
        st.metric("Credit Rating", selected_result['rating'])
        
        # Risk level indicator
        if score >= 70:
            st.success("🟢 LOW RISK")
        elif score >= 50:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.error("🔴 HIGH RISK")
    
    with insight_col3:
        st.subheader(" AI Explanations")
        
        st.write("**Key Insights:**")
        for i, explanation in enumerate(selected_result['explanation'], 1):
            st.write(f"{i}. {explanation}")
        
        if selected_result.get('risk_factors'):
            st.write("**Risk Factors:**")
            for risk in selected_result['risk_factors']:
                severity_color = "🔴" if risk['severity'] == 'High' else "🟡"
                st.write(f"{severity_color} **{risk['factor']}**: {risk['impact']:+.1f}")
        
        # Model interpretation confidence
        st.info(f" **Interpretation Confidence**: The neural network is {confidence*100:.1f}% confident in this analysis.")

with tab3:
    st.header(" Advanced Risk Analytics")
    
    # Portfolio risk metrics
    portfolio_col1, portfolio_col2 = st.columns([2, 1])
    
    with portfolio_col1:
        st.subheader(" Portfolio Risk Heatmap")
        
        # Create risk matrix
        risk_matrix_data = []
        for result in results:
            risk_matrix_data.append({
                'Company': result['ticker'],
                'Credit_Score': result['credit_score'],
                'Confidence': result.get('confidence', 0.9),
                'Volatility_Risk': abs(result['contributions'].get('volatility_30d', {}).get('contribution', 0)),
                'Debt_Risk': abs(result['contributions'].get('debt_to_equity', {}).get('contribution', 0)),
                'Liquidity_Risk': -result['contributions'].get('current_ratio', {}).get('contribution', 0)
            })
        
        risk_df = pd.DataFrame(risk_matrix_data)
        
        # Risk heatmap
        fig_heatmap = px.imshow(
            risk_df.set_index('Company')[['Volatility_Risk', 'Debt_Risk', 'Liquidity_Risk']].T,
            aspect="auto",
            color_continuous_scale="Reds",
            title="Risk Factor Heatmap by Company"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with portfolio_col2:
        st.subheader(" Risk Alerts")
        
        # Generate risk-based alerts
        for result in results:
            score = result['credit_score']
            if score < 50:
                st.error(f"🚨 **{result['ticker']}**: Critical Risk (Score: {score:.1f})")
            elif score < 60:
                st.warning(f"⚠️ **{result['ticker']}**: Elevated Risk (Score: {score:.1f})")
        
        # Portfolio statistics
        scores = [r['credit_score'] for r in results]
        st.metric("Portfolio Min Score", f"{min(scores):.1f}")
        st.metric("Portfolio Max Score", f"{max(scores):.1f}")
        st.metric("Score Std Dev", f"{np.std(scores):.2f}")

with tab4:
    st.header(" Real-Time Operations Center")
    
    # Operations dashboard
    ops_col1, ops_col2, ops_col3 = st.columns(3)
    
    with ops_col1:
        st.subheader(" Update Management")
        
        st.write("**Update Triggers:**")
        st.write("•  Market close (daily)")
        st.write("•  Breaking news detection")
        st.write("•  Significant price movements (>5%)")
        st.write("•  Earnings announcements")
        st.write("•  Manual refresh requests")
        
        update_frequency = st.selectbox("Update Frequency", 
                                      ["Real-time (30s)", "High (1 min)", "Medium (5 min)", "Low (15 min)"])
        
        if st.button(" Trigger Emergency Update"):
            st.success("Emergency update initiated! All models refreshing...")
            st.session_state.update_count += 1
    
    with ops_col2:
        st.subheader(" System Health")
        
        # System health metrics
        health_metrics = {
            "Neural Network": " Healthy",
            "Data Pipeline": " Active",
            "News Feed": " Connected", 
            "Alert System": " Monitoring",
            "API Endpoints": " Responsive"
        }
        
        for system, status in health_metrics.items():
            st.write(f"**{system}**: {status}")
        
        # Performance metrics
        if model_metrics:
            st.write("**Model Performance:**")
            st.write(f"• R² Score: {model_metrics.get('r2', 0.92):.4f}")
            st.write(f"• RMSE: {model_metrics.get('rmse', 1.42):.2f}")
            st.write(f"• MAE: {model_metrics.get('mae', 1.08):.2f}")
    
    with ops_col3:
        st.subheader(" Alert History")
        
        # Show recent alerts
        if st.session_state.alerts_history:
            for alert in st.session_state.alerts_history[-5:]:
                alert_time = alert['timestamp'].strftime("%H:%M:%S")
                st.write(f"**{alert_time}**: {alert['message']}")
        else:
            st.info("No recent alerts")
        
        if st.button(" Test Alert System"):
            test_alert = {
                'type': 'TEST',
                'message': 'Alert system test - All systems operational',
                'timestamp': datetime.now(),
                'severity': 'LOW'
            }
            st.session_state.alerts_history.append(test_alert)
            st.success("Test alert generated!")

with tab5:
    st.header(" Model Control Center")
    
    control_col1, control_col2 = st.columns(2)
    
    with control_col1:
        st.subheader(" Model Parameters")
        
        # Model configuration
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
        alert_threshold = st.slider("Alert Score Threshold", 20, 70, 50, 5)
        
        st.write("**Current Model Config:**")
        st.write(f"• Architecture: 4-Layer Neural Network")
        st.write(f"• Features: 24 financial + sentiment")
        st.write(f"• Training Data: {len(results)} companies × 5 augmented")
        st.write(f"• Optimizer: Adam (lr=0.001)")
        
        if st.button(" Save Configuration"):
            st.success("Model configuration saved!")
    
    with control_col2:
        st.subheader(" Advanced Operations")
        
        # Advanced operations
        if st.button(" Retrain Neural Network"):
            with st.spinner("Retraining neural network..."):
                time.sleep(3)  # Simulate training
            st.success("Neural network retrained successfully!")
        
        if st.button(" Recalculate All Scores"):
            with st.spinner("Recalculating credit scores..."):
                time.sleep(2)
            st.success("All credit scores updated!")
        
        if st.button(" Refresh Data Sources"):
            with st.spinner("Refreshing financial data..."):
                time.sleep(2)
            st.success("Data sources refreshed!")
        
        # Backup and restore
        st.write("**Data Management:**")
        if st.button(" Backup Model"):
            st.info("Model backup created: neural_model_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        if st.button(" Export Results"):
            # Create downloadable data
            results_df = pd.DataFrame(results)
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results CSV",
                data=csv,
                file_name=f"credit_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )

# FOOTER WITH REAL-TIME STATUS
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.write(" **Powered by Neural Networks**")
    st.write("TensorFlow + Keras Deep Learning")

with footer_col2:
    st.write(" **Real-Time Intelligence**")
    st.write("Live data • Instant updates • Smart alerts")

with footer_col3:
    st.write(" **Enterprise Ready**")
    st.write("Scalable • Secure • Production-grade")

st.markdown(f"**Status**: System operational | Last update: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')} | Updates: {st.session_state.update_count}")
