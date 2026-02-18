"""
🎓 AI ENGINEERING KNOWLEDGE ASSISTANT - STREAMLIT UI (DARK MODE)
==================================================================================
A beautiful dark-themed web interface for your Elite Production RAG System.
Features:
- ✅ Dark mode with black background
- ✅ Chat interface with message history
- ✅ Real-time metrics display
- ✅ Source document viewer
- ✅ Retriever mode switching
- ✅ Cost tracking dashboard
- ✅ Export functionality
==================================================================================
"""

import sys
import os
import time
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
from src.rag.rag_pipeline import EnhancedProductionRAG

# Page configuration
st.set_page_config(
    page_title="AI Engineering Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DARK MODE CSS - Black background with vibrant colors
st.markdown("""
<style>
    /* Main background - BLACK */
    .stApp {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Override Streamlit's default white background */
    .main > div {
        background-color: #000000;
    }
    
    /* Sidebar background - Dark gray */
    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #333333;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }
    
    /* Headers with gradient */
    .main-header {
        background: linear-gradient(90deg, #6b11cb 0%, #2575fc 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(107, 17, 203, 0.3);
        border: 1px solid #333333;
    }
    
    /* Metric cards - Dark with glow */
    .metric-card {
        background: #111111;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        text-align: center;
        border: 1px solid #333333;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #6b11cb;
        box-shadow: 0 6px 20px rgba(107, 17, 203, 0.3);
    }
    
    .metric-card label {
        color: #aaaaaa !important;
    }
    
    .metric-card div {
        color: #ffffff !important;
    }
    
    /* Source cards - Dark theme */
    .source-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6b11cb;
        margin-bottom: 0.5rem;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    .source-card small {
        color: #cccccc;
    }
    
    /* Confidence colors */
    .confidence-high {
        color: #00ff88 !important;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffbb00 !important;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #ff4444 !important;
        font-weight: bold;
    }
    
    /* Chat messages - Dark theme */
    .chat-message-user {
        background: linear-gradient(135deg, #6b11cb 0%, #2575fc 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 0 20px;
        margin: 0.5rem 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 4px 15px rgba(107, 17, 203, 0.3);
        border: 1px solid #444444;
    }
    
    .chat-message-assistant {
        background: #1a1a1a;
        color: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 0;
        margin: 0.5rem 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
        border: 1px solid #333333;
    }
    
    .timestamp {
        font-size: 0.8rem;
        color: #888888;
        margin-top: 0.2rem;
    }
    
    /* Input field - Dark theme */
    .stTextInput > div > div > input {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #6b11cb !important;
        box-shadow: 0 0 0 2px rgba(107, 17, 203, 0.2) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #6b11cb 0%, #2575fc 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: bold !important;
        transition: transform 0.2s !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(107, 17, 203, 0.4) !important;
    }
    
    /* Selectbox - Dark theme */
    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #6b11cb !important;
    }
    
    /* Expander - Dark theme */
    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    .streamlit-expanderContent {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-top: none !important;
    }
    
    /* Dataframes - Dark theme */
    .stDataFrame {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
    }
    
    .stDataFrame td {
        color: #ffffff !important;
    }
    
    .stDataFrame th {
        background-color: #111111 !important;
        color: #ffffff !important;
    }
    
    /* Tabs - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111111 !important;
        border-bottom: 1px solid #333333 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a !important;
        border-color: #6b11cb !important;
    }
    
    /* Info boxes - Dark theme */
    .stAlert {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Markdown text */
    .stMarkdown p {
        color: #ffffff !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    /* Sidebar text */
    .css-1d391kg, .css-163ttbj, .css-1wrcr25 {
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        color: #ffffff !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #1a1a1a !important;
        color: #ffffff !important;
        border: 1px solid #333333 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #6b11cb !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        background: #111111;
    }
    
    ::-webkit-scrollbar-track {
        background: #111111;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #333333;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6b11cb;
    }
    
    /* Plotly charts background */
    .js-plotly-plot .plotly {
        background-color: #1a1a1a !important;
    }
    
    /* Success/Info/Warning/Error messages */
    .stSuccess {
        background-color: #1a3a1a !important;
        color: #ffffff !important;
        border-color: #00ff88 !important;
    }
    
    .stInfo {
        background-color: #1a2a3a !important;
        color: #ffffff !important;
        border-color: #2575fc !important;
    }
    
    .stWarning {
        background-color: #3a3a1a !important;
        color: #ffffff !important;
        border-color: #ffbb00 !important;
    }
    
    .stError {
        background-color: #3a1a1a !important;
        color: #ffffff !important;
        border-color: #ff4444 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_system():
    """Initialize RAG system (cached to avoid reloading)"""
    config = {
        "retriever_k": 4,
        "llm_model": "gpt-4o-mini",
        "temperature": 0.1,
        "enable_caching": True,
        "enable_source_filtering": True,
        "verbose": False,
        "enable_ab_testing": False,
    }
    return EnhancedProductionRAG(config=config)


def create_confidence_gauge(confidence):
    """Create a gauge chart for confidence score with dark theme"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score", 'font': {'color': 'white'}},
        number={'font': {'color': 'white'}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': 'white'},
            'bar': {'color': "#6b11cb"},
            'bgcolor': '#1a1a1a',
            'borderwidth': 2,
            'bordercolor': '#333333',
            'steps': [
                {'range': [0, 50], 'color': "#442222"},
                {'range': [50, 75], 'color': "#444422"},
                {'range': [75, 100], 'color': "#224422"}
            ],
            'threshold': {
                'line': {'color': "#ff4444", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor='#111111',
        font={'color': 'white'}
    )
    return fig


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_system" not in st.session_state:
        with st.spinner("🚀 Initializing AI Engineering Knowledge Assistant..."):
            st.session_state.rag_system = init_rag_system()
    if "metrics_history" not in st.session_state:
        st.session_state.metrics_history = []
    if "current_retriever" not in st.session_state:
        st.session_state.current_retriever = "similarity"

    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='color: white; text-align: center;'>🎓 AI Assistant</h1>", unsafe_allow_html=True)
        st.markdown("---")

        # System Status
        st.markdown("<h3 style='color: #cccccc;'>🤖 System Status</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background: #1a1a1a; padding: 0.5rem; border-radius: 5px; text-align: center; border: 1px solid #333333;'>
                <span style='color: #00ff88; font-size: 1.2rem;'>●</span><br>
                <span style='color: #cccccc;'>Status</span><br>
                <span style='color: white; font-weight: bold;'>Online</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background: #1a1a1a; padding: 0.5rem; border-radius: 5px; text-align: center; border: 1px solid #333333;'>
                <span style='color: #2575fc; font-size: 1.2rem;'>⚡</span><br>
                <span style='color: #cccccc;'>Mode</span><br>
                <span style='color: white; font-weight: bold;'>{st.session_state.current_retriever.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

        # Retriever Mode Selection
        st.markdown("<h3 style='color: #cccccc;'>🔄 Retriever Mode</h3>", unsafe_allow_html=True)
        mode = st.selectbox(
            "Select retriever mode",
            options=["similarity", "mmr"],
            index=0 if st.session_state.current_retriever == "similarity" else 1,
            label_visibility="collapsed"
        )
        if mode != st.session_state.current_retriever:
            st.session_state.rag_system.switch_retriever_mode(mode)
            st.session_state.current_retriever = mode
            st.success(f"✅ Switched to {mode} mode")

        st.markdown("---")

        # Quick Actions
        st.markdown("<h3 style='color: #cccccc;'>⚡ Quick Actions</h3>", unsafe_allow_html=True)
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.session_state.rag_system.clear_cache()
            st.success("Cache cleared!")
        
        if st.button("📊 Show ROI Report", use_container_width=True):
            roi_data = st.session_state.rag_system.calculate_roi()
            if "message" in roi_data:
                st.info(roi_data["message"])
            else:
                st.markdown("<h4 style='color: #cccccc;'>💰 ROI Analysis</h4>", unsafe_allow_html=True)
                for key, value in roi_data.items():
                    if isinstance(value, float):
                        if 'cost' in key or 'saved' in key or 'value' in key:
                            st.metric(key.replace('_', ' ').title(), f"${value:,.2f}")
                        elif 'percentage' in key:
                            st.metric(key.replace('_', ' ').title(), f"{value:.1f}%")
                        else:
                            st.metric(key.replace('_', ' ').title(), f"{value:,.2f}")
                    else:
                        st.metric(key.replace('_', ' ').title(), value)

        if st.button("📥 Export Data", use_container_width=True):
            with st.spinner("Exporting..."):
                st.session_state.rag_system.export_data("all")
            st.success("Data exported to 'exports/' directory!")

        st.markdown("---")

        # System Info
        st.markdown("<h3 style='color: #cccccc;'>📋 System Info</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background: #1a1a1a; padding: 1rem; border-radius: 10px; border: 1px solid #333333;'>
            <small style='color: #cccccc;'>
            <b>Architecture:</b><br>
            • Embeddings: HuggingFace<br>
            • Vector DB: Chroma<br>
            • LLM: GPT-4o-mini<br>
            • Retriever: Similarity/MMR
            </small>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown('<div class="main-header">', unsafe_allow_html=True)
        st.markdown("<h1 style='color: white; margin: 0;'>🎓 AI Engineering Knowledge Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: rgba(255,255,255,0.9); margin: 0;'>Ask questions about your MLOps projects and technical documentation</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Total Queries",
            len(st.session_state.rag_system.query_history),
            help="Number of queries in this session"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_cost = sum(q.cost_usd for q in st.session_state.rag_system.query_history)
        st.metric(
            "Total Cost",
            f"${total_cost:.6f}",
            help="Total API cost"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat interface
    chat_container = st.container()

    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-assistant">{message["content"]}</div>', unsafe_allow_html=True)
                if "sources" in message:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"][:3]):
                            st.markdown(f"""
                            <div class="source-card">
                                <b>Source {i+1}:</b> {source.metadata.get('source', 'Unknown')}<br>
                                <small>{source.page_content[:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
                if "metrics" in message:
                    with st.expander("📊 View Metrics"):
                        metrics = message["metrics"]
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Time", f"{metrics.total_time:.2f}s")
                        confidence_color = "high" if metrics.confidence_score > 0.7 else "medium" if metrics.confidence_score > 0.4 else "low"
                        col_b.markdown(f"<span style='color: white;'>Confidence</span><br><span class='confidence-{confidence_color}'>{metrics.confidence_score*100:.1f}%</span>", unsafe_allow_html=True)
                        col_c.metric("Cost", f"${metrics.cost_usd:.6f}")

    # Input area
    st.markdown("---")
    col_input, col_send = st.columns([5, 1])
    with col_input:
        user_question = st.text_input(
            "Ask your question:",
            placeholder="e.g., How do I deploy a model?",
            label_visibility="collapsed"
        )
    with col_send:
        send_button = st.button("🚀 Send", use_container_width=True)

    if send_button and user_question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Get response
        with st.spinner("🔍 Processing your question..."):
            result = st.session_state.rag_system.ask(user_question)

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "metrics": result["metrics"]
        })

        # Add to metrics history
        if result["metrics"]:
            st.session_state.metrics_history.append(result["metrics"])

        # Rerun to update chat
        st.rerun()

    # Analytics Dashboard (expandable)
    with st.expander("📈 Analytics Dashboard", expanded=False):
        if st.session_state.metrics_history:
            # Create metrics dataframe
            metrics_df = pd.DataFrame([
                {
                    "Query": m.query[:50] + "...",
                    "Time": m.total_time,
                    "Confidence": m.confidence_score * 100,
                    "Cost": m.cost_usd,
                    "Sources": m.num_sources,
                    "Tokens": m.tokens_used,
                    "Mode": m.retriever_mode
                }
                for m in st.session_state.metrics_history
            ])

            # Performance charts with dark theme
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Confidence trend
                fig_confidence = px.line(
                    metrics_df,
                    y="Confidence",
                    title="Confidence Score Trend",
                    markers=True
                )
                fig_confidence.update_layout(
                    height=300,
                    paper_bgcolor='#111111',
                    plot_bgcolor='#1a1a1a',
                    font={'color': 'white'},
                    title_font={'color': 'white'},
                    xaxis={'gridcolor': '#333333', 'tickcolor': 'white'},
                    yaxis={'gridcolor': '#333333', 'tickcolor': 'white'}
                )
                fig_confidence.update_traces(line_color='#6b11cb')
                st.plotly_chart(fig_confidence, use_container_width=True)

            with col_chart2:
                # Response time trend
                fig_time = px.line(
                    metrics_df,
                    y="Time",
                    title="Response Time Trend (seconds)",
                    markers=True
                )
                fig_time.update_layout(
                    height=300,
                    paper_bgcolor='#111111',
                    plot_bgcolor='#1a1a1a',
                    font={'color': 'white'},
                    title_font={'color': 'white'},
                    xaxis={'gridcolor': '#333333', 'tickcolor': 'white'},
                    yaxis={'gridcolor': '#333333', 'tickcolor': 'white'}
                )
                fig_time.update_traces(line_color='#2575fc')
                st.plotly_chart(fig_time, use_container_width=True)

            # Metrics table
            st.subheader("📊 Query History")
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100
                    ),
                    "Cost": st.column_config.NumberColumn(
                        "Cost ($)",
                        format="%.6f"
                    )
                }
            )

            # Download button
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Metrics CSV",
                data=csv,
                file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No queries yet. Start asking questions to see analytics!")

    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small style='color: #666666;'>Powered by LangChain • ChromaDB • GPT-4o-mini • Dark Mode</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()