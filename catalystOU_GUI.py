import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import random


###############################################################
#### you need to do the following before you run this code:
#
#
###############################################################


# --- Page Configuration ---
st.set_page_config(
    page_title="SynergyScout - Collaboration Discovery",
    page_icon="🔬",
    layout="wide"
)

# --- MOCK DATABASE ---
MOCK_DATABASE = {
    "prof_reed_ai": { "name": "Dr. Evelyn Reed", "affiliation": "Institute for AI Research", "fields": ["Natural Language Processing", "Explainable AI"], "methods": ["Transformer Networks", "Causal Inference", "Knowledge Graphs"], "keywords": ["LLM Alignment", "Causality", "Interpretability"]},
    "prof_tanaka_bio": { "name": "Dr. Kenji Tanaka", "affiliation": "Center for Biomedical Data Science", "fields": ["Bioinformatics", "Genomics"], "methods": ["Graph Neural Networks", "Dimensionality Reduction", "Survival Analysis"], "keywords": ["Gene Regulatory Networks", "Drug Discovery"]},
    "prof_chen_climate": { "name": "Dr. Anya Chen", "affiliation": "Global Climate Institute", "fields": ["Climate Modeling", "Earth Science"], "methods": ["Time-Series Forecasting", "Satellite Imagery Analysis"], "keywords": ["Sea Level Rise", "Carbon Cycle"]},
    "prof_santos_robotics": { "name": "Dr. Marco Santos", "affiliation": "Advanced Robotics Laboratory", "fields": ["Robotics", "Control Systems"], "methods": ["Reinforcement Learning", "SLAM"], "keywords": ["Autonomous Navigation", "Manipulation"]}
}

# --- DYNAMIC ANALYSIS ENGINE ---
def mock_analysis_engine():
    time.sleep(2) # Simulate delay
    key_a, key_b = random.sample(list(MOCK_DATABASE.keys()), 2)
    profile_a = MOCK_DATABASE[key_a]
    profile_b = MOCK_DATABASE[key_b]
    score = random.randint(75, 95)
    summary = f"Scholars {profile_a['name']} and {profile_b['name']} are highly complementary in methods and problem domains, ideal for theoretical-practical collaboration."
    insight1_text = f"**{profile_a['name']}'s** expertise in **{random.choice(profile_a['methods'])}** complements **{profile_b['name']}'s** focus on **{random.choice(profile_b['fields'])}**, enabling innovative cross-domain solutions."
    insight2_text = f"Both scholars address **AI Ethics** in their work, providing a shared foundation for impactful collaboration."
    insight3_text = f"While **{profile_a['name']}** brings theoretical models, a potential gap may exist in securing the large-scale, specialized datasets that **{profile_b['name']}**'s research typically requires for validation."
    
    return {
        "profile_a": profile_a, "profile_b": profile_b, "potential_score": score,
        "one_liner_summary": summary,
        "radar_data": {'categories': ['Domain Similarity', 'Method Complementarity', 'Technical Overlap', 'Resource Synergy', 'Network Proximity'], 'values': [random.randint(40, 90) for _ in range(5)]},
        "insights": [
            {"title": "Method Complementarity", "text": insight1_text, "icon": "💡"},
            {"title": "Problem Domain Synergy", "text": insight2_text, "icon": "✅"},
             {"title": "Potential Gaps", "text": insight3_text, "icon": "🧩"}
        ]
    }

# --- HELPER FUNCTIONS ---
def create_radar_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=data['values'], theta=data['categories'], fill='toself', name='Score', marker_color='#007AFF'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=300, margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# --- STYLES ---
st.markdown("""
<style>
.insight-card {
    border: 1px solid #e1e1e1;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
    background-color: #fcfcfc;
}
.insight-title {
    font-size: 1.1em;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# --- UI RENDERING LOGIC ---
# ==============================================================================

# Initialize session state to manage the two-page flow
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# --- RENDER INPUT PAGE ---
def render_input_page():
    st.image("https://i.imgur.com/Kqf471Q.png", use_column_width=True) # Display the mockup for context
    
    st.title("SynergyScout: Discover Research Collaboration Opportunities")
    st.markdown("Enter information for any two researchers (for this demo, any text will work). Our AI engine will **randomly select two scholars** from our database to analyze and generate a detailed collaboration potential report.")
    st.markdown("---")

    with st.form(key='input_form'):
        col1, col2 = st.columns(2)
        with col1:
            researcher_a_input = st.text_input("👤 Researcher A", placeholder="Enter arXiv ID, ORCID, etc.", key="researcher_a")
        with col2:
            researcher_b_input = st.text_input("👤 Researcher B", placeholder="Enter arXiv ID, ORCID, etc.", key="researcher_b")
        
        _, col_btn, _ = st.columns([2, 1, 2])
        with col_btn:
            submitted = st.form_submit_button("🚀 Analyze Potential", use_container_width=True)

    if submitted:
        if not researcher_a_input or not researcher_b_input:
            st.error("Please provide input for both researchers.")
        else:
            with st.spinner("AI Engine is analyzing... Please wait."):
                st.session_state.results = mock_analysis_engine()
                st.session_state.analysis_complete = True
            st.rerun()

# --- UI COMPONENTS ---
def create_gauge_chart(score):
    score_on_10 = score / 10.0
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score_on_10,
        number = {'suffix': "/10"},
        title = {'text': "Synergy Score", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 10]},
            'bar': {'color': "#6897FF"}, # Bar color for the score
            # Colored steps for the background
            'steps': [
                {'range': [0, 4], 'color': "#ea4335"},  # Red for low scores
                {'range': [4, 7], 'color': "#fbbc05"},  # Yellow for medium scores
                {'range': [7, 10], 'color': "#34a853"}   # Green for high scores
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- PAGE 2: REPORT PAGE ---
def render_synergy_report():
    results = st.session_state.results
    st.title("Collaboration Synergy Report")
    if st.button("⬅️ Start New Analysis"):
        st.session_state.analysis_complete = False
        if 'results' in st.session_state: del st.session_state.results
        st.rerun()
    st.markdown("---")
    
    # Using a smaller gap for the columns to bring them closer
    col1, col2, col3 = st.columns([1, 1.5, 1], gap="small")
    
    def display_profile_header(profile):
        st.subheader(f"👤 {profile['name']}")
        st.caption(f"🏢 {profile.get('affiliation', '')}")
        st.markdown("**Research Fields:**")
        for field in profile.get('fields', []): st.markdown(f"- {field}")
        st.markdown("**Methods:**")
        for method in profile.get('methods', []): st.markdown(f"- {method}")
        st.markdown("**Keywords:**")
        for keyword in profile.get('keywords', []): st.markdown(f"- {keyword}")

    with col1:
        with st.container(border=True): display_profile_header(results['profile_a'])
    with col2:
        st.plotly_chart(create_gauge_chart(results['potential_score']), use_container_width=True)
        with st.container(border=True):
            st.subheader("💡 Summary", anchor=False)
            st.markdown(results['one_liner_summary'])
    with col3:
        with st.container(border=True): display_profile_header(results['profile_b'])
    st.markdown("---")
    st.header("Detailed Analysis")
    
    tab_titles = [f"{insight['icon']} {insight['title']}" for insight in results['insights']]
    tabs = st.tabs(tab_titles)
    
    for i, insight in enumerate(results['insights']):
        with tabs[i]:
            st.subheader(insight['title'])
            st.markdown(insight['text'])
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Start New Analysis", use_container_width=True):
            st.session_state.analysis_complete = False
            st.rerun()
    with col2:
        # Placeholder for Export functionality
        st.button("Export as PDF", use_container_width=True, type="primary")

     
# --- MAIN LOGIC to switch between pages ---
if st.session_state.analysis_complete:
    render_synergy_report()
else:
    render_input_page()
