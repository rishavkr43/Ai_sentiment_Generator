# main.py
"""
AI Sentiment-Based Text Generator
A Streamlit application that analyzes sentiment and generates aligned text
"""

import streamlit as st
from sentiment import SentimentAnalyzer
from text_generator import TextGenerator
import time

# Page configuration
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        padding: 1rem 0;
        border-bottom: 2px solid #2E86AB;
        margin-bottom: 2rem;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 1rem 0;
    }
    .positive {
        background-color: #D4EDDA;
        color: #155724;
    }
    .negative {
        background-color: #F8D7DA;
        color: #721C24;
    }
    .neutral {
        background-color: #D1ECF1;
        color: #0C5460;
    }
    .generated-text {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        margin-top: 1rem;
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_resource
def load_models():
    try:
        with st.spinner("Loading models..."):
            sentiment_analyzer = SentimentAnalyzer()
            text_generator = TextGenerator()
        return sentiment_analyzer, text_generator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models
sentiment_analyzer, text_generator = load_models()

# Main header
st.markdown("<h1 class='main-header'>🤖 AI Sentiment-Based Text Generator</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Generation settings
    st.subheader("Generation Options")
    
    generation_mode = st.radio(
        "Generation Mode",
        ["Automatic Sentiment", "Manual Sentiment"],
        help="Choose whether to detect sentiment automatically or select manually"
    )
    
    if generation_mode == "Manual Sentiment":
        manual_sentiment = st.selectbox(
            "Select Sentiment",
            ["positive", "negative", "neutral"],
            format_func=lambda x: x.capitalize()
        )
    
    st.divider()
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        show_scores = st.checkbox("Show detailed sentiment scores", value=False)
        clear_history = st.button("Clear History", type="secondary")
        
        if clear_history:
            st.session_state.history = []
            st.success("History cleared!")
    
    st.divider()
    
    # Info section
    st.info("""
    **How it works:**
    1. Enter a prompt in the text area
    2. The AI analyzes sentiment (or use manual selection)
    3. Text is generated matching that sentiment
    4. View your generation history below
    """)

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("📝 Input Prompt")
    
    # Text input area
    prompt = st.text_area(
        "Enter your prompt:",
        height=120,
        placeholder="Type your prompt here... The AI will analyze its sentiment and generate text accordingly.",
        help="The AI will detect the sentiment of your prompt and generate text that aligns with that sentiment."
    )
    
    # Generate button
    generate_col1, generate_col2, generate_col3 = st.columns([1, 1, 2])
    
    with generate_col1:
        generate_btn = st.button("🚀 Generate Text", type="primary", use_container_width=True)
    
    with generate_col2:
        if st.button("🔄 Clear Input", use_container_width=True):
            st.rerun()

# Generation logic
if generate_btn:
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt to generate text.")
    else:
        with col1:
            # Progress indication
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Sentiment Analysis
            status_text.text("Analyzing sentiment...")
            progress_bar.progress(33)
            
            if generation_mode == "Automatic Sentiment":
                sentiment = sentiment_analyzer.detect_sentiment(prompt)
                if show_scores:
                    scores = sentiment_analyzer.get_sentiment_scores(prompt)
            else:
                sentiment = manual_sentiment
            
            time.sleep(0.5)  # Brief pause for UI feedback
            
            # Step 2: Text Generation
            status_text.text("Generating text...")
            progress_bar.progress(66)
            
            generated = text_generator.generate_text(prompt, sentiment)
            
            time.sleep(0.5)
            
            # Step 3: Complete
            status_text.text("Complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success("✅ Text generated successfully!")
            
            # Show sentiment badge
            sentiment_class = sentiment.lower()
            st.markdown(
                f"<div class='sentiment-badge {sentiment_class}'>Detected Sentiment: {sentiment.upper()}</div>",
                unsafe_allow_html=True
            )
            
            # Show detailed scores if enabled
            if show_scores and generation_mode == "Automatic Sentiment":
                with st.expander("📊 Detailed Sentiment Analysis"):
                    st.json(scores)
            
            # Display generated text
            st.subheader("📄 Generated Text")
            st.markdown(
                f"<div class='generated-text'>{generated}</div>",
                unsafe_allow_html=True
            )
            
            # Add to history
            st.session_state.history.append({
                'prompt': prompt,
                'sentiment': sentiment,
                'generated': generated,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })

# History section
with col2:
    st.subheader("📚 Generation History")
    
    if st.session_state.history:
        # Display history in reverse order (newest first)
        for idx, item in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5
            with st.expander(f"🕐 {item['timestamp']}", expanded=idx==0):
                st.write(f"**Prompt:** {item['prompt'][:100]}...")
                st.write(f"**Sentiment:** {item['sentiment'].capitalize()}")
                st.write(f"**Generated:** {item['generated'][:200]}...")
    else:
        st.info("No generation history yet. Start by entering a prompt!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Built with ❤️ using Streamlit, Transformers, and NLTK</p>
    <p style='font-size: 0.9rem;'>AI Text Generator v1.0 | Sentiment-Aligned Content Generation</p>
</div>
""", unsafe_allow_html=True)