"""
News Bias Analyzer - Streamlit Web App
Detects political bias (Left/Center/Right) in news articles
"""
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import time

# Page configuration
st.set_page_config(
    page_title="News Bias Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1F77B4;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.2rem;
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.bias-left {
    background-color: #3498db;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
}
.bias-center {
    background-color: #95a5a6;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
}
.bias-right {
    background-color: #e74c3c;
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: bold;
}
.confidence-bar {
    background-color: #ecf0f1;
    border-radius: 10px;
    height: 30px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load the trained bias detection model"""
    try:
        # Update this path to where your model is saved
        model_path = "Kaptain2026/news-bias-detector"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return tokenizer, model, True
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        st.info("💡 Make sure you've trained the model and saved it to './final_bias_detector'")
        return None, None, False

def scrape_article(url):
    """
    Scrape article text from URL
    Returns: (title, text, success)
    """
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None, "Invalid URL format", False
        
        # Set headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to get title
        title = soup.find('title')
        title = title.get_text() if title else "Unknown Title"
        
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Get text - try multiple strategies
        article_text = ""
        
        # Strategy 1: Look for article tag
        article = soup.find('article')
        if article:
            article_text = article.get_text(separator=' ', strip=True)
        
        # Strategy 2: Look for common content divs
        if not article_text:
            for tag in ['div', 'section']:
                for class_name in ['article-body', 'article-content', 'story-body', 'entry-content', 'post-content', 'content']:
                    content = soup.find(tag, class_=re.compile(class_name, re.I))
                    if content:
                        article_text = content.get_text(separator=' ', strip=True)
                        break
                if article_text:
                    break
        
        # Strategy 3: Get all paragraph text
        if not article_text:
            paragraphs = soup.find_all('p')
            article_text = ' '.join([p.get_text() for p in paragraphs])
        
        # Clean text
        article_text = re.sub(r'\s+', ' ', article_text).strip()
        
        if len(article_text) < 100:
            return title, "Could not extract enough text from the article. The page might require JavaScript or have anti-scraping protection.", False
        
        return title, article_text, True
        
    except requests.exceptions.Timeout:
        return None, "Request timed out. The website took too long to respond.", False
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching URL: {str(e)}", False
    except Exception as e:
        return None, f"Error processing article: {str(e)}", False

def predict_bias(text, tokenizer, model):
    """
    Predict political bias of text
    Returns: (prediction, probabilities, confidence)
    """
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            prediction = probs.argmax().item()
            confidence = probs[prediction].item()
        
        return prediction, probs.numpy(), confidence
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Header
st.markdown('<div class="main-header">📰 News Bias Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Detect political bias in news articles using AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This tool uses a fine-tuned **DeBERTa-v3** model to analyze news articles and detect political bias.
    
    **How it works:**
    1. Enter a news article URL
    2. AI scrapes and analyzes the full article
    3. Get bias classification (Left/Center/Right)
    4. See confidence scores
    
    **Accuracy:** 90%+ on balanced test set
    """)
    
    st.header("📊 Model Info")
    st.write("""
    - **Model:** DeBERTa-v3-base
    - **Classes:** Left, Center, Right
    - **Context:** Full article (512 tokens)
    - **Training:** MBIC Dataset
    """)
    
    st.header("⚠️ Disclaimer")
    st.write("""
    This tool provides AI-based analysis and should not be considered definitive. Always verify information from multiple sources.
    """)

# Load model
if not st.session_state.model_loaded:
    with st.spinner("🔄 Loading AI model..."):
        tokenizer, model, success = load_model()
        if success:
            st.session_state.tokenizer = tokenizer
            st.session_state.model = model
            st.session_state.model_loaded = True
            st.success("✅ Model loaded successfully!")
        else:
            st.error("❌ Failed to load model. Please check the model path.")
            st.stop()

# Main interface
st.header("🔍 Analyze an Article")

# Input methods
input_method = st.radio(
    "Choose input method:",
    ["URL", "Paste Text"],
    horizontal=True
)

if input_method == "URL":
    url = st.text_input(
        "Enter article URL:",
        placeholder="https://example.com/news-article",
        help="Paste the full URL of a news article"
    )
    
    if st.button("🚀 Analyze Article", type="primary", use_container_width=True):
        if not url:
            st.warning("⚠️ Please enter a URL")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Scrape article
            status_text.text("📥 Fetching article...")
            progress_bar.progress(25)
            
            title, article_text, success = scrape_article(url)
            
            if not success:
                st.error(f"❌ {article_text}")
                st.stop()
            
            # Step 2: Display article info
            progress_bar.progress(50)
            status_text.text("📄 Processing article...")
            
            st.subheader("📰 Article Information")
            st.write(f"**Title:** {title}")
            st.write(f"**Length:** {len(article_text)} characters, {len(article_text.split())} words")
            
            with st.expander("📖 View Article Text"):
                st.write(article_text[:2000] + "..." if len(article_text) > 2000 else article_text)
            
            # Step 3: Predict bias
            progress_bar.progress(75)
            status_text.text("🤖 Analyzing bias...")
            
            prediction, probabilities, confidence = predict_bias(
                article_text,
                st.session_state.tokenizer,
                st.session_state.model
            )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            if prediction is not None:
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                bias_labels = ['Left-Leaning', 'Center/Neutral', 'Right-Leaning']
                predicted_bias = bias_labels[prediction]
                
                # Main prediction box
                if prediction == 0:
                    st.markdown(f'<div class="bias-left">🔵 {predicted_bias}</div>', unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f'<div class="bias-center">⚪ {predicted_bias}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bias-right">🔴 {predicted_bias}</div>', unsafe_allow_html=True)
                
                st.write("")  # Spacing
                
                # Confidence meter
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                with col2:
                    st.progress(confidence)
                
                # Detailed probabilities
                st.subheader("📈 Detailed Scores")
                cols = st.columns(3)
                for idx, (label, prob) in enumerate(zip(bias_labels, probabilities)):
                    with cols[idx]:
                        st.metric(
                            label=label,
                            value=f"{prob*100:.1f}%",
                            delta=None
                        )
                        st.progress(float(prob))
                
                # Interpretation
                st.markdown("---")
                st.subheader("💡 Interpretation")
                
                if confidence > 0.8:
                    st.success(f"✅ High confidence: The model is quite certain this article is **{predicted_bias.lower()}**.")
                elif confidence > 0.6:
                    st.info(f"ℹ️ Moderate confidence: The article appears to be **{predicted_bias.lower()}**, but consider checking other sources.")
                else:
                    st.warning(f"⚠️ Low confidence: The bias is unclear. The article might be mixed or have subtle bias.")
                
                # Additional insights
                if prediction == 1 and confidence > 0.7:
                    st.write("📌 This article appears to maintain neutrality in its reporting.")
                elif max(probabilities) - sorted(probabilities)[-2] < 0.2:
                    st.write("📌 The article shows mixed signals. It may contain elements from multiple perspectives.")

else:  # Paste Text
    article_text = st.text_area(
        "Paste article text:",
        height=300,
        placeholder="Paste the full article text here...",
        help="Paste the complete article text for analysis"
    )
    
    if st.button("🚀 Analyze Text", type="primary", use_container_width=True):
        if not article_text or len(article_text) < 100:
            st.warning("⚠️ Please enter at least 100 characters of text")
        else:
            # Progress
            with st.spinner("🤖 Analyzing text..."):
                prediction, probabilities, confidence = predict_bias(
                    article_text,
                    st.session_state.tokenizer,
                    st.session_state.model
                )
            
            if prediction is not None:
                # Display results (same as above)
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                bias_labels = ['Left-Leaning', 'Center/Neutral', 'Right-Leaning']
                predicted_bias = bias_labels[prediction]
                
                if prediction == 0:
                    st.markdown(f'<div class="bias-left">🔵 {predicted_bias}</div>', unsafe_allow_html=True)
                elif prediction == 1:
                    st.markdown(f'<div class="bias-center">⚪ {predicted_bias}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="bias-right">🔴 {predicted_bias}</div>', unsafe_allow_html=True)
                
                st.write("")
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                with col2:
                    st.progress(confidence)
                
                st.subheader("📈 Detailed Scores")
                cols = st.columns(3)
                for idx, (label, prob) in enumerate(zip(bias_labels, probabilities)):
                    with cols[idx]:
                        st.metric(label=label, value=f"{prob*100:.1f}%")
                        st.progress(float(prob))
                
                st.markdown("---")
                st.subheader("💡 Interpretation")
                
                if confidence > 0.8:
                    st.success(f"✅ High confidence: The model is quite certain this text is **{predicted_bias.lower()}**.")
                elif confidence > 0.6:
                    st.info(f"ℹ️ Moderate confidence: The text appears to be **{predicted_bias.lower()}**, but consider checking other sources.")
                else:
                    st.warning(f"⚠️ Low confidence: The bias is unclear. The text might be mixed or have subtle bias.")

# Example URLs section
st.markdown("---")
st.subheader("🔗 Try Example Articles")
st.write("Click on any example to analyze:")

example_cols = st.columns(3)
examples = [
    ("CNN Politics", "https://www.cnn.com/politics"),
    ("BBC News", "https://www.bbc.com/news"),
    ("Fox News Politics", "https://www.foxnews.com/politics"),
]

for col, (name, url) in zip(example_cols, examples):
    with col:
        if st.button(name, use_container_width=True):
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit and DeBERTa-v3</p>
    <p>⚠️ For educational purposes. Always verify information from multiple sources.</p>
</div>
""", unsafe_allow_html=True)