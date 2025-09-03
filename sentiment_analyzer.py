import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit Page Configuration ---
# This should be the first Streamlit command in your script
st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CORE LOGIC (Functions) ---

def fetch_news(api_key, query):
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    url = (f'https://newsapi.org/v2/everything?'
           f'q={query}&'
           f'from={from_date}&'
           f'sortBy=publishedAt&'
           f'language=en&'
           f'apiKey={api_key}')
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        articles = data.get('articles', [])
        headlines = [article['title'] for article in articles if article.get('title')]
        return list(set(headlines))
    else:
        # Gracefully handle API errors in the UI
        st.error(f"API Error: {response.json().get('message', 'Failed to fetch news.')}")
        return []

@st.cache_resource
def get_sentiment_pipeline():
    # Using a specific version for stability
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(headlines):
    sentiment_pipeline = get_sentiment_pipeline()
    results = sentiment_pipeline(headlines)
    df = pd.DataFrame({
        'headline': headlines,
        'sentiment': [result['label'] for result in results],
        'confidence': [result['score'] for result in results]
    })
    return df

# --- WEB APP INTERFACE ---

# --- Sidebar ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)
    st.header("⚙️ User Input")
    search_query_input = st.text_input("Company Name or Ticker", "Tesla", help="Enter a company name like 'Apple' or a ticker like 'TSLA'.")
    
    if st.button("Analyze Sentiment", type="primary"):
        st.session_state.run_analysis = True
        st.session_state.query = search_query_input
    
    st.markdown("---")
    st.info("This app fetches recent news and uses AI to analyze its sentiment. It's a portfolio project showcasing data app development skills.")

# --- Main Page ---
st.title("📈 Financial News Sentiment Analyzer")
st.markdown("This tool provides a high-level overview of the market sentiment for a specific company based on the latest news headlines.")

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

if st.session_state.run_analysis:
    # Access the API key from secrets
    try:
        api_key = st.secrets["api_key"]
    except (FileNotFoundError, KeyError):
        st.error("API key not found. Please create a `.streamlit/secrets.toml` file with your `api_key`.")
        st.stop()

    query = st.session_state.query
    with st.spinner(f"Fetching and analyzing news for '{query}'..."):
        headlines = fetch_news(api_key, query)
        
        if headlines:
            df = analyze_sentiment(headlines)
            
            # --- NEW: Dashboard Metrics ---
            total_headlines = len(df)
            positive_count = df[df['sentiment'] == 'POSITIVE'].shape[0]
            negative_count = df[df['sentiment'] == 'NEGATIVE'].shape[0]
            positive_pct = (positive_count / total_headlines) * 100 if total_headlines > 0 else 0

            st.markdown("---")
            st.subheader(f"Sentiment Dashboard for '{query}'")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Headlines", f"{total_headlines}")
            col2.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            col3.metric("Negative Sentiment", f"{(100-positive_pct):.1f}%")
            
            # --- Visualizations ---
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sentiment Distribution")
                fig_pie, ax_pie = plt.subplots()
                df['sentiment'].value_counts().plot.pie(ax=ax_pie, autopct='%1.1f%%', startangle=90, colors=['#34A853', '#EA4335'])
                ax_pie.set_ylabel('') # Hide the y-label
                st.pyplot(fig_pie)

            with col2:
                st.subheader("Sentiment Counts")
                fig_bar, ax_bar = plt.subplots()
                sns.countplot(ax=ax_bar, x='sentiment', data=df, palette={'POSITIVE': '#34A853', 'NEGATIVE': '#EA4335'}, order=['POSITIVE', 'NEGATIVE'])
                ax_bar.set_title('')
                ax_bar.set_ylabel('Number of Headlines')
                ax_bar.set_xlabel('Sentiment')
                st.pyplot(fig_bar)
            
            # --- NEW: Expander for Data Table ---
            with st.expander("📰 View Analyzed Headlines Data"):
                st.dataframe(df)
        else:
            st.warning("No recent headlines found. Please try a different query.")
    
    # Reset the state so it doesn't run again automatically
    st.session_state.run_analysis = False
else:
    st.info("Enter a company name in the sidebar and click 'Analyze Sentiment' to begin.")
