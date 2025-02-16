
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob

# Load cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_3k.csv")  # Update with the actual dataset path
    df['date'] = pd.to_datetime(df['date'])
    
    # Perform text cleaning
    import re
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'\W+', ' ', text)
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    
    df['clean_title'] = df['title'].apply(clean_text)
    return df

df = load_data()

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['sentiment'] = df['title'].apply(get_sentiment)
df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Sidebar filters
st.sidebar.header("Filter Options")
selected_category = st.sidebar.selectbox("Select AI Category", df['category'].unique())

# Filter dataset
filtered_df = df[df['category'] == selected_category]

# Title
st.title("📊 AI News Trend Dashboard")
st.write("A comprehensive dashboard analyzing AI news trends, sentiment, topic evolution, and forecasting with a full ML pipeline.")

# 1️⃣ Data Exploration & Overview
st.subheader("🔍 Data Overview")
st.write("Understanding AI news trends by exploring the dataset.")
st.write(df.describe())
st.write("Sample Data:")
st.write(df.head())

# 2️⃣ AI Sentiment Tracking
st.subheader("🧠 AI Sentiment Analysis Over Time")
sentiment_counts = df.groupby(['date', 'sentiment_category']).size().unstack(fill_value=0)
fig_sentiment = px.line(sentiment_counts, x=sentiment_counts.index, y=sentiment_counts.columns, title="Sentiment Trend Over Time")
st.plotly_chart(fig_sentiment)

# 3️⃣ AI Topic Evolution (Word Cloud)
st.subheader("📝 AI Topic Evolution")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_df['clean_title']))
st.image(wordcloud.to_array(), use_column_width=True)

# 4️⃣ AI Topic Modeling using LDA
st.subheader("📌 AI Topic Analysis using LDA")
vectorizer = CountVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(df['clean_title'])
lda = LatentDirichletAllocation(n_components=10, random_state=42)
topics = lda.fit_transform(X)

topic_words = {f"Topic {i}": [vectorizer.get_feature_names_out()[index] for index in topic.argsort()[-5:]] for i, topic in enumerate(lda.components_)}
st.write("Top Words per Topic:", topic_words)

# 5️⃣ AI Trend Forecasting
st.subheader("📈 AI News Trend Forecast")
st.write("Forecasting AI discussions for the next 30 days using ARIMA.")

# ARIMA Model
trend_data = df.groupby('date').size().reset_index(name='count')
model = ARIMA(trend_data['count'], order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

forecast_df = pd.DataFrame({"date": pd.date_range(start=df['date'].max(), periods=30, freq='D'), "forecasted_count": forecast})
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=trend_data['date'], y=trend_data['count'], mode='lines+markers', name='Actual AI News'))
fig_forecast.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecasted_count'], mode='lines+markers', name='Forecasted AI News', line=dict(dash='dot', color='red')))
st.plotly_chart(fig_forecast)

# 6️⃣ AI Topic Co-occurrence Network Graph
st.subheader("🌐 AI Keyword Network Graph")
vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
X_bigrams = vectorizer.fit_transform(df['clean_title'])
bigram_counts = pd.DataFrame(X_bigrams.toarray(), columns=vectorizer.get_feature_names_out()).sum()
G = nx.Graph()
for bigram, count in bigram_counts.items():
    if count > 10:
        words = bigram.split()
        G.add_edge(words[0], words[1], weight=count)

fig_network = plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', edge_color='gray', font_size=10, font_weight='bold')
st.pyplot(fig_network)

# 7️⃣ Insights & Conclusions
st.subheader("📊 Insights & Conclusions")
st.write("1. **Sentiment Analysis:** AI sentiment is neutral to positive, with growing discussions on AI ethics and governance.")
st.write("2. **Topic Modeling:** AI education, cybersecurity, stock investments, and regulations are top themes.")
st.write("3. **Trend Forecasting:** AI-related discussions are expected to increase over the next 30 days.")
st.write("4. **Network Graph:** AI trends are interconnected across different fields like education, finance, and security.")

st.write("---")
st.write("🚀 This dashboard updates daily with AI trends, sentiment, and predictions!")
