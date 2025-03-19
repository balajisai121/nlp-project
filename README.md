# nlp-project
Project Overview
This repository contains our submission for RAISE-25, an AI-driven competition hosted by Rutgers Bloustein MPI. Our project explores the impact of AI on education, careers, and society using news headlines as the primary dataset. By leveraging Natural Language Processing (NLP) techniques, we analyze sentiment, themes, and trends to determine whether AI is shaping a Utopian or Dystopian future.

Dataset Description
The dataset consists of AI-related news headlines with the following key attributes:

Date (publication date)
Title (headline text)
Source (news publisher)
Character & Word Counts (headline length metrics)
Day, Month, Year, Quarter, Weekend Indicator (temporal insights)
Language (multilingual support)
Category:
Education (learning, teaching, students, universities)
Society (ethics, governance, policies, public perception)
Career (jobs, recruitment, workforce trends)
Other (miscellaneous headlines)
Note: The dataset categorization is algorithmic, meaning some headlines may be misclassified due to context ambiguity.

Project Goals
Extract Meaningful Insights: Use text analytics and NLP to understand AI’s impact on different domains.
Sentiment Analysis: Classify headlines as positive, negative, or neutral to determine AI’s perceived impact.
Topic Modeling & Trend Analysis: Identify dominant AI-related topics and their evolution over time.
Comparative Analysis Across Domains: Examine education, career, and society separately to highlight key concerns and opportunities.
Methods & Tools Used
1. Exploratory Data Analysis (EDA)
Word Clouds & N-grams: Identify common words and phrases.
Category & Source Distribution: Visualize how AI-related news is covered across media platforms.
2. Natural Language Processing (NLP)
Sentiment Analysis: Using VADER & TextBlob for polarity scoring.
Topic Modeling: LDA & BERTopic to uncover hidden themes.
Named Entity Recognition (NER): Identifying key organizations, people, and AI technologies in the news.
3. Machine Learning & AI Integration
Supervised Classification: Predict if an article leans Utopian or Dystopian using Random Forest, XGBoost.
Deep Learning Approaches: Experiment with Transformer-based models (BERT, GPT-3.5) for context-based analysis.
Clustering & Trends: Group similar headlines using K-Means & DBSCAN.
4. Comparative Analysis
Education Sector: AI in classrooms, online learning, automation in teaching.
Workforce Impact: Job displacement vs. AI-driven opportunities.
Society & Governance: Ethical dilemmas, misinformation, policy regulations.
Results & Key Insights
✅ Positive Sentiment dominates AI in education, highlighting benefits like personalized learning.
⚠️ Negative Sentiment in career-related headlines points to fears of job loss & automation risks.
📊 Topic Evolution: The perception of AI shifted from optimism (pre-2020) to skepticism (post-2021).
🌎 Regional Differences: News sentiment varies significantly across languages and sources.

Future Work
Fine-tune NLP models to reduce misclassification in sentiment & topic analysis.
Expand analysis to more sources for a global perspective.
Automate report generation using AI-generated summaries.
