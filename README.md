
# Large-Scale Social Media Intelligence Platform

**DS-03 Capstone Project | Data Science Internship**

## Live Dashboard
Explore the deployed application:
https://social-media-intelligence-hhj6ruva5nrvgb43nboqty.streamlit.app/

## Project Overview
This project demonstrates a social media text analysis pipeline covering data preparation, sentiment analysis, topic modeling, sentiment trends, graph analysis, and an illustrative misinformation classifier.

## Data Sources
- **Synthetic development dataset:** 600 illustrative records used for pipeline development.
- **YouTube API sample:** 50 real video metadata records collected using YouTube Data API v3. The dataset includes video titles and descriptions, not viewer comments or verified audience opinions.

## Technologies
- Python and Pandas
- Hugging Face Transformers (DistilBERT)
- BERTopic
- NetworkX
- Scikit-learn
- Streamlit

## Key Features
- Sentiment classification of video titles and descriptions
- Exploratory topic modeling
- Sentiment trends by publication year
- Video-topic network metrics
- Baseline text classification for illustrative misinformation examples
- Interactive dashboard for exploring outputs

## Important Limitations
The datasets are small, and the development records and misinformation labels are synthetic. Sentiment predictions are not verified audience opinions, and the misinformation classifier is not a reliable fact-checking system. Network metrics describe video-topic associations, not actual user influence or interactions.

## Running Locally
Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```
