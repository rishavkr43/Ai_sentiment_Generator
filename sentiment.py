# sentiment.py
"""
Sentiment Analysis Module
Uses Hugging Face transformers for sentiment analysis
"""

from transformers import pipeline

class SentimentAnalyzer:
    """
    Analyzes sentiment of input text using Hugging Face transformers
    Returns: 'positive', 'negative', or 'neutral'
    """
    
    def __init__(self):
        """Initialize the sentiment analysis pipeline"""
        self._pipeline = pipeline("sentiment-analysis")
    
    def detect_sentiment(self, text):
        """
        Detect sentiment and normalize labels to lowercase.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            str: One of 'positive', 'negative', or 'neutral'
        """
        if not text or not text.strip():
            return "neutral"
        
        result = self._pipeline(text)[0]
        label = result.get("label", "").lower()
        
        if "pos" in label:
            return "positive"
        if "neg" in label:
            return "negative"
        return "neutral"
    
    def get_sentiment_scores(self, text):
        """
        Get detailed sentiment scores
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Sentiment scores with label and confidence
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0}
        
        return self._pipeline(text)[0]