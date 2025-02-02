import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import re
from datetime import datetime
import logging
from typing import Dict, List, Union, Optional
import json

class BotDetectionSystem:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Initialize TF-IDF vectorizer for content analysis
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Define thresholds for rule-based detection
        self.thresholds = {
            'spam_words': 0.3,
            'post_frequency': 50,  # posts per day
            'engagement_rate': 0.01,
            'link_ratio': 0.7,
            'duplicate_content': 0.8
        }
        
        # Load spam words list (you can expand this list)
        self.spam_words = [
            'buy now', 'click here', 'free', 'discount', 'limited time',
            'earn money', 'make money', 'win', 'winner', 'congratulations',
            'limited offer', 'best price', 'cheap', 'discount', 'sale'
        ]

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text content"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()

    def calculate_posting_frequency(self, timestamps: List[datetime]) -> float:
        """Calculate average posting frequency per day"""
        if not timestamps or len(timestamps) < 2:
            return 0.0
            
        timestamps = sorted(timestamps)
        time_diff = timestamps[-1] - timestamps[0]
        days_diff = time_diff.total_seconds() / (24 * 3600)
        return len(timestamps) / max(days_diff, 1)

    def calculate_engagement_rate(self, likes: List[int], followers: int) -> float:
        """Calculate average engagement rate"""
        if not likes or followers == 0:
            return 0.0
        return sum(likes) / (len(likes) * max(followers, 1))

    def analyze_content_patterns(self, texts: List[str]) -> Dict[str, float]:
        """Analyze content for spam patterns and sentiment"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Calculate spam score
        spam_scores = []
        for text in processed_texts:
            spam_word_count = sum(1 for word in self.spam_words if word in text)
            spam_scores.append(spam_word_count / max(len(text.split()), 1))
        
        # Calculate sentiment scores
        try:
            sentiment_results = self.sentiment_analyzer(texts[:100])  # Limit to prevent overload
            sentiment_scores = [result['score'] for result in sentiment_results]
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            sentiment_scores = [0.5] * len(texts)  # Default neutral sentiment
        
        # Check for duplicate content
        tfidf_matrix = self.tfidf.fit_transform(processed_texts)
        duplicate_score = tfidf_matrix.toarray().std(axis=0).mean()
        
        return {
            'spam_score': np.mean(spam_scores),
            'sentiment_score': np.mean(sentiment_scores),
            'duplicate_content_score': duplicate_score
        }

    def analyze_behavioral_patterns(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze user behavioral patterns"""
        patterns = {}
        
        # Posting frequency analysis
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            patterns['posting_frequency'] = self.calculate_posting_frequency(timestamps)
        else:
            patterns['posting_frequency'] = 0.0
        
        # Engagement analysis
        if all(col in data.columns for col in ['likes', 'followers']):
            patterns['engagement_rate'] = self.calculate_engagement_rate(
                data['likes'].tolist(),
                data['followers'].iloc[0] if len(data) > 0 else 0
            )
        else:
            patterns['engagement_rate'] = 0.0
            
        # Link ratio analysis
        if 'content' in data.columns:
            link_counts = data['content'].str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
            patterns['link_ratio'] = link_counts.mean() if not pd.isna(link_counts.mean()) else 0.0
        else:
            patterns['link_ratio'] = 0.0
            
        return patterns

    def calculate_bot_probability(self, 
                                content_patterns: Dict[str, float],
                                behavioral_patterns: Dict[str, float]) -> Dict[str, Union[float, int]]:
        """Calculate final bot probability and matched rules"""
        rule_weights = {
            'spam_score': 0.3,
            'duplicate_content': 0.2,
            'posting_frequency': 0.2,
            'engagement_rate': 0.15,
            'link_ratio': 0.15
        }
        
        matched_rules = []
        rule_scores = []
        
        # Check content patterns
        if content_patterns['spam_score'] > self.thresholds['spam_words']:
            matched_rules.append('High spam content detected')
            rule_scores.append(content_patterns['spam_score'] * rule_weights['spam_score'])
            
        if content_patterns['duplicate_content_score'] > self.thresholds['duplicate_content']:
            matched_rules.append('Duplicate content detected')
            rule_scores.append(content_patterns['duplicate_content_score'] * rule_weights['duplicate_content'])
        
        # Check behavioral patterns
        if behavioral_patterns['posting_frequency'] > self.thresholds['post_frequency']:
            matched_rules.append('Unusual posting frequency')
            rule_scores.append(min(behavioral_patterns['posting_frequency'] / self.thresholds['post_frequency'], 1) 
                             * rule_weights['posting_frequency'])
            
        if behavioral_patterns['engagement_rate'] < self.thresholds['engagement_rate']:
            matched_rules.append('Low engagement rate')
            rule_scores.append((1 - behavioral_patterns['engagement_rate'] / self.thresholds['engagement_rate']) 
                             * rule_weights['engagement_rate'])
            
        if behavioral_patterns['link_ratio'] > self.thresholds['link_ratio']:
            matched_rules.append('High link sharing rate')
            rule_scores.append(behavioral_patterns['link_ratio'] * rule_weights['link_ratio'])
        
        # Calculate final probability
        bot_probability = sum(rule_scores) / sum(rule_weights.values()) if rule_scores else 0.0
        
        return {
            'bot_probability': bot_probability,
            'matched_rules': matched_rules,
            'matched_rules_count': len(matched_rules)
        }

    def analyze_profile(self, data: Union[pd.DataFrame, Dict, List]) -> Dict:
        """Main analysis pipeline"""
        try:
            # Convert input to DataFrame if necessary
            if isinstance(data, (dict, list)):
                data = pd.DataFrame(data)
            
            # Ensure required columns exist
            required_columns = ['content', 'timestamp', 'likes', 'followers']
            for col in required_columns:
                if col not in data.columns:
                    data[col] = None
            
            # Run analysis components
            content_patterns = self.analyze_content_patterns(
                data['content'].dropna().tolist()
            )
            
            behavioral_patterns = self.analyze_behavioral_patterns(data)
            
            # Calculate final results
            result = self.calculate_bot_probability(
                content_patterns,
                behavioral_patterns
            )
            
            # Format response for frontend
            return {
                'bot_probability': result['bot_probability'],
                'content_patterns': {
                    'spam_score': content_patterns['spam_score'],
                    'sentiment_score': content_patterns['sentiment_score']
                },
                'behavioral_metrics': {
                    'posting_frequency': behavioral_patterns['posting_frequency'],
                    'engagement_rate': behavioral_patterns['engagement_rate']
                },
                'matched_rules': result['matched_rules'],
                'matched_rules_count': result['matched_rules_count']
            }
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}")

    def process_social_links(self, links: List[str]) -> pd.DataFrame:
        """Process social media links into analyzable data"""
        # Placeholder for social media API integration
        # You would implement actual social media data fetching here
        pass

    def save_analysis_results(self, results: Dict, filepath: str) -> None:
        """Save analysis results to a file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = BotDetectionSystem()
    
    # Sample data
    sample_data = {
        'content': ['Check out this amazing deal! www.example.com',
                   'Buy now at discounted price! Limited time offer!'],
        'timestamp': ['2024-02-01 10:00:00', '2024-02-01 10:05:00'],
        'likes': [5, 3],
        'followers': [100, 100]
    }
    
    # Run analysis
    try:
        results = detector.analyze_profile(sample_data)
        print("Analysis Results:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")