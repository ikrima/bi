import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import re
from datetime import datetime
from typing import List, Dict, Tuple

class MessageClusterer:
    def __init__(self, min_cluster_size=10):
        self.clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        self.vectorizer = TfidfVectorizer(
            max_features=200,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
    
    def cluster_messages(self, embeddings: List[List[float]], messages: List[str], 
                        timestamps: List[str] = None, authors: List[str] = None) -> Dict:
        """Advanced clustering with temporal and author analysis"""
        if len(embeddings) < 5:
            return {"clusters": [], "summary": {"total_messages": len(messages)}}
        
        embeddings_array = np.array(embeddings)
        
        # Perform HDBSCAN clustering
        labels = self.clusterer.fit_predict(embeddings_array)
        
        # Analyze clusters
        clusters = []
        noise_count = 0
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Noise
                noise_count = sum(1 for l in labels if l == -1)
                continue
                
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            cluster_messages = [messages[i] for i in cluster_indices]
            cluster_authors = [authors[i] if authors else f"user_{i}" for i in cluster_indices]
            cluster_timestamps = [timestamps[i] if timestamps else None for i in cluster_indices]
            
            # Extract detailed insights
            theme = self._extract_theme(cluster_messages)
            sentiment = self._analyze_sentiment(cluster_messages)
            urgency = self._detect_urgency(cluster_messages)
            author_diversity = len(set(cluster_authors))
            
            # Temporal analysis
            temporal_info = self._analyze_temporal_patterns(cluster_timestamps) if timestamps else {}
            
            clusters.append({
                'id': int(cluster_id),
                'theme': theme,
                'sentiment': sentiment,
                'urgency': urgency,
                'size': len(cluster_messages),
                'author_diversity': author_diversity,
                'sample_messages': cluster_messages[:3],
                'keywords': self._extract_keywords(cluster_messages),
                'temporal': temporal_info,
                'confidence': float(self.clusterer.probabilities_[cluster_indices].mean()) if hasattr(self.clusterer, 'probabilities_') else 0.8
            })
        
        # Sort clusters by importance (size * urgency * confidence)
        clusters.sort(key=lambda x: x['size'] * x['urgency'] * x['confidence'], reverse=True)
        
        return {
            "clusters": clusters[:10],  # Top 10 clusters
            "summary": {
                "total_messages": len(messages),
                "clustered_messages": len(messages) - noise_count,
                "noise_messages": noise_count,
                "cluster_count": len(clusters),
                "overall_sentiment": self._calculate_overall_sentiment(clusters)
            }
        }
    
    def _extract_theme(self, messages: List[str]) -> str:
        """Enhanced theme extraction using TF-IDF"""
        if not messages:
            return "empty"
        
        try:
            # Use TF-IDF for better keyword extraction
            tfidf_matrix = self.vectorizer.fit_transform(messages)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top TF-IDF terms
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = mean_scores.argsort()[-5:][::-1]
            top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0.1]
            
            if top_terms:
                return " + ".join(top_terms[:3])
            else:
                # Fallback to word frequency
                return self._fallback_theme_extraction(messages)
        except:
            return self._fallback_theme_extraction(messages)
    
    def _fallback_theme_extraction(self, messages: List[str]) -> str:
        """Fallback theme extraction using word frequency"""
        text = ' '.join(messages).lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        word_freq = Counter(words)
        
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'to', 'of', 'in', 
            'for', 'with', 'this', 'that', 'are', 'was', 'will', 'been', 'have',
            'has', 'had', 'can', 'could', 'should', 'would', 'may', 'might'
        }
        
        filtered_words = [
            word for word, count in word_freq.most_common(20) 
            if word not in stop_words and len(word) > 2 and count > 1
        ]
        
        return ' + '.join(filtered_words[:3]) if filtered_words else "general discussion"
    
    def _extract_keywords(self, messages: List[str]) -> List[str]:
        """Extract important keywords from cluster messages"""
        text = ' '.join(messages).lower()
        
        # Game-specific terms
        game_terms = re.findall(r'\b(?:bug|glitch|crash|lag|fps|ping|server|disconnect|error|broken|fix|update|patch|nerf|buff|balance|op|overpowered|meta|tier|rank|skill|noob|pro|git gud|gg|rip|lol|omg|wtf)\b', text)
        
        # Technical terms
        tech_terms = re.findall(r'\b(?:cpu|gpu|ram|memory|graphics|settings|config|install|download|steam|epic|launcher|driver|nvidia|amd|intel)\b', text)
        
        # Emotional terms
        emotion_terms = re.findall(r'\b(?:love|hate|amazing|awesome|terrible|awful|great|bad|good|excellent|worst|best|frustrating|annoying|fun|boring|exciting)\b', text)
        
        all_keywords = game_terms + tech_terms + emotion_terms
        return list(set(all_keywords))[:10]
    
    def _analyze_sentiment(self, messages: List[str]) -> str:
        """Enhanced sentiment analysis with scoring"""
        text = ' '.join(messages).lower()
        
        positive_words = {
            'love', 'great', 'awesome', 'amazing', 'good', 'excellent', 'fantastic', 
            'wonderful', 'perfect', 'best', 'fun', 'enjoy', 'like', 'happy', 'glad',
            'thanks', 'thank', 'appreciate', 'cool', 'nice', 'sweet', 'epic'
        }
        
        negative_words = {
            'hate', 'bad', 'awful', 'terrible', 'broken', 'sucks', 'horrible', 
            'worst', 'annoying', 'frustrating', 'stupid', 'dumb', 'trash', 'garbage',
            'boring', 'lame', 'disappointing', 'useless', 'pathetic', 'rage', 'angry'
        }
        
        urgent_words = {
            'crash', 'bug', 'error', 'broken', 'cant', 'cannot', 'wont', 'will not',
            'doesnt work', 'not working', 'help', 'fix', 'urgent', 'emergency'
        }
        
        positive_count = sum(text.count(word) for word in positive_words)
        negative_count = sum(text.count(word) for word in negative_words)
        urgent_count = sum(text.count(word) for word in urgent_words)
        
        # Weighted scoring
        sentiment_score = positive_count - negative_count - (urgent_count * 0.5)
        
        if sentiment_score > 2:
            return 'very_positive'
        elif sentiment_score > 0:
            return 'positive'
        elif sentiment_score < -2:
            return 'very_negative'
        elif sentiment_score < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def _detect_urgency(self, messages: List[str]) -> float:
        """Detect urgency level (0.0 to 1.0)"""
        text = ' '.join(messages).lower()
        
        urgent_indicators = {
            'crash': 1.0, 'bug': 0.8, 'error': 0.7, 'broken': 0.8,
            'cant play': 1.0, 'wont start': 0.9, 'not working': 0.7,
            'help': 0.5, 'fix': 0.6, 'urgent': 1.0, 'emergency': 1.0,
            'stuck': 0.4, 'lost': 0.3, 'confused': 0.2
        }
        
        urgency_score = 0.0
        word_count = 0
        
        for indicator, weight in urgent_indicators.items():
            count = text.count(indicator)
            if count > 0:
                urgency_score += weight * min(count, 3)  # Cap impact of repeated words
                word_count += count
        
        # Normalize by message length and word frequency
        total_words = len(text.split())
        if total_words > 0:
            urgency_score = min(urgency_score / max(total_words * 0.01, 1), 1.0)
        
        return round(urgency_score, 2)
    
    def _analyze_temporal_patterns(self, timestamps: List[str]) -> Dict:
        """Analyze temporal patterns in cluster messages"""
        if not timestamps or not any(timestamps):
            return {}
        
        try:
            # Convert timestamps to datetime objects
            valid_timestamps = []
            for ts in timestamps:
                if ts:
                    try:
                        # Assume ISO format, adjust as needed
                        valid_timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    except:
                        continue
            
            if len(valid_timestamps) < 2:
                return {}
            
            # Calculate time span
            time_span = (max(valid_timestamps) - min(valid_timestamps)).total_seconds() / 3600  # hours
            
            # Peak activity detection
            hour_counts = defaultdict(int)
            for ts in valid_timestamps:
                hour_counts[ts.hour] += 1
            
            peak_hour = max(hour_counts.keys(), key=lambda h: hour_counts[h])
            
            return {
                "time_span_hours": round(time_span, 2),
                "peak_hour": peak_hour,
                "message_frequency": round(len(valid_timestamps) / max(time_span, 0.1), 2),
                "recency": "recent" if time_span < 24 else "older"
            }
        except Exception:
            return {}
    
    def _calculate_overall_sentiment(self, clusters: List[Dict]) -> str:
        """Calculate overall sentiment across all clusters"""
        if not clusters:
            return 'neutral'
        
        sentiment_weights = {
            'very_positive': 2, 'positive': 1, 'neutral': 0, 
            'negative': -1, 'very_negative': -2
        }
        
        weighted_score = 0
        total_messages = 0
        
        for cluster in clusters:
            weight = sentiment_weights.get(cluster['sentiment'], 0)
            size = cluster['size']
            weighted_score += weight * size
            total_messages += size
        
        if total_messages == 0:
            return 'neutral'
        
        avg_sentiment = weighted_score / total_messages
        
        if avg_sentiment > 0.5:
            return 'positive'
        elif avg_sentiment < -0.5:
            return 'negative'
        else:
            return 'neutral'