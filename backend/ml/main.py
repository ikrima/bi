from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from collections import Counter

app = FastAPI(title="Player Intelligence ML Service")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model_loaded = True
except Exception as e:
    print(f"Warning: Could not load sentence transformer model: {e}")
    model = None
    model_loaded = False

class EmbeddingRequest(BaseModel):
    texts: List[str]

class ClusterRequest(BaseModel):
    embeddings: List[List[float]]
    messages: List[str]
    timestamps: List[str] = None
    authors: List[str] = None

class PersonaRequest(BaseModel):
    user_data: List[Dict[str, Any]]

class PredictionRequest(BaseModel):
    change_description: str
    affected_areas: List[str]
    personas: List[Dict[str, Any]]
    historical_reactions: List[Dict[str, Any]] = []

class CompetitorAnalysisRequest(BaseModel):
    messages: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": str(np.datetime64('now'))
    }

@app.post("/embed")
def generate_embeddings(request: EmbeddingRequest):
    if not model_loaded:
        return {"error": "Model not loaded", "embeddings": []}
    
    try:
        embeddings = model.encode(request.texts)
        return {"embeddings": embeddings.tolist()}
    except Exception as e:
        return {"error": str(e), "embeddings": []}

@app.post("/cluster")
def cluster_messages(request: ClusterRequest):
    try:
        from clustering import MessageClusterer
        
        if len(request.embeddings) < 5:
            return {"clusters": [], "summary": {"total_messages": len(request.messages)}}
        
        # Use advanced clustering
        clusterer = MessageClusterer(min_cluster_size=max(3, len(request.embeddings) // 20))
        result = clusterer.cluster_messages(
            request.embeddings,
            request.messages,
            getattr(request, 'timestamps', None),
            getattr(request, 'authors', None)
        )
        
        return result
        
    except Exception as e:
        return {"error": str(e), "clusters": [], "summary": {"error": True}}

def extract_theme(messages):
    """Extract theme from cluster messages"""
    if not messages:
        return "empty"
    
    text = ' '.join(messages).lower()
    words = re.findall(r'\b[a-z]+\b', text)
    word_freq = Counter(words)
    
    # Remove common words
    stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'to', 'of', 'in', 'for', 'with', 'this', 'that'}
    filtered_words = [w for w, _ in word_freq.most_common(20) 
                     if w not in stop_words and len(w) > 2]
    
    if not filtered_words:
        return "general discussion"
    
    return ' + '.join(filtered_words[:3])

def analyze_sentiment(messages):
    """Simple sentiment analysis"""
    positive_words = {'love', 'great', 'awesome', 'amazing', 'good', 'excellent', 'fantastic', 'wonderful'}
    negative_words = {'hate', 'bad', 'awful', 'terrible', 'broken', 'sucks', 'horrible', 'worst'}
    
    text = ' '.join(messages).lower()
    
    positive_count = sum(word in text for word in positive_words)
    negative_count = sum(word in text for word in negative_words)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    return 'neutral'

@app.post("/personas/discover")
def discover_personas(request: PersonaRequest):
    """Discover player personas from user behavioral data"""
    try:
        from personas import AdvancedPersonaDiscovery
        
        if len(request.user_data) < 10:
            return {
                "success": False,
                "error": "Insufficient users for persona discovery",
                "minimum_required": 10,
                "current_users": len(request.user_data)
            }
        
        persona_discovery = AdvancedPersonaDiscovery()
        personas = persona_discovery.discover_personas(request.user_data)
        
        return {
            "success": True,
            "data": personas,
            "users_analyzed": len(request.user_data)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": []
        }

@app.post("/predict/change-impact")
def predict_change_impact(request: PredictionRequest):
    """Predict impact of game changes on player community"""
    try:
        from predictions import PredictiveAnalytics, WhatIfAnalyzer
        
        # Initialize analytics engines
        predictor = PredictiveAnalytics()
        what_if = WhatIfAnalyzer()
        
        # Run scenario analysis
        scenario_results = what_if.analyze_scenario({
            'change_type': 'game_update',
            'description': request.change_description,
            'affected_areas': request.affected_areas
        }, request.personas)
        
        # Calculate overall confidence based on historical data
        confidence = min(0.9, max(0.3, len(request.historical_reactions) * 0.1))
        
        return {
            "success": True,
            "data": scenario_results,
            "confidence": confidence,
            "personas_analyzed": len(request.personas)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": {}
        }

@app.post("/competitive/analyze")
def analyze_competitor_sentiment(request: CompetitorAnalysisRequest):
    """Analyze sentiment in competitor mentions"""
    try:
        messages_text = [msg.get('content', '') for msg in request.messages]
        
        if not messages_text:
            return {
                "success": True,
                "data": {
                    "average_sentiment": 0.5,
                    "message_count": 0,
                    "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0}
                }
            }
        
        # Simple sentiment analysis for competitor mentions
        total_sentiment = 0
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        
        for text in messages_text:
            sentiment = analyze_sentiment([text])
            sentiment_counts[sentiment] += 1
            
            if sentiment == "positive":
                total_sentiment += 0.8
            elif sentiment == "negative":
                total_sentiment += 0.2
            else:
                total_sentiment += 0.5
        
        average_sentiment = total_sentiment / len(messages_text) if messages_text else 0.5
        
        return {
            "success": True,
            "data": {
                "average_sentiment": average_sentiment,
                "message_count": len(messages_text),
                "sentiment_distribution": sentiment_counts
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": {}
        }

@app.post("/trends/detect")
def detect_anomalies(data: Dict[str, List[float]]):
    """Detect anomalies and trends in time series data"""
    try:
        results = {}
        
        for metric_name, values in data.items():
            if len(values) < 3:
                results[metric_name] = {"trend": "insufficient_data", "anomalies": []}
                continue
            
            # Simple trend calculation
            n = len(values)
            x = list(range(n))
            
            # Linear regression for trend
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(xi * yi for xi, yi in zip(x, values))
            sum_x2 = sum(xi * xi for xi in x)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                slope = 0
            
            # Normalize slope by mean for trend classification
            mean_value = sum_y / n if n > 0 else 0
            normalized_slope = slope / mean_value if mean_value != 0 else 0
            
            if normalized_slope > 0.05:
                trend = "increasing"
            elif normalized_slope < -0.05:
                trend = "decreasing"
            else:
                trend = "stable"
            
            # Simple anomaly detection (values beyond 2 standard deviations)
            if len(values) > 5:
                mean = np.mean(values)
                std = np.std(values)
                anomalies = [
                    {"index": i, "value": val, "deviation": abs(val - mean) / std}
                    for i, val in enumerate(values)
                    if abs(val - mean) > 2 * std
                ]
            else:
                anomalies = []
            
            results[metric_name] = {
                "trend": trend,
                "slope": slope,
                "anomalies": anomalies
            }
        
        return {"success": True, "data": results}
        
    except Exception as e:
        return {"success": False, "error": str(e), "data": {}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)