import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json
from collections import defaultdict, Counter

class PredictiveAnalytics:
    def __init__(self):
        self.sentiment_predictor = None
        self.engagement_predictor = None
        self.churn_predictor = None
        self.trained = False
        
    def train_models(self, historical_data: List[Dict]):
        """Train predictive models on historical data"""
        if len(historical_data) < 100:
            return {"success": False, "error": "Insufficient historical data for training"}
        
        # Prepare training data
        features, sentiment_targets, engagement_targets, churn_targets = self._prepare_training_data(historical_data)
        
        if len(features) < 50:
            return {"success": False, "error": "Insufficient feature data"}
        
        # Train sentiment prediction model
        X_train, X_test, y_sent_train, y_sent_test = train_test_split(
            features, sentiment_targets, test_size=0.2, random_state=42
        )
        
        self.sentiment_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.sentiment_predictor.fit(X_train, y_sent_train)
        sent_score = self.sentiment_predictor.score(X_test, y_sent_test)
        
        # Train engagement prediction model  
        _, _, y_eng_train, y_eng_test = train_test_split(
            features, engagement_targets, test_size=0.2, random_state=42
        )
        
        self.engagement_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.engagement_predictor.fit(X_train, y_eng_train)
        eng_score = self.engagement_predictor.score(X_test, y_eng_test)
        
        # Train churn prediction model
        _, _, y_churn_train, y_churn_test = train_test_split(
            features, churn_targets, test_size=0.2, random_state=42
        )
        
        self.churn_predictor = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.churn_predictor.fit(X_train, y_churn_train)
        churn_score = accuracy_score(y_churn_test, self.churn_predictor.predict(X_test))
        
        self.trained = True
        
        return {
            "success": True,
            "model_performance": {
                "sentiment_r2": round(sent_score, 3),
                "engagement_r2": round(eng_score, 3), 
                "churn_accuracy": round(churn_score, 3)
            },
            "training_samples": len(features)
        }
    
    def _prepare_training_data(self, historical_data: List[Dict]) -> Tuple[List[List[float]], List[float], List[float], List[int]]:
        """Prepare features and targets from historical data"""
        features = []
        sentiment_targets = []
        engagement_targets = []
        churn_targets = []
        
        for record in historical_data:
            # Extract features
            feature_vector = self._extract_features(record)
            if feature_vector:
                features.append(feature_vector)
                
                # Extract targets
                sentiment_targets.append(record.get('sentiment_score', 50) / 100.0)
                engagement_targets.append(record.get('engagement_score', 0.5))
                churn_targets.append(1 if record.get('churned', False) else 0)
        
        return features, sentiment_targets, engagement_targets, churn_targets
    
    def _extract_features(self, record: Dict) -> Optional[List[float]]:
        """Extract numerical features from a data record"""
        try:
            features = [
                record.get('message_count', 0) / 100.0,  # Normalized message count
                record.get('unique_authors', 0) / 50.0,  # Normalized author count
                record.get('avg_message_length', 0) / 200.0,  # Normalized message length
                record.get('question_ratio', 0),  # Ratio of questions
                record.get('exclamation_ratio', 0),  # Ratio of exclamations
                record.get('positive_keywords', 0) / 10.0,  # Normalized positive keywords
                record.get('negative_keywords', 0) / 10.0,  # Normalized negative keywords
                record.get('technical_keywords', 0) / 10.0,  # Normalized technical keywords
                record.get('hour_of_day', 12) / 24.0,  # Normalized hour
                record.get('day_of_week', 3) / 7.0,  # Normalized day
                record.get('previous_sentiment', 50) / 100.0,  # Previous sentiment
                record.get('trend_direction', 0),  # -1, 0, or 1 for trend
            ]
            return features
        except:
            return None


class WhatIfAnalyzer:
    def __init__(self, predictor: PredictiveAnalytics):
        self.predictor = predictor
        
    def analyze_game_change_impact(self, change_description: str, 
                                 current_community_state: Dict,
                                 personas: List[Dict]) -> Dict:
        """Analyze the predicted impact of a game change"""
        
        if not self.predictor.trained:
            return {"error": "Predictive models not trained"}
        
        # Categorize the change
        change_category = self._categorize_change(change_description)
        change_magnitude = self._assess_change_magnitude(change_description)
        
        # Predict impact on each persona
        persona_impacts = []
        for persona in personas:
            impact = self._predict_persona_reaction(persona, change_category, change_magnitude)
            persona_impacts.append(impact)
        
        # Calculate overall community impact
        overall_impact = self._calculate_overall_impact(persona_impacts, personas)
        
        # Generate recommendations
        recommendations = self._generate_change_recommendations(change_category, persona_impacts)
        
        return {
            "change_analysis": {
                "description": change_description,
                "category": change_category,
                "magnitude": change_magnitude
            },
            "persona_impacts": persona_impacts,
            "overall_impact": overall_impact,
            "recommendations": recommendations,
            "confidence": self._calculate_confidence(personas, current_community_state)
        }
    
    def _categorize_change(self, description: str) -> str:
        """Categorize the type of game change"""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['nerf', 'reduce', 'decrease', 'weaken']):
            return 'nerf'
        elif any(word in desc_lower for word in ['buff', 'increase', 'strengthen', 'improve']):
            return 'buff'  
        elif any(word in desc_lower for word in ['new', 'add', 'feature', 'content']):
            return 'content_addition'
        elif any(word in desc_lower for word in ['fix', 'bug', 'patch', 'repair']):
            return 'bug_fix'
        elif any(word in desc_lower for word in ['balance', 'adjust', 'tweak']):
            return 'balance_change'
        elif any(word in desc_lower for word in ['remove', 'delete', 'disable']):
            return 'content_removal'
        else:
            return 'general_change'
    
    def _assess_change_magnitude(self, description: str) -> float:
        """Assess the magnitude of the change (0.0 to 1.0)"""
        desc_lower = description.lower()
        
        high_impact_words = ['major', 'significant', 'complete', 'entirely', 'drastically']
        medium_impact_words = ['moderate', 'noticeable', 'somewhat', 'partially']
        low_impact_words = ['minor', 'slight', 'small', 'tiny', 'barely']
        
        if any(word in desc_lower for word in high_impact_words):
            return 0.8
        elif any(word in desc_lower for word in medium_impact_words):
            return 0.5
        elif any(word in desc_lower for word in low_impact_words):
            return 0.2
        else:
            return 0.4  # Default moderate impact
    
    def _predict_persona_reaction(self, persona: Dict, change_category: str, magnitude: float) -> Dict:
        """Predict how a specific persona will react to a change"""
        
        persona_type = persona['characteristics']['play_style']
        engagement = persona['characteristics']['engagement_level']
        
        # Base reaction based on persona type and change category
        reaction_matrix = {
            'competitive': {
                'nerf': -0.6, 'buff': 0.4, 'balance_change': -0.2,
                'content_addition': 0.2, 'bug_fix': 0.3, 'content_removal': -0.4
            },
            'casual': {
                'nerf': -0.2, 'buff': 0.2, 'balance_change': 0.0,
                'content_addition': 0.6, 'bug_fix': 0.4, 'content_removal': -0.2
            },
            'creative': {
                'nerf': -0.1, 'buff': 0.1, 'balance_change': 0.0,
                'content_addition': 0.8, 'bug_fix': 0.2, 'content_removal': -0.6
            },
            'social': {
                'nerf': -0.3, 'buff': 0.3, 'balance_change': -0.1,
                'content_addition': 0.4, 'bug_fix': 0.5, 'content_removal': -0.3
            }
        }
        
        base_reaction = reaction_matrix.get(persona_type, {}).get(change_category, 0.0)
        
        # Adjust for magnitude
        adjusted_reaction = base_reaction * magnitude
        
        # Adjust for engagement level
        engagement_multipliers = {
            'very_high': 1.3, 'high': 1.1, 'medium': 1.0, 'low': 0.8, 'very_low': 0.6
        }
        engagement_mult = engagement_multipliers.get(engagement, 1.0)
        final_reaction = adjusted_reaction * engagement_mult
        
        # Predict sentiment change
        current_sentiment = persona['behavioral_patterns']['sentiment_profile']['dominant']
        predicted_sentiment_change = self._predict_sentiment_change(final_reaction, current_sentiment)
        
        # Predict engagement change
        predicted_engagement_change = self._predict_engagement_change(final_reaction, persona_type)
        
        # Predict churn risk change
        churn_risk_change = max(-0.3, min(0.3, -final_reaction * 0.5))  # Negative reaction increases churn risk
        
        return {
            'persona_name': persona['name'],
            'persona_size': persona['size'],
            'predicted_reaction': round(final_reaction, 2),
            'sentiment_change': predicted_sentiment_change,
            'engagement_change': predicted_engagement_change,
            'churn_risk_change': round(churn_risk_change, 2),
            'reaction_explanation': self._explain_reaction(persona_type, change_category, final_reaction)
        }
    
    def _predict_sentiment_change(self, reaction_score: float, current_sentiment: str) -> Dict:
        """Predict change in sentiment"""
        sentiment_values = {'very_negative': -2, 'negative': -1, 'neutral': 0, 'positive': 1, 'very_positive': 2}
        current_value = sentiment_values.get(current_sentiment, 0)
        
        change_amount = reaction_score * 2  # Scale to sentiment range
        new_value = max(-2, min(2, current_value + change_amount))
        
        new_sentiment = {v: k for k, v in sentiment_values.items()}[round(new_value)]
        
        return {
            'from': current_sentiment,
            'to': new_sentiment,
            'change_magnitude': round(abs(change_amount), 2)
        }
    
    def _predict_engagement_change(self, reaction_score: float, persona_type: str) -> Dict:
        """Predict change in engagement"""
        # Negative reactions tend to decrease engagement more than positive reactions increase it
        if reaction_score < 0:
            engagement_change = reaction_score * 1.2  # Negative bias
        else:
            engagement_change = reaction_score * 0.8
        
        change_percentage = round(engagement_change * 100, 1)
        
        return {
            'direction': 'increase' if engagement_change > 0 else 'decrease',
            'magnitude': abs(change_percentage),
            'explanation': f"Engagement expected to {'increase' if engagement_change > 0 else 'decrease'} by {abs(change_percentage)}%"
        }
    
    def _calculate_overall_impact(self, persona_impacts: List[Dict], personas: List[Dict]) -> Dict:
        """Calculate weighted overall community impact"""
        total_users = sum(p['size'] for p in personas)
        
        if total_users == 0:
            return {"error": "No persona data available"}
        
        # Weight impacts by persona size
        weighted_reaction = sum(
            impact['predicted_reaction'] * personas[i]['size'] 
            for i, impact in enumerate(persona_impacts)
        ) / total_users
        
        # Calculate risk metrics
        high_churn_risk_users = sum(
            personas[i]['size'] for i, impact in enumerate(persona_impacts)
            if impact['churn_risk_change'] > 0.1
        )
        
        negative_reaction_users = sum(
            personas[i]['size'] for i, impact in enumerate(persona_impacts)
            if impact['predicted_reaction'] < -0.3
        )
        
        return {
            'overall_sentiment_impact': round(weighted_reaction, 2),
            'impact_level': self._categorize_impact_level(weighted_reaction),
            'at_risk_users': high_churn_risk_users,
            'negative_reaction_users': negative_reaction_users,
            'risk_percentage': round((high_churn_risk_users / total_users) * 100, 1),
            'summary': self._generate_impact_summary(weighted_reaction, high_churn_risk_users, total_users)
        }
    
    def _categorize_impact_level(self, impact_score: float) -> str:
        """Categorize the overall impact level"""
        if impact_score > 0.4:
            return 'very_positive'
        elif impact_score > 0.1:
            return 'positive'
        elif impact_score > -0.1:
            return 'neutral'
        elif impact_score > -0.4:
            return 'negative'
        else:
            return 'very_negative'
    
    def _generate_change_recommendations(self, change_category: str, persona_impacts: List[Dict]) -> List[Dict]:
        """Generate recommendations based on predicted impacts"""
        recommendations = []
        
        # Check for high-risk reactions
        high_risk_personas = [p for p in persona_impacts if p['predicted_reaction'] < -0.4]
        
        if high_risk_personas:
            recommendations.append({
                'priority': 'high',
                'type': 'risk_mitigation',
                'title': 'Address Negative Reactions',
                'description': f"Consider communication strategy for {len(high_risk_personas)} persona(s) with strong negative reactions",
                'affected_personas': [p['persona_name'] for p in high_risk_personas]
            })
        
        # Check for positive opportunities
        positive_personas = [p for p in persona_impacts if p['predicted_reaction'] > 0.3]
        
        if positive_personas:
            recommendations.append({
                'priority': 'medium',
                'type': 'amplification',
                'title': 'Leverage Positive Reception',
                'description': f"Highlight change benefits for {len(positive_personas)} enthusiastic persona(s)",
                'affected_personas': [p['persona_name'] for p in positive_personas]
            })
        
        # Category-specific recommendations
        if change_category == 'nerf':
            recommendations.append({
                'priority': 'medium',
                'type': 'communication',
                'title': 'Explain Nerf Rationale',
                'description': 'Provide clear reasoning for balance changes to maintain player trust',
                'strategy': 'transparency'
            })
        
        return recommendations
    
    def _explain_reaction(self, persona_type: str, change_category: str, reaction_score: float) -> str:
        """Explain why a persona reacted a certain way"""
        explanations = {
            'competitive': {
                'nerf': "Competitive players strongly dislike nerfs that affect their preferred strategies",
                'buff': "Appreciate buffs that create new strategic options",
                'balance_change': "Concerned about meta shifts affecting their performance"
            },
            'casual': {
                'content_addition': "Excited about new content to explore and enjoy",
                'bug_fix': "Happy when game issues are resolved",
                'nerf': "Less affected by balance changes, more focused on fun"
            },
            'creative': {
                'content_addition': "Thrilled by new tools and possibilities for creation",
                'content_removal': "Frustrated when creative options are taken away"
            },
            'social': {
                'bug_fix': "Values stability for group activities",
                'content_addition': "Enjoys new ways to interact with community"
            }
        }
        
        specific_explanation = explanations.get(persona_type, {}).get(change_category)
        if specific_explanation:
            return specific_explanation
        
        # Generic explanations
        if reaction_score > 0.3:
            return f"{persona_type.title()} players generally respond positively to this type of change"
        elif reaction_score < -0.3:
            return f"{persona_type.title()} players tend to be concerned about this type of change"
        else:
            return f"{persona_type.title()} players show mixed reactions to this change"
    
    def _generate_impact_summary(self, weighted_reaction: float, at_risk_users: int, total_users: int) -> str:
        """Generate human-readable impact summary"""
        impact_level = self._categorize_impact_level(weighted_reaction)
        
        if impact_level == 'very_positive':
            return f"Community will likely respond very positively to this change"
        elif impact_level == 'positive':
            return f"Generally positive community reaction expected"
        elif impact_level == 'neutral':
            return f"Mixed community reaction expected with minimal overall impact"
        elif impact_level == 'negative':
            return f"Some negative community reaction expected, monitor {at_risk_users} at-risk users"
        else:
            return f"Strong negative reaction predicted, {at_risk_users}/{total_users} users at churn risk"
    
    def _calculate_confidence(self, personas: List[Dict], community_state: Dict) -> float:
        """Calculate prediction confidence based on data quality"""
        total_users = sum(p['size'] for p in personas)
        persona_quality = sum(p['confidence'] for p in personas) / len(personas) if personas else 0
        
        # More users and higher persona quality = higher confidence
        user_factor = min(total_users / 100.0, 1.0)  # Normalize to 1.0
        confidence = (user_factor * 0.6 + persona_quality * 0.4)
        
        return round(min(confidence, 0.95), 2)  # Cap at 95%