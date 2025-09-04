import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import json

class AdvancedPersonaDiscovery:
    def __init__(self, min_cluster_size=15):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)
        self.clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=5,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        self.personas = {}
        self.persona_history = []
    
    def discover_personas(self, user_embeddings: List[List[float]], 
                         user_messages: List[List[str]], 
                         user_metadata: List[Dict]) -> Dict:
        """
        Advanced persona discovery with behavioral analysis
        """
        if len(user_embeddings) < 10:
            return {"personas": [], "summary": {"error": "Insufficient data for persona analysis"}}
        
        # Normalize embeddings
        embeddings_array = np.array(user_embeddings)
        scaled_embeddings = self.scaler.fit_transform(embeddings_array)
        
        # Dimensionality reduction for visualization
        reduced_embeddings = self.pca.fit_transform(scaled_embeddings)
        
        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(scaled_embeddings)
        
        # Calculate cluster quality
        if len(set(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(scaled_embeddings, cluster_labels)
        else:
            silhouette_avg = 0.0
        
        # Analyze each persona
        personas = []
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Noise cluster
                continue
            
            cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            cluster_messages = [user_messages[i] for i in cluster_indices]
            cluster_metadata = [user_metadata[i] for i in cluster_indices]
            cluster_coords = reduced_embeddings[cluster_indices]
            
            persona = self._analyze_persona(
                cluster_id, 
                cluster_messages, 
                cluster_metadata, 
                cluster_coords
            )
            personas.append(persona)
        
        # Sort by influence score
        personas.sort(key=lambda x: x['influence_score'], reverse=True)
        
        # Generate persona insights
        insights = self._generate_persona_insights(personas)
        
        return {
            "personas": personas,
            "summary": {
                "total_users": len(user_embeddings),
                "identified_personas": len(personas),
                "noise_users": sum(1 for l in cluster_labels if l == -1),
                "cluster_quality": round(silhouette_avg, 3),
                "insights": insights
            }
        }
    
    def _analyze_persona(self, cluster_id: int, messages: List[List[str]], 
                        metadata: List[Dict], coordinates: np.ndarray) -> Dict:
        """Comprehensive persona analysis"""
        
        # Flatten messages for analysis
        all_messages = [msg for user_msgs in messages for msg in user_msgs]
        
        # Basic characteristics
        play_style = self._detect_play_style(all_messages)
        engagement_level = self._measure_engagement(messages, metadata)
        sentiment_profile = self._analyze_sentiment_profile(all_messages)
        expertise_level = self._assess_expertise(all_messages)
        
        # Behavioral patterns
        activity_patterns = self._analyze_activity_patterns(metadata)
        communication_style = self._analyze_communication_style(all_messages)
        influence_score = self._calculate_influence_score(messages, metadata)
        
        # Predictive traits
        churn_risk = self._predict_churn_risk(messages, metadata, sentiment_profile)
        value_potential = self._assess_value_potential(engagement_level, influence_score)
        
        # Generate persona name and description
        persona_name = self._generate_persona_name(play_style, engagement_level, expertise_level)
        description = self._generate_persona_description(play_style, engagement_level, sentiment_profile)
        
        return {
            'id': cluster_id,
            'name': persona_name,
            'description': description,
            'size': len(messages),
            'characteristics': {
                'play_style': play_style,
                'engagement_level': engagement_level,
                'expertise_level': expertise_level,
                'communication_style': communication_style
            },
            'behavioral_patterns': {
                'activity_patterns': activity_patterns,
                'sentiment_profile': sentiment_profile,
                'primary_topics': self._extract_primary_topics(all_messages)
            },
            'predictive_metrics': {
                'churn_risk': churn_risk,
                'value_potential': value_potential,
                'influence_score': influence_score
            },
            'coordinates': {
                'centroid': coordinates.mean(axis=0).tolist()[:2],  # 2D for visualization
                'spread': float(coordinates.std())
            },
            'sample_messages': all_messages[:3],
            'confidence': float(len(messages) / max(10, len(messages)))  # Confidence based on sample size
        }
    
    def _detect_play_style(self, messages: List[str]) -> str:
        """Detect primary play style from messages"""
        text = ' '.join(messages).lower()
        
        competitive_indicators = ['rank', 'ranked', 'competitive', 'ladder', 'tournament', 'pro', 'tryhard', 'meta', 'optimal']
        casual_indicators = ['fun', 'casual', 'chill', 'relaxing', 'friends', 'social', 'story', 'single player']
        creative_indicators = ['build', 'creative', 'design', 'mod', 'custom', 'sandbox', 'creation']
        explorer_indicators = ['explore', 'discover', 'adventure', 'quest', 'world', 'lore', 'easter egg']
        social_indicators = ['guild', 'clan', 'group', 'team', 'community', 'party', 'together']
        
        scores = {
            'competitive': sum(text.count(word) for word in competitive_indicators),
            'casual': sum(text.count(word) for word in casual_indicators),
            'creative': sum(text.count(word) for word in creative_indicators),
            'explorer': sum(text.count(word) for word in explorer_indicators),
            'social': sum(text.count(word) for word in social_indicators)
        }
        
        return max(scores.keys(), key=lambda k: scores[k]) if max(scores.values()) > 0 else 'general'
    
    def _measure_engagement(self, messages: List[List[str]], metadata: List[Dict]) -> str:
        """Measure engagement level"""
        total_messages = sum(len(user_msgs) for user_msgs in messages)
        avg_messages_per_user = total_messages / len(messages)
        
        if avg_messages_per_user > 20:
            return 'very_high'
        elif avg_messages_per_user > 10:
            return 'high'
        elif avg_messages_per_user > 5:
            return 'medium'
        elif avg_messages_per_user > 2:
            return 'low'
        else:
            return 'very_low'
    
    def _analyze_sentiment_profile(self, messages: List[str]) -> Dict:
        """Analyze sentiment patterns"""
        text = ' '.join(messages).lower()
        
        positive_words = {'love', 'great', 'awesome', 'amazing', 'good', 'excellent', 'fantastic', 'fun', 'enjoy'}
        negative_words = {'hate', 'bad', 'awful', 'terrible', 'broken', 'sucks', 'horrible', 'worst', 'annoying'}
        frustrated_words = {'frustrated', 'angry', 'rage', 'quit', 'unfair', 'cheating', 'stupid', 'dumb'}
        excited_words = {'excited', 'hyped', 'pumped', 'amazing', 'incredible', 'wow', 'omg', 'epic'}
        
        counts = {
            'positive': sum(text.count(word) for word in positive_words),
            'negative': sum(text.count(word) for word in negative_words),
            'frustrated': sum(text.count(word) for word in frustrated_words),
            'excited': sum(text.count(word) for word in excited_words)
        }
        
        total = sum(counts.values())
        if total == 0:
            return {'dominant': 'neutral', 'distribution': counts}
        
        percentages = {k: round(v / total * 100, 1) for k, v in counts.items()}
        dominant = max(percentages.keys(), key=lambda k: percentages[k])
        
        return {'dominant': dominant, 'distribution': percentages}
    
    def _assess_expertise(self, messages: List[str]) -> str:
        """Assess player expertise level"""
        text = ' '.join(messages).lower()
        
        expert_indicators = ['meta', 'optimal', 'strategy', 'build order', 'frame data', 'patch notes', 'tier list', 'guide']
        intermediate_indicators = ['tips', 'help', 'advice', 'learn', 'improve', 'practice', 'tutorial']
        beginner_indicators = ['new', 'beginner', 'noob', 'how to', 'confused', 'dont understand', 'help me']
        
        expert_score = sum(text.count(phrase) for phrase in expert_indicators)
        intermediate_score = sum(text.count(phrase) for phrase in intermediate_indicators)
        beginner_score = sum(text.count(phrase) for phrase in beginner_indicators)
        
        if expert_score > intermediate_score and expert_score > beginner_score:
            return 'expert'
        elif intermediate_score > beginner_score:
            return 'intermediate'
        elif beginner_score > 0:
            return 'beginner'
        else:
            return 'intermediate'  # Default
    
    def _analyze_activity_patterns(self, metadata: List[Dict]) -> Dict:
        """Analyze temporal activity patterns"""
        # Mock implementation - would analyze actual timestamps
        return {
            'most_active_hours': [19, 20, 21],  # 7-9 PM
            'most_active_days': ['saturday', 'sunday'],
            'consistency': 'regular',
            'session_length': 'medium'
        }
    
    def _analyze_communication_style(self, messages: List[str]) -> str:
        """Analyze communication patterns"""
        text = ' '.join(messages).lower()
        total_chars = len(text)
        
        if total_chars == 0:
            return 'silent'
        
        question_count = text.count('?')
        exclamation_count = text.count('!')
        avg_message_length = total_chars / len(messages)
        
        if question_count > len(messages) * 0.3:
            return 'inquisitive'
        elif exclamation_count > len(messages) * 0.2:
            return 'enthusiastic'
        elif avg_message_length > 100:
            return 'detailed'
        elif avg_message_length < 20:
            return 'concise'
        else:
            return 'balanced'
    
    def _calculate_influence_score(self, messages: List[List[str]], metadata: List[Dict]) -> float:
        """Calculate user influence within the community"""
        total_messages = sum(len(user_msgs) for user_msgs in messages)
        unique_topics = len(set(self._extract_primary_topics(
            [msg for user_msgs in messages for msg in user_msgs]
        )))
        
        # Simple influence calculation
        message_factor = min(total_messages / 100.0, 1.0)  # Normalize to 1.0
        topic_diversity = min(unique_topics / 10.0, 1.0)  # Normalize to 1.0
        
        return round((message_factor * 0.6 + topic_diversity * 0.4), 2)
    
    def _predict_churn_risk(self, messages: List[List[str]], metadata: List[Dict], 
                           sentiment_profile: Dict) -> float:
        """Predict likelihood of player churn"""
        
        # Factors that increase churn risk
        negative_sentiment = sentiment_profile['distribution'].get('negative', 0)
        frustrated_sentiment = sentiment_profile['distribution'].get('frustrated', 0)
        
        recent_activity = len([m for m in messages[-5:] if m])  # Last 5 users
        total_activity = len(messages)
        
        # Calculate risk factors
        sentiment_risk = (negative_sentiment + frustrated_sentiment * 1.5) / 100.0
        activity_decline = 1.0 - (recent_activity / max(total_activity * 0.2, 1))
        
        churn_risk = min((sentiment_risk * 0.7 + activity_decline * 0.3), 1.0)
        return round(churn_risk, 2)
    
    def _assess_value_potential(self, engagement_level: str, influence_score: float) -> str:
        """Assess potential value of persona to community"""
        engagement_scores = {
            'very_high': 5, 'high': 4, 'medium': 3, 'low': 2, 'very_low': 1
        }
        
        engagement_value = engagement_scores.get(engagement_level, 3)
        combined_score = engagement_value + influence_score * 5
        
        if combined_score >= 8:
            return 'high'
        elif combined_score >= 6:
            return 'medium'
        else:
            return 'low'
    
    def _extract_primary_topics(self, messages: List[str]) -> List[str]:
        """Extract primary discussion topics"""
        text = ' '.join(messages).lower()
        
        gaming_topics = {
            'balance': ['balance', 'nerf', 'buff', 'op', 'overpowered', 'underpowered'],
            'bugs': ['bug', 'glitch', 'broken', 'error', 'crash'],
            'gameplay': ['gameplay', 'mechanics', 'controls', 'difficulty'],
            'content': ['update', 'patch', 'new content', 'dlc', 'expansion'],
            'community': ['community', 'players', 'toxic', 'friendly', 'team'],
            'competitive': ['rank', 'tournament', 'esports', 'pro scene', 'meta']
        }
        
        topic_scores = {}
        for topic, keywords in gaming_topics.items():
            score = sum(text.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        # Return top 3 topics
        return sorted(topic_scores.keys(), key=lambda k: topic_scores[k], reverse=True)[:3]
    
    def _generate_persona_name(self, play_style: str, engagement: str, expertise: str) -> str:
        """Generate descriptive persona name"""
        style_names = {
            'competitive': 'Competitor',
            'casual': 'Casual Player', 
            'creative': 'Creator',
            'explorer': 'Explorer',
            'social': 'Community Member'
        }
        
        engagement_modifiers = {
            'very_high': 'Hardcore',
            'high': 'Dedicated', 
            'medium': 'Regular',
            'low': 'Occasional',
            'very_low': 'Lurker'
        }
        
        expertise_modifiers = {
            'expert': 'Expert',
            'intermediate': 'Experienced',
            'beginner': 'Novice'
        }
        
        base_name = style_names.get(play_style, 'Player')
        engagement_mod = engagement_modifiers.get(engagement, '')
        expertise_mod = expertise_modifiers.get(expertise, '')
        
        if engagement_mod and expertise_mod:
            return f"{engagement_mod} {expertise_mod} {base_name}"
        elif engagement_mod:
            return f"{engagement_mod} {base_name}"
        elif expertise_mod:
            return f"{expertise_mod} {base_name}"
        else:
            return base_name
    
    def _generate_persona_description(self, play_style: str, engagement: str, sentiment: Dict) -> str:
        """Generate persona description"""
        descriptions = {
            'competitive': "Focused on winning and improving their rank through strategic gameplay",
            'casual': "Plays for fun and relaxation without intense focus on competition", 
            'creative': "Enjoys building, modding, and creating custom content",
            'explorer': "Loves discovering new areas, secrets, and lore within the game",
            'social': "Values community interaction and playing with others"
        }
        
        base_desc = descriptions.get(play_style, "Engages with the game in their own unique way")
        
        engagement_notes = {
            'very_high': "Extremely active in community discussions",
            'high': "Regularly participates in community conversations", 
            'medium': "Moderately active in discussions",
            'low': "Occasionally contributes to conversations",
            'very_low': "Rarely posts but may be actively reading"
        }
        
        engagement_note = engagement_notes.get(engagement, "")
        sentiment_note = f"Generally {sentiment['dominant']} in their communications"
        
        return f"{base_desc}. {engagement_note}. {sentiment_note}."
    
    def _generate_persona_insights(self, personas: List[Dict]) -> List[str]:
        """Generate actionable insights from personas"""
        insights = []
        
        if not personas:
            return ["No distinct personas identified - community may need more diverse engagement"]
        
        # Analyze persona distribution
        total_users = sum(p['size'] for p in personas)
        largest_persona = max(personas, key=lambda x: x['size'])
        
        insights.append(f"Largest persona: {largest_persona['name']} ({largest_persona['size']} users, "
                       f"{round(largest_persona['size']/total_users*100, 1)}% of community)")
        
        # High-value personas
        high_value = [p for p in personas if p['predictive_metrics']['value_potential'] == 'high']
        if high_value:
            insights.append(f"High-value personas identified: {len(high_value)} groups with strong engagement")
        
        # Churn risk analysis
        high_churn = [p for p in personas if p['predictive_metrics']['churn_risk'] > 0.6]
        if high_churn:
            insights.append(f"Churn risk detected: {len(high_churn)} personas show signs of potential disengagement")
        
        # Expertise distribution
        expert_count = sum(1 for p in personas if p['characteristics']['expertise_level'] == 'expert')
        beginner_count = sum(1 for p in personas if p['characteristics']['expertise_level'] == 'beginner')
        
        if expert_count > beginner_count * 2:
            insights.append("Community skews toward experienced players - consider newcomer onboarding")
        elif beginner_count > expert_count * 2:
            insights.append("Large beginner population - opportunity for mentorship programs")
        
        return insights