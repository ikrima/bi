#!/usr/bin/env python3
"""
Player Intelligence MVP - The Embarrassing Launch
This is the entire backend. 200 lines. Ships today.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
import discord
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import time
from collections import Counter
import re

# Config - Use environment variables in production
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', 'your-bot-token')
DISCORD_CHANNEL_ID = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
EMAIL_FROM = os.getenv('EMAIL_FROM', 'digest@playerintel.ai')
EMAIL_TO = os.getenv('EMAIL_TO', 'gamedev@studio.com')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER', 'your-email@gmail.com')
SMTP_PASS = os.getenv('SMTP_PASS', 'your-app-password')

# Initialize model once (this is the slow part)
print("Loading model... (this takes 30 seconds, only happens once)")
model = SentenceTransformer('all-MiniLM-L6-v2')

class PlayerIntelligence:
    def __init__(self):
        self.messages = []
        self.client = discord.Client(intents=discord.Intents.default())
        
    async def fetch_messages(self, limit=1000):
        """Fetch yesterday's messages from Discord"""
        print(f"Fetching messages from Discord...")
        
        @self.client.event
        async def on_ready():
            channel = self.client.get_channel(DISCORD_CHANNEL_ID)
            if not channel:
                print(f"Channel {DISCORD_CHANNEL_ID} not found!")
                return
            
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            async for message in channel.history(limit=limit, after=yesterday):
                if not message.author.bot:  # Ignore bot messages
                    self.messages.append({
                        'content': message.content,
                        'author': str(message.author),
                        'timestamp': message.created_at.isoformat(),
                        'reactions': len(message.reactions)
                    })
            
            print(f"Fetched {len(self.messages)} messages")
            await self.client.close()
        
        await self.client.start(DISCORD_TOKEN)
    
    def cluster_messages(self, n_clusters=5):
        """Cluster messages into themes"""
        if len(self.messages) < n_clusters:
            return []
        
        print("Clustering messages...")
        
        # Extract just the text
        texts = [m['content'] for m in self.messages]
        
        # Generate embeddings
        embeddings = model.encode(texts)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(texts)))
        labels = kmeans.fit_predict(embeddings)
        
        # Group messages by cluster
        clusters = []
        for i in range(n_clusters):
            cluster_messages = [texts[j] for j, label in enumerate(labels) if label == i]
            if cluster_messages:
                clusters.append(self._summarize_cluster(cluster_messages))
        
        return clusters
    
    def _summarize_cluster(self, messages):
        """Extract theme from a cluster of messages"""
        # Simple word frequency approach
        all_text = ' '.join(messages).lower()
        
        # Remove common words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'an', 'as', 'are', 
                     'was', 'were', 'been', 'be', 'being', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                     'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'we',
                     'they', 'he', 'she', 'it', 'to', 'of', 'in', 'for', 'with', 'my'}
        
        words = re.findall(r'\b[a-z]+\b', all_text)
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Get top words
        word_freq = Counter(words)
        top_words = word_freq.most_common(3)
        
        theme = ' + '.join([word for word, _ in top_words])
        
        # Detect sentiment (super simple)
        positive_words = {'love', 'great', 'awesome', 'amazing', 'good', 'best', 'fun', 'enjoy'}
        negative_words = {'hate', 'bad', 'awful', 'terrible', 'worst', 'broken', 'bug', 'crash'}
        
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        
        if negative_count > positive_count:
            sentiment = "😟 Negative"
        elif positive_count > negative_count:
            sentiment = "😊 Positive"
        else:
            sentiment = "😐 Neutral"
        
        return {
            'theme': theme or 'general discussion',
            'sentiment': sentiment,
            'message_count': len(messages),
            'sample': messages[0][:200] if messages else ''
        }
    
    def find_urgent_issues(self):
        """Find messages that need immediate attention"""
        urgent_keywords = ['crash', 'broken', 'bug', 'cant play', 'wont start', 
                          'lost progress', 'error', 'stuck', 'freeze', 'lag']
        
        urgent = []
        for msg in self.messages:
            content_lower = msg['content'].lower()
            if any(keyword in content_lower for keyword in urgent_keywords):
                urgent.append(msg)
        
        return urgent[:5]  # Top 5 urgent issues
    
    def generate_digest(self):
        """Generate the email digest"""
        if not self.messages:
            return "No messages to analyze today."
        
        clusters = self.cluster_messages()
        urgent = self.find_urgent_issues()
        
        # Calculate overall metrics
        total_messages = len(self.messages)
        unique_authors = len(set(m['author'] for m in self.messages))
        
        # Build email HTML
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #1a1a1a;">🎮 Your Daily Player Intelligence</h1>
            <p style="color: #666;">Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 20px 0;">
                <h2 style="color: #1a1a1a; margin-top: 0;">📊 24-Hour Overview</h2>
                <p><strong>{total_messages}</strong> messages from <strong>{unique_authors}</strong> players</p>
            </div>
            
            <h2 style="color: #1a1a1a;">🎯 Main Topics Players Discussed</h2>
            <ol>
        """
        
        for cluster in clusters:
            html += f"""
                <li style="margin-bottom: 15px;">
                    <strong>{cluster['theme'].title()}</strong> {cluster['sentiment']}<br>
                    <span style="color: #666;">{cluster['message_count']} messages</span><br>
                    <em style="color: #888; font-size: 0.9em;">"{cluster['sample']}..."</em>
                </li>
            """
        
        html += "</ol>"
        
        if urgent:
            html += """
                <div style="background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffc107;">
                    <h2 style="color: #856404; margin-top: 0;">⚠️ Urgent Issues to Address</h2>
                    <ul style="margin: 10px 0;">
            """
            
            for msg in urgent:
                html += f"""
                    <li style="margin-bottom: 10px;">
                        <strong>{msg['author']}</strong>: "{msg['content'][:100]}..."
                    </li>
                """
            
            html += """
                    </ul>
                </div>
            """
        
        html += """
            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                <p style="color: #888; font-size: 0.9em;">
                    This digest saved you ~2 hours of Discord scrolling.<br>
                    <a href="mailto:support@playerintel.ai">Feedback?</a> | 
                    <a href="#">Upgrade to Pro</a>
                </p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def send_email(self, html_content):
        """Send the email digest"""
        print("Sending email digest...")
        
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🎮 Player Intelligence - {datetime.now().strftime('%B %d')}"
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        try:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
            print(f"✅ Digest sent to {EMAIL_TO}")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
    
    async def run_daily_digest(self):
        """The main function that runs everything"""
        print("\n🚀 Starting Player Intelligence Daily Digest...\n")
        
        # Fetch messages
        await self.fetch_messages()
        
        if not self.messages:
            print("No messages found. Skipping digest.")
            return
        
        # Generate and send digest
        digest = self.generate_digest()
        self.send_email(digest)
        
        # Save a local copy for debugging
        with open(f"digest_{datetime.now().strftime('%Y%m%d')}.html", 'w') as f:
            f.write(digest)
        
        print("\n✅ Daily digest complete!\n")


def main():
    """Run once for testing, or schedule for daily execution"""
    pi = PlayerIntelligence()
    
    # For testing - run immediately
    asyncio.run(pi.run_daily_digest())
    
    # For production - schedule daily at 8 AM
    # schedule.every().day.at("08:00").do(lambda: asyncio.run(pi.run_daily_digest()))
    # 
    # print("📅 Scheduled daily digest for 8:00 AM")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)


if __name__ == "__main__":
    main()
