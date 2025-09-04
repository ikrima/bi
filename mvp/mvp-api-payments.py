#!/usr/bin/env python3
"""
API Server - Connect everything together
This replaces the script with a proper API
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import stripe
import os
from datetime import datetime, timedelta
import asyncio
import json

# Import our existing code
from main import PlayerIntelligence

# Initialize FastAPI
app = FastAPI(title="Player Intelligence API")

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stripe setup (get from https://dashboard.stripe.com/test/apikeys)
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_...')
STRIPE_PRICE_ID = os.getenv('STRIPE_PRICE_ID', 'price_...')  # Create in Stripe Dashboard
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET', 'whsec_...')

# Simple in-memory database (replace with PostgreSQL later)
customers = {}
digests = {}

# Initialize our intelligence system
pi = PlayerIntelligence()

# --- Data Models ---

class Customer(BaseModel):
    email: str
    discord_channel_id: int
    stripe_customer_id: Optional[str] = None
    subscription_status: str = "trial"  # trial, active, cancelled
    created_at: datetime = datetime.now()

class DigestRequest(BaseModel):
    customer_email: str
    channel_id: Optional[int] = None

class SubscribeRequest(BaseModel):
    email: str
    payment_method_id: str
    discord_channel_id: int

# --- Routes ---

@app.get("/")
async def root():
    return {
        "name": "Player Intelligence API",
        "version": "0.1.0 (Embarrassing MVP)",
        "status": "Probably working",
        "message": "Ship it anyway"
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "customers": len(customers),
        "digests_today": len([d for d in digests.values() 
                             if d.get('created_at', '') > (datetime.now() - timedelta(days=1)).isoformat()])
    }

@app.post("/api/generate-digest")
async def generate_digest(request: DigestRequest, background_tasks: BackgroundTasks):
    """Generate a digest for a customer (manual trigger)"""
    
    # Check if customer exists
    if request.customer_email not in customers:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    customer = customers[request.customer_email]
    
    # Use provided channel or customer's default
    channel_id = request.channel_id or customer['discord_channel_id']
    
    # Generate digest in background
    background_tasks.add_task(generate_digest_task, customer['email'], channel_id)
    
    return {
        "status": "generating",
        "message": "Digest generation started",
        "email": customer['email']
    }

async def generate_digest_task(email: str, channel_id: int):
    """Background task to generate digest"""
    try:
        # Temporarily set the channel ID
        original_channel = os.getenv('DISCORD_CHANNEL_ID')
        os.environ['DISCORD_CHANNEL_ID'] = str(channel_id)
        
        # Run the digest
        await pi.run_daily_digest()
        
        # Store digest result
        digests[email] = {
            "created_at": datetime.now().isoformat(),
            "status": "sent",
            "channel_id": channel_id
        }
        
        # Restore original
        if original_channel:
            os.environ['DISCORD_CHANNEL_ID'] = original_channel
            
    except Exception as e:
        digests[email] = {
            "created_at": datetime.now().isoformat(),
            "status": "failed",
            "error": str(e)
        }

@app.get("/api/digest/{email}")
async def get_digest(email: str):
    """Get the latest digest for a customer"""
    
    if email not in customers:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    # For MVP, return mock data (replace with real data storage)
    return {
        "email": email,
        "digest": {
            "overview": {
                "totalMessages": 1247,
                "uniquePlayers": 89,
                "sentiment": 72,
                "urgentIssues": 3
            },
            "themes": [
                {"theme": "weapon balance", "sentiment": "Negative", "count": 234},
                {"theme": "new map", "sentiment": "Positive", "count": 189},
                {"theme": "server issues", "sentiment": "Negative", "count": 156}
            ],
            "generated_at": datetime.now().isoformat()
        }
    }

@app.post("/api/subscribe")
async def subscribe(request: SubscribeRequest):
    """Subscribe a new customer (with payment)"""
    
    try:
        # Create Stripe customer
        stripe_customer = stripe.Customer.create(
            email=request.email,
            payment_method=request.payment_method_id,
            invoice_settings={"default_payment_method": request.payment_method_id}
        )
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=stripe_customer.id,
            items=[{"price": STRIPE_PRICE_ID}],
            expand=["latest_invoice.payment_intent"]
        )
        
        # Store customer
        customers[request.email] = {
            "email": request.email,
            "discord_channel_id": request.discord_channel_id,
            "stripe_customer_id": stripe_customer.id,
            "stripe_subscription_id": subscription.id,
            "subscription_status": subscription.status,
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "status": "success",
            "subscription_id": subscription.id,
            "customer_id": stripe_customer.id,
            "message": "Welcome to Player Intelligence!"
        }
        
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/stripe-webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks (payment confirmations, cancellations, etc.)"""
    
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        # Activate customer subscription
        email = session.get('customer_email')
        if email and email in customers:
            customers[email]['subscription_status'] = 'active'
            
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        # Cancel customer access
        for email, customer in customers.items():
            if customer.get('stripe_subscription_id') == subscription.id:
                customer['subscription_status'] = 'cancelled'
                break
    
    return {"status": "success"}

@app.get("/api/customers")
async def list_customers():
    """List all customers (for admin dashboard)"""
    return {
        "customers": list(customers.values()),
        "total": len(customers),
        "active": len([c for c in customers.values() if c.get('subscription_status') == 'active'])
    }

@app.post("/api/quick-start")
async def quick_start(email: str):
    """Quick start for testing - creates a trial customer"""
    
    if email in customers:
        return {"message": "Customer already exists", "customer": customers[email]}
    
    customers[email] = {
        "email": email,
        "discord_channel_id": int(os.getenv('DISCORD_CHANNEL_ID', '0')),
        "subscription_status": "trial",
        "created_at": datetime.now().isoformat()
    }
    
    return {
        "message": "Trial started! You have 7 days free.",
        "customer": customers[email],
        "next_steps": [
            "1. Add our bot to your Discord",
            "2. Send us your channel ID", 
            "3. Get your first digest tomorrow at 8 AM",
            "4. Upgrade to keep receiving digests after 7 days"
        ]
    }

@app.get("/api/stats")
async def get_stats():
    """Get system stats for the dashboard"""
    
    # Calculate real stats
    total_customers = len(customers)
    active_customers = len([c for c in customers.values() if c.get('subscription_status') == 'active'])
    trial_customers = len([c for c in customers.values() if c.get('subscription_status') == 'trial'])
    mrr = active_customers * 499  # $499/month per customer
    
    # Growth rate (mock for now)
    growth_rate = 50 if total_customers > 0 else 0
    
    return {
        "metrics": {
            "total_customers": total_customers,
            "active_customers": active_customers,
            "trial_customers": trial_customers,
            "mrr": mrr,
            "growth_rate": growth_rate,
            "digests_sent_today": len([d for d in digests.values() 
                                      if d.get('created_at', '') > (datetime.now() - timedelta(days=1)).isoformat()])
        },
        "chart_data": {
            "daily_signups": [
                {"day": "Mon", "signups": 2},
                {"day": "Tue", "signups": 3},
                {"day": "Wed", "signups": 5},
                {"day": "Thu", "signups": 4},
                {"day": "Fri", "signups": 7},
                {"day": "Sat", "signups": 6},
                {"day": "Sun", "signups": 8}
            ]
        }
    }

# --- Scheduled Tasks ---

async def run_daily_digests():
    """Run all customer digests (call this from a scheduler)"""
    
    for email, customer in customers.items():
        if customer.get('subscription_status') in ['active', 'trial']:
            try:
                await generate_digest_task(email, customer['discord_channel_id'])
                print(f"✅ Digest sent to {email}")
            except Exception as e:
                print(f"❌ Failed to send digest to {email}: {e}")

# --- Startup Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    print("""
    🚀 Player Intelligence API Starting...
    
    Remember:
    - Ship it before it's ready
    - Charge money immediately  
    - Talk to users every day
    - Default alive > default dead
    
    API running at http://localhost:8000
    Docs at http://localhost:8000/docs
    """)
    
    # Load any persisted data (implement later)
    # For now, create a test customer
    if os.getenv('TEST_MODE'):
        customers["test@example.com"] = {
            "email": "test@example.com",
            "discord_channel_id": int(os.getenv('DISCORD_CHANNEL_ID', '0')),
            "subscription_status": "trial",
            "created_at": datetime.now().isoformat()
        }

# --- Run the server ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)