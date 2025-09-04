#!/bin/bash
# launch.sh - Ship the MVP in one command
# Run this and you're live in 10 minutes

set -e  # Exit on error

echo "🚀 PLAYER INTELLIGENCE MVP LAUNCHER"
echo "===================================="
echo "This script will have you live in 10 minutes."
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install it first."
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Install it first."
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# Setup backend
echo "🔧 Setting up backend..."

# Create project directory
mkdir -p player-intelligence-mvp
cd player-intelligence-mvp

# Create Python files
cat > main.py << 'PYTHON_END'
# [Previous main.py content would go here]
# Copy from the main.py artifact above
PYTHON_END

cat > api.py << 'API_END'
# [Previous api.py content would go here]  
# Copy from the api.py artifact above
API_END

# Create requirements.txt
cat > requirements.txt << 'EOF'
discord.py==2.3.2
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.24.3
schedule==1.2.0
python-dotenv==1.0.0
fastapi==0.104.1
uvicorn==0.24.0
stripe==7.4.0
EOF

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python packages (this takes 2-3 minutes)..."
pip install -r requirements.txt

# Create .env template
cat > .env.template << 'EOF'
# Discord Configuration
DISCORD_TOKEN=your-bot-token-here
DISCORD_CHANNEL_ID=your-channel-id-here

# Email Configuration  
EMAIL_FROM=digest@yourdomain.com
EMAIL_TO=your-email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-gmail@gmail.com
SMTP_PASS=your-app-password-here

# Stripe Configuration (optional for MVP)
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PRICE_ID=price_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Testing
TEST_MODE=true
EOF

echo ""
echo "⚠️  IMPORTANT: Copy .env.template to .env and fill in your values!"
echo ""

# Setup dashboard
echo "🎨 Setting up dashboard..."

npx create-next-app@latest dashboard --typescript --tailwind --app --no-git --yes

cd dashboard

# Update package.json with correct scripts
cat > package.json << 'EOF'
{
  "name": "dashboard",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "14.0.0",
    "react": "^18",
    "react-dom": "^18",
    "typescript": "^5",
    "tailwindcss": "^3",
    "autoprefixer": "^10",
    "postcss": "^8"
  }
}
EOF

npm install

# Create the dashboard page
cat > app/page.tsx << 'TYPESCRIPT_END'
// [Previous dashboard code would go here]
// Copy from the dashboard artifact above
TYPESCRIPT_END

cd ..

# Create Docker Compose for easy running (optional)
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    command: python api.py
    
  worker:
    build: .
    env_file:
      - .env
    command: python main.py
    
  dashboard:
    build: ./dashboard
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
EOF

# Create Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the ML model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

CMD ["python", "api.py"]
EOF

# Create startup script
cat > run.sh << 'EOF'
#!/bin/bash
# Start everything locally

echo "Starting Player Intelligence MVP..."

# Start API server
echo "Starting API server..."
python api.py &
API_PID=$!

# Start dashboard
echo "Starting dashboard..."
cd dashboard && npm run dev &
DASHBOARD_PID=$!

echo ""
echo "🎉 Everything is running!"
echo ""
echo "API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Dashboard: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $API_PID $DASHBOARD_PID" INT
wait
EOF

chmod +x run.sh

# Create quick test script
cat > test_discord.py << 'EOF'
#!/usr/bin/env python3
"""Quick test to verify Discord connection"""

import os
import asyncio
import discord
from dotenv import load_dotenv

load_dotenv()

client = discord.Client(intents=discord.Intents.default())

@client.event
async def on_ready():
    print(f'✅ Connected as {client.user}')
    
    channel_id = int(os.getenv('DISCORD_CHANNEL_ID', '0'))
    channel = client.get_channel(channel_id)
    
    if channel:
        print(f'✅ Found channel: {channel.name}')
        
        # Count recent messages
        count = 0
        async for message in channel.history(limit=100):
            count += 1
        
        print(f'✅ Can read messages: {count} messages found')
    else:
        print(f'❌ Channel {channel_id} not found!')
    
    await client.close()

client.run(os.getenv('DISCORD_TOKEN'))
EOF

chmod +x test_discord.py

# Create deployment script
cat > deploy.sh << 'EOF'
#!/bin/bash
# Deploy to production

echo "🚀 Deploying to production..."

# Deploy API to Heroku
echo "Deploying API to Heroku..."
heroku create player-intel-api-$RANDOM
heroku config:set $(cat .env | grep -v '^#' | xargs)
git init
git add .
git commit -m "Deploy MVP"
git push heroku main

# Deploy dashboard to Vercel
echo "Deploying dashboard to Vercel..."
cd dashboard
vercel --prod

echo "✅ Deployment complete!"
EOF

chmod +x deploy.sh

# Final instructions
cat > README.md << 'EOF'
# Player Intelligence MVP

## Quick Start (5 minutes)

1. **Configure Discord Bot**
   - Copy `.env.template` to `.env`
   - Add your Discord bot token and channel ID
   - Run `python test_discord.py` to verify connection

2. **Start Everything**
   ```bash
   ./run.sh
   ```

3. **Access Services**
   - API: http://localhost:8000
   - Dashboard: http://localhost:3000
   - API Docs: http://localhost:8000/docs

## First Customer Setup

1. Create trial account:
   ```bash
   curl -X POST "http://localhost:8000/api/quick-start?email=customer@example.com"
   ```

2. Generate their first digest:
   ```bash
   curl -X POST "http://localhost:8000/api/generate-digest" \
     -H "Content-Type: application/json" \
     -d '{"customer_email": "customer@example.com"}'
   ```

3. Check the digest was sent:
   - Check email inbox
   - Check dashboard at http://localhost:3000

## Deploy to Production

```bash
./deploy.sh
```

## Daily Operations

- Check metrics: http://localhost:8000/api/stats
- View customers: http://localhost:8000/api/customers  
- Generate digest manually: POST to /api/generate-digest

## Remember

- Ship before ready
- Charge money immediately
- Talk to users daily
- Default alive > default dead

Built with 🚀 by YC mindset
EOF

echo ""
echo "========================================="
echo "✅ MVP SETUP COMPLETE!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.template to .env"
echo "2. Fill in your Discord and email credentials"
echo "3. Run: ./run.sh"
echo "4. Open: http://localhost:3000"
echo ""
echo "To get your first customer:"
echo "1. Share your bot invite link"
echo "2. They add it to their Discord"
echo "3. You add their email with: curl -X POST 'http://localhost:8000/api/quick-start?email=their@email.com'"
echo "4. They get digest tomorrow at 8 AM"
echo "5. They pay you $499/month"
echo ""
echo "🎯 Goal: First paying customer by Friday"
echo ""