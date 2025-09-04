# 🚀 Launch Your MVP in 30 Minutes

Stop reading. Start shipping. Here's exactly how to get this live.

## Prerequisites (5 minutes)

```bash
# You need:
- Python 3.8+
- Node.js 16+
- A Discord server you admin
- A Gmail account (for sending emails)
```

## Step 1: Discord Bot Setup (5 minutes)

1. Go to https://discord.com/developers/applications
2. Click "New Application" → Name it "PlayerIntel"
3. Go to "Bot" section → Click "Add Bot"
4. Copy the token (you'll need this)
5. Under "Privileged Gateway Intents" enable:
   - MESSAGE CONTENT INTENT
   - SERVER MEMBERS INTENT
6. Go to OAuth2 → URL Generator:
   - Scopes: Select "bot"
   - Bot Permissions: Select "Read Messages/View Channels" and "Read Message History"
7. Copy the generated URL and open it to add bot to your server

## Step 2: Gmail App Password (3 minutes)

1. Go to https://myaccount.google.com/security
2. Enable 2-factor authentication (if not already)
3. Search for "App passwords"
4. Generate new app password for "Mail"
5. Copy the 16-character password

## Step 3: Backend Setup (10 minutes)

```bash
# Clone or create directory
mkdir player-intelligence
cd player-intelligence

# Create the main.py file (copy from artifact above)
# Create requirements.txt:
cat > requirements.txt << EOF
discord.py==2.3.2
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.24.3
schedule==1.2.0
python-dotenv==1.0.0
EOF

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (this takes 2-3 minutes)
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
DISCORD_TOKEN=your-bot-token-here
DISCORD_CHANNEL_ID=your-channel-id-here
EMAIL_FROM=digest@yourdomain.com
EMAIL_TO=your-email@gmail.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-gmail@gmail.com
SMTP_PASS=your-app-password-here
EOF

# Test run
python main.py
```

## Step 4: Dashboard Setup (7 minutes)

```bash
# In a new terminal
npx create-next-app@latest dashboard --typescript --tailwind --app
cd dashboard

# Replace app/page.tsx with the code from artifact above

# Install additional dependencies
npm install recharts

# Create app/globals.css (if not exists)
cat > app/globals.css << EOF
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

# Run the dashboard
npm run dev

# Open http://localhost:3000
```

## Step 5: Deploy to Production (10 minutes)

### Backend Deployment (Heroku - Free tier works)

```bash
# In the backend directory
heroku create your-app-name
heroku config:set DISCORD_TOKEN=your-token
heroku config:set DISCORD_CHANNEL_ID=your-channel-id
# ... set all other env variables

# Create Procfile
echo "worker: python main.py" > Procfile

# Deploy
git init
git add .
git commit -m "Launch"
git push heroku main

# Scale the worker
heroku ps:scale worker=1
```

### Dashboard Deployment (Vercel - Free)

```bash
# In the dashboard directory
npm install -g vercel
vercel

# Follow prompts, accept all defaults
# Your dashboard is now live at https://your-app.vercel.app
```

## Step 6: Schedule Daily Digest (2 minutes)

Edit `main.py` and uncomment the scheduler section:

```python
# Change this:
asyncio.run(pi.run_daily_digest())

# To this:
schedule.every().day.at("08:00").do(lambda: asyncio.run(pi.run_daily_digest()))
while True:
    schedule.run_pending()
    time.sleep(60)
```

Redeploy to Heroku:
```bash
git add .
git commit -m "Enable daily schedule"
git push heroku main
```

## You're Live! 🎉

Total time: ~30 minutes

What you now have:
- ✅ Discord bot reading messages
- ✅ Daily email digest at 8 AM
- ✅ Web dashboard showing insights
- ✅ Everything deployed and running

## First Customer Checklist

1. **Add to their Discord** (2 minutes)
   - Send them your bot invite link
   - Have them add it to their server
   - Get their channel ID

2. **Configure their digest** (1 minute)
   - Add their email to EMAIL_TO
   - Set their DISCORD_CHANNEL_ID
   - Redeploy

3. **Send test digest** (1 minute)
   ```bash
   heroku run python main.py
   ```

4. **Charge them money** (1 minute)
   - Send Stripe payment link
   - $499/month
   - No free trials

## Quick Fixes for Common Issues

**"Model download is slow"**
- First run downloads 90MB model. Only happens once.

**"No messages found"**
- Check bot has permission to read channel
- Check channel ID is correct
- Try increasing the limit in fetch_messages()

**"Email not sending"**
- Check Gmail app password (not your regular password)
- Check 2FA is enabled on Gmail
- Try smtp.sendgrid.net if Gmail blocks you

**"Dashboard shows mock data"**
- That's intentional for MVP
- Implement /api/digest endpoint when you have 10 customers

## Next Steps (After First Customer Pays)

Week 1 Priority:
```python
# Add to main.py:
- Multiple Discord servers support
- Better clustering (use HDBSCAN)
- Store digests in PostgreSQL
- Add Stripe webhook for payments
```

Week 2 Priority:
```javascript
// Add to dashboard:
- Real data from API
- User authentication (Clerk)
- Customer settings page
- Export to PDF feature
```

Week 3 Priority:
```python
# Add intelligence:
- Persona discovery
- Trend detection
- Webhook for real-time alerts
- Slack integration
```

## The Paul Graham Test

Ask yourself every morning:
1. Did we talk to users yesterday? (You better have)
2. Did we ship code yesterday? (You better have)
3. Are we default alive? (Check your math)

## Stop Reading. Start Shipping.

Your competition is reading documentation.
You're getting customers.

First customer by Friday or you're doing it wrong.

---

**Remember**: This MVP is embarrassing. That's the point. Ship it anyway.

**Remember**: Code quality doesn't matter if no one uses it.

**Remember**: You can refactor after you have revenue.

Now go. Launch. Today.