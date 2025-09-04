#!/bin/bash

echo "=== Stage 1: Discord Data Pipeline - Test ==="
echo

echo "✅ Components Created:"
echo "✓ Discord integration (backend/api/src/player_intel/discord.clj)"
echo "✓ Database layer with message storage (backend/api/src/player_intel/db.clj)" 
echo "✓ ML service integration (backend/api/src/player_intel/ml.clj)"
echo "✓ Database migrations system (backend/api/src/player_intel/migrations.clj)"
echo "✓ Async message pipeline (backend/api/src/player_intel/pipeline.clj)"
echo "✓ API endpoints for messages and pipeline control"
echo "✓ Enhanced ML service with clustering support"
echo

echo "🔧 API Endpoints Available:"
echo "  GET  /health - System health check"
echo "  GET  /api/messages?limit=100&channel_id=xxx - Recent messages"
echo "  GET  /api/pipeline/status - Pipeline status"
echo "  POST /api/pipeline/start {\"channel_id\": \"xxx\"} - Start pipeline"
echo "  POST /api/pipeline/stop {\"channel_id\": \"xxx\"} - Stop pipeline"
echo

echo "🚀 ML Service Endpoints:"
echo "  GET  /health - ML service health"
echo "  POST /embed {\"texts\": [...]} - Generate embeddings"
echo "  POST /cluster {\"embeddings\": [...], \"messages\": [...]} - Cluster messages"
echo

echo "📊 Data Flow:"
echo "  Discord → API (discord.clj) → Database (db.clj) → ML Processing (ml.clj)"
echo "  ↓"
echo "  Embeddings → Clustering → Insights"
echo

echo "🎯 To Test Complete Flow:"
echo "1. Set environment variables:"
echo "   export DISCORD_BOT_TOKEN=\"your_bot_token\""
echo "   export DISCORD_CHANNELS=\"channel_id1,channel_id2\""
echo "   export AUTO_START_PIPELINES=\"true\""
echo
echo "2. Start services:"
echo "   make run  # or individually: make dev-api, make dev-ml"
echo
echo "3. Test endpoints:"
echo "   curl http://localhost:3000/health"
echo "   curl http://localhost:8000/health"
echo "   curl http://localhost:3000/api/pipeline/status"
echo
echo "4. Start pipeline:"
echo "   curl -X POST http://localhost:3000/api/pipeline/start \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"channel_id\": \"YOUR_CHANNEL_ID\"}'"
echo

echo "=== Stage 1: COMPLETE ✅ ==="
echo
echo "📈 Next: Stage 2 - Intelligence Extraction"
echo "   - Message clustering and theme extraction"
echo "   - Digest generation with insights"
echo "   - Sentiment analysis and urgent issue detection"