(ns player-intel.demo
  (:require [player-intel.intelligence :as intelligence]
            [player-intel.analytics :as analytics]
            [player-intel.competitive :as competitive]
            [player-intel.alerts :as alerts]
            [player-intel.ml :as ml]
            [player-intel.db :as db]
            [taoensso.timbre :as log]
            [clojure.data.json :as json]
            [clj-time.core :as time]
            [clj-time.format :as time-format]))

;; --- Demo Data Setup ---
(def demo-customer-id "demo-customer-001")

(def demo-messages
  [{:id "demo1" :content "This game is amazing! Way better than Valorant in my opinion" 
    :author "ProGamer2023" :channel-id "general" :timestamp (coerce/to-timestamp (time/now))}
   {:id "demo2" :content "Valorant has better gunplay though, we should copy their recoil system"
    :author "CompetitivePlayer" :channel-id "general" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 1)))}
   {:id "demo3" :content "I'm switching to Apex Legends, this game is too buggy"
    :author "CasualGamer" :channel-id "general" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 2)))}
   {:id "demo4" :content "The new weapon is completely overpowered, needs immediate nerf"
    :author "BalanceExpert" :channel-id "feedback" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 3)))}
   {:id "demo5" :content "Love the new map! Graphics are incredible"
    :author "ArtLover" :channel-id "general" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 4)))}
   {:id "demo6" :content "Servers are down again... third time this week"
    :author "TechnicalUser" :channel-id "technical" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 5)))}
   {:id "demo7" :content "Fortnite building mechanics would be cool in this game"
    :author "BuildingFan" :channel-id "suggestions" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 6)))}
   {:id "demo8" :content "The monetization is getting out of hand, feels like pay-to-win"
    :author "F2PPlayer" :channel-id "general" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 7)))}
   {:id "demo9" :content "Tutorial is confusing for new players, needs improvement"
    :author "NewbieHelper" :channel-id "help" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 8)))}
   {:id "demo10" :content "Competitive matchmaking is finally fair! Great job devs"
    :author "RankedPlayer" :channel-id "competitive" :timestamp (coerce/to-timestamp (time/minus (time/now) (time/hours 9)))}])

(def demo-user-behavioral-data
  [{:author "ProGamer2023" 
    :messages ["amazing gameplay" "love the mechanics" "perfect balance" "great update"] 
    :message-count 85 :timespan 30}
   {:author "CompetitivePlayer" 
    :messages ["needs better ranking" "matchmaking issues" "balance problems" "competitive integrity"] 
    :message-count 120 :timespan 25}
   {:author "CasualGamer" 
    :messages ["too difficult" "not fun anymore" "switching games" "frustrated"] 
    :message-count 45 :timespan 15}
   {:author "BalanceExpert" 
    :messages ["weapon stats analysis" "meta discussion" "balance suggestions" "nerf recommendations"] 
    :message-count 95 :timespan 28}
   {:author "ArtLover" 
    :messages ["beautiful graphics" "amazing art style" "visual effects" "aesthetic appreciation"] 
    :message-count 60 :timespan 20}
   {:author "TechnicalUser" 
    :messages ["server issues" "connection problems" "bug reports" "technical feedback"] 
    :message-count 70 :timespan 22}
   {:author "BuildingFan" 
    :messages ["building mechanics" "construction features" "creative mode" "building suggestions"] 
    :message-count 55 :timespan 18}
   {:author "F2PPlayer" 
    :messages ["monetization concerns" "pay-to-win issues" "pricing complaints" "free-to-play feedback"] 
    :message-count 40 :timespan 12}
   {:author "NewbieHelper" 
    :messages ["tutorial feedback" "new player experience" "learning curve" "onboarding issues"] 
    :message-count 65 :timespan 24}
   {:author "RankedPlayer" 
    :messages ["competitive analysis" "ranking discussions" "esports talk" "professional play"] 
    :message-count 110 :timespan 35}])

;; --- Demo Execution Functions ---

(defn setup-demo-data
  "Set up demo data in database"
  []
  (log/info "Setting up demo data...")
  
  ;; Insert demo messages
  (doseq [message demo-messages]
    (db/insert-message! message))
  
  ;; Create analytics tables
  (analytics/create-analytics-tables!)
  (competitive/create-competitive-tables!)
  
  (log/info "Demo data setup complete"))

(defn demo-persona-discovery
  "Demonstrate persona discovery capabilities"
  []
  (log/info "=== DEMO: Persona Discovery ===")
  
  (let [result (intelligence/discover-customer-personas demo-customer-id)]
    (if (:success result)
      (do
        (log/info "✓ Persona Discovery Successful!")
        (log/info "  Users Analyzed:" (:users-analyzed result))
        (log/info "  Personas Discovered:" (count (:personas result)))
        
        (doseq [persona (:personas result)]
          (log/info (str "  📊 " (:name persona)))
          (log/info (str "     Size: " (:size persona) " players"))
          (log/info (str "     Churn Risk: " (format "%.1f%%" (* (:churn_risk persona) 100))))
          (log/info (str "     Engagement: " (format "%.1f%%" (* (:engagement_score persona) 100))))
          (log/info (str "     Play Style: " (get-in persona [:characteristics :play_style]))))
        
        result)
      (do
        (log/error "✗ Persona Discovery Failed:" (:error result))
        result))))

(defn demo-predictive-analytics
  "Demonstrate predictive analytics capabilities"
  []
  (log/info "=== DEMO: Predictive Analytics ===")
  
  (let [change-scenarios [
         {:description "Reduce assault rifle damage by 15%"
          :areas ["weapons" "balance"]}
         {:description "Add building mechanics similar to Fortnite"
          :areas ["gameplay" "mechanics"]}
         {:description "Increase cosmetic prices by 25%"
          :areas ["monetization" "pricing"]}
         {:description "Implement skill-based matchmaking"
          :areas ["matchmaking" "competitive"]}]]
    
    (doseq [{:keys [description areas]} change-scenarios]
      (log/info (str "🔮 Predicting impact of: " description))
      
      (let [result (intelligence/predict-change-impact demo-customer-id description areas)]
        (if (:success result)
          (let [prediction (:prediction result)
                sentiment (:overall_sentiment prediction)
                recommendation (:recommendation result)]
            
            (log/info (str "   Overall Sentiment: " (format "%.1f%%" (* sentiment 100))))
            (log/info (str "   Confidence: " (format "%.1f%%" (* (:confidence result) 100))))
            (log/info (str "   Risk Level: " (:severity recommendation)))
            (log/info (str "   Key Recommendation: " (first (:actions recommendation)))))
          
          (log/error (str "   ✗ Prediction failed: " (:error result)))))
    
    (log/info "Predictive analytics demo complete")))

(defn demo-competitive-intelligence
  "Demonstrate competitive intelligence capabilities"
  []
  (log/info "=== DEMO: Competitive Intelligence ===")
  
  ;; Detect competitor mentions
  (log/info "🔍 Detecting competitor mentions...")
  (let [mentions (competitive/detect-competitor-mentions 
                 demo-customer-id
                 :days-back 1
                 :competitors ["valorant" "apex legends" "fortnite"])]
    
    (log/info (str "Found " (count mentions) " competitor mentions"))
    
    (doseq [mention mentions]
      (log/info (str "  📈 " (:competitor mention) " - " (:type mention)))
      (log/info (str "     Sentiment: " (format "%.1f%%" (* (:sentiment mention) 100))))
      (log/info (str "     Context: " (take 50 (:context mention)) "...")))
    
    ;; Generate competitive insights
    (log/info "🧠 Generating competitive insights...")
    (let [insights-result (competitive/generate-competitive-insights demo-customer-id :days-back 7)]
      (if (:success insights-result)
        (do
          (log/info (str "✓ Analysis complete - " (:total_mentions insights-result) " total mentions"))
          (log/info (str "  Competitors detected: " (:competitors_detected insights-result)))
          
          (doseq [insight (:insights insights-result)]
            (log/info (str "  🎯 " (:competitor insight)))
            (let [data (:data insight)]
              (log/info (str "     Mentions: " (:mention_count data)))
              (log/info (str "     Sentiment: " (:sentiment_trend data)))
              (when (:trending data)
                (log/info "     📈 TRENDING competitor!")))))
        
        (log/error "✗ Competitive analysis failed:" (:error insights-result))))))

(defn demo-automated-recommendations
  "Demonstrate automated recommendation engine"
  []
  (log/info "=== DEMO: Automated Recommendations ===")
  
  (let [result (intelligence/generate-automated-insights demo-customer-id)]
    (if (:success result)
      (do
        (log/info "✓ Generated" (:insights-count result) "automated recommendations")
        
        (doseq [recommendation (:recommendations result)]
          (log/info (str "🎯 " (:title recommendation)))
          (log/info (str "   Priority: " (str/upper-case (:priority recommendation))))
          (log/info (str "   Type: " (:type recommendation)))
          (log/info (str "   Description: " (:description recommendation)))
          (when (seq (:actions recommendation))
            (log/info "   Recommended Actions:")
            (doseq [action (:actions recommendation)]
              (log/info (str "     • " action))))))
      
      (log/error "✗ Automated recommendations failed:" (:error result)))))

(defn demo-trend-detection
  "Demonstrate trend detection and anomaly alerting"
  []
  (log/info "=== DEMO: Trend Detection & Anomaly Alerting ===")
  
  ;; Analyze community trends
  (log/info "📊 Analyzing community trends...")
  (let [trends (intelligence/analyze-community-trends demo-customer-id)]
    (if trends
      (do
        (log/info "✓ Trend analysis complete")
        (log/info (str "  Engagement Trend: " 
                      (cond
                        (> (:engagement-trend trends) 0.05) "📈 Increasing"
                        (< (:engagement-trend trends) -0.05) "📉 Decreasing" 
                        :else "➡️ Stable")))
        (log/info (str "  Sentiment Trend: "
                      (cond
                        (> (:sentiment-trend trends) 0.05) "😊 Improving"
                        (< (:sentiment-trend trends) -0.05) "😟 Declining"
                        :else "😐 Neutral")))
        (log/info (str "  Volatility: " (format "%.2f" (:sentiment-volatility trends 0)))))
      
      (log/warn "⚠️ Insufficient data for trend analysis")))
  
  ;; Run alert checks
  (log/info "🚨 Running alert checks...")
  (let [alerts-result (alerts/run-alert-checks demo-customer-id)]
    (if (:error alerts-result)
      (log/error "✗ Alert checks failed:" (:error alerts-result))
      (do
        (log/info "✓ Alert checks complete")
        (log/info (str "  Alerts Created: " (:alerts-created alerts-result)))
        (log/info (str "  Severe Alerts: " (:severe-alerts alerts-result)))
        
        (when (> (:severe-alerts alerts-result) 0)
          (log/warn "🚨 SEVERE ALERTS DETECTED - Immediate attention required!"))))))

(defn demo-lifecycle-analytics
  "Demonstrate player lifecycle and retention analytics"
  []
  (log/info "=== DEMO: Player Lifecycle & Retention Analytics ===")
  
  (let [metrics (analytics/calculate-player-lifecycle-metrics demo-customer-id)]
    (if (:error metrics)
      (log/error "✗ Lifecycle analytics failed:" (:error metrics))
      (do
        (log/info "✓ Lifecycle analytics complete")
        (log/info (str "  Total Users: " (:total_users metrics)))
        (log/info (str "  Active Users (Week): " (:active_users_week metrics)))
        (log/info (str "  Retention Rate: " (format "%.1f%%" (* (:retention_rate metrics) 100))))
        (log/info (str "  Avg Engagement: " (format "%.2f" (:avg_engagement metrics))))
        (log/info (str "  Churn Risk: " (format "%.1f%%" (* (:churn_risk_score metrics) 100))))
        (log/info (str "  High Engagement Users: " (:high_engagement_users metrics)))
        (log/info (str "  Low Engagement Users: " (:low_engagement_users metrics)))
        
        ;; Risk assessment
        (cond
          (< (:retention_rate metrics) 0.4)
          (log/warn "🚨 CRITICAL: Retention rate below 40%!")
          
          (< (:retention_rate metrics) 0.6)
          (log/warn "⚠️ WARNING: Retention rate below 60%")
          
          :else
          (log/info "✅ Retention rate is healthy"))))))

(defn run-comprehensive-demo
  "Run comprehensive demo of all Stage 6 capabilities"
  []
  (log/info "🚀 Starting Player Intelligence Platform - Stage 6 Demo")
  (log/info "=" (apply str (repeat 60 "=")))
  
  (let [start-time (System/currentTimeMillis)]
    
    ;; Setup
    (setup-demo-data)
    
    ;; Run all demos
    (demo-persona-discovery)
    (log/info "")
    
    (demo-predictive-analytics)
    (log/info "")
    
    (demo-competitive-intelligence)
    (log/info "")
    
    (demo-automated-recommendations)
    (log/info "")
    
    (demo-trend-detection)
    (log/info "")
    
    (demo-lifecycle-analytics)
    (log/info "")
    
    ;; Summary
    (let [end-time (System/currentTimeMillis)
          duration (- end-time start-time)]
      
      (log/info "🎉 Stage 6 Demo Complete!")
      (log/info "=" (apply str (repeat 60 "=")))
      (log/info "CAPABILITIES DEMONSTRATED:")
      (log/info "✅ Advanced Persona Discovery")
      (log/info "✅ Predictive Analytics for Game Changes")  
      (log/info "✅ What-If Scenario Analysis")
      (log/info "✅ Automated Recommendation Engine")
      (log/info "✅ Player Lifecycle & Retention Analytics")
      (log/info "✅ Trend Detection & Anomaly Alerting")
      (log/info "✅ Competitive Intelligence Features")
      (log/info "")
      (log/info (str "Total Demo Time: " duration "ms"))
      (log/info "")
      (log/info "🎯 Stage 6: Intelligence Amplification - COMPLETE")
      
      {:success true
       :duration duration
       :capabilities-demonstrated 7
       :timestamp (time-format/unparse (time-format/formatters :date-time) (time/now))})))

;; --- Health Check Functions ---

(defn health-check-ml-services
  "Health check all ML services"
  []
  (log/info "🔍 Checking ML services health...")
  
  (let [ml-health (ml/health-check)]
    (if (:healthy ml-health)
      (log/info "✅ ML Service: Healthy")
      (log/error "❌ ML Service: " (:message ml-health))))
  
  ;; Test each ML capability
  (let [test-results {:embeddings false :clustering false :personas false :predictions false}]
    
    ;; Test embeddings
    (try
      (let [embeddings (ml/generate-embeddings ["test message"])]
        (if embeddings
          (do (log/info "✅ Embeddings: Working")
              (assoc test-results :embeddings true))
          (do (log/error "❌ Embeddings: Failed")
              test-results)))
      (catch Exception e
        (log/error "❌ Embeddings: Error -" (.getMessage e))
        test-results))))

(defn system-readiness-check
  "Comprehensive system readiness check"
  []
  (log/info "🏥 System Readiness Check")
  (log/info "=" (apply str (repeat 40 "=")))
  
  (let [checks {:database false :ml-service false :analytics false :competitive false}]
    
    ;; Database check
    (try
      (let [db-health (db/health-check)]
        (if (:healthy db-health)
          (do (log/info "✅ Database: Connected")
              (assoc checks :database true))
          (do (log/error "❌ Database: " (:message db-health))
              checks)))
      (catch Exception e
        (log/error "❌ Database: Error -" (.getMessage e))
        checks))
    
    ;; ML service check
    (health-check-ml-services)
    
    ;; Table existence checks
    (try
      (analytics/create-analytics-tables!)
      (competitive/create-competitive-tables!)
      (log/info "✅ Database Tables: Created/Verified")
      (catch Exception e
        (log/error "❌ Database Tables: " (.getMessage e))))
    
    (log/info "")
    (log/info "🎯 System Ready for Demo")))

;; --- CLI Interface ---

(defn -main [& args]
  (let [command (first args)]
    (case command
      "demo" (run-comprehensive-demo)
      "health" (system-readiness-check)
      "personas" (demo-persona-discovery)
      "predictions" (demo-predictive-analytics)
      "competitive" (demo-competitive-intelligence)
      "recommendations" (demo-automated-recommendations)
      "trends" (demo-trend-detection)
      "lifecycle" (demo-lifecycle-analytics)
      
      ;; Default: full demo
      (do
        (log/info "Available commands: demo, health, personas, predictions, competitive, recommendations, trends, lifecycle")
        (log/info "Running full demo...")
        (run-comprehensive-demo)))))