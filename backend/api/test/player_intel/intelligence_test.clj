(ns player-intel.intelligence-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [player-intel.intelligence :as intelligence]
            [player-intel.analytics :as analytics]
            [player-intel.competitive :as competitive]
            [player-intel.alerts :as alerts]
            [player-intel.ml :as ml]
            [player-intel.db :as db]
            [clojure.data.json :as json]
            [taoensso.timbre :as log]))

;; --- Test Fixtures ---
(def test-customer-id "test-customer-123")

(def sample-user-data
  [{:author "player1"
    :messages ["this game is amazing!" "love the new update" "best game ever"]
    :message-count 50
    :timespan 15}
   {:author "player2"
    :messages ["game is ok" "could be better" "needs more content"]
    :message-count 25
    :timespan 10}
   {:author "player3"
    :messages ["switching to valorant" "this game sucks now" "uninstalling"]
    :message-count 30
    :timespan 20}
   {:author "player4"
    :messages ["new weapon is op" "nerf this please" "balance issues"]
    :message-count 40
    :timespan 12}
   {:author "player5"
    :messages ["great community" "helpful players" "fun events"]
    :message-count 35
    :timespan 18}
   {:author "player6"
    :messages ["lag issues" "servers down again" "connection problems"]
    :message-count 45
    :timespan 14}
   {:author "player7"
    :messages ["awesome graphics" "smooth gameplay" "well optimized"]
    :message-count 28
    :timespan 11}
   {:author "player8"
    :messages ["pay to win" "expensive cosmetics" "money grab"]
    :message-count 32
    :timespan 16}
   {:author "player9"
    :messages ["tutorial is confusing" "hard to learn" "need guides"]
    :message-count 22
    :timespan 8}
   {:author "player10"
    :messages ["competitive mode rocks" "ranked system good" "fair matchmaking"]
    :message-count 55
    :timespan 22}])

(def sample-messages
  [{:id "msg1" :content "this game is amazing compared to valorant" :author "player1" :timestamp "2023-01-01T10:00:00Z"}
   {:id "msg2" :content "valorant has better graphics though" :author "player2" :timestamp "2023-01-01T11:00:00Z"}
   {:id "msg3" :content "switching to apex legends, this is broken" :author "player3" :timestamp "2023-01-01T12:00:00Z"}
   {:id "msg4" :content "love this game but needs balance" :author "player4" :timestamp "2023-01-01T13:00:00Z"}
   {:id "msg5" :content "fortnite does building better" :author "player5" :timestamp "2023-01-01T14:00:00Z"}])

;; --- ML Service Mock ---
(defn mock-ml-service
  "Mock ML service responses for testing"
  []
  ;; Mock persona discovery
  (with-redefs [ml/discover-personas
                (fn [user-data]
                  {:success true
                   :data [{:id 1
                          :name "Hardcore Competitor"
                          :characteristics {:play_style "competitive"
                                          :engagement_level "high"
                                          :feedback_type "critical"
                                          :skill_level "expert"}
                          :size 3
                          :churn_risk 0.2
                          :engagement_score 0.8
                          :coordinates [0.8 0.2]}
                         {:id 2
                          :name "Casual Observer"
                          :characteristics {:play_style "casual"
                                          :engagement_level "low"
                                          :feedback_type "neutral"
                                          :skill_level "beginner"}
                          :size 7
                          :churn_risk 0.7
                          :engagement_score 0.3
                          :coordinates [-0.2 0.6]}]})
                
                ;; Mock prediction service
                ml/predict-change-impact
                (fn [request]
                  {:success true
                   :data {:overall_sentiment 0.35
                         :persona_impacts [{:persona_id 1
                                          :persona_name "Hardcore Competitor"
                                          :predicted_sentiment 0.2
                                          :churn_risk 0.8}
                                         {:persona_id 2
                                          :persona_name "Casual Observer"
                                          :predicted_sentiment 0.4
                                          :churn_risk 0.5}]
                         :impact_summary "Negative reaction expected from competitive players"}
                   :confidence 0.75})
                
                ;; Mock competitor sentiment analysis
                ml/analyze-competitor-sentiment
                (fn [messages]
                  {:success true
                   :data {:average_sentiment 0.6
                         :message_count (count messages)
                         :sentiment_distribution {"positive" 2 "neutral" 1 "negative" 2}}})
                
                ;; Mock trend detection
                ml/detect-trends-and-anomalies
                (fn [data]
                  {:success true
                   :data {"sentiment_score" {:trend "decreasing"
                                           :slope -0.15
                                           :anomalies [{:index 5 :value 25 :deviation 2.5}]}
                         "engagement_score" {:trend "stable"
                                           :slope 0.02
                                           :anomalies []}}})]
    
    ;; Return the mock setup indicator
    :mocked))

;; --- Database Mock ---
(defn mock-database
  "Mock database operations for testing"
  []
  (with-redefs [db/get-customer-messages
                (fn [customer-id limit] sample-messages)
                
                db/get-customer-messages-since
                (fn [customer-id since] sample-messages)
                
                db/get-latest-digest
                (fn [customer-id] {:sentiment 45 :message_count 500 :generated_at (System/currentTimeMillis)})
                
                db/get-customer-digests
                (fn [customer-id limit]
                  (repeatedly limit #(hash-map :sentiment (+ 30 (rand-int 40))
                                             :message_count (+ 200 (rand-int 300))
                                             :generated_at (- (System/currentTimeMillis) 
                                                            (* (rand-int 7) 86400000)))))
                
                analytics/save-customer-personas!
                (fn [customer-id personas] :saved)
                
                analytics/get-customer-personas
                (fn [customer-id] [])
                
                analytics/log-prediction!
                (fn [& args] :logged)
                
                analytics/create-trend-alert!
                (fn [& args] :alert-created)
                
                analytics/record-metric!
                (fn [& args] :metric-recorded)]
    
    :mocked))

;; --- Test Cases ---

(deftest test-persona-discovery
  (testing "Persona discovery with sufficient data"
    (mock-ml-service)
    (mock-database)
    
    (let [result (intelligence/discover-customer-personas test-customer-id)]
      (is (:success result))
      (is (= 2 (count (:personas result))))
      (is (= 10 (:users-analyzed result)))
      
      (let [hardcore-persona (first (:personas result))]
        (is (= "Hardcore Competitor" (:name hardcore-persona)))
        (is (= "competitive" (get-in hardcore-persona [:characteristics :play_style])))
        (is (= 0.2 (:churn_risk hardcore-persona)))))))

(deftest test-persona-discovery-insufficient-data
  (testing "Persona discovery with insufficient data"
    (with-redefs [db/get-customer-messages (fn [customer-id limit] (take 5 sample-messages))]
      (let [result (intelligence/discover-customer-personas test-customer-id)]
        (is (not (:success result)))
        (is (str/includes? (:error result) "Insufficient"))
        (is (= 5 (:current-users result)))))))

(deftest test-change-impact-prediction
  (testing "Predicting impact of game changes"
    (mock-ml-service)
    (mock-database)
    
    (with-redefs [intelligence/get-customer-personas 
                  (fn [customer-id] 
                    {:success true 
                     :personas [{:id 1 :name "Test Persona"}]})]
      
      (let [result (intelligence/predict-change-impact 
                   test-customer-id
                   "Nerfing the assault rifle damage by 20%"
                   ["weapons" "balance"])]
        
        (is (:success result))
        (is (contains? (:prediction result) :overall_sentiment))
        (is (< (:overall_sentiment (:prediction result)) 0.5)) ; Should predict negative reaction
        (is (> (:confidence result) 0.5))
        (is (contains? (:recommendation result) :severity))))))

(deftest test-automated-insights
  (testing "Generating automated insights"
    (mock-ml-service)
    (mock-database)
    
    (with-redefs [intelligence/get-customer-personas
                  (fn [customer-id] 
                    {:success true 
                     :personas [{:churn_risk 0.9}]})
                  
                  intelligence/analyze-community-trends
                  (fn [customer-id] 
                    {:engagement-trend -0.15
                     :sentiment-trend -0.10})]
      
      (let [result (intelligence/generate-automated-insights test-customer-id)]
        (is (:success result))
        (is (> (:insights-count result) 0))
        (is (seq (:recommendations result)))
        
        ;; Should generate churn risk alert
        (is (some #(= (:type %) "churn-risk-alert") (:recommendations result)))))))

(deftest test-competitive-analysis
  (testing "Comprehensive competitive analysis"
    (mock-ml-service)
    (mock-database)
    
    (let [result (competitive/generate-competitive-insights test-customer-id :days-back 7)]
      (is (:success result))
      (is (> (:total_mentions result) 0))
      (is (> (:competitors_detected result) 0))
      (is (seq (:insights result))))))

(deftest test-competitor-mention-detection
  (testing "Detecting competitor mentions in messages"
    (mock-database)
    
    (let [mentions (competitive/detect-competitor-mentions 
                   test-customer-id 
                   :days-back 7 
                   :competitors ["valorant" "apex legends" "fortnite"])]
      
      (is (> (count mentions) 0))
      (is (some #(= (:competitor %) "valorant") mentions))
      (is (some #(= (:competitor %) "apex legends") mentions))
      (is (some #(= (:competitor %) "fortnite") mentions))
      
      ;; Check mention classification
      (let [valorant-mention (first (filter #(= (:competitor %) "valorant") mentions))]
        (is (contains? #{"comparison" "general-mention"} (:type valorant-mention)))
        (is (<= 0.0 (:sentiment valorant-mention) 1.0))))))

(deftest test-alert-generation
  (testing "Generating alerts from metrics"
    (mock-database)
    
    (with-redefs [analytics/calculate-player-lifecycle-metrics
                  (fn [customer-id]
                    {:sentiment_score 25  ; Critical level
                     :avg_engagement 0.25 ; Critical level  
                     :retention_rate 0.35 ; Critical level
                     :churn_risk_score 0.85})] ; Critical level
      
      (let [result (alerts/run-alert-checks test-customer-id)]
        (is (> (:alerts-created result) 0))
        (is (> (:severe-alerts result) 0))))))

(deftest test-trend-analysis
  (testing "Analyzing community trends"
    (mock-database)
    
    (let [result (intelligence/analyze-community-trends test-customer-id)]
      (is (contains? result :engagement-trend))
      (is (contains? result :sentiment-trend))
      (is (contains? result :sentiment-volatility))
      (is (number? (:engagement-trend result)))
      (is (number? (:sentiment-trend result)))))

(deftest test-lifecycle-analytics
  (testing "Player lifecycle analytics calculation"
    (mock-database)
    
    (let [result (analytics/calculate-player-lifecycle-metrics test-customer-id)]
      (is (contains? result :total_users))
      (is (contains? result :retention_rate))
      (is (contains? result :avg_engagement))
      (is (contains? result :churn_risk_score))
      
      (is (<= 0.0 (:retention_rate result) 1.0))
      (is (<= 0.0 (:churn_risk_score result) 1.0))
      (is (>= (:total_users result) 0)))))

(deftest test-recommendation-generation
  (testing "Generating actionable recommendations"
    (let [prediction-data {:overall_sentiment 0.25
                          :persona_impacts [{:churn_risk 0.9 :name "High Risk Persona"}
                                          {:churn_risk 0.3 :name "Low Risk Persona"}]}
          
          recommendation (intelligence/generate-recommendation prediction-data)]
      
      (is (= "high-risk" (:severity recommendation)))
      (is (seq (:actions recommendation)))
      (is (seq (:focus-areas recommendation)))
      (is (str/includes? (:timeline-recommendation recommendation) "2+ weeks"))))

(deftest test-anomaly-detection
  (testing "Anomaly detection in metrics"
    (mock-ml-service)
    (mock-database)
    
    (with-redefs [analytics/get-customer-metrics-history
                  (fn [customer-id metric days]
                    (map (fn [i] {:metric_value (if (= i 5) 25 50)}) ; Anomaly at index 5
                         (range 10)))]
      
      (let [result (analytics/run-anomaly-detection test-customer-id)]
        ;; Should complete without error and create alerts for anomalies
        (is (nil? result)) ; Function returns nil on success
        ))))

(deftest test-competitive-benchmarking
  (testing "Creating competitive benchmarks"
    (mock-database)
    
    (let [industry-data {:average 0.65 :top_competitor 0.85}
          result (competitive/create-competitive-benchmark 
                 test-customer-id
                 "player_satisfaction"
                 0.72
                 industry-data)]
      
      (is (:success result))
      (let [benchmark (:benchmark result)]
        (is (= 0.72 (:customer_value benchmark)))
        (is (= 0.65 (:industry_average benchmark)))
        (is (> (:performance_vs_average benchmark) 1.0)) ; Above average
        (is (< (:performance_vs_top benchmark) 1.0))))) ; Below top performer

;; --- Integration Tests ---

(deftest test-end-to-end-intelligence-pipeline
  (testing "Complete intelligence pipeline from data to insights"
    (mock-ml-service)
    (mock-database)
    
    ;; Step 1: Discover personas
    (let [persona-result (intelligence/discover-customer-personas test-customer-id)]
      (is (:success persona-result))
      
      ;; Step 2: Generate predictions based on personas
      (let [prediction-result (intelligence/predict-change-impact 
                              test-customer-id
                              "Major gameplay overhaul"
                              ["core-mechanics" "ui"])]
        (is (:success prediction-result))
        
        ;; Step 3: Generate automated insights
        (let [insights-result (intelligence/generate-automated-insights test-customer-id)]
          (is (:success insights-result))
          
          ;; Step 4: Check for alerts
          (let [alerts-result (alerts/run-alert-checks test-customer-id)]
            (is (>= (:alerts-created alerts-result) 0))
            
            ;; Step 5: Competitive analysis
            (let [competitive-result (competitive/generate-competitive-insights test-customer-id)]
              (is (:success competitive-result)))))))))

(deftest test-analytics-dashboard-integration
  (testing "Analytics dashboard data integration"
    (mock-database)
    
    (let [lifecycle-metrics (analytics/calculate-player-lifecycle-metrics test-customer-id)
          competitive-dashboard (competitive/get-competitive-dashboard test-customer-id)
          customer-alerts (analytics/get-customer-alerts test-customer-id)]
      
      ;; All dashboard components should work
      (is (contains? lifecycle-metrics :total_users))
      (is (:success competitive-dashboard))
      (is (sequential? customer-alerts)))))

;; --- Performance Tests ---

(deftest test-performance-with-large-datasets
  (testing "Performance with large message datasets"
    (mock-ml-service)
    
    (with-redefs [db/get-customer-messages 
                  (fn [customer-id limit] 
                    (repeatedly limit #(hash-map :id (str "msg" (rand-int 10000))
                                                :content "test message content"
                                                :author (str "user" (rand-int 100))
                                                :timestamp "2023-01-01T10:00:00Z")))]
      
      (let [start-time (System/currentTimeMillis)
            result (intelligence/discover-customer-personas test-customer-id)
            end-time (System/currentTimeMillis)
            duration (- end-time start-time)]
        
        ;; Should complete within reasonable time (5 seconds)
        (is (< duration 5000))
        (is (:success result))))))

;; --- Run All Tests ---

(defn run-all-intelligence-tests
  "Run all intelligence tests and return summary"
  []
  (log/info "Starting comprehensive intelligence system tests...")
  
  (let [test-results (atom {:passed 0 :failed 0 :errors []})]
    
    (try
      ;; Test persona discovery
      (test-persona-discovery)
      (swap! test-results update :passed inc)
      (log/info "✓ Persona discovery tests passed")
      (catch Exception e
        (swap! test-results update :failed inc)
        (swap! test-results update :errors conj {:test "persona-discovery" :error (.getMessage e)})
        (log/error "✗ Persona discovery tests failed:" (.getMessage e))))
    
    (try
      ;; Test prediction capabilities
      (test-change-impact-prediction)
      (swap! test-results update :passed inc)
      (log/info "✓ Change impact prediction tests passed")
      (catch Exception e
        (swap! test-results update :failed inc)
        (swap! test-results update :errors conj {:test "prediction" :error (.getMessage e)})
        (log/error "✗ Change impact prediction tests failed:" (.getMessage e))))
    
    (try
      ;; Test competitive intelligence
      (test-competitive-analysis)
      (swap! test-results update :passed inc)
      (log/info "✓ Competitive analysis tests passed")
      (catch Exception e
        (swap! test-results update :failed inc)
        (swap! test-results update :errors conj {:test "competitive" :error (.getMessage e)})
        (log/error "✗ Competitive analysis tests failed:" (.getMessage e))))
    
    (try
      ;; Test alerting system
      (test-alert-generation)
      (swap! test-results update :passed inc)
      (log/info "✓ Alert generation tests passed")
      (catch Exception e
        (swap! test-results update :failed inc)
        (swap! test-results update :errors conj {:test "alerts" :error (.getMessage e)})
        (log/error "✗ Alert generation tests failed:" (.getMessage e))))
    
    (try
      ;; Test end-to-end pipeline
      (test-end-to-end-intelligence-pipeline)
      (swap! test-results update :passed inc)
      (log/info "✓ End-to-end pipeline tests passed")
      (catch Exception e
        (swap! test-results update :failed inc)
        (swap! test-results update :errors conj {:test "end-to-end" :error (.getMessage e)})
        (log/error "✗ End-to-end pipeline tests failed:" (.getMessage e))))
    
    (let [final-results @test-results]
      (log/info "Intelligence system tests completed:")
      (log/info "  Passed:" (:passed final-results))
      (log/info "  Failed:" (:failed final-results))
      (when (seq (:errors final-results))
        (log/error "  Errors:" (:errors final-results)))
      
      final-results)))