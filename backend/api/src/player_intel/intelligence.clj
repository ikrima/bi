(ns player-intel.intelligence
  (:require [player-intel.ml :as ml]
            [player-intel.db :as db]
            [player-intel.customers :as customers]
            [clojure.data.json :as json]
            [taoensso.timbre :as log]
            [clj-time.core :as time]
            [clj-time.coerce :as coerce]))

;; --- Persona Discovery Integration ---
(defn discover-customer-personas
  "Discover personas for a specific customer's community"
  [customer-id]
  (try
    (let [customer (customers/get-customer-by-id customer-id)
          messages (db/get-customer-messages customer-id 5000) ; Last 5000 messages
          user-groups (group-by :author messages)
          user-data (map (fn [[author msgs]]
                          {:author author
                           :messages (map :content msgs)
                           :message-count (count msgs)
                           :timespan (time/in-days 
                                     (time/interval 
                                      (coerce/from-sql-time (:timestamp (first msgs)))
                                      (coerce/from-sql-time (:timestamp (last msgs)))))})
                        user-groups)]
      
      (if (< (count user-data) 10) ; Need minimum users for analysis
        {:success false 
         :error "Insufficient data for persona discovery"
         :minimum-required 10
         :current-users (count user-data)}
        
        (let [ml-response (ml/discover-personas user-data)]
          (if (:success ml-response)
            (do
              ;; Cache personas in database
              (db/save-customer-personas! customer-id (:data ml-response))
              {:success true
               :personas (:data ml-response)
               :generated-at (System/currentTimeMillis)
               :users-analyzed (count user-data)})
            ml-response))))
    
    (catch Exception e
      (log/error e "Failed to discover personas for customer" customer-id)
      {:success false :error (.getMessage e)})))

(defn get-customer-personas
  "Get cached personas or generate new ones"
  [customer-id & {:keys [refresh?] :or {refresh? false}}]
  (if refresh?
    (discover-customer-personas customer-id)
    (if-let [cached (db/get-customer-personas customer-id)]
      {:success true :personas cached :from-cache true}
      (discover-customer-personas customer-id))))

;; --- Predictive Analytics Integration ---
(defn predict-change-impact
  "Predict impact of a game change on community"
  [customer-id change-description affected-areas]
  (try
    (let [personas (get-customer-personas customer-id)
          historical-data (db/get-customer-historical-reactions customer-id)]
      
      (if-not (:success personas)
        personas ; Return the error from persona discovery
        
        (let [prediction-request {:change-description change-description
                                :affected-areas affected-areas
                                :personas (:personas personas)
                                :historical-reactions historical-data}
              ml-response (ml/predict-change-impact prediction-request)]
          
          (if (:success ml-response)
            (do
              ;; Log prediction for future learning
              (db/log-prediction! customer-id change-description (:data ml-response))
              
              {:success true
               :prediction (:data ml-response)
               :confidence (:confidence ml-response)
               :generated-at (System/currentTimeMillis)
               :recommendation (generate-recommendation (:data ml-response))})
            ml-response))))
    
    (catch Exception e
      (log/error e "Failed to predict change impact for customer" customer-id)
      {:success false :error (.getMessage e)})))

(defn generate-recommendation
  "Generate actionable recommendations based on prediction"
  [prediction-data]
  (let [overall-sentiment (:overall-sentiment prediction-data)
        high-risk-personas (filter #(> (:churn-risk %) 0.7) (:persona-impacts prediction-data))
        negative-personas (filter #(< (:predicted-sentiment %) 0.4) (:persona-impacts prediction-data))]
    
    {:severity (cond
                (or (< overall-sentiment 0.3) (> (count high-risk-personas) 2))
                "high-risk"
                
                (or (< overall-sentiment 0.5) (> (count high-risk-personas) 1))
                "medium-risk"
                
                :else "low-risk")
     
     :actions (cond
                (< overall-sentiment 0.3)
                ["Consider delaying this change"
                 "Prepare extensive communication campaign"
                 "Plan compensation mechanisms"
                 "Schedule community feedback sessions"]
                
                (< overall-sentiment 0.5)
                ["Communicate change rationale clearly"
                 "Provide advance notice"
                 "Monitor community closely post-launch"
                 "Prepare quick response plan"]
                
                :else
                ["Standard communication approach sufficient"
                 "Monitor for unexpected reactions"])
     
     :focus-areas (map :name high-risk-personas)
     :timeline-recommendation (if (< overall-sentiment 0.4)
                               "2+ weeks advance notice recommended"
                               "1 week advance notice sufficient")}))

;; --- Automated Recommendation Engine ---
(defn generate-automated-insights
  "Generate automated insights and recommendations"
  [customer-id]
  (try
    (let [recent-digest (db/get-latest-digest customer-id)
          personas (get-customer-personas customer-id)
          trends (analyze-community-trends customer-id)
          recommendations []]
      
      ;; Sentiment-based recommendations
      (let [recommendations 
            (if (and recent-digest (< (:sentiment recent-digest) 50))
              (conj recommendations
                    {:type "sentiment-alert"
                     :priority "high"
                     :title "Community Sentiment Declining"
                     :description "Recent sentiment has dropped below 50%"
                     :actions ["Review recent changes"
                              "Increase community engagement"
                              "Address top concerns in digest"]})
              recommendations)]
        
        ;; Engagement-based recommendations
        (let [recommendations
              (if (and trends (< (:engagement-trend trends) -0.1))
                (conj recommendations
                      {:type "engagement-alert"
                       :priority "medium"
                       :title "Player Engagement Dropping"
                       :description "Engagement has declined by more than 10%"
                       :actions ["Create engaging content"
                                "Host community events"
                                "Survey players for feedback"]})
                recommendations)]
          
          ;; Persona-based recommendations
          (let [recommendations
                (if (and (:success personas) 
                        (some #(> (:churn-risk %) 0.8) (:personas personas)))
                  (conj recommendations
                        {:type "churn-risk-alert"
                         :priority "high"
                         :title "High Churn Risk Detected"
                         :description "Some player segments show high churn risk"
                         :actions ["Target retention campaigns"
                                  "Personalize engagement"
                                  "Address specific concerns"]})
                  recommendations)]
            
            {:success true
             :recommendations recommendations
             :generated-at (System/currentTimeMillis)
             :insights-count (count recommendations)}))))
    
    (catch Exception e
      (log/error e "Failed to generate automated insights for customer" customer-id)
      {:success false :error (.getMessage e)})))

(defn analyze-community-trends
  "Analyze trends in community activity and sentiment"
  [customer-id]
  (try
    (let [past-digests (db/get-customer-digests customer-id 30) ; Last 30 digests
          message-counts (map :message-count past-digests)
          sentiment-scores (map :sentiment past-digests)]
      
      (when (> (count past-digests) 5) ; Need minimum data points
        (let [engagement-trend (calculate-trend message-counts)
              sentiment-trend (calculate-trend sentiment-scores)
              volatility (calculate-volatility sentiment-scores)]
          
          {:engagement-trend engagement-trend
           :sentiment-trend sentiment-trend
           :sentiment-volatility volatility
           :data-points (count past-digests)
           :period-days 30})))
    
    (catch Exception e
      (log/error e "Failed to analyze community trends for customer" customer-id)
      nil)))

(defn calculate-trend
  "Calculate linear trend from time series data"
  [values]
  (when (> (count values) 2)
    (let [n (count values)
          x-values (range n)
          sum-x (reduce + x-values)
          sum-y (reduce + values)
          sum-xy (reduce + (map * x-values values))
          sum-x2 (reduce + (map * x-values x-values))
          slope (/ (- (* n sum-xy) (* sum-x sum-y))
                  (- (* n sum-x2) (* sum-x sum-x)))]
      (/ slope (/ sum-y n))))) ; Normalize by mean

(defn calculate-volatility
  "Calculate volatility (standard deviation) of values"
  [values]
  (when (> (count values) 1)
    (let [mean (/ (reduce + values) (count values))
          squared-diffs (map #(Math/pow (- % mean) 2) values)
          variance (/ (reduce + squared-diffs) (count values))]
      (Math/sqrt variance))))

;; --- Competitive Intelligence ---
(defn analyze-competitive-landscape
  "Analyze competitive mentions and sentiment"
  [customer-id competitor-games]
  (try
    (let [messages (db/get-customer-messages customer-id 2000)
          competitor-mentions (filter (fn [msg]
                                      (some #(re-find (re-pattern (str "(?i)" %)) 
                                                     (:content msg))
                                           competitor-games))
                                    messages)]
      
      (if (empty? competitor-mentions)
        {:success true
         :competitor-mentions 0
         :analysis "No competitor mentions found"}
        
        (let [mention-analysis (group-by (fn [msg]
                                         (first (filter #(re-find (re-pattern (str "(?i)" %)) 
                                                                 (:content msg))
                                                       competitor-games)))
                                        competitor-mentions)
              ml-response (ml/analyze-competitor-sentiment competitor-mentions)]
          
          {:success true
           :competitor-mentions (count competitor-mentions)
           :mentions-by-game (into {} (map (fn [[game msgs]] 
                                           [game (count msgs)]) 
                                          mention-analysis))
           :sentiment-analysis (:data ml-response)
           :insights (generate-competitive-insights mention-analysis ml-response)})))
    
    (catch Exception e
      (log/error e "Failed to analyze competitive landscape for customer" customer-id)
      {:success false :error (.getMessage e)})))

(defn generate-competitive-insights
  "Generate insights from competitive analysis"
  [mention-analysis sentiment-analysis]
  (let [top-mentioned (first (sort-by (comp count second) > mention-analysis))
        insights []]
    
    (cond-> insights
      (> (count mention-analysis) 3)
      (conj {:type "high-competition"
             :message "Multiple competitors mentioned frequently"
             :recommendation "Monitor competitive positioning closely"})
      
      (and top-mentioned (> (count (second top-mentioned)) 20))
      (conj {:type "dominant-competitor"
             :competitor (first top-mentioned)
             :mentions (count (second top-mentioned))
             :recommendation "Analyze what players like about this competitor"})
      
      (< (:average-sentiment sentiment-analysis 0.5) 0.3)
      (conj {:type "negative-comparison"
             :message "Players comparing unfavorably to competitors"
             :recommendation "Address specific concerns mentioned in comparisons"}))))

;; --- API Endpoints ---
(defn intelligence-routes []
  [["/api/intelligence"
    {:middleware [auth/customer-middleware]}
    
    ["/personas"
     {:get {:handler (fn [req]
                      (let [customer-id (get-in req [:customer :id])
                            refresh? (get-in req [:query-params "refresh"])
                            result (get-customer-personas customer-id 
                                                        :refresh? (= refresh? "true"))]
                        {:status (if (:success result) 200 400)
                         :body result}))}}]
    
    ["/predict"
     {:post {:handler (fn [req]
                       (let [customer-id (get-in req [:customer :id])
                             {:keys [change-description affected-areas]} (:body req)
                             result (predict-change-impact customer-id 
                                                         change-description 
                                                         affected-areas)]
                         {:status (if (:success result) 200 400)
                          :body result}))}}]
    
    ["/recommendations"
     {:get {:handler (fn [req]
                      (let [customer-id (get-in req [:customer :id])
                            result (generate-automated-insights customer-id)]
                        {:status (if (:success result) 200 400)
                         :body result}))}}]
    
    ["/trends"
     {:get {:handler (fn [req]
                      (let [customer-id (get-in req [:customer :id])
                            trends (analyze-community-trends customer-id)]
                        {:status 200
                         :body (or trends {:message "Insufficient data for trend analysis"})}))}}]
    
    ["/competitive"
     {:post {:handler (fn [req]
                       (let [customer-id (get-in req [:customer :id])
                             {:keys [competitors]} (:body req)
                             result (analyze-competitive-landscape customer-id competitors)]
                         {:status (if (:success result) 200 400)
                          :body result}))}}]]])