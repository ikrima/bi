(ns player-intel.competitive
  (:require [player-intel.db :as db]
            [player-intel.ml :as ml]
            [clojure.string :as str]
            [clojure.data.json :as json]
            [taoensso.timbre :as log]
            [clj-time.core :as time]
            [clj-time.coerce :as coerce]
            [honey.sql :as sql]
            [next.jdbc :as jdbc]))

;; --- Competitive Intelligence Database Schema ---
(defn create-competitive-tables!
  "Create competitive intelligence database tables"
  []
  (try
    ;; Competitor mentions table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS competitor_mentions ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "message_id VARCHAR(255),"
            "competitor_name VARCHAR(100),"
            "mention_context TEXT,"
            "sentiment_score FLOAT,"
            "mention_type VARCHAR(50)," ; comparison, feature-request, complaint, praise
            "detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "processed BOOLEAN DEFAULT FALSE)")])
    
    ;; Competitive insights table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS competitive_insights ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "competitor_name VARCHAR(100),"
            "insight_type VARCHAR(50),"
            "insight_data JSONB,"
            "confidence FLOAT,"
            "generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "expires_at TIMESTAMP)")])
    
    ;; Competitive benchmarks table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS competitive_benchmarks ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "metric_name VARCHAR(100),"
            "customer_value FLOAT,"
            "industry_average FLOAT,"
            "top_competitor_value FLOAT,"
            "benchmark_date DATE,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")])
    
    (log/info "Competitive intelligence tables created successfully")
    (catch Exception e
      (log/error e "Failed to create competitive intelligence tables"))))

;; --- Competitor Detection ---
(def default-competitors
  {:gaming ["fortnite" "apex legends" "valorant" "overwatch" "call of duty" "counter-strike"
            "league of legends" "dota" "world of warcraft" "destiny" "minecraft" "roblox"]
   :mobile ["clash royale" "candy crush" "pokemon go" "clash of clans" "honor of kings"]
   :indie ["among us" "fall guys" "phasmophobia" "valheim" "hades"]})

(defn detect-competitor-mentions
  "Detect competitor mentions in recent messages"
  [customer-id & {:keys [days-back competitors] :or {days-back 7}}]
  (try
    (let [since-date (coerce/to-timestamp (time/minus (time/now) (time/days days-back)))
          messages (db/get-customer-messages-since customer-id since-date)
          competitor-list (or competitors 
                             (apply concat (vals default-competitors)))
          mentions []]
      
      (doseq [message messages]
        (let [content (str/lower-case (:content message))]
          (doseq [competitor competitor-list]
            (when (and (re-find (re-pattern (str "\\b" (str/lower-case competitor) "\\b")) content)
                      ;; Avoid false positives on short names
                      (or (> (count competitor) 4)
                          (re-find (re-pattern (str "\\b" (str/lower-case competitor) "\\s+(game|player|better|worse|like)")) content)))
              
              (let [mention-context (extract-mention-context content competitor)
                    mention-type (classify-mention-type content competitor)
                    sentiment (analyze-competitive-sentiment content competitor)]
                
                ;; Store mention
                (jdbc/execute! (db/get-db)
                  (sql/format {:insert-into :competitor_mentions
                               :values [{:customer_id customer-id
                                        :message_id (:id message)
                                        :competitor_name competitor
                                        :mention_context mention-context
                                        :sentiment_score sentiment
                                        :mention_type mention-type}]}))
                
                (conj mentions {:competitor competitor
                               :message_id (:id message)
                               :context mention-context
                               :type mention-type
                               :sentiment sentiment
                               :timestamp (:timestamp message)}))))))
      
      (log/info "Detected" (count mentions) "competitor mentions for customer" customer-id)
      mentions)
    
    (catch Exception e
      (log/error e "Failed to detect competitor mentions")
      [])))

(defn extract-mention-context
  "Extract context around a competitor mention"
  [content competitor]
  (let [words (str/split content #"\s+")
        competitor-words (str/split (str/lower-case competitor) #"\s+")
        competitor-indices (for [i (range (count words))
                                :when (some #(str/includes? (str/lower-case (nth words i "")) %) competitor-words)]
                            i)]
    (when (seq competitor-indices)
      (let [start-idx (max 0 (- (first competitor-indices) 10))
            end-idx (min (count words) (+ (last competitor-indices) 10))]
        (str/join " " (subvec (vec words) start-idx end-idx))))))

(defn classify-mention-type
  "Classify the type of competitor mention"
  [content competitor]
  (let [content-lower (str/lower-case content)]
    (cond
      (some #(str/includes? content-lower %) ["better than" "worse than" "compared to" "vs" "versus"])
      "comparison"
      
      (some #(str/includes? content-lower %) ["should add" "needs" "like in" "has this feature"])
      "feature-request"
      
      (some #(str/includes? content-lower %) ["switching to" "moving to" "quit for" "leaving for"])
      "churn-intent"
      
      (some #(str/includes% content-lower %) ["love" "amazing" "great" "awesome" "prefer"])
      "praise"
      
      (some #(str/includes? content-lower %) ["hate" "terrible" "broken" "sucks"])
      "complaint"
      
      :else "general-mention")))

(defn analyze-competitive-sentiment
  "Analyze sentiment of competitor mention"
  [content competitor]
  (let [content-lower (str/lower-case content)
        positive-indicators ["better" "great" "love" "awesome" "prefer" "amazing" "excellent"]
        negative-indicators ["worse" "hate" "terrible" "sucks" "awful" "broken" "bad"]]
    
    (let [positive-score (count (filter #(str/includes? content-lower %) positive-indicators))
          negative-score (count (filter #(str/includes? content-lower %) negative-indicators))]
      
      (cond
        (> positive-score negative-score) 0.8
        (> negative-score positive-score) 0.2
        :else 0.5))))

;; --- Competitive Analysis ---
(defn generate-competitive-insights
  "Generate comprehensive competitive insights"
  [customer-id & {:keys [days-back] :or {days-back 30}}]
  (try
    (let [since-date (coerce/to-sql-date (time/minus (time/now) (time/days days-back)))
          mentions (jdbc/execute! (db/get-db)
                    (sql/format {:select :*
                                :from :competitor_mentions
                                :where [:and
                                       [:= :customer_id customer-id]
                                       [:>= :detected_at since-date]]
                                :order-by [[:detected_at :desc]]}))
          
          ;; Group mentions by competitor
          by-competitor (group-by :competitor_name mentions)
          
          insights []]
      
      ;; Generate insights for each competitor
      (doseq [[competitor competitor-mentions] by-competitor]
        (let [mention-count (count competitor-mentions)
              avg-sentiment (/ (reduce + (map :sentiment_score competitor-mentions)) 
                              mention-count)
              mention-types (frequencies (map :mention_type competitor-mentions))
              
              ;; Key insights
              insight-data {:mention_count mention-count
                           :average_sentiment avg-sentiment
                           :mention_types mention-types
                           :trending (> mention-count 5)
                           :sentiment_trend (if (> avg-sentiment 0.6) "positive" 
                                           (if (< avg-sentiment 0.4) "negative" "neutral"))
                           :top_concerns (extract-top-concerns competitor-mentions)
                           :feature_requests (filter #(= (:mention_type %) "feature-request") competitor-mentions)}]
          
          ;; Store insight
          (jdbc/execute! (db/get-db)
            (sql/format {:insert-into :competitive_insights
                         :values [{:customer_id customer-id
                                  :competitor_name competitor
                                  :insight_type "mention-analysis"
                                  :insight_data (json/write-str insight-data)
                                  :confidence (min 0.9 (/ mention-count 10))
                                  :expires_at (coerce/to-timestamp 
                                              (time/plus (time/now) (time/days 7)))}]}))
          
          (conj insights {:competitor competitor
                         :data insight-data})))
      
      ;; Generate market positioning insight
      (when (> (count by-competitor) 1)
        (let [positioning-data {:most_mentioned (first (sort-by (comp count second) > by-competitor))
                               :sentiment_leader (first (sort-by #(/ (reduce + (map :sentiment_score (second %)))
                                                                    (count (second %))) > by-competitor))
                               :feature_gap_analysis (analyze-feature-gaps mentions)
                               :churn_risk_analysis (analyze-churn-risk mentions)}]
          
          (jdbc/execute! (db/get-db)
            (sql/format {:insert-into :competitive_insights
                         :values [{:customer_id customer-id
                                  :competitor_name "MARKET_ANALYSIS"
                                  :insight_type "market-positioning"
                                  :insight_data (json/write-str positioning-data)
                                  :confidence 0.8
                                  :expires_at (coerce/to-timestamp 
                                              (time/plus (time/now) (time/days 7)))}]}))
          
          (conj insights {:competitor "MARKET_ANALYSIS"
                         :data positioning-data})))
      
      {:success true
       :insights insights
       :total_mentions (count mentions)
       :competitors_detected (count by-competitor)
       :analysis_period_days days-back})
    
    (catch Exception e
      (log/error e "Failed to generate competitive insights")
      {:success false :error (.getMessage e)})))

(defn extract-top-concerns
  "Extract top concerns from competitor mentions"
  [mentions]
  (let [complaint-mentions (filter #(= (:mention_type %) "complaint") mentions)
        contexts (map :mention_context complaint-mentions)]
    
    ;; Simple keyword extraction for concerns
    (->> contexts
         (mapcat #(str/split (str/lower-case %) #"\s+"))
         (filter #(> (count %) 4))
         frequencies
         (sort-by second >)
         (take 5)
         (map first))))

(defn analyze-feature-gaps
  "Analyze feature gaps based on competitor mentions"
  [mentions]
  (let [feature-requests (filter #(= (:mention_type %) "feature-request") mentions)
        features (map :mention_context feature-requests)]
    
    ;; Extract commonly requested features
    (when (seq features)
      {:common_requests (take 3 (keys (sort-by val > (frequencies features))))
       :request_count (count feature-requests)
       :top_competitors_for_features (frequencies (map :competitor_name feature-requests))})))

(defn analyze-churn-risk
  "Analyze churn risk based on competitor mentions"
  [mentions]
  (let [churn-mentions (filter #(= (:mention_type %) "churn-intent") mentions)]
    {:churn_intent_mentions (count churn-mentions)
     :churn_risk_score (min 1.0 (/ (count churn-mentions) 20.0))
     :top_churn_destinations (take 3 (keys (sort-by val > (frequencies (map :competitor_name churn-mentions)))))
     :recent_churn_spike (> (count churn-mentions) 5)}))

;; --- Competitive Benchmarking ---
(defn create-competitive-benchmark
  "Create competitive benchmark for metrics"
  [customer-id metric-name customer-value industry-data]
  (try
    (let [industry-average (:average industry-data)
          top-competitor-value (:top_competitor industry-data)]
      
      (jdbc/execute! (db/get-db)
        (sql/format {:insert-into :competitive_benchmarks
                     :values [{:customer_id customer-id
                               :metric_name metric-name
                               :customer_value customer-value
                               :industry_average industry-average
                               :top_competitor_value top-competitor-value
                               :benchmark_date (coerce/to-sql-date (time/now))}]}))
      
      {:success true
       :benchmark {:metric metric-name
                  :customer_value customer-value
                  :industry_average industry-average
                  :performance_vs_average (/ customer-value industry-average)
                  :performance_vs_top (/ customer-value top-competitor-value)}})
    
    (catch Exception e
      (log/error e "Failed to create competitive benchmark")
      {:success false :error (.getMessage e)})))

(defn get-competitive-dashboard
  "Get comprehensive competitive intelligence dashboard"
  [customer-id]
  (try
    (let [recent-insights (jdbc/execute! (db/get-db)
                           (sql/format {:select :*
                                       :from :competitive_insights
                                       :where [:and
                                              [:= :customer_id customer-id]
                                              [:> :expires_at (coerce/to-timestamp (time/now))]]
                                       :order-by [[:generated_at :desc]]
                                       :limit 10}))
          
          recent-mentions (jdbc/execute! (db/get-db)
                          (sql/format {:select [:competitor_name [:%count.* :mention_count] 
                                               [:%avg.sentiment_score :avg_sentiment]]
                                      :from :competitor_mentions
                                      :where [:and
                                             [:= :customer_id customer-id]
                                             [:>= :detected_at (coerce/to-timestamp 
                                                               (time/minus (time/now) (time/days 7)))]]
                                      :group-by :competitor_name
                                      :order-by [[:mention_count :desc]]}))
          
          benchmarks (jdbc/execute! (db/get-db)
                     (sql/format {:select :*
                                 :from :competitive_benchmarks
                                 :where [:= :customer_id customer-id]
                                 :order-by [[:benchmark_date :desc]]
                                 :limit 5}))]
      
      {:success true
       :insights (map (fn [insight]
                       {:competitor (:competitor_name insight)
                        :type (:insight_type insight)
                        :data (json/read-str (:insight_data insight) :key-fn keyword)
                        :confidence (:confidence insight)
                        :generated_at (:generated_at insight)})
                     recent-insights)
       :recent_mentions recent-mentions
       :benchmarks benchmarks
       :summary {:total_competitors_mentioned (count recent-mentions)
                :most_mentioned_competitor (when (seq recent-mentions) 
                                            (:competitor_name (first recent-mentions)))
                :overall_competitive_sentiment (when (seq recent-mentions)
                                               (/ (reduce + (map :avg_sentiment recent-mentions))
                                                 (count recent-mentions)))
                :generated_at (System/currentTimeMillis)}})
    
    (catch Exception e
      (log/error e "Failed to get competitive dashboard")
      {:success false :error (.getMessage e)})))

;; --- Scheduled Competitive Analysis ---
(defn run-daily-competitive-analysis
  "Run competitive analysis for all customers"
  []
  (try
    (let [active-customers (db/get-active-customers)]
      (log/info "Running competitive analysis for" (count active-customers) "customers")
      
      (doseq [customer active-customers]
        (try
          ;; Detect new mentions
          (detect-competitor-mentions (:id customer) :days-back 1)
          
          ;; Generate weekly insights
          (when (= (mod (System/currentTimeMillis) (* 7 24 60 60 1000)) 0) ; Weekly
            (generate-competitive-insights (:id customer) :days-back 7))
          
          (log/info "Completed competitive analysis for customer" (:id customer))
          
          (catch Exception e
            (log/error e "Failed competitive analysis for customer" (:id customer)))))
      
      (log/info "Completed daily competitive analysis"))
    
    (catch Exception e
      (log/error e "Failed to run daily competitive analysis"))))

;; --- API Routes ---
(defn competitive-routes []
  [["/api/competitive"
    {:middleware [auth/customer-middleware]}
    
    ["/dashboard"
     {:get {:handler (fn [req]
                      (let [customer-id (get-in req [:customer :id])
                            dashboard (get-competitive-dashboard customer-id)]
                        {:status (if (:success dashboard) 200 400)
                         :body dashboard}))}}]
    
    ["/mentions/detect"
     {:post {:handler (fn [req]
                       (let [customer-id (get-in req [:customer :id])
                             {:keys [days-back competitors]} (:body req)
                             mentions (detect-competitor-mentions customer-id
                                                                :days-back (or days-back 7)
                                                                :competitors competitors)]
                         {:status 200
                          :body {:success true
                                :mentions mentions
                                :count (count mentions)}}))}}]
    
    ["/insights/generate"
     {:post {:handler (fn [req]
                       (let [customer-id (get-in req [:customer :id])
                             {:keys [days-back]} (:body req)
                             insights (generate-competitive-insights customer-id
                                                                   :days-back (or days-back 30))]
                         {:status (if (:success insights) 200 400)
                          :body insights}))}}]
    
    ["/benchmark"
     {:post {:handler (fn [req]
                       (let [customer-id (get-in req [:customer :id])
                             {:keys [metric-name customer-value industry-data]} (:body req)
                             benchmark (create-competitive-benchmark customer-id
                                                                   metric-name
                                                                   customer-value
                                                                   industry-data)]
                         {:status (if (:success benchmark) 200 400)
                          :body benchmark}))}}]]])