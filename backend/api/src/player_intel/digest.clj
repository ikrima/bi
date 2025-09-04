(ns player-intel.digest
  (:require [player-intel.db :as db]
            [player-intel.ml :as ml]
            [player-intel.cache :as cache]
            [clojure.string :as str]
            [taoensso.timbre :as log]
            [clojure.data.json :as json]
            [honey.sql :as sql]
            [next.jdbc :as jdbc]))

(defn generate-digest
  "Generate a comprehensive digest from recent messages"
  ([] (generate-digest 1000))
  ([message-limit] (generate-digest message-limit nil))
  ([message-limit channel-id]
   (log/info "Generating digest for channel:" channel-id "with limit:" message-limit)
   
   (let [messages (db/get-recent-messages message-limit channel-id)]
     (if (empty? messages)
       {:timestamp (System/currentTimeMillis)
        :channel-id channel-id
        :message-count 0
        :clusters []
        :urgent-issues []
        :sentiment {:score 50 :label "neutral"}
        :summary "No messages found for analysis"}
       
       (let [texts (map :content messages)
             authors (map :author messages)
             timestamps (map :timestamp messages)
             
             ;; Generate embeddings
             embeddings (ml/generate-embeddings texts)]
         
         (if embeddings
           ;; Perform clustering analysis
           (let [clustering-result (ml/cluster-messages embeddings texts timestamps authors)
                 clusters (get clustering-result :clusters [])
                 summary (get clustering-result :summary {})]
             
             {:timestamp (System/currentTimeMillis)
              :channel-id channel-id
              :message-count (count messages)
              :clusters (take 5 clusters)  ; Top 5 themes
              :urgent-issues (find-urgent-issues messages clusters)
              :sentiment (calculate-overall-sentiment clusters summary)
              :insights (generate-insights clusters messages)
              :recommendations (generate-recommendations clusters)
              :summary (generate-summary clusters (count messages))})
           
           ;; Fallback without ML
           (generate-fallback-digest messages)))))))

(defn find-urgent-issues
  "Find messages that need immediate attention"
  [messages clusters]
  (let [high-urgency-clusters (filter #(> (:urgency %) 0.6) clusters)
        urgent-keywords ["crash" "broken" "bug" "error" "cant play" "wont start" "not working" "help" "stuck"]]
    
    ;; Combine cluster-based and keyword-based urgent detection
    (concat
      ;; From high-urgency clusters
      (mapcat (fn [cluster]
                (map (fn [msg] 
                       {:content msg
                        :urgency (:urgency cluster)
                        :theme (:theme cluster)
                        :source "cluster-analysis"})
                     (:sample-messages cluster)))
              high-urgency-clusters)
      
      ;; Keyword-based detection
      (->> messages
           (filter #(some (fn [kw] 
                           (str/includes? (str/lower-case (:content %)) kw))
                         urgent-keywords))
           (take 5)
           (map #(assoc (select-keys % [:content :author :timestamp])
                       :urgency 0.8
                       :source "keyword-detection"))))))

(defn calculate-overall-sentiment
  "Calculate overall sentiment with detailed breakdown"
  [clusters summary]
  (if (empty? clusters)
    {:score 50 :label "neutral" :confidence 0.0}
    
    (let [sentiment-scores {"very_positive" 90
                           "positive" 75
                           "neutral" 50
                           "negative" 25
                           "very_negative" 10}
          
          weighted-score (/ (reduce + (map (fn [cluster]
                                            (* (:size cluster)
                                               (sentiment-scores (:sentiment cluster) 50)))
                                          clusters))
                           (max 1 (reduce + (map :size clusters))))
          
          confidence (/ (reduce + (map :confidence clusters))
                       (max 1 (count clusters)))
          
          label (cond
                  (>= weighted-score 80) "very positive"
                  (>= weighted-score 60) "positive"
                  (<= weighted-score 20) "very negative"
                  (<= weighted-score 40) "negative"
                  :else "neutral")]
      
      {:score (Math/round weighted-score)
       :label label
       :confidence (Math/round (* confidence 100))
       :breakdown (frequencies (map :sentiment clusters))})))

(defn generate-insights
  "Generate actionable insights from the analysis"
  [clusters messages]
  (let [total-messages (count messages)
        cluster-count (count clusters)]
    
    (cond-> []
      ;; Community health insights
      (> total-messages 100)
      (conj {:type "community-health"
             :insight "High community engagement detected"
             :details (str total-messages " messages analyzed")
             :action "Consider acknowledging active community members"})
      
      ;; Topic diversity
      (> cluster-count 3)
      (conj {:type "topic-diversity"
             :insight "Diverse discussion topics"
             :details (str cluster-count " distinct themes identified")
             :action "Community discussions are well-distributed across topics"})
      
      ;; Urgent attention needed
      (some #(> (:urgency %) 0.7) clusters)
      (conj {:type "urgent-attention"
             :insight "Critical issues require immediate attention"
             :details "High-urgency clusters detected"
             :action "Review urgent issues and provide timely responses"})
      
      ;; Sentiment concerns
      (< (:score (calculate-overall-sentiment clusters {})) 40)
      (conj {:type "sentiment-concern"
             :insight "Community sentiment trending negative"
             :details "Consider addressing player concerns"
             :action "Engage with community to understand and resolve issues"})
      
      ;; Positive momentum
      (> (:score (calculate-overall-sentiment clusters {})) 70)
      (conj {:type "positive-momentum"
             :insight "Strong positive community sentiment"
             :details "Players are generally satisfied and engaged"
             :action "Maintain current engagement strategies"}))))

(defn generate-recommendations
  "Generate actionable recommendations for developers"
  [clusters]
  (let [urgent-clusters (filter #(> (:urgency %) 0.5) clusters)
        negative-clusters (filter #(contains? #{"negative" "very_negative"} (:sentiment %)) clusters)
        popular-topics (take 3 (sort-by :size > clusters))]
    
    (cond-> []
      ;; Urgent action items
      (seq urgent-clusters)
      (conj {:priority "high"
             :category "urgent-fixes"
             :title "Address Critical Issues"
             :description (str "Focus on " (count urgent-clusters) " urgent issues")
             :themes (map :theme urgent-clusters)})
      
      ;; Sentiment improvement
      (seq negative-clusters)
      (conj {:priority "medium"
             :category "sentiment-improvement"
             :title "Improve Player Satisfaction"
             :description (str "Address concerns in " (count negative-clusters) " negative themes")
             :themes (map :theme negative-clusters)})
      
      ;; Popular topic engagement
      (seq popular-topics)
      (conj {:priority "low"
             :category "community-engagement"
             :title "Engage with Popular Topics"
             :description "Build on popular discussion themes"
             :themes (map :theme popular-topics)}))))

(defn generate-summary
  "Generate a human-readable summary"
  [clusters message-count]
  (if (empty? clusters)
    "No significant themes identified in the messages."
    
    (let [top-theme (:theme (first clusters))
          urgent-count (count (filter #(> (:urgency %) 0.5) clusters))
          sentiment-dist (frequencies (map :sentiment clusters))]
      
      (str "Analyzed " message-count " messages revealing " (count clusters) " key themes. "
           "Primary discussion focus: " top-theme ". "
           (when (> urgent-count 0)
             (str urgent-count " themes require urgent attention. "))
           "Community sentiment is primarily " 
           (name (key (apply max-key val sentiment-dist))) "."))))

(defn generate-fallback-digest
  "Generate a basic digest when ML services are unavailable"
  [messages]
  (log/warn "Generating fallback digest without ML analysis")
  
  (let [message-count (count messages)
        urgent-keywords ["crash" "broken" "bug" "error" "cant play"]
        urgent-messages (->> messages
                            (filter #(some (fn [kw] 
                                            (str/includes? (str/lower-case (:content %)) kw))
                                          urgent-keywords))
                            (take 5))]
    
    {:timestamp (System/currentTimeMillis)
     :message-count message-count
     :clusters []
     :urgent-issues (map #(select-keys % [:content :author :timestamp]) urgent-messages)
     :sentiment {:score 50 :label "unknown" :confidence 0}
     :insights [{:type "service-limitation"
                 :insight "Limited analysis available"
                 :details "ML service unavailable, using basic keyword analysis"
                 :action "Check ML service connectivity"}]
     :recommendations []
     :summary (str "Basic analysis of " message-count " messages. ML services unavailable.")}))

;; Digest Storage Functions

(defn save-digest!
  "Save digest to database for historical tracking"
  [digest customer-id]
  (try
    (let [digest-record {:customer_id customer-id
                        :channel_id (:channel-id digest)
                        :message_count (:message-count digest)
                        :clusters (json/write-str (:clusters digest))
                        :urgent_issues (json/write-str (:urgent-issues digest))
                        :sentiment_score (get-in digest [:sentiment :score])}]
      (jdbc/execute! (db/get-db)
        (sql/format {:insert-into :digests
                     :values [digest-record]})))
    (catch Exception e
      (log/error e "Failed to save digest to database"))))

(defn get-recent-digests
  "Get recent digests for a customer"
  [customer-id limit]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:select :*
                   :from :digests
                   :where [:= :customer_id customer-id]
                   :order-by [[:generated_at :desc]]
                   :limit limit}))
    (catch Exception e
      (log/error e "Failed to get recent digests")
      [])))

;; Cached Digest Functions

(defn get-digest-cached
  "Get digest with caching"
  ([channel-id] (get-digest-cached channel-id 1000))
  ([channel-id message-limit]
   (let [cache-key (cache/cache-key channel-id "digest")]
     (cache/with-cache cache-key 300 ; 5 minute cache
       generate-digest message-limit channel-id))))

(defn get-digest-for-channel
  "Generate digest specifically for a channel with caching"
  [channel-id]
  (get-digest-cached channel-id 500))

(defn invalidate-digest-cache!
  "Clear digest cache for a channel"
  [channel-id]
  (cache/cache-del! (cache/cache-key channel-id "digest"))
  (log/info "Invalidated digest cache for channel:" channel-id))

(defn get-digest-summary
  "Get a lightweight digest summary"
  [channel-id]
  (let [cache-key (cache/cache-key channel-id "summary")]
    (cache/with-cache cache-key 120 ; 2 minute cache
      (fn []
        (let [messages (db/get-recent-messages 100 channel-id)
              message-count (count messages)
              urgent-count (count (find-urgent-issues messages []))]
          {:channel-id channel-id
           :message-count message-count
           :urgent-count urgent-count
           :last-updated (System/currentTimeMillis)})))))

(defn schedule-digest-generation!
  "Schedule digest generation for all active channels"
  []
  (log/info "Starting scheduled digest generation")
  ;; This would integrate with the customer system
  ;; For now, just clear caches to force regeneration
  (cache/cache-del! (cache/cache-key nil "digest"))
  (log/info "Cleared digest caches"))