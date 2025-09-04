(ns player-intel.analytics
  (:require [player-intel.db :as db]
            [player-intel.ml :as ml]
            [honey.sql :as sql]
            [next.jdbc :as jdbc]
            [taoensso.timbre :as log]
            [clj-time.core :as time]
            [clj-time.coerce :as coerce]
            [clojure.data.json :as json]))

;; --- Player Lifecycle Analytics ---
(defn create-analytics-tables!
  "Create analytics-related database tables"
  []
  (try
    ;; Player personas table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS player_personas ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "persona_name VARCHAR(100),"
            "characteristics JSONB,"
            "player_count INTEGER,"
            "churn_risk FLOAT,"
            "engagement_score FLOAT,"
            "coordinates JSONB,"
            "generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "expires_at TIMESTAMP)")])
    
    ;; Predictions log table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS predictions_log ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "prediction_type VARCHAR(50),"
            "input_data JSONB,"
            "prediction_result JSONB,"
            "confidence FLOAT,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
            "actual_outcome JSONB,"
            "outcome_recorded_at TIMESTAMP)")])
    
    ;; Customer analytics table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS customer_analytics ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "metric_name VARCHAR(100),"
            "metric_value FLOAT,"
            "measurement_date DATE,"
            "metadata JSONB,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")])
    
    ;; Trend alerts table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS trend_alerts ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "alert_type VARCHAR(50),"
            "severity VARCHAR(20),"
            "title VARCHAR(200),"
            "description TEXT,"
            "data JSONB,"
            "acknowledged BOOLEAN DEFAULT FALSE,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")])
    
    (log/info "Analytics tables created successfully")
    (catch Exception e
      (log/error e "Failed to create analytics tables"))))

(defn save-customer-personas!
  "Save discovered personas for a customer"
  [customer-id personas]
  (try
    ;; Clear existing personas
    (jdbc/execute! (db/get-db)
      (sql/format {:delete-from :player_personas
                   :where [:= :customer_id customer-id]}))
    
    ;; Insert new personas
    (doseq [persona personas]
      (jdbc/execute! (db/get-db)
        (sql/format {:insert-into :player_personas
                     :values [{:customer_id customer-id
                               :persona_name (:name persona)
                               :characteristics (json/write-str (:characteristics persona))
                               :player_count (:size persona)
                               :churn_risk (:churn_risk persona 0.0)
                               :engagement_score (:engagement_score persona 0.5)
                               :coordinates (json/write-str (:coordinates persona))
                               :expires_at (coerce/to-timestamp 
                                           (time/plus (time/now) (time/days 7)))}]})))
    
    (log/info "Saved" (count personas) "personas for customer" customer-id)
    (catch Exception e
      (log/error e "Failed to save personas for customer" customer-id))))

(defn get-customer-personas
  "Get saved personas for a customer"
  [customer-id]
  (try
    (let [personas (jdbc/execute! (db/get-db)
                    (sql/format {:select :*
                                :from :player_personas
                                :where [:and
                                       [:= :customer_id customer-id]
                                       [:> :expires_at (coerce/to-timestamp (time/now))]]
                                :order-by [[:generated_at :desc]]}))]
      (map (fn [persona]
             {:id (:id persona)
              :name (:persona_name persona)
              :characteristics (json/read-str (:characteristics persona) :key-fn keyword)
              :size (:player_count persona)
              :churn_risk (:churn_risk persona)
              :engagement_score (:engagement_score persona)
              :coordinates (json/read-str (:coordinates persona) :key-fn keyword)
              :generated_at (:generated_at persona)})
           personas))
    (catch Exception e
      (log/error e "Failed to get personas for customer" customer-id)
      [])))

(defn log-prediction!
  "Log a prediction for future accuracy tracking"
  [customer-id prediction-type input-data prediction-result confidence]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:insert-into :predictions_log
                   :values [{:customer_id customer-id
                             :prediction_type prediction-type
                             :input_data (json/write-str input-data)
                             :prediction_result (json/write-str prediction-result)
                             :confidence confidence}]}))
    (log/info "Logged prediction for customer" customer-id "type:" prediction-type)
    (catch Exception e
      (log/error e "Failed to log prediction"))))

(defn get-customer-historical-reactions
  "Get historical reactions for prediction training"
  [customer-id]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:select :*
                   :from :predictions_log
                   :where [:and
                          [:= :customer_id customer-id]
                          [:is-not :actual_outcome nil]]
                   :order-by [[:created_at :desc]]
                   :limit 50}))
    (catch Exception e
      (log/error e "Failed to get historical reactions for customer" customer-id)
      [])))

(defn record-metric!
  "Record a customer analytics metric"
  [customer-id metric-name value & [metadata]]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:insert-into :customer_analytics
                   :values [{:customer_id customer-id
                             :metric_name metric-name
                             :metric_value value
                             :measurement_date (coerce/to-sql-date (time/now))
                             :metadata (when metadata (json/write-str metadata))}]}))
    (catch Exception e
      (log/error e "Failed to record metric" metric-name "for customer" customer-id))))

(defn get-customer-metrics-history
  "Get historical metrics for a customer"
  [customer-id metric-name days-back]
  (try
    (let [start-date (coerce/to-sql-date (time/minus (time/now) (time/days days-back)))]
      (jdbc/execute! (db/get-db)
        (sql/format {:select [:measurement_date :metric_value :metadata]
                     :from :customer_analytics
                     :where [:and
                            [:= :customer_id customer-id]
                            [:= :metric_name metric-name]
                            [:>= :measurement_date start-date]]
                     :order-by [[:measurement_date :asc]]})))
    (catch Exception e
      (log/error e "Failed to get metrics history for customer" customer-id)
      [])))

(defn calculate-player-lifecycle-metrics
  "Calculate comprehensive player lifecycle metrics"
  [customer-id]
  (try
    (let [now (time/now)
          month-ago (time/minus now (time/months 1))
          week-ago (time/minus now (time/weeks 1))
          
          ;; Get message activity by user
          messages-month (db/get-customer-messages-since customer-id (coerce/to-timestamp month-ago))
          messages-week (db/get-customer-messages-since customer-id (coerce/to-timestamp week-ago))
          
          ;; Group by author
          users-month (group-by :author messages-month)
          users-week (group-by :author messages-week)
          
          ;; Calculate metrics
          total-users-month (count users-month)
          active-users-week (count users-week)
          retention-rate (if (> total-users-month 0)
                          (/ active-users-week total-users-month)
                          0.0)
          
          ;; Calculate engagement distribution
          engagement-scores (map (fn [[author msgs]]
                                  (let [message-count (count msgs)
                                        unique-days (count (distinct (map #(coerce/to-date (:timestamp %)) msgs)))]
                                    {:author author
                                     :messages message-count
                                     :active_days unique-days
                                     :engagement_score (+ (* message-count 0.7) (* unique-days 0.3))}))
                                users-month)
          
          avg-engagement (if (seq engagement-scores)
                          (/ (reduce + (map :engagement_score engagement-scores)) 
                             (count engagement-scores))
                          0.0)
          
          ;; Identify player segments
          high-engagement (filter #(> (:engagement_score %) (* avg-engagement 1.5)) engagement-scores)
          low-engagement (filter #(< (:engagement_score %) (* avg-engagement 0.5)) engagement-scores)
          
          metrics {:total_users total-users-month
                  :active_users_week active-users-week
                  :retention_rate retention-rate
                  :avg_engagement avg-engagement
                  :high_engagement_users (count high-engagement)
                  :low_engagement_users (count low-engagement)
                  :churn_risk_score (max 0.0 (min 1.0 (- 1.0 retention-rate)))
                  :generated_at (coerce/to-timestamp now)}]
      
      ;; Record metrics in database
      (doseq [[metric value] metrics]
        (when (number? value)
          (record-metric! customer-id (name metric) value)))
      
      metrics)
    
    (catch Exception e
      (log/error e "Failed to calculate lifecycle metrics for customer" customer-id)
      {:error (.getMessage e)})))

(defn create-trend-alert!
  "Create a trend alert for customer"
  [customer-id alert-type severity title description data]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:insert-into :trend_alerts
                   :values [{:customer_id customer-id
                             :alert_type alert-type
                             :severity severity
                             :title title
                             :description description
                             :data (json/write-str data)}]}))
    (log/info "Created trend alert for customer" customer-id "type:" alert-type)
    (catch Exception e
      (log/error e "Failed to create trend alert"))))

(defn get-customer-alerts
  "Get unacknowledged alerts for customer"
  [customer-id & {:keys [limit] :or {limit 10}}]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:select :*
                   :from :trend_alerts
                   :where [:and
                          [:= :customer_id customer-id]
                          [:= :acknowledged false]]
                   :order-by [[:created_at :desc]]
                   :limit limit}))
    (catch Exception e
      (log/error e "Failed to get alerts for customer" customer-id)
      [])))

(defn acknowledge-alert!
  "Mark alert as acknowledged"
  [alert-id]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:update :trend_alerts
                   :set {:acknowledged true}
                   :where [:= :id alert-id]}))
    (log/info "Acknowledged alert" alert-id)
    (catch Exception e
      (log/error e "Failed to acknowledge alert" alert-id))))

(defn run-anomaly-detection
  "Run anomaly detection on customer metrics"
  [customer-id]
  (try
    (let [metrics-to-check ["sentiment_score" "message_count" "active_users" "engagement_score"]
          anomalies []]
      
      (doseq [metric metrics-to-check]
        (let [history (get-customer-metrics-history customer-id metric 30)
              values (map :metric_value history)]
          
          (when (> (count values) 5)
            (let [ml-result (ml/detect-trends-and-anomalies {metric values})]
              (when (:success ml-result)
                (let [metric-analysis (get-in ml-result [:data metric])]
                  (when (and metric-analysis (seq (:anomalies metric-analysis)))
                    ;; Create alert for anomalies
                    (create-trend-alert! customer-id
                                       "anomaly-detected"
                                       "medium"
                                       (str "Anomaly detected in " metric)
                                       (str "Unusual values detected in " metric " over the past 30 days")
                                       {:metric metric
                                        :anomalies (:anomalies metric-analysis)
                                        :trend (:trend metric-analysis)}))))))))
      
      (log/info "Completed anomaly detection for customer" customer-id))
    
    (catch Exception e
      (log/error e "Failed to run anomaly detection for customer" customer-id))))

;; --- Batch Analytics Jobs ---
(defn run-daily-analytics
  "Run daily analytics for all active customers"
  []
  (try
    (let [active-customers (db/get-active-customers)]
      (log/info "Running daily analytics for" (count active-customers) "customers")
      
      (doseq [customer active-customers]
        (try
          ;; Calculate lifecycle metrics
          (calculate-player-lifecycle-metrics (:id customer))
          
          ;; Run anomaly detection
          (run-anomaly-detection (:id customer))
          
          (log/info "Completed analytics for customer" (:id customer))
          
          (catch Exception e
            (log/error e "Failed analytics for customer" (:id customer)))))
      
      (log/info "Completed daily analytics run"))
    
    (catch Exception e
      (log/error e "Failed to run daily analytics"))))

;; --- Analytics API ---
(defn analytics-routes []
  [["/api/analytics"
    {:middleware [auth/customer-middleware]}
    
    ["/lifecycle"
     {:get {:handler (fn [req]
                      (let [customer-id (get-in req [:customer :id])
                            metrics (calculate-player-lifecycle-metrics customer-id)]
                        {:status 200
                         :body metrics}))}}]
    
    ["/alerts"
     {:get {:handler (fn [req]
                      (let [customer-id (get-in req [:customer :id])
                            alerts (get-customer-alerts customer-id)]
                        {:status 200
                         :body {:alerts alerts}}))}}]
    
    ["/alerts/:alert-id/acknowledge"
     {:post {:handler (fn [req]
                       (let [alert-id (get-in req [:path-params :alert-id])]
                         (acknowledge-alert! alert-id)
                         {:status 200
                          :body {:success true}}))}}]
    
    ["/metrics/:metric-name"
     {:get {:parameters {:query {:days integer?}}
            :handler (fn [req]
                      (let [customer-id (get-in req [:customer :id])
                            metric-name (get-in req [:path-params :metric-name])
                            days (get-in req [:query-params :days] 30)
                            history (get-customer-metrics-history customer-id metric-name days)]
                        {:status 200
                         :body {:metric metric-name
                               :data history}}))}}]]])