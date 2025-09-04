(ns player-intel.db
  (:require [next.jdbc :as jdbc]
            [honey.sql :as sql]
            [environ.core :refer [env]]
            [clojure.data.json :as json]
            [taoensso.timbre :as log])
  (:import [java.time Instant]))

(def db-spec
  {:dbtype "postgresql"
   :dbname (env :db-name "player_intel")
   :host (env :db-host "localhost")
   :port (env :db-port 5432)
   :user (env :db-user "admin")
   :password (env :db-password "secret")})

(def ds (delay (jdbc/get-datasource db-spec)))

(defn get-db [] @ds)

(defn execute! [query]
  (try
    (jdbc/execute! (get-db) query)
    (catch Exception e
      (log/error e "Database error executing query:" query)
      nil)))

(defn create-tables! []
  (log/info "Creating database tables...")
  (execute!
    [(str "CREATE TABLE IF NOT EXISTS messages ("
          "id VARCHAR(255) PRIMARY KEY,"
          "content TEXT NOT NULL,"
          "author VARCHAR(255),"
          "channel_id VARCHAR(255),"
          "timestamp TIMESTAMP,"
          "embedding JSONB,"
          "cluster_id INTEGER,"
          "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")]))

(defn insert-message! [message]
  (try
    (execute!
      (sql/format {:insert-into :messages
                   :values [message]
                   :on-conflict :id
                   :do-nothing true}))
    (log/debug "Inserted message:" (:id message))
    (catch Exception e
      (log/error e "Failed to insert message:" message))))

(defn get-recent-messages 
  ([limit] (get-recent-messages limit nil))
  ([limit channel-id]
   (try
     (let [query (cond-> {:select :*
                         :from :messages
                         :order-by [[:created_at :desc]]
                         :limit limit}
                   channel-id (assoc :where [:= :channel_id channel-id]))]
       (jdbc/execute! (get-db) (sql/format query)))
     (catch Exception e
       (log/error e "Failed to get recent messages")
       []))))

(defn get-messages-without-embeddings [limit]
  (try
    (jdbc/execute! (get-db)
      (sql/format {:select :*
                   :from :messages  
                   :where [:is :embedding nil]
                   :limit limit}))
    (catch Exception e
      (log/error e "Failed to get messages without embeddings")
      [])))

(defn update-message-embedding! [message-id embedding]
  (try
    (execute!
      (sql/format {:update :messages
                   :set {:embedding (json/write-str embedding)}
                   :where [:= :id message-id]}))
    (catch Exception e
      (log/error e "Failed to update embedding for message:" message-id))))

(defn get-message-count []
  (try
    (-> (jdbc/execute-one! (get-db)
          (sql/format {:select [[:%count.* :count]]
                       :from :messages}))
        :count)
    (catch Exception e
      (log/error e "Failed to get message count")
      0)))

(defn health-check []
  (try
    (jdbc/execute-one! (get-db) ["SELECT 1 as status"])
    {:healthy true :message "Database connection successful"}
    (catch Exception e
      (log/error e "Database health check failed")
      {:healthy false :message (.getMessage e)})))

(defn get-customer-messages
  "Get messages for a specific customer (via channel mapping)"
  [customer-id limit]
  (try
    ;; For now, get all messages - in production would join with customer->channel mapping
    (jdbc/execute! (get-db)
      (sql/format {:select :*
                   :from :messages
                   :order-by [[:created_at :desc]]
                   :limit limit}))
    (catch Exception e
      (log/error e "Failed to get customer messages for" customer-id)
      [])))

(defn get-customer-messages-since
  "Get customer messages since a specific timestamp"
  [customer-id since-timestamp]
  (try
    (jdbc/execute! (get-db)
      (sql/format {:select :*
                   :from :messages
                   :where [:>= :timestamp since-timestamp]
                   :order-by [[:timestamp :desc]]}))
    (catch Exception e
      (log/error e "Failed to get customer messages since timestamp")
      [])))

(defn get-latest-digest
  "Get the latest digest for a customer"
  [customer-id]
  (try
    ;; This would typically come from a digests table
    ;; For now, return a placeholder
    {:sentiment 65
     :message_count 450
     :generated_at (System/currentTimeMillis)}
    (catch Exception e
      (log/error e "Failed to get latest digest for customer" customer-id)
      nil)))

(defn get-customer-digests
  "Get recent digests for a customer"
  [customer-id limit]
  (try
    ;; Placeholder - would come from digests table
    (repeatedly limit 
                #(hash-map :sentiment (+ 40 (rand-int 40))
                          :message_count (+ 200 (rand-int 300))
                          :generated_at (- (System/currentTimeMillis) 
                                          (* (rand-int 30) 86400000))))
    (catch Exception e
      (log/error e "Failed to get customer digests")
      [])))

(defn get-active-customers
  "Get all active customers"
  []
  (try
    ;; Placeholder - would come from customers table
    [{:id "customer-1" :email "test@example.com"}
     {:id "customer-2" :email "test2@example.com"}]
    (catch Exception e
      (log/error e "Failed to get active customers")
      [])))