(ns player-intel.migrations
  (:require [player-intel.db :as db]
            [taoensso.timbre :as log]))

(def migrations
  [{:id "001-create-messages-table"
    :description "Create messages table with basic fields"
    :up-fn (fn []
             (db/execute!
               [(str "CREATE TABLE IF NOT EXISTS messages ("
                     "id VARCHAR(255) PRIMARY KEY,"
                     "content TEXT NOT NULL,"
                     "author VARCHAR(255),"
                     "channel_id VARCHAR(255),"
                     "timestamp TIMESTAMP,"
                     "embedding JSONB,"
                     "cluster_id INTEGER,"
                     "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")]))}
   
   {:id "002-create-customers-table"
    :description "Create customers table for user management"
    :up-fn (fn []
             (db/execute!
               [(str "CREATE TABLE IF NOT EXISTS customers ("
                     "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
                     "email VARCHAR(255) UNIQUE NOT NULL,"
                     "discord_channel_id VARCHAR(255),"
                     "plan VARCHAR(50) DEFAULT 'trial',"
                     "stripe_subscription_id VARCHAR(255),"
                     "trial_ends_at TIMESTAMP,"
                     "upgraded_at TIMESTAMP,"
                     "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")]))}
   
   {:id "003-create-digests-table" 
    :description "Create digests table for storing generated insights"
    :up-fn (fn []
             (db/execute!
               [(str "CREATE TABLE IF NOT EXISTS digests ("
                     "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
                     "customer_id UUID REFERENCES customers(id),"
                     "channel_id VARCHAR(255),"
                     "message_count INTEGER,"
                     "clusters JSONB,"
                     "urgent_issues JSONB,"
                     "sentiment_score INTEGER,"
                     "generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")]))}])

(defn create-migrations-table! []
  (db/execute!
    [(str "CREATE TABLE IF NOT EXISTS schema_migrations ("
          "id VARCHAR(255) PRIMARY KEY,"
          "description TEXT,"
          "applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")]))

(defn get-applied-migrations []
  (try
    (set (map :id (db/execute! ["SELECT id FROM schema_migrations"])))
    (catch Exception e
      (log/info "Migrations table doesn't exist yet, creating it...")
      (create-migrations-table!)
      #{})))

(defn mark-migration-applied! [migration-id description]
  (db/execute!
    [(str "INSERT INTO schema_migrations (id, description) VALUES (?, ?)")
     migration-id description]))

(defn run-migrations! []
  (log/info "Running database migrations...")
  (let [applied (get-applied-migrations)]
    (doseq [{:keys [id description up-fn]} migrations]
      (if (contains? applied id)
        (log/info "Migration already applied:" id)
        (do
          (log/info "Applying migration:" id "-" description)
          (try
            (up-fn)
            (mark-migration-applied! id description)
            (log/info "Successfully applied migration:" id)
            (catch Exception e
              (log/error e "Failed to apply migration:" id)
              (throw e)))))))
  (log/info "All migrations completed."))