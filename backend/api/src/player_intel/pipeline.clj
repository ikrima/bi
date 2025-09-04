(ns player-intel.pipeline
  (:require [player-intel.discord :as discord]
            [player-intel.db :as db]
            [player-intel.ml :as ml]
            [clojure.core.async :as async]
            [environ.core :refer [env]]
            [taoensso.timbre :as log]))

(def pipeline-state (atom {:running? false
                          :channels {}
                          :stats {:processed 0
                                 :errors 0
                                 :last-run nil}}))

(defn update-stats! [key]
  (swap! pipeline-state update-in [:stats key] 
         (if (= key :processed) inc inc)))

(defn start-embedding-processor!
  "Background processor for generating embeddings"
  []
  (async/go-loop []
    (try
      (let [messages (db/get-messages-without-embeddings 10)]
        (when (seq messages)
          (log/info "Processing" (count messages) "messages for embeddings")
          (when-let [processed-messages (ml/process-message-embeddings messages)]
            (doseq [msg processed-messages]
              (db/update-message-embedding! (:id msg) (:embedding msg)))
            (log/info "Updated embeddings for" (count processed-messages) "messages"))))
      (catch Exception e
        (log/error e "Error in embedding processor")
        (update-stats! :errors)))
    
    (async/<! (async/timeout 30000)) ; Process every 30 seconds
    (recur)))

(defn message-handler
  "Handler for processing individual messages"
  [message]
  (try
    (log/debug "Processing message:" (:id message))
    (db/insert-message! message)
    (update-stats! :processed)
    (swap! pipeline-state assoc-in [:stats :last-run] (java.time.Instant/now))
    (catch Exception e
      (log/error e "Error processing message:" message)
      (update-stats! :errors))))

(defn start-discord-pipeline!
  "Start the Discord message pipeline for a channel"
  [channel-id]
  (let [token (env :discord-bot-token)]
    (if-not token
      (log/error "DISCORD_BOT_TOKEN environment variable not set")
      (do
        (log/info "Starting Discord pipeline for channel:" channel-id)
        (let [pipeline-ch (discord/start-message-pipeline token channel-id message-handler)]
          (swap! pipeline-state assoc-in [:channels channel-id] pipeline-ch)
          pipeline-ch)))))

(defn stop-discord-pipeline!
  "Stop the Discord pipeline for a channel"
  [channel-id]
  (when-let [ch (get-in @pipeline-state [:channels channel-id])]
    (log/info "Stopping Discord pipeline for channel:" channel-id)
    (async/close! ch)
    (swap! pipeline-state update :channels dissoc channel-id)))

(defn start-all-pipelines!
  "Start all configured pipelines"
  []
  (log/info "Starting all pipelines...")
  (swap! pipeline-state assoc :running? true)
  
  ;; Start embedding processor
  (start-embedding-processor!)
  
  ;; Start Discord pipelines for configured channels
  (let [channels (env :discord-channels)]
    (when channels
      (doseq [channel-id (clojure.string/split channels #",")]
        (start-discord-pipeline! (clojure.string/trim channel-id)))))
  
  (log/info "All pipelines started"))

(defn stop-all-pipelines!
  "Stop all running pipelines"
  []
  (log/info "Stopping all pipelines...")
  (swap! pipeline-state assoc :running? false)
  
  ;; Close all Discord channels
  (doseq [[channel-id ch] (:channels @pipeline-state)]
    (async/close! ch))
  
  (swap! pipeline-state assoc :channels {})
  (log/info "All pipelines stopped"))

(defn get-pipeline-status
  "Get current pipeline status"
  []
  {:running? (:running? @pipeline-state)
   :active-channels (keys (:channels @pipeline-state))
   :stats (:stats @pipeline-state)
   :message-count (db/get-message-count)})