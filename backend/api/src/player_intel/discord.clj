(ns player-intel.discord
  (:require [clj-http.client :as http]
            [clojure.data.json :as json]
            [clojure.core.async :as async]
            [taoensso.timbre :as log]))

(def discord-api "https://discord.com/api/v10")

(defn fetch-channel-messages
  "Fetch messages from a Discord channel"
  [token channel-id & {:keys [limit] :or {limit 100}}]
  (try
    (let [response (http/get (str discord-api "/channels/" channel-id "/messages")
                            {:headers {"Authorization" (str "Bot " token)}
                             :query-params {"limit" limit}
                             :as :json
                             :throw-exceptions false})]
      (if (= 200 (:status response))
        (:body response)
        (do
          (log/error "Failed to fetch Discord messages:" (:status response) (:body response))
          [])))
    (catch Exception e
      (log/error e "Error fetching Discord messages")
      [])))

(defn process-messages
  "Transform Discord messages into our data structure"
  [messages]
  (map (fn [msg]
         {:id (:id msg)
          :content (:content msg)
          :author (get-in msg [:author :username])
          :timestamp (:timestamp msg)
          :channel-id (:channel_id msg)
          :created-at (java.time.Instant/now)})
       (filter #(and (:content %) 
                     (not (clojure.string/blank? (:content %))))
               messages)))

(defn start-message-pipeline
  "Start async pipeline for processing Discord messages"
  [token channel-id message-handler]
  (let [messages-ch (async/chan 100)]
    (async/go-loop []
      (log/info "Fetching messages from Discord channel:" channel-id)
      (when-let [messages (fetch-channel-messages token channel-id :limit 50)]
        (let [processed (process-messages messages)]
          (log/info "Processed" (count processed) "messages")
          (doseq [msg processed]
            (async/>! messages-ch msg))))
      (async/<! (async/timeout 60000)) ; Poll every minute
      (recur))
    
    ;; Message processor
    (async/go-loop []
      (when-let [msg (async/<! messages-ch)]
        (try
          (message-handler msg)
          (catch Exception e
            (log/error e "Error processing message:" msg)))
        (recur)))
    
    messages-ch))

(defn validate-discord-config
  "Validate Discord bot token and channel ID"
  [token channel-id]
  (if (and token channel-id)
    (let [test-messages (fetch-channel-messages token channel-id :limit 1)]
      (if (seq test-messages)
        {:valid true :message "Discord integration configured successfully"}
        {:valid false :message "Unable to fetch messages. Check token and channel ID."}))
    {:valid false :message "Missing Discord token or channel ID"}))