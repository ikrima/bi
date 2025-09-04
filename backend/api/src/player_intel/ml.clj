(ns player-intel.ml
  (:require [clj-http.client :as http]
            [clojure.data.json :as json]
            [environ.core :refer [env]]
            [taoensso.timbre :as log]))

(def ml-service-url (or (env :ml-service-url) 
                        "http://localhost:8000"))

(defn generate-embeddings
  "Call Python ML service to generate embeddings"
  [texts]
  (try
    (let [response (http/post (str ml-service-url "/embed")
                             {:body (json/write-str {:texts texts})
                              :headers {"Content-Type" "application/json"}
                              :as :json
                              :throw-exceptions false
                              :socket-timeout 30000
                              :connection-timeout 10000})]
      (if (= 200 (:status response))
        (get-in response [:body :embeddings])
        (do
          (log/error "ML service error:" (:status response) (:body response))
          nil)))
    (catch Exception e
      (log/error e "Failed to generate embeddings")
      nil)))

(defn cluster-messages
  "Call ML service to cluster messages"
  [embeddings messages]
  (try
    (let [response (http/post (str ml-service-url "/cluster")
                             {:body (json/write-str {:embeddings embeddings
                                                     :messages messages})
                              :headers {"Content-Type" "application/json"}
                              :as :json
                              :throw-exceptions false
                              :socket-timeout 60000
                              :connection-timeout 10000})]
      (if (= 200 (:status response))
        (get-in response [:body :clusters])
        (do
          (log/error "ML service clustering error:" (:status response) (:body response))
          nil)))
    (catch Exception e
      (log/error e "Failed to cluster messages")
      nil)))

(defn health-check []
  "Check if ML service is healthy"
  (try
    (let [response (http/get (str ml-service-url "/health")
                            {:as :json
                             :throw-exceptions false
                             :socket-timeout 5000
                             :connection-timeout 5000})]
      (if (= 200 (:status response))
        {:healthy true :message "ML service is healthy"}
        {:healthy false :message (str "ML service returned status: " (:status response))}))
    (catch Exception e
      (log/error e "ML service health check failed")
      {:healthy false :message (.getMessage e)})))

(defn process-message-embeddings
  "Process a batch of messages to generate embeddings"
  [messages]
  (when (seq messages)
    (log/info "Processing" (count messages) "messages for embeddings")
    (let [texts (map :content messages)
          embeddings (generate-embeddings texts)]
      (when embeddings
        (map (fn [message embedding]
               (assoc message :embedding embedding))
             messages embeddings)))))

(defn discover-personas
  "Discover player personas from user behavioral data"
  [user-data]
  (try
    (let [response (http/post (str ml-service-url "/personas/discover")
                             {:body (json/write-str {:user_data user-data})
                              :headers {"Content-Type" "application/json"}
                              :as :json
                              :throw-exceptions false
                              :socket-timeout 120000
                              :connection-timeout 10000})]
      (if (= 200 (:status response))
        (:body response)
        (do
          (log/error "Persona discovery error:" (:status response) (:body response))
          {:success false :error "ML service error"})))
    (catch Exception e
      (log/error e "Failed to discover personas")
      {:success false :error (.getMessage e)})))

(defn predict-change-impact
  "Predict impact of game changes on player community"
  [prediction-request]
  (try
    (let [response (http/post (str ml-service-url "/predict/change-impact")
                             {:body (json/write-str prediction-request)
                              :headers {"Content-Type" "application/json"}
                              :as :json
                              :throw-exceptions false
                              :socket-timeout 60000
                              :connection-timeout 10000})]
      (if (= 200 (:status response))
        (:body response)
        (do
          (log/error "Prediction error:" (:status response) (:body response))
          {:success false :error "ML service error"})))
    (catch Exception e
      (log/error e "Failed to predict change impact")
      {:success false :error (.getMessage e)})))

(defn analyze-competitor-sentiment
  "Analyze sentiment in competitor mentions"
  [messages]
  (try
    (let [response (http/post (str ml-service-url "/competitive/analyze")
                             {:body (json/write-str {:messages messages})
                              :headers {"Content-Type" "application/json"}
                              :as :json
                              :throw-exceptions false
                              :socket-timeout 30000
                              :connection-timeout 10000})]
      (if (= 200 (:status response))
        (:body response)
        (do
          (log/error "Competitor analysis error:" (:status response) (:body response))
          {:success false :error "ML service error"})))
    (catch Exception e
      (log/error e "Failed to analyze competitor sentiment")
      {:success false :error (.getMessage e)})))

(defn detect-trends-and-anomalies
  "Detect trends and anomalies in time series data"
  [metrics-data]
  (try
    (let [response (http/post (str ml-service-url "/trends/detect")
                             {:body (json/write-str metrics-data)
                              :headers {"Content-Type" "application/json"}
                              :as :json
                              :throw-exceptions false
                              :socket-timeout 30000
                              :connection-timeout 10000})]
      (if (= 200 (:status response))
        (:body response)
        (do
          (log/error "Trend detection error:" (:status response) (:body response))
          {:success false :error "ML service error"})))
    (catch Exception e
      (log/error e "Failed to detect trends and anomalies")
      {:success false :error (.getMessage e)})))

(defn get-current-personas
  "Get current personas for a customer (placeholder)"
  []
  ;; This would typically fetch from a cache or database
  ;; For now, return empty list
  [])