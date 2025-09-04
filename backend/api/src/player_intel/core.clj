(ns player-intel.core
  (:require [org.httpkit.server :as server]
            [reitit.ring :as ring]
            [muuntaja.core :as m]
            [reitit.ring.middleware.muuntaja :as muuntaja]
            [reitit.ring.middleware.parameters :as parameters]
            [player-intel.db :as db]
            [player-intel.ml :as ml]
            [player-intel.pipeline :as pipeline]
            [player-intel.migrations :as migrations]
            [player-intel.digest :as digest]
            [player-intel.cache :as cache]
            [player-intel.customers :as customers]
            [player-intel.scheduler :as scheduler]
            [player-intel.payments :as payments]
            [player-intel.billing :as billing]
            [taoensso.timbre :as log]
            [environ.core :refer [env]]))

;; Route handlers
(defn health-handler [_]
  (let [db-health (db/health-check)
        ml-health (ml/health-check)]
    {:status (if (and (:healthy db-health) (:healthy ml-health)) 200 503)
     :body {:status "healthy"
            :timestamp (System/currentTimeMillis)
            :services {:database db-health
                      :ml-service ml-health}}}))

(defn messages-handler [request]
  (let [params (:query-params request)
        limit (Integer/parseInt (get params "limit" "100"))
        channel-id (get params "channel_id")]
    {:status 200
     :body {:messages (db/get-recent-messages limit channel-id)
            :count (db/get-message-count)}}))

(defn pipeline-status-handler [_]
  {:status 200
   :body (pipeline/get-pipeline-status)})

(defn start-pipeline-handler [request]
  (let [channel-id (get-in request [:body-params :channel_id])]
    (if channel-id
      (do
        (pipeline/start-discord-pipeline! channel-id)
        {:status 200
         :body {:message "Pipeline started" :channel_id channel-id}})
      {:status 400
       :body {:error "channel_id required"}})))

(defn stop-pipeline-handler [request]
  (let [channel-id (get-in request [:body-params :channel_id])]
    (if channel-id
      (do
        (pipeline/stop-discord-pipeline! channel-id)
        {:status 200
         :body {:message "Pipeline stopped" :channel_id channel-id}})
      {:status 400
       :body {:error "channel_id required"}})))

;; Digest handlers
(defn digest-handler [request]
  (let [params (:query-params request)
        channel-id (get params "channel_id")
        limit (Integer/parseInt (get params "limit" "1000"))]
    {:status 200
     :body (digest/get-digest-cached channel-id limit)}))

(defn digest-summary-handler [request]
  (let [params (:query-params request)
        channel-id (get params "channel_id")]
    {:status 200
     :body (digest/get-digest-summary channel-id)}))

(defn invalidate-cache-handler [request]
  (let [channel-id (get-in request [:body-params :channel_id])]
    (if channel-id
      (do
        (digest/invalidate-digest-cache! channel-id)
        {:status 200
         :body {:message "Cache invalidated" :channel_id channel-id}})
      {:status 400
       :body {:error "channel_id required"}})))

(defn cache-stats-handler [_]
  {:status 200
   :body (cache/get-cache-stats)})

;; Customer onboarding handlers
(defn signup-handler [request]
  (let [body (:body-params request)
        email (:email body)
        discord-channel-id (:discord_channel_id body)
        game-name (:game_name body)]
    (if (and email discord-channel-id)
      (let [result (customers/create-trial-account! email discord-channel-id game-name)]
        {:status (if (:success result) 201 400)
         :body result})
      {:status 400
       :body {:success false 
              :message "Email and Discord channel ID are required"}})))

(defn customer-status-handler [request]
  (let [customer-id (get-in request [:path-params :customer-id])]
    (if customer-id
      (if-let [summary (customers/get-customer-summary (java.util.UUID/fromString customer-id))]
        {:status 200 :body summary}
        {:status 404 :body {:error "Customer not found"}})
      {:status 400 :body {:error "Customer ID required"}})))

(defn validate-channel-handler [request]
  (let [body (:body-params request)
        channel-id (:channel_id body)
        bot-token (env :discord-bot-token)]
    (if channel-id
      (let [result (customers/validate-discord-channel channel-id bot-token)]
        {:status 200 :body result})
      {:status 400 :body {:error "Channel ID required"}})))

(defn customer-metrics-handler [_]
  {:status 200
   :body (customers/get-customer-metrics)})

;; Scheduler handlers
(defn scheduler-status-handler [_]
  {:status 200
   :body (scheduler/get-scheduler-status)})

(defn run-digest-handler [request]
  (let [body (:body-params request)
        customer-email (:customer_email body)]
    (if customer-email
      (let [result (scheduler/run-digest-for-customer customer-email)]
        {:status 200 :body result})
      (let [result (scheduler/run-digest-for-all-customers)]
        {:status 200 :body result}))))

(defn scheduler-health-handler [_]
  {:status 200
   :body (scheduler/scheduler-health-check)})

;; Billing and payments handlers
(defn plans-handler [_]
  {:status 200
   :body {:plans (billing/get-plan-comparison)}})

(defn create-checkout-handler [request]
  (let [body (:body-params request)
        customer-email (:customer_email body)
        plan (:plan body)
        success-url (get body :success_url "https://app.playerintel.ai/success")
        cancel-url (get body :cancel_url "https://app.playerintel.ai/cancel")]
    (if (and customer-email plan)
      (let [result (payments/create-checkout-session customer-email (keyword plan) success-url cancel-url)]
        {:status (if (:success result) 200 400)
         :body result})
      {:status 400
       :body {:success false :error "Customer email and plan are required"}})))

(defn billing-info-handler [request]
  (let [customer-id (get-in request [:path-params :customer-id])]
    (if customer-id
      (if-let [billing-info (billing/get-customer-billing-info (java.util.UUID/fromString customer-id))]
        {:status 200 :body billing-info}
        {:status 404 :body {:error "Customer not found"}})
      {:status 400 :body {:error "Customer ID required"}})))

(defn upgrade-handler [request]
  (let [body (:body-params request)
        customer-id (:customer_id body)
        new-plan (:plan body)]
    (if (and customer-id new-plan)
      (let [result (billing/initiate-plan-change (java.util.UUID/fromString customer-id) (keyword new-plan))]
        {:status (if (:success result) 200 400)
         :body result})
      {:status 400
       :body {:error "Customer ID and plan are required"}})))

(defn cancel-subscription-handler [request]
  (let [body (:body-params request)
        customer-id (:customer_id body)
        reason (get body :reason "customer-request")]
    (if customer-id
      (let [result (billing/cancel-subscription (java.util.UUID/fromString customer-id) reason)]
        {:status 200 :body result})
      {:status 400 :body {:error "Customer ID required"}})))

(defn revenue-report-handler [_]
  {:status 200
   :body (billing/get-revenue-report)})

(defn revenue-trend-handler [_]
  {:status 200
   :body {:trend (billing/get-monthly-revenue-trend)}})

(defn stripe-webhook-handler [request]
  (let [body (:body request)
        signature (get-in request [:headers "stripe-signature"])
        webhook-secret (env :stripe-webhook-secret)]
    (if (and body signature webhook-secret)
      (if (payments/verify-stripe-signature body signature webhook-secret)
        (try
          (let [event (json/read-str body :key-fn keyword)
                result (payments/handle-webhook-event event)]
            {:status 200 :body result})
          (catch Exception e
            (log/error e "Error processing webhook")
            {:status 400 :body {:error "Invalid webhook data"}}))
        {:status 400 :body {:error "Invalid signature"}})
      {:status 400 :body {:error "Missing required webhook data"}})))

;; Routes
(def routes
  [["/health" {:get health-handler}]
   ["/api"
    ["/messages" {:get messages-handler}]
    ["/pipeline"
     ["/status" {:get pipeline-status-handler}]
     ["/start" {:post start-pipeline-handler}]
     ["/stop" {:post stop-pipeline-handler}]]
    ["/digest" {:get digest-handler}]
    ["/digest-summary" {:get digest-summary-handler}]
    ["/cache"
     ["/stats" {:get cache-stats-handler}]
     ["/invalidate" {:post invalidate-cache-handler}]]
    ["/customers"
     ["/signup" {:post signup-handler}]
     ["/metrics" {:get customer-metrics-handler}]
     ["/validate-channel" {:post validate-channel-handler}]
     ["/:customer-id/status" {:get customer-status-handler}]]
    ["/scheduler"
     ["/status" {:get scheduler-status-handler}]
     ["/health" {:get scheduler-health-handler}]
     ["/run-digest" {:post run-digest-handler}]]
    ["/billing"
     ["/plans" {:get plans-handler}]
     ["/checkout" {:post create-checkout-handler}]
     ["/upgrade" {:post upgrade-handler}]
     ["/cancel" {:post cancel-subscription-handler}]
     ["/:customer-id/info" {:get billing-info-handler}]]
    ["/revenue"
     ["/report" {:get revenue-report-handler}]
     ["/trend" {:get revenue-trend-handler}]]
    ["/webhooks"
     ["/stripe" {:post stripe-webhook-handler}]]]])

(def app
  (ring/ring-handler
    (ring/router routes
      {:data {:muuntaja m/instance
              :middleware [muuntaja/format-middleware
                          parameters/parameters-middleware]}})
    (ring/create-default-handler)))

(defn initialize-system! []
  "Initialize the system on startup"
  (log/info "Initializing Player Intelligence system...")
  
  ;; Run database migrations
  (try
    (migrations/run-migrations!)
    (billing/create-billing-tables!)
    (log/info "Database migrations completed")
    (catch Exception e
      (log/error e "Failed to run migrations")))
  
  ;; Start pipelines if configured
  (when (env :auto-start-pipelines)
    (pipeline/start-all-pipelines!))
  
  ;; Start scheduler if configured
  (when (env :auto-start-scheduler)
    (scheduler/start-scheduler!))
  
  (log/info "System initialization completed"))

(defn -main [& args]
  (log/info "Starting Player Intelligence API on port 3000...")
  (initialize-system!)
  (server/run-server app {:port 3000})
  (log/info "Server started on port 3000"))