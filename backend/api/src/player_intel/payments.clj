(ns player-intel.payments
  (:require [clj-http.client :as http]
            [clojure.data.json :as json]
            [clojure.string :as str]
            [environ.core :refer [env]]
            [taoensso.timbre :as log]
            [player-intel.customers :as customers]
            [player-intel.email :as email]
            [clj-time.core :as time]
            [clj-time.coerce :as coerce])
  (:import [java.security.MessageDigest]
           [javax.crypto.Mac]
           [javax.crypto.spec.SecretKeySpec]
           [java.util Base64]))

;; Stripe configuration
(def stripe-config
  {:api-key (env :stripe-secret-key)
   :publishable-key (env :stripe-publishable-key)
   :webhook-secret (env :stripe-webhook-secret)
   :api-url "https://api.stripe.com/v1"})

;; Pricing configuration
(def pricing-plans
  {:basic {:price-id (env :stripe-basic-price-id "price_basic")
           :amount 2499  ; $24.99
           :name "Basic Plan"
           :features ["Daily community digests"
                     "Sentiment analysis"
                     "Theme clustering"
                     "Urgent issue detection"
                     "Email support"]}
   
   :pro {:price-id (env :stripe-pro-price-id "price_pro")
         :amount 4999  ; $49.99
         :name "Pro Plan"
         :features ["Everything in Basic"
                   "Real-time alerts"
                   "Advanced analytics"
                   "API access"
                   "Priority support"
                   "Custom branding"]}})

;; --- Stripe API Helpers ---
(defn stripe-request
  "Make authenticated request to Stripe API"
  [method endpoint params]
  (let [auth-header (str "Bearer " (:api-key stripe-config))
        url (str (:api-url stripe-config) endpoint)]
    (try
      (let [response (case method
                       :get (http/get url 
                                     {:headers {"Authorization" auth-header}
                                      :query-params params
                                      :as :json})
                       :post (http/post url
                                       {:headers {"Authorization" auth-header
                                                 "Content-Type" "application/x-www-form-urlencoded"}
                                        :form-params params
                                        :as :json})
                       :delete (http/delete url
                                           {:headers {"Authorization" auth-header}
                                            :as :json}))]
        (if (= 200 (:status response))
          {:success true :data (:body response)}
          {:success false :error (get-in response [:body :error])}))
      (catch Exception e
        (log/error e "Stripe API request failed:" method endpoint)
        {:success false :error (.getMessage e)}))))

;; --- Checkout Session Management ---
(defn create-checkout-session
  "Create Stripe checkout session for subscription upgrade"
  [customer-email plan success-url cancel-url]
  (let [plan-config (pricing-plans plan)]
    (if-not plan-config
      {:success false :error "Invalid plan selected"}
      
      (let [params {:customer_email customer-email
                   :line_items [{:price (:price-id plan-config)
                                :quantity 1}]
                   :mode "subscription"
                   :success_url success-url
                   :cancel_url cancel-url
                   :allow_promotion_codes true
                   :billing_address_collection "required"
                   :metadata {:plan (name plan)
                             :customer_email customer-email}}]
        
        (log/info "Creating checkout session for:" customer-email "plan:" plan)
        (stripe-request :post "/checkout/sessions" params)))))

(defn get-checkout-session
  "Retrieve checkout session details"
  [session-id]
  (stripe-request :get (str "/checkout/sessions/" session-id) {}))

;; --- Subscription Management ---
(defn create-customer-in-stripe
  "Create customer in Stripe"
  [email metadata]
  (let [params (merge {:email email} metadata)]
    (stripe-request :post "/customers" params)))

(defn get-subscription
  "Get subscription details from Stripe"
  [subscription-id]
  (stripe-request :get (str "/subscriptions/" subscription-id) {}))

(defn cancel-subscription
  "Cancel subscription in Stripe"
  [subscription-id reason]
  (let [params {:cancellation_details {:comment reason}}]
    (stripe-request :delete (str "/subscriptions/" subscription-id) params)))

(defn update-subscription
  "Update subscription (change plan, etc.)"
  [subscription-id params]
  (stripe-request :post (str "/subscriptions/" subscription-id) params))

;; --- Payment Processing ---
(defn process-successful-payment
  "Process successful payment and upgrade customer"
  [checkout-session]
  (let [customer-email (get-in checkout-session [:customer_details :email])
        subscription-id (:subscription checkout-session)
        plan (get-in checkout-session [:metadata :plan])]
    
    (log/info "Processing successful payment for:" customer-email "subscription:" subscription-id)
    
    (if-let [customer (customers/get-customer-by-email customer-email)]
      (do
        ;; Upgrade customer account
        (customers/upgrade-customer! (:id customer) plan subscription-id)
        
        ;; Send upgrade confirmation email
        (try
          (send-upgrade-confirmation-email customer plan)
          (catch Exception e
            (log/error e "Failed to send upgrade confirmation email")))
        
        ;; Log revenue event
        (log-revenue-event customer-email plan 
                          (get-in (pricing-plans (keyword plan)) [:amount]))
        
        {:success true
         :customer customer
         :plan plan
         :subscription-id subscription-id})
      
      {:success false
       :error "Customer not found"
       :email customer-email})))

(defn process-failed-payment
  "Handle failed payment"
  [invoice]
  (let [customer-id (:customer invoice)
        subscription-id (:subscription invoice)]
    (log/warn "Payment failed for subscription:" subscription-id)
    
    ;; Could implement dunning management, retry logic, etc.
    {:success true :action "logged"}))

;; --- Webhook Handling ---
(defn verify-stripe-signature
  "Verify Stripe webhook signature"
  [payload signature webhook-secret]
  (try
    (let [elements (str/split signature #",")
          timestamp (-> (filter #(str/starts-with? % "t=") elements)
                       first
                       (str/replace #"t=" ""))
          signature-hash (-> (filter #(str/starts-with? % "v1=") elements)
                            first
                            (str/replace #"v1=" ""))
          signed-payload (str timestamp "." payload)
          mac (Mac/getInstance "HmacSHA256")
          secret-key (SecretKeySpec. (.getBytes webhook-secret) "HmacSHA256")]
      
      (.init mac secret-key)
      (let [computed-hash (-> (.doFinal mac (.getBytes signed-payload))
                             (Base64/getEncoder)
                             (.encodeToString))]
        (= signature-hash computed-hash)))
    (catch Exception e
      (log/error e "Failed to verify Stripe signature")
      false)))

(defn handle-webhook-event
  "Process Stripe webhook events"
  [event]
  (let [event-type (:type event)
        event-data (:data event)]
    
    (log/info "Processing Stripe webhook:" event-type)
    
    (case event-type
      "checkout.session.completed"
      (let [session (:object event-data)]
        (if (= "subscription" (:mode session))
          (process-successful-payment session)
          {:success true :action "ignored-non-subscription"}))
      
      "invoice.payment_failed"
      (process-failed-payment (:object event-data))
      
      "customer.subscription.deleted"
      (let [subscription (:object event-data)
            customer-email (get-in subscription [:metadata :customer_email])]
        (when-let [customer (customers/get-customer-by-email customer-email)]
          (customers/cancel-customer! (:id customer) "subscription-cancelled"))
        {:success true :action "customer-cancelled"})
      
      "invoice.payment_succeeded"
      (let [invoice (:object event-data)]
        (log/info "Payment succeeded for subscription:" (:subscription invoice))
        {:success true :action "payment-logged"})
      
      ;; Default case for unhandled events
      (do
        (log/info "Unhandled webhook event:" event-type)
        {:success true :action "ignored"}))))

;; --- Revenue Analytics ---
(def revenue-events (atom []))

(defn log-revenue-event
  "Log revenue event for analytics"
  [customer-email plan amount]
  (let [event {:timestamp (System/currentTimeMillis)
               :customer-email customer-email
               :plan plan
               :amount amount
               :currency "usd"
               :type "subscription-upgrade"}]
    (swap! revenue-events conj event)
    (log/info "Revenue event logged:" event)))

(defn get-revenue-metrics
  "Calculate revenue metrics"
  []
  (let [events @revenue-events
        total-revenue (reduce + (map :amount events))
        this-month-start (.toEpochMilli (coerce/to-long (time/first-day-of-the-month (time/now))))
        this-month-events (filter #(> (:timestamp %) this-month-start) events)
        monthly-revenue (reduce + (map :amount this-month-events))
        
        plan-breakdown (->> events
                           (group-by :plan)
                           (map (fn [[plan events]]
                                  [plan {:count (count events)
                                        :revenue (reduce + (map :amount events))}]))
                           (into {}))
        
        avg-revenue-per-customer (if (> (count events) 0)
                                  (/ total-revenue (count events))
                                  0)]
    
    {:total-revenue total-revenue
     :monthly-revenue monthly-revenue
     :total-upgrades (count events)
     :monthly-upgrades (count this-month-events)
     :avg-revenue-per-customer avg-revenue-per-customer
     :plan-breakdown plan-breakdown
     :conversion-rate (customers/get-customer-metrics)}))

;; --- Email Templates ---
(defn send-upgrade-confirmation-email
  "Send upgrade confirmation email to customer"
  [customer plan]
  (let [plan-config (pricing-plans (keyword plan))
        subject (str "🎉 Welcome to " (:name plan-config) "!")
        html-body (str "<!DOCTYPE html>
<html>
<head><title>Upgrade Confirmation</title></head>
<body style='font-family: sans-serif; line-height: 1.6; color: #333;'>
    <div style='max-width: 600px; margin: 0 auto; padding: 20px;'>
        <h1 style='color: #3b82f6;'>🎉 Upgrade Successful!</h1>
        
        <p>Hi " (or (:game_name customer) (:email customer)) "!</p>
        
        <p>Thank you for upgrading to <strong>" (:name plan-config) "</strong>! Your subscription is now active and you have access to all premium features.</p>
        
        <div style='background: #f0f9ff; padding: 20px; border-radius: 8px; border-left: 4px solid #3b82f6; margin: 20px 0;'>
            <h3 style='margin: 0 0 10px 0;'>Your " (:name plan-config) " includes:</h3>
            <ul style='margin: 0; padding-left: 20px;'>
                " (str/join "\n" (map #(str "<li>" % "</li>") (:features plan-config))) "
            </ul>
        </div>
        
        <p>Your enhanced daily digests will continue as scheduled, now with premium features enabled.</p>
        
        <p><a href='https://app.playerintel.ai/dashboard' style='display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600;'>Access Your Dashboard →</a></p>
        
        <p>Questions about your subscription? <a href='mailto:support@playerintel.ai'>Contact our support team</a> - we're here to help!</p>
        
        <p>Best regards,<br>The Player Intelligence Team</p>
        
        <hr style='margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;'>
        <p style='color: #6b7280; font-size: 14px; text-align: center;'>
            Manage your subscription: <a href='https://app.playerintel.ai/billing'>Billing Settings</a><br>
            Player Intelligence • <a href='https://playerintel.ai'>playerintel.ai</a>
        </p>
    </div>
</body>
</html>")
        text-body (str "Upgrade Successful!\n\n"
                      "Thank you for upgrading to " (:name plan-config) "!\n\n"
                      "Your premium features are now active:\n"
                      (str/join "\n" (map #(str "• " %) (:features plan-config))) "\n\n"
                      "Access your dashboard: https://app.playerintel.ai/dashboard\n\n"
                      "Questions? Contact support@playerintel.ai")]
    
    (email/send-email! (:email customer) subject html-body text-body)))

;; --- Plan Management ---
(defn get-available-plans
  "Get available subscription plans"
  []
  (map (fn [[plan-key plan-config]]
         (assoc plan-config 
                :key plan-key
                :monthly-price (/ (:amount plan-config) 100)
                :currency "USD"))
       pricing-plans))

(defn get-upgrade-url
  "Generate upgrade URL for customer"
  [customer-email plan]
  (let [success-url "https://app.playerintel.ai/success"
        cancel-url "https://app.playerintel.ai/cancel"
        session-result (create-checkout-session customer-email plan success-url cancel-url)]
    (if (:success session-result)
      {:success true :checkout-url (get-in session-result [:data :url])}
      session-result)))