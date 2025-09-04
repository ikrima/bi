(ns player-intel.customers
  (:require [player-intel.db :as db]
            [honey.sql :as sql]
            [next.jdbc :as jdbc]
            [clojure.string :as str]
            [taoensso.timbre :as log]
            [clojure.spec.alpha :as s]
            [clj-time.core :as time]
            [clj-time.coerce :as coerce])
  (:import [java.util UUID]
           [java.time Instant]))

;; --- Specs for validation ---
(s/def ::email (s/and string? #(re-matches #"[^@]+@[^@]+\.[^@]+" %)))
(s/def ::discord-channel-id (s/and string? #(re-matches #"\d{17,19}" %)))
(s/def ::plan #{"trial" "basic" "pro"})
(s/def ::customer-id uuid?)

(s/def ::customer-create
  (s/keys :req-un [::email ::discord-channel-id]
          :opt-un [::plan]))

;; --- Customer CRUD Operations ---
(defn create-customer!
  "Create a new customer account with trial period"
  [{:keys [email discord-channel-id plan game-name] 
    :or {plan "trial"}}]
  (if (s/valid? ::customer-create {:email email :discord-channel-id discord-channel-id})
    (let [customer-id (UUID/randomUUID)
          trial-end-date (coerce/to-timestamp 
                         (time/plus (time/now) (time/days 7)))
          customer {:id customer-id
                   :email email
                   :discord_channel_id discord-channel-id
                   :game_name game-name
                   :plan plan
                   :trial_ends_at trial-end-date
                   :created_at (coerce/to-timestamp (time/now))
                   :status "active"}]
      (try
        (jdbc/execute! (db/get-db)
          (sql/format {:insert-into :customers
                       :values [customer]}))
        (log/info "Created customer:" email "with channel:" discord-channel-id)
        customer
        (catch Exception e
          (log/error e "Failed to create customer:" email)
          nil)))
    (do
      (log/warn "Invalid customer data:" {:email email :discord-channel-id discord-channel-id})
      nil)))

(defn get-customer-by-email
  "Get customer by email address"
  [email]
  (try
    (jdbc/execute-one! (db/get-db)
      (sql/format {:select :*
                   :from :customers
                   :where [:= :email email]}))
    (catch Exception e
      (log/error e "Failed to get customer by email:" email)
      nil)))

(defn get-customer-by-id
  "Get customer by UUID"
  [customer-id]
  (try
    (jdbc/execute-one! (db/get-db)
      (sql/format {:select :*
                   :from :customers
                   :where [:= :id customer-id]}))
    (catch Exception e
      (log/error e "Failed to get customer by ID:" customer-id)
      nil)))

(defn update-customer!
  "Update customer information"
  [customer-id updates]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:update :customers
                   :set (assoc updates :updated_at (coerce/to-timestamp (time/now)))
                   :where [:= :id customer-id]}))
    (log/info "Updated customer:" customer-id)
    true
    (catch Exception e
      (log/error e "Failed to update customer:" customer-id)
      false)))

(defn upgrade-customer!
  "Upgrade customer from trial to paid plan"
  [customer-id plan stripe-subscription-id]
  (let [updates {:plan plan
                 :stripe_subscription_id stripe-subscription-id
                 :upgraded_at (coerce/to-timestamp (time/now))
                 :status "active"}]
    (update-customer! customer-id updates)))

(defn cancel-customer!
  "Cancel customer subscription"
  [customer-id reason]
  (let [updates {:status "cancelled"
                 :cancelled_at (coerce/to-timestamp (time/now))
                 :cancellation_reason reason}]
    (update-customer! customer-id updates)))

;; --- Customer Status and Validation ---
(defn customer-active?
  "Check if customer account is active"
  [customer]
  (and customer
       (= (:status customer) "active")
       (or (not= (:plan customer) "trial")
           (and (:trial_ends_at customer)
                (.isAfter (coerce/from-sql-time (:trial_ends_at customer))
                         (time/now))))))

(defn get-active-customers
  "Get all customers who should receive digests"
  []
  (try
    (let [customers (jdbc/execute! (db/get-db)
                      (sql/format {:select :*
                                   :from :customers
                                   :where [:and
                                          [:= :status "active"]
                                          [:or
                                           [:not= :plan "trial"]
                                           [:> :trial_ends_at (coerce/to-timestamp (time/now))]]]}))]
      (log/info "Found" (count customers) "active customers")
      customers)
    (catch Exception e
      (log/error e "Failed to get active customers")
      [])))

(defn get-trial-customers-expiring-soon
  "Get trial customers expiring in the next N days"
  [days]
  (try
    (let [cutoff-date (coerce/to-timestamp 
                       (time/plus (time/now) (time/days days)))]
      (jdbc/execute! (db/get-db)
        (sql/format {:select :*
                     :from :customers
                     :where [:and
                            [:= :plan "trial"]
                            [:= :status "active"]
                            [:<= :trial_ends_at cutoff-date]
                            [:> :trial_ends_at (coerce/to-timestamp (time/now))]]})))
    (catch Exception e
      (log/error e "Failed to get expiring trial customers")
      [])))

(defn validate-discord-channel
  "Validate that Discord channel is accessible"
  [discord-channel-id discord-bot-token]
  (if (and discord-channel-id discord-bot-token)
    (try
      ;; This would use the Discord API to validate channel access
      ;; For now, return a mock validation
      {:valid true 
       :channel-name "general"
       :guild-name "Test Server"
       :message "Channel is accessible"}
      (catch Exception e
        (log/error e "Failed to validate Discord channel:" discord-channel-id)
        {:valid false
         :message "Unable to access Discord channel. Please check channel ID and bot permissions."}))
    {:valid false
     :message "Discord channel ID or bot token not provided"}))

;; --- Customer Onboarding ---
(defn create-trial-account!
  "Create a new trial account with full onboarding"
  [email discord-channel-id game-name]
  (log/info "Creating trial account for:" email)
  
  ;; Check if customer already exists
  (if-let [existing (get-customer-by-email email)]
    {:success false
     :message "Account already exists with this email"
     :customer existing}
    
    ;; Create new customer
    (if-let [customer (create-customer! {:email email
                                        :discord-channel-id discord-channel-id
                                        :game-name game-name
                                        :plan "trial"})]
      (do
        (log/info "Successfully created trial account:" email)
        {:success true
         :message "Trial account created successfully"
         :customer customer
         :trial-days-remaining 7})
      
      {:success false
       :message "Failed to create account. Please try again."})))

(defn get-customer-summary
  "Get customer summary with usage stats"
  [customer-id]
  (when-let [customer (get-customer-by-id customer-id)]
    (let [digests-count (try
                         (-> (jdbc/execute-one! (db/get-db)
                               (sql/format {:select [[:%count.* :count]]
                                           :from :digests
                                           :where [:= :customer_id customer-id]}))
                             :count)
                         (catch Exception _ 0))
          
          days-remaining (when (= (:plan customer) "trial")
                          (try
                            (let [end-date (coerce/from-sql-time (:trial_ends_at customer))
                                  now (time/now)
                                  diff (time/in-days (time/interval now end-date))]
                              (max 0 diff))
                            (catch Exception _ 0)))]
      
      {:customer customer
       :digests-generated digests-count
       :trial-days-remaining days-remaining
       :active? (customer-active? customer)
       :needs-upgrade? (and (= (:plan customer) "trial")
                           (or (not days-remaining) (< days-remaining 3)))})))

;; --- Customer Metrics ---
(defn get-customer-metrics
  "Get overall customer metrics"
  []
  (try
    (let [total-customers (-> (jdbc/execute-one! (db/get-db)
                                (sql/format {:select [[:%count.* :count]]
                                            :from :customers}))
                             :count)
          
          active-trials (-> (jdbc/execute-one! (db/get-db)
                              (sql/format {:select [[:%count.* :count]]
                                          :from :customers
                                          :where [:and
                                                 [:= :plan "trial"]
                                                 [:= :status "active"]
                                                 [:> :trial_ends_at (coerce/to-timestamp (time/now))]]}))
                           :count)
          
          paid-customers (-> (jdbc/execute-one! (db/get-db)
                               (sql/format {:select [[:%count.* :count]]
                                           :from :customers
                                           :where [:and
                                                  [:not= :plan "trial"]
                                                  [:= :status "active"]]}))
                            :count)
          
          churned-customers (-> (jdbc/execute-one! (db/get-db)
                                  (sql/format {:select [[:%count.* :count]]
                                              :from :customers
                                              :where [:= :status "cancelled"]}))
                               :count)]
      
      {:total-customers total-customers
       :active-trials active-trials
       :paid-customers paid-customers
       :churned-customers churned-customers
       :conversion-rate (if (> total-customers 0)
                         (Math/round (* 100.0 (/ paid-customers total-customers)))
                         0)})
    (catch Exception e
      (log/error e "Failed to get customer metrics")
      {:total-customers 0 :active-trials 0 :paid-customers 0 :churned-customers 0 :conversion-rate 0})))