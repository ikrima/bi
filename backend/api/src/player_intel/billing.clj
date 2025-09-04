(ns player-intel.billing
  (:require [player-intel.customers :as customers]
            [player-intel.payments :as payments]
            [player-intel.db :as db]
            [honey.sql :as sql]
            [next.jdbc :as jdbc]
            [taoensso.timbre :as log]
            [clj-time.core :as time]
            [clj-time.coerce :as coerce]
            [clojure.data.json :as json]))

;; --- Billing History ---
(defn create-billing-tables!
  "Create billing-related database tables"
  []
  (try
    ;; Invoice table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS invoices ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "stripe_invoice_id VARCHAR(255),"
            "amount INTEGER NOT NULL,"
            "currency VARCHAR(3) DEFAULT 'usd',"
            "status VARCHAR(50),"
            "plan VARCHAR(50),"
            "period_start TIMESTAMP,"
            "period_end TIMESTAMP,"
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")])
    
    ;; Payment events table
    (jdbc/execute! (db/get-db)
      [(str "CREATE TABLE IF NOT EXISTS payment_events ("
            "id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "customer_id UUID REFERENCES customers(id),"
            "event_type VARCHAR(100),"
            "stripe_event_id VARCHAR(255) UNIQUE,"
            "amount INTEGER,"
            "currency VARCHAR(3),"
            "plan VARCHAR(50),"
            "metadata JSONB,"
            "processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")])
    
    (log/info "Billing tables created successfully")
    (catch Exception e
      (log/error e "Failed to create billing tables"))))

(defn record-payment-event!
  "Record a payment event for analytics"
  [customer-id event-type stripe-event-id amount currency plan metadata]
  (try
    (jdbc/execute! (db/get-db)
      (sql/format {:insert-into :payment_events
                   :values [{:customer_id customer-id
                            :event_type event-type
                            :stripe_event_id stripe-event-id
                            :amount amount
                            :currency currency
                            :plan plan
                            :metadata (json/write-str metadata)}]
                   :on-conflict :stripe_event_id
                   :do-nothing true}))
    (log/info "Payment event recorded:" event-type "for customer:" customer-id)
    (catch Exception e
      (log/error e "Failed to record payment event"))))

(defn create-invoice!
  "Create invoice record"
  [customer-id stripe-invoice-id amount currency status plan period-start period-end]
  (try
    (let [invoice {:customer_id customer-id
                   :stripe_invoice_id stripe-invoice-id
                   :amount amount
                   :currency currency
                   :status status
                   :plan plan
                   :period_start (when period-start (coerce/to-timestamp period-start))
                   :period_end (when period-end (coerce/to-timestamp period-end))}]
      (jdbc/execute! (db/get-db)
        (sql/format {:insert-into :invoices
                     :values [invoice]}))
      (log/info "Invoice created:" stripe-invoice-id))
    (catch Exception e
      (log/error e "Failed to create invoice"))))

;; --- Customer Billing Info ---
(defn get-customer-billing-info
  "Get comprehensive billing information for customer"
  [customer-id]
  (when-let [customer (customers/get-customer-by-id customer-id)]
    (let [;; Get recent invoices
          invoices (try
                    (jdbc/execute! (db/get-db)
                      (sql/format {:select :*
                                   :from :invoices
                                   :where [:= :customer_id customer-id]
                                   :order-by [[:created_at :desc]]
                                   :limit 12}))
                    (catch Exception e
                      (log/error e "Failed to get invoices")
                      []))
          
          ;; Get payment events
          payment-events (try
                          (jdbc/execute! (db/get-db)
                            (sql/format {:select :*
                                        :from :payment_events
                                        :where [:= :customer_id customer-id]
                                        :order-by [[:processed_at :desc]]
                                        :limit 20}))
                          (catch Exception e
                            (log/error e "Failed to get payment events")
                            []))
          
          ;; Calculate spending
          total-spent (reduce + (map :amount invoices))
          
          ;; Get subscription info from Stripe if available
          subscription-info (when (:stripe_subscription_id customer)
                             (payments/get-subscription (:stripe_subscription_id customer)))]
      
      {:customer customer
       :subscription subscription-info
       :invoices invoices
       :payment-events payment-events
       :total-spent total-spent
       :currency "usd"
       :billing-status (cond
                        (= (:plan customer) "trial") :trial
                        (:stripe_subscription_id customer) :active
                        :else :inactive)})))

(defn get-customer-usage-stats
  "Get usage statistics for billing purposes"
  [customer-id]
  (let [customer (customers/get-customer-by-id customer-id)
        
        ;; Get digest count this month
        month-start (coerce/to-timestamp (time/first-day-of-the-month (time/now)))
        digest-count (try
                      (-> (jdbc/execute-one! (db/get-db)
                            (sql/format {:select [[:%count.* :count]]
                                        :from :digests
                                        :where [:and
                                               [:= :customer_id customer-id]
                                               [:>= :generated_at month-start]]}))
                          :count)
                      (catch Exception _ 0))
        
        ;; Calculate other usage metrics
        days-active (if (= (:plan customer) "trial")
                     (try
                       (let [created (coerce/from-sql-time (:created_at customer))
                             now (time/now)
                             diff (time/in-days (time/interval created now))]
                         (min diff 7))
                       (catch Exception _ 0))
                     nil)]
    
    {:digests-this-month digest-count
     :plan (:plan customer)
     :days-active days-active
     :trial-days-remaining (when (= (:plan customer) "trial")
                            (customers/get-customer-summary customer-id))}))

;; --- Subscription Management ---
(defn initiate-plan-change
  "Start process to change customer's plan"
  [customer-id new-plan]
  (let [customer (customers/get-customer-by-id customer-id)]
    (cond
      (not customer)
      {:success false :error "Customer not found"}
      
      (= (:plan customer) (name new-plan))
      {:success false :error "Customer already on this plan"}
      
      (= (:plan customer) "trial")
      ;; Trial to paid - create checkout session
      (payments/get-upgrade-url (:email customer) new-plan)
      
      ;; Existing subscription - would need to modify in Stripe
      (:stripe_subscription_id customer)
      {:success false :error "Plan changes for existing subscriptions not implemented yet"}
      
      :else
      {:success false :error "Invalid plan change scenario"})))

(defn cancel-subscription
  "Cancel customer subscription"
  [customer-id reason]
  (let [customer (customers/get-customer-by-id customer-id)]
    (cond
      (not customer)
      {:success false :error "Customer not found"}
      
      (= (:plan customer) "trial")
      (do
        (customers/cancel-customer! customer-id reason)
        {:success true :message "Trial cancelled"})
      
      (:stripe_subscription_id customer)
      (let [result (payments/cancel-subscription (:stripe_subscription_id customer) reason)]
        (if (:success result)
          (do
            (customers/cancel-customer! customer-id reason)
            {:success true :message "Subscription cancelled"})
          result))
      
      :else
      {:success false :error "No active subscription to cancel"})))

;; --- Revenue Reporting ---
(defn get-revenue-report
  "Generate comprehensive revenue report"
  []
  (try
    (let [;; Get all payment events
          payment-events (jdbc/execute! (db/get-db)
                          (sql/format {:select :*
                                      :from :payment_events
                                      :where [:= :event_type "subscription-upgrade"]
                                      :order-by [[:processed_at :desc]]}))
          
          ;; Calculate metrics
          total-revenue (reduce + (map :amount payment-events))
          
          ;; Monthly revenue
          month-start (coerce/to-timestamp (time/first-day-of-the-month (time/now)))
          monthly-events (filter #(.isAfter (coerce/from-sql-time (:processed_at %)) 
                                           (coerce/from-sql-time month-start)) 
                                payment-events)
          monthly-revenue (reduce + (map :amount monthly-events))
          
          ;; Plan breakdown
          plan-breakdown (->> payment-events
                             (group-by :plan)
                             (map (fn [[plan events]]
                                    [plan {:count (count events)
                                          :revenue (reduce + (map :amount events))}]))
                             (into {}))
          
          ;; Customer metrics
          customer-metrics (customers/get-customer-metrics)
          
          ;; MRR calculation (simplified)
          active-subscriptions (:paid-customers customer-metrics)
          estimated-mrr (* active-subscriptions 2499) ; Assuming average basic plan
          
          ;; Growth metrics
          last-month-start (coerce/to-timestamp 
                           (time/first-day-of-the-month 
                            (time/minus (time/now) (time/months 1))))
          last-month-events (filter #(let [event-time (coerce/from-sql-time (:processed_at %))]
                                      (and (.isAfter event-time (coerce/from-sql-time last-month-start))
                                           (.isBefore event-time (coerce/from-sql-time month-start))))
                                   payment-events)
          last-month-revenue (reduce + (map :amount last-month-events))
          growth-rate (if (> last-month-revenue 0)
                       (* 100 (/ (- monthly-revenue last-month-revenue) last-month-revenue))
                       0)]
      
      {:total-revenue (/ total-revenue 100.0)
       :monthly-revenue (/ monthly-revenue 100.0)
       :last-month-revenue (/ last-month-revenue 100.0)
       :growth-rate (Math/round growth-rate)
       :estimated-mrr (/ estimated-mrr 100.0)
       :total-transactions (count payment-events)
       :monthly-transactions (count monthly-events)
       :avg-transaction-value (if (> (count payment-events) 0)
                               (/ total-revenue (count payment-events) 100.0)
                               0)
       :plan-breakdown (into {} (map (fn [[k v]] [k (update v :revenue #(/ % 100.0))]) 
                                    plan-breakdown))
       :customer-metrics customer-metrics})
    
    (catch Exception e
      (log/error e "Failed to generate revenue report")
      {:error "Failed to generate revenue report"})))

(defn get-monthly-revenue-trend
  "Get revenue trend over the last 12 months"
  []
  (try
    (let [months (for [i (range 12)]
                   (time/minus (time/now) (time/months i)))
          
          revenue-by-month (for [month months]
                            (let [month-start (coerce/to-timestamp (time/first-day-of-the-month month))
                                  month-end (coerce/to-timestamp (time/last-day-of-the-month month))
                                  
                                  month-events (jdbc/execute! (db/get-db)
                                                (sql/format {:select :*
                                                            :from :payment_events
                                                            :where [:and
                                                                   [:= :event_type "subscription-upgrade"]
                                                                   [:>= :processed_at month-start]
                                                                   [:<= :processed_at month-end]]}))
                                  
                                  month-revenue (reduce + (map :amount month-events))]
                              
                              {:month (time/year-month month)
                               :revenue (/ month-revenue 100.0)
                               :transactions (count month-events)}))]
      
      (reverse revenue-by-month))
    
    (catch Exception e
      (log/error e "Failed to get revenue trend")
      [])))

;; --- Billing Utilities ---
(defn format-currency
  "Format currency amount for display"
  [amount currency]
  (case (keyword currency)
    :usd (format "$%.2f" (/ amount 100.0))
    :eur (format "€%.2f" (/ amount 100.0))
    (format "%.2f %s" (/ amount 100.0) (clojure.string/upper-case currency))))

(defn get-plan-comparison
  "Get plan comparison data for upgrade flows"
  []
  (let [plans (payments/get-available-plans)]
    (map (fn [plan]
           (assoc plan 
                  :formatted-price (format-currency (:amount plan) "usd")
                  :billing-cycle "monthly"
                  :recommended (= (:key plan) :basic)))
         plans)))