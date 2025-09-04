(ns player-intel.scheduler
  (:require [player-intel.customers :as customers]
            [player-intel.digest :as digest]
            [player-intel.email :as email]
            [clojure.core.async :as async]
            [taoensso.timbre :as log]
            [clj-time.core :as time]
            [clj-time.coerce :as coerce]
            [clj-time.periodic :as periodic]
            [environ.core :refer [env]])
  (:import [java.util.concurrent Executors ScheduledExecutorService TimeUnit]))

;; Scheduler state
(defonce scheduler-state (atom {:running? false
                               :executor nil
                               :tasks {}
                               :stats {:digests-sent 0
                                      :emails-sent 0
                                      :errors 0
                                      :last-run nil}}))

;; --- Core Digest Generation ---
(defn send-customer-digest
  "Generate and send digest for a single customer"
  [customer]
  (log/info "Generating digest for customer:" (:email customer))
  
  (try
    (let [channel-id (:discord_channel_id customer)
          digest-data (digest/get-digest-for-channel channel-id)]
      
      (if (and digest-data (> (:message-count digest-data 0) 0))
        (do
          ;; Save digest to database
          (digest/save-digest! digest-data (:id customer))
          
          ;; Send email
          (let [email-result (email/send-digest! customer digest-data)]
            (if (:success email-result)
              (do
                (log/info "Successfully sent digest to:" (:email customer))
                (swap! scheduler-state update-in [:stats :digests-sent] inc)
                (swap! scheduler-state update-in [:stats :emails-sent] inc)
                {:success true :customer (:email customer)})
              (do
                (log/error "Failed to send email to:" (:email customer) (:message email-result))
                (swap! scheduler-state update-in [:stats :errors] inc)
                {:success false :customer (:email customer) :error (:message email-result)}))))
        
        (do
          (log/warn "No digest data available for customer:" (:email customer))
          {:success false :customer (:email customer) :error "No digest data available"})))
    
    (catch Exception e
      (log/error e "Error generating digest for customer:" (:email customer))
      (swap! scheduler-state update-in [:stats :errors] inc)
      {:success false :customer (:email customer) :error (.getMessage e)})))

(defn run-daily-digests
  "Send digests to all active customers"
  []
  (log/info "Starting daily digest generation")
  (swap! scheduler-state assoc-in [:stats :last-run] (System/currentTimeMillis))
  
  (let [customers (customers/get-active-customers)
        start-time (System/currentTimeMillis)]
    
    (log/info "Found" (count customers) "active customers")
    
    (if (empty? customers)
      (log/info "No active customers found - skipping digest generation")
      
      (let [results (doall (map send-customer-digest customers))
            successful (count (filter :success results))
            failed (count (filter #(not (:success %)) results))
            duration (- (System/currentTimeMillis) start-time)]
        
        (log/info "Completed daily digest run:"
                 "customers:" (count customers)
                 "successful:" successful  
                 "failed:" failed
                 "duration:" (str duration "ms"))
        
        {:total-customers (count customers)
         :successful-digests successful
         :failed-digests failed
         :duration-ms duration
         :results results}))))

;; --- Trial Management ---
(defn send-trial-reminders
  "Send reminder emails to customers with expiring trials"
  []
  (log/info "Checking for expiring trials")
  
  (let [expiring-in-3-days (customers/get-trial-customers-expiring-soon 3)
        expiring-in-1-day (customers/get-trial-customers-expiring-soon 1)]
    
    ;; Send 3-day reminders
    (doseq [customer expiring-in-3-days]
      (try
        (email/send-trial-expiring-email! customer 3)
        (log/info "Sent 3-day trial reminder to:" (:email customer))
        (catch Exception e
          (log/error e "Failed to send trial reminder to:" (:email customer)))))
    
    ;; Send 1-day reminders
    (doseq [customer expiring-in-1-day]
      (try
        (email/send-trial-expiring-email! customer 1)
        (log/info "Sent 1-day trial reminder to:" (:email customer))
        (catch Exception e
          (log/error e "Failed to send trial reminder to:" (:email customer)))))
    
    (log/info "Trial reminder check complete:"
             "3-day reminders:" (count expiring-in-3-days)
             "1-day reminders:" (count expiring-in-1-day))))

;; --- Scheduler Management ---
(defn create-scheduled-executor
  "Create a scheduled executor service"
  []
  (Executors/newScheduledThreadPool 2))

(defn schedule-daily-task
  "Schedule a task to run daily at specified hour"
  [executor task-fn hour-of-day task-name]
  (let [now (time/now)
        target-time (time/today-at hour-of-day 0)
        target-time (if (time/before? target-time now)
                     (time/plus target-time (time/days 1))
                     target-time)
        delay-ms (- (coerce/to-long target-time) (coerce/to-long now))
        delay-seconds (/ delay-ms 1000)]
    
    (log/info "Scheduling" task-name "to run in" (int (/ delay-seconds 3600)) "hours")
    
    (.scheduleAtFixedRate executor
                         (fn []
                           (try
                             (log/info "Running scheduled task:" task-name)
                             (task-fn)
                             (catch Exception e
                               (log/error e "Error in scheduled task:" task-name))))
                         delay-seconds
                         (* 24 60 60) ; 24 hours in seconds
                         TimeUnit/SECONDS)))

(defn start-scheduler!
  "Start the digest and notification scheduler"
  []
  (log/info "Starting Player Intelligence scheduler")
  
  (when-let [existing-executor (:executor @scheduler-state)]
    (.shutdown existing-executor))
  
  (let [executor (create-scheduled-executor)
        digest-hour (Integer/parseInt (env :digest-hour "8"))  ; 8 AM default
        reminder-hour (Integer/parseInt (env :reminder-hour "10"))] ; 10 AM default
    
    ;; Schedule daily digest generation
    (let [digest-task (schedule-daily-task executor run-daily-digests digest-hour "daily-digests")]
      (swap! scheduler-state assoc-in [:tasks :digest] digest-task))
    
    ;; Schedule trial reminder checks
    (let [reminder-task (schedule-daily-task executor send-trial-reminders reminder-hour "trial-reminders")]
      (swap! scheduler-state assoc-in [:tasks :reminders] reminder-task))
    
    (swap! scheduler-state assoc 
           :running? true
           :executor executor)
    
    (log/info "Scheduler started successfully"
             "- Daily digests at" (str digest-hour ":00")
             "- Trial reminders at" (str reminder-hour ":00"))))

(defn stop-scheduler!
  "Stop the scheduler"
  []
  (log/info "Stopping Player Intelligence scheduler")
  
  (when-let [executor (:executor @scheduler-state)]
    (.shutdown executor)
    (try
      (when-not (.awaitTermination executor 5 TimeUnit/SECONDS)
        (.shutdownNow executor))
      (catch InterruptedException _
        (.shutdownNow executor))))
  
  (swap! scheduler-state assoc 
         :running? false
         :executor nil
         :tasks {})
  
  (log/info "Scheduler stopped"))

(defn get-scheduler-status
  "Get current scheduler status and statistics"
  []
  (let [state @scheduler-state]
    {:running? (:running? state)
     :active-tasks (keys (:tasks state))
     :stats (:stats state)
     :next-digest-run (when (:running? state)
                       ;; Calculate next run time
                       (let [digest-hour (Integer/parseInt (env :digest-hour "8"))
                             now (time/now)
                             next-run (time/today-at digest-hour 0)]
                         (if (time/before? next-run now)
                           (time/plus next-run (time/days 1))
                           next-run)))}))

;; --- Manual Operations ---
(defn run-digest-for-customer
  "Manually run digest for a specific customer"
  [customer-email]
  (log/info "Running manual digest for:" customer-email)
  
  (if-let [customer (customers/get-customer-by-email customer-email)]
    (if (customers/customer-active? customer)
      (send-customer-digest customer)
      {:success false :error "Customer account is not active"})
    {:success false :error "Customer not found"}))

(defn run-digest-for-all-customers
  "Manually run digests for all active customers"
  []
  (log/info "Running manual digest for all customers")
  (run-daily-digests))

;; --- Health Checks ---
(defn scheduler-health-check
  "Check scheduler health and recent activity"
  []
  (let [state @scheduler-state
        stats (:stats state)
        last-run (:last-run stats)
        now (System/currentTimeMillis)
        hours-since-last-run (when last-run (/ (- now last-run) 1000 60 60))]
    
    {:healthy? (:running? state)
     :last-run last-run
     :hours-since-last-run (when hours-since-last-run (Math/round hours-since-last-run))
     :stats stats
     :status (cond
               (not (:running? state)) "stopped"
               (not last-run) "waiting-for-first-run"
               (> hours-since-last-run 25) "overdue"
               :else "healthy")}))