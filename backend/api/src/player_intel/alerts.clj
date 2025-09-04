(ns player-intel.alerts
  (:require [player-intel.analytics :as analytics]
            [player-intel.email :as email]
            [player-intel.customers :as customers]
            [taoensso.timbre :as log]
            [clj-time.core :as time]
            [clj-time.format :as time-format]
            [clojure.string :as str]))

;; --- Alert System ---
(def alert-thresholds
  {:sentiment {:severe 30 :warning 40}
   :engagement {:severe 0.3 :warning 0.5}
   :churn-risk {:severe 0.8 :warning 0.6}
   :retention-rate {:severe 0.4 :warning 0.6}})

(defn check-sentiment-alerts
  "Check for sentiment-based alerts"
  [customer-id metrics]
  (let [sentiment (:sentiment_score metrics 50)]
    (cond
      (< sentiment (get-in alert-thresholds [:sentiment :severe]))
      {:type "sentiment-alert"
       :severity "severe"
       :title "Critical Community Sentiment Drop"
       :description (str "Community sentiment has dropped to " sentiment "% (Critical threshold: <30%)")
       :recommendation "Immediate community engagement required. Review recent changes and address concerns."
       :data {:current_sentiment sentiment :threshold 30}}
      
      (< sentiment (get-in alert-thresholds [:sentiment :warning]))
      {:type "sentiment-alert"
       :severity "warning"
       :title "Community Sentiment Declining"
       :description (str "Community sentiment has dropped to " sentiment "% (Warning threshold: <40%)")
       :recommendation "Monitor community closely and consider proactive communication."
       :data {:current_sentiment sentiment :threshold 40}}
      
      :else nil)))

(defn check-engagement-alerts
  "Check for engagement-based alerts"
  [customer-id metrics]
  (let [engagement (:avg_engagement metrics 0.5)]
    (cond
      (< engagement (get-in alert-thresholds [:engagement :severe]))
      {:type "engagement-alert"
       :severity "severe"
       :title "Critical Drop in Player Engagement"
       :description (str "Average engagement has dropped to " (format "%.2f" engagement))
       :recommendation "Launch engagement campaigns, events, or content updates immediately."
       :data {:current_engagement engagement :threshold 0.3}}
      
      (< engagement (get-in alert-thresholds [:engagement :warning]))
      {:type "engagement-alert"
       :severity "warning"
       :title "Player Engagement Declining"
       :description (str "Average engagement has dropped to " (format "%.2f" engagement))
       :recommendation "Consider new content or engagement initiatives."
       :data {:current_engagement engagement :threshold 0.5}}
      
      :else nil)))

(defn check-retention-alerts
  "Check for retention-based alerts"
  [customer-id metrics]
  (let [retention (:retention_rate metrics 0.7)]
    (cond
      (< retention (get-in alert-thresholds [:retention-rate :severe]))
      {:type "retention-alert"
       :severity "severe"
       :title "Critical Player Retention Issue"
       :description (str "Player retention has dropped to " (format "%.1f%%" (* retention 100)))
       :recommendation "Urgent retention campaign needed. Review recent changes and player feedback."
       :data {:current_retention retention :threshold 0.4}}
      
      (< retention (get-in alert-thresholds [:retention-rate :warning]))
      {:type "retention-alert"
       :severity "warning"
       :title "Player Retention Declining"
       :description (str "Player retention has dropped to " (format "%.1f%%" (* retention 100)))
       :recommendation "Monitor retention trends and consider retention initiatives."
       :data {:current_retention retention :threshold 0.6}}
      
      :else nil)))

(defn check-churn-risk-alerts
  "Check for churn risk alerts"
  [customer-id metrics]
  (let [churn-risk (:churn_risk_score metrics 0.2)]
    (cond
      (> churn-risk (get-in alert-thresholds [:churn-risk :severe]))
      {:type "churn-risk-alert"
       :severity "severe"
       :title "High Churn Risk Detected"
       :description (str "Churn risk score is " (format "%.1f%%" (* churn-risk 100)))
       :recommendation "Implement targeted retention strategies for high-risk player segments."
       :data {:churn_risk churn-risk :threshold 0.8}}
      
      (> churn-risk (get-in alert-thresholds [:churn-risk :warning]))
      {:type "churn-risk-alert"
       :severity "warning"
       :title "Elevated Churn Risk"
       :description (str "Churn risk score is " (format "%.1f%%" (* churn-risk 100)))
       :recommendation "Monitor player satisfaction and consider proactive engagement."
       :data {:churn_risk churn-risk :threshold 0.6}}
      
      :else nil)))

(defn run-alert-checks
  "Run all alert checks for a customer"
  [customer-id]
  (try
    (let [metrics (analytics/calculate-player-lifecycle-metrics customer-id)
          alerts (filter some? [(check-sentiment-alerts customer-id metrics)
                              (check-engagement-alerts customer-id metrics)
                              (check-retention-alerts customer-id metrics)
                              (check-churn-risk-alerts customer-id metrics)])]
      
      ;; Create alerts in database
      (doseq [alert alerts]
        (analytics/create-trend-alert! customer-id
                                     (:type alert)
                                     (:severity alert)
                                     (:title alert)
                                     (:description alert)
                                     (:data alert)))
      
      ;; Send severe alerts via email immediately
      (let [severe-alerts (filter #(= (:severity %) "severe") alerts)]
        (when (seq severe-alerts)
          (send-alert-email customer-id severe-alerts)))
      
      {:alerts-created (count alerts)
       :severe-alerts (count (filter #(= (:severity %) "severe") alerts))})
    
    (catch Exception e
      (log/error e "Failed to run alert checks for customer" customer-id)
      {:error (.getMessage e)})))

(defn send-alert-email
  "Send alert notification email"
  [customer-id alerts]
  (try
    (let [customer (customers/get-customer-by-id customer-id)
          severe-count (count (filter #(= (:severity %) "severe") alerts))
          warning-count (count (filter #(= (:severity %) "warning") alerts))
          
          subject (if (> severe-count 0)
                   (str "🚨 URGENT: " severe-count " Critical Alert" 
                        (when (> severe-count 1) "s") " for Your Community")
                   (str "⚠️ Warning: " warning-count " Alert" 
                        (when (> warning-count 1) "s") " for Your Community"))
          
          html-body (generate-alert-email-html alerts customer)
          text-body (generate-alert-email-text alerts customer)]
      
      (when customer
        (email/send-email! (:email customer) subject html-body text-body)
        (log/info "Sent alert email to customer" customer-id)))
    
    (catch Exception e
      (log/error e "Failed to send alert email"))))

(defn generate-alert-email-html
  "Generate HTML email body for alerts"
  [alerts customer]
  (let [severe-alerts (filter #(= (:severity %) "severe") alerts)
        warning-alerts (filter #(= (:severity %) "warning") alerts)
        timestamp (time-format/unparse (time-format/formatters :date-time) (time/now))]
    
    (str "<!DOCTYPE html>
<html>
<head><title>Community Alert</title></head>
<body style='font-family: sans-serif; line-height: 1.6; color: #333;'>
    <div style='max-width: 600px; margin: 0 auto; padding: 20px;'>
        <h1 style='color: " (if (seq severe-alerts) "#dc2626" "#f59e0b") ";'>
            " (if (seq severe-alerts) "🚨 Critical Community Alerts" "⚠️ Community Warnings") "
        </h1>
        
        <p>Hi " (or (:game_name customer) (:email customer)) ",</p>
        
        <p>We've detected some important changes in your community that require your attention:</p>"
        
        ;; Severe alerts section
        (when (seq severe-alerts)
          (str "
        <div style='background: #fef2f2; border: 2px solid #dc2626; padding: 20px; border-radius: 8px; margin: 20px 0;'>
            <h2 style='color: #dc2626; margin: 0 0 15px 0;'>🚨 Critical Issues (Immediate Action Required)</h2>"
            (str/join "\n"
              (map (fn [alert]
                     (str "<div style='margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #fca5a5;'>
                              <h3 style='margin: 0; color: #991b1b;'>" (:title alert) "</h3>
                              <p style='margin: 5px 0;'>" (:description alert) "</p>
                              <p style='margin: 5px 0; font-weight: 600; color: #7c2d12;'>Recommendation: " (:recommendation alert) "</p>
                          </div>"))
                   severe-alerts))
            "</div>"))
        
        ;; Warning alerts section  
        (when (seq warning-alerts)
          (str "
        <div style='background: #fefbf2; border: 2px solid #f59e0b; padding: 20px; border-radius: 8px; margin: 20px 0;'>
            <h2 style='color: #f59e0b; margin: 0 0 15px 0;'>⚠️ Warnings (Monitor Closely)</h2>"
            (str/join "\n"
              (map (fn [alert]
                     (str "<div style='margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #fcd34d;'>
                              <h3 style='margin: 0; color: #92400e;'>" (:title alert) "</h3>
                              <p style='margin: 5px 0;'>" (:description alert) "</p>
                              <p style='margin: 5px 0; font-weight: 600; color: #78350f;'>Recommendation: " (:recommendation alert) "</p>
                          </div>"))
                   warning-alerts))
            "</div>"))
        
        "<p><a href='https://app.playerintel.ai/alerts' style='display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600;'>View Full Dashboard →</a></p>
        
        <p>These alerts are generated based on real-time analysis of your community data. Taking action on them can help maintain healthy community engagement.</p>
        
        <p>Questions? <a href='mailto:support@playerintel.ai'>Contact our support team</a></p>
        
        <hr style='margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;'>
        <p style='color: #6b7280; font-size: 14px;'>
            Alert generated at " timestamp "<br>
            Player Intelligence • <a href='https://playerintel.ai'>playerintel.ai</a>
        </p>
    </div>
</body>
</html>")))

(defn generate-alert-email-text
  "Generate plain text email body for alerts"
  [alerts customer]
  (let [severe-alerts (filter #(= (:severity %) "severe") alerts)
        warning-alerts (filter #(= (:severity %) "warning") alerts)
        timestamp (time-format/unparse (time-format/formatters :date-time) (time/now))]
    
    (str "COMMUNITY ALERT NOTIFICATION\n\n"
         "Hi " (or (:game_name customer) (:email customer)) ",\n\n"
         "We've detected important changes in your community that require attention:\n\n"
         
         (when (seq severe-alerts)
           (str "CRITICAL ISSUES (Immediate Action Required):\n"
                (str/join "\n" 
                  (map (fn [alert]
                         (str "- " (:title alert) "\n"
                              "  " (:description alert) "\n"
                              "  Recommendation: " (:recommendation alert) "\n"))
                       severe-alerts))
                "\n"))
         
         (when (seq warning-alerts)
           (str "WARNINGS (Monitor Closely):\n"
                (str/join "\n"
                  (map (fn [alert]
                         (str "- " (:title alert) "\n"
                              "  " (:description alert) "\n"
                              "  Recommendation: " (:recommendation alert) "\n"))
                       warning-alerts))
                "\n"))
         
         "View your full dashboard: https://app.playerintel.ai/alerts\n\n"
         "These alerts help you maintain healthy community engagement.\n\n"
         "Questions? Contact: support@playerintel.ai\n\n"
         "---\n"
         "Alert generated at " timestamp "\n"
         "Player Intelligence • https://playerintel.ai")))

;; --- Scheduled Alert Processing ---
(defn run-daily-alert-checks
  "Run alert checks for all active customers"
  []
  (try
    (let [active-customers (analytics/get-active-customers)]
      (log/info "Running alert checks for" (count active-customers) "customers")
      
      (let [results (map (fn [customer]
                          (let [result (run-alert-checks (:id customer))]
                            (assoc result :customer-id (:id customer))))
                        active-customers)
            
            total-alerts (reduce + (map :alerts-created results))
            severe-alerts (reduce + (map :severe-alerts results))]
        
        (log/info "Alert check completed:"
                 "Total alerts:" total-alerts
                 "Severe alerts:" severe-alerts)
        
        {:customers-checked (count active-customers)
         :total-alerts total-alerts
         :severe-alerts severe-alerts
         :results results}))
    
    (catch Exception e
      (log/error e "Failed to run daily alert checks")
      {:error (.getMessage e)})))

;; --- Alert Configuration ---
(defn update-alert-thresholds!
  "Update alert thresholds for a customer (future feature)"
  [customer-id new-thresholds]
  ;; Would store custom thresholds per customer
  (log/info "Alert thresholds updated for customer" customer-id new-thresholds))

;; --- Alert API Routes ---
(defn alert-routes []
  [["/api/alerts"
    {:middleware [auth/customer-middleware]}
    
    ["/check"
     {:post {:handler (fn [req]
                       (let [customer-id (get-in req [:customer :id])
                             result (run-alert-checks customer-id)]
                         {:status 200
                          :body result}))}}]
    
    ["/thresholds"
     {:get {:handler (fn [req]
                      {:status 200
                       :body {:thresholds alert-thresholds}})}
      
      :post {:handler (fn [req]
                       (let [customer-id (get-in req [:customer :id])
                             new-thresholds (:body req)]
                         (update-alert-thresholds! customer-id new-thresholds)
                         {:status 200
                          :body {:success true}}))}}]]])