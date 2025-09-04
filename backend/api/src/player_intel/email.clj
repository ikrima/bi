(ns player-intel.email
  (:require [clojure.string :as str]
            [clojure.data.json :as json]
            [taoensso.timbre :as log]
            [environ.core :refer [env]]
            [clj-time.format :as time-format]
            [clj-time.core :as time]))

;; Email service configuration
(def email-config
  {:smtp-host (env :smtp-host "localhost")
   :smtp-port (env :smtp-port 587)
   :smtp-user (env :smtp-user)
   :smtp-password (env :smtp-password)
   :from-email (env :from-email "noreply@playerintel.ai")
   :from-name (env :from-name "Player Intelligence")})

;; --- Email Templates ---
(defn format-sentiment-label
  [sentiment-data]
  (let [score (:score sentiment-data 50)
        label (:label sentiment-data "neutral")]
    (str score "% " (str/capitalize label))))

(defn format-urgent-issues
  [urgent-issues]
  (if (empty? urgent-issues)
    "✅ No urgent issues detected - your community is stable!"
    (str "⚠️ " (count urgent-issues) " urgent issues requiring attention:\n\n"
         (str/join "\n"
           (map-indexed 
             (fn [idx issue]
               (str (inc idx) ". " (:content issue)
                    (when (:theme issue) (str " [" (:theme issue) "]"))
                    (when (:urgency issue) 
                      (str " (Urgency: " (Math/round (* 100 (:urgency issue))) "%)"))))
             (take 5 urgent-issues))))))

(defn format-top-themes
  [clusters]
  (if (empty? clusters)
    "No major themes identified in recent messages."
    (str/join "\n"
      (map-indexed
        (fn [idx cluster]
          (str (inc idx) ". " (:theme cluster)
               " (" (:size cluster) " messages, " 
               (str/capitalize (:sentiment cluster)) " sentiment"
               (when (> (:urgency cluster 0) 0.5)
                 (str ", " (Math/round (* 100 (:urgency cluster))) "% urgency"))
               ")"))
        (take 5 clusters)))))

(defn format-recommendations
  [recommendations]
  (if (empty? recommendations)
    "No specific recommendations at this time."
    (str/join "\n\n"
      (map (fn [rec]
             (str "• " (:title rec) " [" (str/upper-case (:priority rec)) " PRIORITY]\n"
                  "  " (:description rec)
                  (when (:themes rec)
                    (str "\n  Related themes: " (str/join ", " (:themes rec))))))
           (take 3 recommendations)))))

(defn generate-digest-email-html
  "Generate HTML email template for digest"
  [customer digest]
  (let [customer-name (or (:game_name customer) (:email customer))
        date-str (time-format/unparse (time-format/formatter "MMMM d, yyyy") (time/now))
        sentiment (format-sentiment-label (:sentiment digest))
        themes (format-top-themes (:clusters digest))
        urgent (format-urgent-issues (:urgent-issues digest))
        recommendations (format-recommendations (:recommendations digest))]
    
    (str "<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>Your Daily Player Intelligence Digest</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 0; background-color: #f8fafc; }
        .container { max-width: 600px; margin: 0 auto; background: white; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px 20px; text-align: center; }
        .header h1 { margin: 0; font-size: 28px; font-weight: 600; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; font-size: 16px; }
        .content { padding: 30px 20px; }
        .metric-card { background: #f8fafc; border-radius: 8px; padding: 20px; margin: 20px 0; border-left: 4px solid #3b82f6; }
        .metric-title { font-size: 18px; font-weight: 600; color: #1f2937; margin-bottom: 10px; }
        .metric-value { font-size: 24px; font-weight: 700; color: #3b82f6; margin-bottom: 5px; }
        .urgent-card { border-left-color: #ef4444; background: #fef2f2; }
        .urgent-title { color: #dc2626; }
        .positive-sentiment { color: #10b981; }
        .negative-sentiment { color: #ef4444; }
        .neutral-sentiment { color: #f59e0b; }
        .theme-list { list-style: none; padding: 0; }
        .theme-item { background: white; border: 1px solid #e5e7eb; border-radius: 6px; padding: 15px; margin: 10px 0; }
        .cta-button { display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600; margin: 20px 0; }
        .footer { background: #f9fafb; padding: 20px; text-align: center; color: #6b7280; font-size: 14px; border-top: 1px solid #e5e7eb; }
        .footer a { color: #3b82f6; text-decoration: none; }
        pre { background: #f3f4f6; padding: 15px; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class='container'>
        <div class='header'>
            <h1>🎮 Player Intelligence</h1>
            <p>Your Daily Community Digest • " date-str "</p>
        </div>
        
        <div class='content'>
            <h2>Hello " customer-name "! 👋</h2>
            <p>Here's your daily intelligence digest analyzing <strong>" (:message-count digest) " messages</strong> from your Discord community.</p>
            
            <div class='metric-card'>
                <div class='metric-title'>📊 Community Overview</div>
                <div class='metric-value'>" (:message-count digest) " messages analyzed</div>
                <div class='metric-value'>" (count (:clusters digest)) " discussion themes identified</div>
                <div class='metric-value " (cond 
                                                (> (:score (:sentiment digest) 50) 70) "positive-sentiment"
                                                (< (:score (:sentiment digest) 50) 40) "negative-sentiment"
                                                :else "neutral-sentiment") "'>" sentiment "</div>
            </div>
            
            " (when (seq (:urgent-issues digest))
                (str "<div class='metric-card urgent-card'>
                        <div class='metric-title urgent-title'>⚠️ Urgent Issues</div>
                        <pre>" urgent "</pre>
                     </div>")) "
            
            <div class='metric-card'>
                <div class='metric-title'>🎯 Top Discussion Themes</div>
                <pre>" themes "</pre>
            </div>
            
            " (when (seq (:recommendations digest))
                (str "<div class='metric-card'>
                        <div class='metric-title'>💡 Recommendations</div>
                        <pre>" recommendations "</pre>
                     </div>")) "
            
            <div class='metric-card'>
                <div class='metric-title'>📝 Summary</div>
                <p>" (:summary digest "No summary available.") "</p>
            </div>
            
            <p><a href='https://app.playerintel.ai/dashboard' class='cta-button'>View Full Dashboard →</a></p>
            
            <p><small>This digest was generated at " (time-format/unparse (time-format/formatter "HH:mm") (time/now)) " and covers the most recent community activity.</small></p>
        </div>
        
        <div class='footer'>
            <p>Player Intelligence • Transform Discord chaos into actionable insights</p>
            <p><a href='https://playerintel.ai'>Visit Website</a> | <a href='mailto:support@playerintel.ai'>Support</a> | <a href='%unsubscribe_url%'>Unsubscribe</a></p>
        </div>
    </div>
</body>
</html>")))

(defn generate-digest-email-text
  "Generate plain text email for digest"
  [customer digest]
  (let [customer-name (or (:game_name customer) (:email customer))
        date-str (time-format/unparse (time-format/formatter "MMMM d, yyyy") (time/now))]
    
    (str "PLAYER INTELLIGENCE - Daily Community Digest\n"
         "===============================================\n\n"
         
         "Hello " customer-name "!\n\n"
         
         "Here's your daily intelligence digest for " date-str ":\n\n"
         
         "📊 COMMUNITY OVERVIEW\n"
         "--------------------\n"
         "• " (:message-count digest) " messages analyzed\n"
         "• " (count (:clusters digest)) " discussion themes identified\n"
         "• Community sentiment: " (format-sentiment-label (:sentiment digest)) "\n\n"
         
         (when (seq (:urgent-issues digest))
           (str "⚠️ URGENT ISSUES\n"
                "---------------\n"
                (format-urgent-issues (:urgent-issues digest)) "\n\n"))
         
         "🎯 TOP DISCUSSION THEMES\n"
         "-----------------------\n"
         (format-top-themes (:clusters digest)) "\n\n"
         
         (when (seq (:recommendations digest))
           (str "💡 RECOMMENDATIONS\n"
                "------------------\n"
                (format-recommendations (:recommendations digest)) "\n\n"))
         
         "📝 SUMMARY\n"
         "----------\n"
         (:summary digest "No summary available.") "\n\n"
         
         "View your full dashboard: https://app.playerintel.ai/dashboard\n\n"
         
         "---\n"
         "Player Intelligence\n"
         "Transform Discord chaos into actionable insights\n"
         "https://playerintel.ai | support@playerintel.ai")))

;; --- Email Sending Functions ---
(defn send-email!
  "Send email using configured SMTP (mock implementation)"
  [to-email subject html-body text-body]
  (log/info "Sending email to:" to-email "Subject:" subject)
  
  ;; Mock implementation - in production would use actual SMTP
  (try
    (let [email-data {:to to-email
                     :from (:from-email email-config)
                     :subject subject
                     :html html-body
                     :text text-body
                     :timestamp (System/currentTimeMillis)}]
      
      ;; Simulate email sending delay
      (Thread/sleep 100)
      
      (log/info "Email sent successfully to:" to-email)
      {:success true :message "Email sent successfully"})
    
    (catch Exception e
      (log/error e "Failed to send email to:" to-email)
      {:success false :message (.getMessage e)})))

(defn send-digest!
  "Send digest email to customer"
  [customer digest]
  (let [subject (str "🎮 Daily Community Digest - " 
                    (or (:game_name customer) "Your Discord")
                    " (" (time-format/unparse (time-format/formatter "MMM d") (time/now)) ")")
        html-body (generate-digest-email-html customer digest)
        text-body (generate-digest-email-text customer digest)]
    
    (send-email! (:email customer) subject html-body text-body)))

(defn send-welcome-email!
  "Send welcome email to new trial customers"
  [customer]
  (let [subject "Welcome to Player Intelligence! 🎮 Your 7-Day Trial Starts Now"
        html-body (str "<!DOCTYPE html>
<html>
<head><title>Welcome to Player Intelligence</title></head>
<body style='font-family: sans-serif; line-height: 1.6; color: #333;'>
    <div style='max-width: 600px; margin: 0 auto; padding: 20px;'>
        <h1 style='color: #3b82f6;'>🎮 Welcome to Player Intelligence!</h1>
        
        <p>Hi there! 👋</p>
        
        <p>Thank you for starting your <strong>7-day free trial</strong> of Player Intelligence! We're excited to help you transform Discord chaos into actionable insights.</p>
        
        <h2>🚀 What happens next?</h2>
        <ul>
            <li>📊 We're already analyzing your Discord channel: <code>" (:discord_channel_id customer) "</code></li>
            <li>📧 You'll receive your first daily digest within 24 hours</li>
            <li>🎯 Each digest will show community themes, sentiment, and urgent issues</li>
            <li>💡 Get actionable recommendations for community management</li>
        </ul>
        
        <h2>⚙️ Quick Setup Tips</h2>
        <p>Make sure our bot has access to your Discord channel for the best results. If you need help with bot permissions, <a href='mailto:support@playerintel.ai'>contact our support team</a>.</p>
        
        <p style='background: #f0f9ff; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;'>
            <strong>💎 Your trial includes:</strong><br>
            ✓ Daily community digests<br>
            ✓ Sentiment analysis<br>
            ✓ Urgent issue detection<br>
            ✓ Theme clustering<br>
            ✓ Developer recommendations
        </p>
        
        <p><a href='https://app.playerintel.ai/dashboard' style='display: inline-block; background: #3b82f6; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; font-weight: 600;'>View Your Dashboard →</a></p>
        
        <p>Questions? Just reply to this email - we're here to help!</p>
        
        <p>Best regards,<br>The Player Intelligence Team</p>
        
        <hr style='margin: 30px 0; border: none; border-top: 1px solid #e5e7eb;'>
        <p style='color: #6b7280; font-size: 14px; text-align: center;'>
            Player Intelligence • Transform Discord chaos into actionable insights<br>
            <a href='https://playerintel.ai'>playerintel.ai</a> | <a href='mailto:support@playerintel.ai'>support@playerintel.ai</a>
        </p>
    </div>
</body>
</html>")
        text-body (str "Welcome to Player Intelligence!\n\n"
                      "Thank you for starting your 7-day free trial! We're excited to help you transform Discord chaos into actionable insights.\n\n"
                      "What happens next:\n"
                      "• We're analyzing your Discord channel: " (:discord_channel_id customer) "\n"
                      "• You'll receive your first daily digest within 24 hours\n"
                      "• Get community themes, sentiment, and urgent issues\n"
                      "• Receive actionable recommendations\n\n"
                      "View your dashboard: https://app.playerintel.ai/dashboard\n\n"
                      "Questions? Reply to this email!\n\n"
                      "Best regards,\nThe Player Intelligence Team\n"
                      "https://playerintel.ai")]
    
    (send-email! (:email customer) subject html-body text-body)))

(defn send-trial-expiring-email!
  "Send trial expiring reminder email"
  [customer days-remaining]
  (let [subject (str "⏰ Your Player Intelligence trial expires in " days-remaining " days")
        html-body (str "<!DOCTYPE html><html><head><title>Trial Expiring</title></head><body>"
                      "<p>Your Player Intelligence trial expires in " days-remaining " days.</p>"
                      "<p>Upgrade now to continue receiving daily community insights:</p>"
                      "<a href='https://app.playerintel.ai/upgrade'>Upgrade Now →</a>"
                      "</body></html>")
        text-body (str "Your Player Intelligence trial expires in " days-remaining " days.\n"
                      "Upgrade now: https://app.playerintel.ai/upgrade")]
    
    (send-email! (:email customer) subject html-body text-body)))