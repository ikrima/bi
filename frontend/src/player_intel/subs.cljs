(ns player-intel.subs
  (:require [re-frame.core :as rf]
            [clojure.string :as str]))

;; --- Basic subscriptions ---
(rf/reg-sub
 :loading?
 (fn [db _]
   (:loading? db)))

(rf/reg-sub
 :error
 (fn [db _]
   (:error db)))

(rf/reg-sub
 :last-updated
 (fn [db _]
   (:last-updated db)))

(rf/reg-sub
 :auto-refresh?
 (fn [db _]
   (:auto-refresh? db)))

(rf/reg-sub
 :selected-channel
 (fn [db _]
   (:selected-channel db)))

(rf/reg-sub
 :view-mode
 (fn [db _]
   (:view-mode db)))

(rf/reg-sub
 :theme
 (fn [db _]
   (:theme db)))

;; --- Digest subscriptions ---
(rf/reg-sub
 :digest
 (fn [db _]
   (:digest db)))

(rf/reg-sub
 :digest-summary
 (fn [db _]
   (:digest-summary db)))

(rf/reg-sub
 :message-count
 :<- [:digest]
 (fn [digest _]
   (:message-count digest 0)))

(rf/reg-sub
 :clusters
 :<- [:digest]
 (fn [digest _]
   (:clusters digest [])))

(rf/reg-sub
 :top-themes
 :<- [:clusters]
 (fn [clusters _]
   (->> clusters
        (take 5)
        (map #(select-keys % [:theme :size :sentiment :urgency])))))

(rf/reg-sub
 :urgent-issues
 :<- [:digest]
 (fn [digest _]
   (:urgent-issues digest [])))

(rf/reg-sub
 :high-urgency-issues
 :<- [:urgent-issues]
 (fn [urgent-issues _]
   (filter #(> (:urgency % 0) 0.7) urgent-issues)))

(rf/reg-sub
 :sentiment
 :<- [:digest]
 (fn [digest _]
   (:sentiment digest {:score 50 :label "neutral"})))

(rf/reg-sub
 :overall-sentiment-score
 :<- [:sentiment]
 (fn [sentiment _]
   (:score sentiment 50)))

(rf/reg-sub
 :sentiment-label
 :<- [:sentiment]
 (fn [sentiment _]
   (:label sentiment "neutral")))

(rf/reg-sub
 :sentiment-color
 :<- [:overall-sentiment-score]
 (fn [score _]
   (cond
     (>= score 80) "text-green-600"
     (>= score 60) "text-green-500"
     (<= score 20) "text-red-600"
     (<= score 40) "text-red-500"
     :else "text-yellow-500")))

(rf/reg-sub
 :insights
 :<- [:digest]
 (fn [digest _]
   (:insights digest [])))

(rf/reg-sub
 :recommendations
 :<- [:digest]
 (fn [digest _]
   (:recommendations digest [])))

(rf/reg-sub
 :high-priority-recommendations
 :<- [:recommendations]
 (fn [recommendations _]
   (filter #(= (:priority %) "high") recommendations)))

(rf/reg-sub
 :digest-summary-text
 :<- [:digest]
 (fn [digest _]
   (:summary digest "")))

;; --- Pipeline subscriptions ---
(rf/reg-sub
 :pipeline-status
 (fn [db _]
   (:pipeline-status db)))

(rf/reg-sub
 :pipeline-running?
 :<- [:pipeline-status]
 (fn [status _]
   (:running? status false)))

(rf/reg-sub
 :active-channels
 :<- [:pipeline-status]
 (fn [status _]
   (:active-channels status [])))

(rf/reg-sub
 :pipeline-stats
 :<- [:pipeline-status]
 (fn [status _]
   (:stats status {})))

(rf/reg-sub
 :messages-processed
 :<- [:pipeline-stats]
 (fn [stats _]
   (:processed stats 0)))

(rf/reg-sub
 :pipeline-errors
 :<- [:pipeline-stats]
 (fn [stats _]
   (:errors stats 0)))

;; --- Cache subscriptions ---
(rf/reg-sub
 :cache-stats
 (fn [db _]
   (:cache-stats db)))

(rf/reg-sub
 :cache-size
 :<- [:cache-stats]
 (fn [stats _]
   (:size stats 0)))

;; --- Computed subscriptions ---
(rf/reg-sub
 :dashboard-health
 :<- [:pipeline-running?]
 :<- [:pipeline-errors]
 :<- [:message-count]
 (fn [[running? errors messages] _]
   (cond
     (not running?) :stopped
     (> errors 5) :degraded
     (< messages 10) :low-activity
     :else :healthy)))

(rf/reg-sub
 :health-indicator
 :<- [:dashboard-health]
 (fn [health _]
   (case health
     :healthy {:color "bg-green-500" :text "System Healthy" :icon "✓"}
     :degraded {:color "bg-yellow-500" :text "Degraded Performance" :icon "⚠"}
     :low-activity {:color "bg-blue-500" :text "Low Activity" :icon "ℹ"}
     :stopped {:color "bg-red-500" :text "Pipeline Stopped" :icon "✗"})))

(rf/reg-sub
 :urgent-count
 :<- [:urgent-issues]
 (fn [urgent-issues _]
   (count urgent-issues)))

(rf/reg-sub
 :cluster-count
 :<- [:clusters]
 (fn [clusters _]
   (count clusters)))

(rf/reg-sub
 :sentiment-distribution
 :<- [:clusters]
 (fn [clusters _]
   (let [sentiments (map :sentiment clusters)
         total (count sentiments)]
     (when (> total 0)
       (->> sentiments
            (frequencies)
            (map (fn [[sentiment count]]
                   {:sentiment sentiment
                    :count count
                    :percentage (Math/round (* 100 (/ count total)))}))
            (sort-by :count >))))))

(rf/reg-sub
 :theme-popularity
 :<- [:clusters]
 (fn [clusters _]
   (->> clusters
        (map #(select-keys % [:theme :size :sentiment]))
        (sort-by :size >)
        (take 10))))

(rf/reg-sub
 :activity-summary
 :<- [:message-count]
 :<- [:cluster-count]
 :<- [:urgent-count]
 (fn [[messages clusters urgent] _]
   {:messages messages
    :themes clusters
    :urgent urgent
    :activity-level (cond
                      (> messages 500) :high
                      (> messages 100) :medium
                      (> messages 10) :low
                      :else :very-low)}))

;; --- Time-based subscriptions ---
(rf/reg-sub
 :last-updated-formatted
 :<- [:last-updated]
 (fn [last-updated _]
   (when last-updated
     (.toLocaleTimeString last-updated))))

(rf/reg-sub
 :data-freshness
 :<- [:last-updated]
 (fn [last-updated _]
   (if last-updated
     (let [now (js/Date.)
           diff (- (.getTime now) (.getTime last-updated))
           minutes (Math/floor (/ diff 60000))]
       (cond
         (< minutes 1) "Just now"
         (< minutes 5) (str minutes " minutes ago")
         (< minutes 60) (str minutes " minutes ago")
         :else "Over an hour ago"))
     "Never")))