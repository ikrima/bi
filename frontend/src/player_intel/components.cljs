(ns player-intel.components
  (:require [re-frame.core :as rf]
            [reagent.core :as r]
            [clojure.string :as str]))

;; --- Utility Components ---
(defn loading-spinner []
  [:div.flex.items-center.justify-center.p-8
   [:div.animate-spin.rounded-full.h-12.w-12.border-b-2.border-blue-600]])

(defn error-message [error]
  [:div.bg-red-50.border.border-red-200.rounded-md.p-4.mb-4
   [:div.flex
    [:div.flex-shrink-0
     [:svg.h-5.w-5.text-red-400 {:fill "currentColor" :viewBox "0 0 20 20"}
      [:path {:fill-rule "evenodd" 
              :d "M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"}]]]
    [:div.ml-3
     [:h3.text-sm.font-medium.text-red-800 "Error"]
     [:p.text-sm.text-red-700.mt-1 error]
     [:button.text-sm.text-red-600.underline.mt-2
      {:on-click #(rf/dispatch [:clear-error])}
      "Dismiss"]]]])

(defn metric-card [{:keys [title value subtitle color icon trend]}]
  [:div.bg-white.rounded-lg.shadow-sm.border.p-6.hover:shadow-md.transition-shadow
   [:div.flex.items-center.justify-between
    [:div
     [:p.text-sm.font-medium.text-gray-600 title]
     [:p.text-2xl.font-semibold.mt-2
      {:class (or color "text-gray-900")}
      value]
     (when subtitle
       [:p.text-xs.text-gray-500.mt-1 subtitle])]
    (when icon
      [:div.text-2xl.opacity-60 icon])]])

(defn status-indicator [{:keys [status text]}]
  (let [colors {:healthy "bg-green-100 text-green-800"
                :degraded "bg-yellow-100 text-yellow-800"
                :error "bg-red-100 text-red-800"
                :inactive "bg-gray-100 text-gray-800"}]
    [:span.inline-flex.items-center.px-2.5.py-0.5.rounded-full.text-xs.font-medium
     {:class (colors status :inactive)}
     text]))

;; --- Dashboard Header ---
(defn dashboard-header []
  (let [last-updated @(rf/subscribe [:data-freshness])
        auto-refresh? @(rf/subscribe [:auto-refresh?])
        health @(rf/subscribe [:health-indicator])]
    [:header.bg-white.shadow-sm.border-b
     [:div.max-w-7xl.mx-auto.px-4.sm:px-6.lg:px-8.py-4
      [:div.flex.justify-between.items-center
       [:div
        [:h1.text-2xl.font-semibold.text-gray-900 "Player Intelligence"]
        [:p.text-sm.text-gray-500.mt-1
         "Last updated: " last-updated]]
       [:div.flex.items-center.space-x-4
        ;; Health indicator
        [:div.flex.items-center.space-x-2
         [:div.h-3.w-3.rounded-full {:class (:color health)}]
         [:span.text-sm.text-gray-600 (:text health)]]
        
        ;; Auto-refresh toggle
        [:label.flex.items-center.cursor-pointer
         [:input.sr-only {:type "checkbox" 
                         :checked auto-refresh?
                         :on-change #(rf/dispatch [:toggle-auto-refresh])}]
         [:div.relative
          [:div.block.bg-gray-600.w-14.h-8.rounded-full]
          [:div.dot.absolute.left-1.top-1.bg-white.w-6.h-6.rounded-full.transition
           {:class (when auto-refresh? "transform translate-x-6")}]]
         [:span.ml-3.text-sm.text-gray-600 "Auto-refresh"]]
        
        ;; Refresh button
        [:button.bg-blue-600.hover:bg-blue-700.text-white.px-4.py-2.rounded-md.text-sm.font-medium.transition-colors
         {:on-click #(rf/dispatch [:refresh-data])}
         "Refresh Now"]]]]]))

;; --- Metrics Overview ---
(defn metrics-overview []
  (let [activity @(rf/subscribe [:activity-summary])
        sentiment-score @(rf/subscribe [:overall-sentiment-score])
        sentiment-color @(rf/subscribe [:sentiment-color])
        pipeline-running? @(rf/subscribe [:pipeline-running?])
        messages-processed @(rf/subscribe [:messages-processed])]
    [:div.grid.grid-cols-1.md:grid-cols-2.lg:grid-cols-4.gap-6.mb-8
     [metric-card {:title "Messages Analyzed"
                   :value (:messages activity)
                   :subtitle (str "Activity: " (name (:activity-level activity)))
                   :icon "📊"}]
     
     [metric-card {:title "Discussion Themes"
                   :value (:themes activity)
                   :subtitle "Distinct topics identified"
                   :icon "🎯"}]
     
     [metric-card {:title "Community Sentiment"
                   :value (str sentiment-score "%")
                   :color sentiment-color
                   :subtitle @(rf/subscribe [:sentiment-label])
                   :icon "😊"}]
     
     [metric-card {:title "Urgent Issues"
                   :value (:urgent activity)
                   :color (if (> (:urgent activity) 0) "text-red-600" "text-green-600")
                   :subtitle "Require attention"
                   :icon "⚠️"}]]))

;; --- Theme List Component ---
(defn theme-list []
  (let [themes @(rf/subscribe [:top-themes])
        loading? @(rf/subscribe [:loading?])]
    [:div.bg-white.rounded-lg.shadow-sm.border
     [:div.px-6.py-4.border-b.bg-gray-50
      [:h2.text-lg.font-semibold.text-gray-900.flex.items-center
       [:span.mr-2 "🎯"]
       "Main Discussion Topics"]]
     [:div.p-6
      (if loading?
        [loading-spinner]
        (if (empty? themes)
          [:p.text-gray-500.text-center.py-8 "No themes detected yet"]
          [:div.space-y-4
           (for [[idx theme] (map-indexed vector themes)]
             ^{:key idx}
             [:div.border-l-4.pl-4.py-2
              {:class (case (:sentiment theme)
                        "positive" "border-green-500 bg-green-50"
                        "negative" "border-red-500 bg-red-50"
                        "neutral" "border-gray-500 bg-gray-50"
                        "border-blue-500 bg-blue-50")}
              [:div.flex.justify-between.items-start
               [:div.flex-1
                [:h4.font-medium.text-gray-900 (:theme theme)]
                [:div.flex.items-center.space-x-4.mt-1
                 [:span.text-sm.text-gray-600 
                  (str (:size theme) " messages")]
                 [:span.text-xs.px-2.py-1.rounded.capitalize
                  {:class (case (:sentiment theme)
                            "positive" "bg-green-100 text-green-800"
                            "negative" "bg-red-100 text-red-800"
                            "bg-gray-100 text-gray-800")}
                  (:sentiment theme)]
                 (when (and (:urgency theme) (> (:urgency theme) 0.5))
                   [:span.text-xs.px-2.py-1.rounded.bg-orange-100.text-orange-800
                    (str "Urgency: " (Math/round (* 100 (:urgency theme))) "%")])]]
               [:div.text-right.text-sm.text-gray-500
                (str "#" (inc idx))]]])]))]))

;; --- Urgent Issues Component ---
(defn urgent-issues []
  (let [urgent-issues @(rf/subscribe [:urgent-issues])
        high-priority @(rf/subscribe [:high-urgency-issues])]
    [:div.bg-white.rounded-lg.shadow-sm.border
     [:div.px-6.py-4.border-b
      {:class (if (seq high-priority) "bg-red-50" "bg-gray-50")}
      [:h2.text-lg.font-semibold.flex.items-center
       {:class (if (seq high-priority) "text-red-900" "text-gray-900")}
       [:span.mr-2 "⚠️"]
       "Urgent Issues"
       (when (seq urgent-issues)
         [:span.ml-2.px-2.py-1.text-xs.rounded-full
          {:class (if (seq high-priority) 
                    "bg-red-200 text-red-800" 
                    "bg-yellow-200 text-yellow-800")}
          (count urgent-issues)])]]
     [:div.p-6
      (if (empty? urgent-issues)
        [:div.text-center.py-8
         [:div.text-4xl.mb-2 "✅"]
         [:p.text-gray-600 "No urgent issues detected"]
         [:p.text-sm.text-gray-500 "Community discussions appear stable"]]
        [:div.space-y-4
         (for [[idx issue] (map-indexed vector (take 5 urgent-issues))]
           ^{:key idx}
           [:div.border-l-4.pl-4.py-3.rounded-r-lg
            {:class (cond
                      (> (:urgency issue 0) 0.8) "border-red-600 bg-red-50"
                      (> (:urgency issue 0) 0.6) "border-orange-500 bg-orange-50"
                      :else "border-yellow-500 bg-yellow-50")}
            [:div.flex.justify-between.items-start.mb-2
             [:div.flex.items-center.space-x-2
              (when (:author issue)
                [:span.text-sm.font-medium.text-gray-700 (:author issue)])
              [:span.text-xs.px-2.py-1.rounded
               {:class (cond
                         (> (:urgency issue 0) 0.8) "bg-red-200 text-red-800"
                         (> (:urgency issue 0) 0.6) "bg-orange-200 text-orange-800"
                         :else "bg-yellow-200 text-yellow-800")}
               (str "Urgency: " (Math/round (* 100 (:urgency issue 0))) "%")]]
             (when (:theme issue)
               [:span.text-xs.text-gray-500 (:theme issue)])]
            [:p.text-sm.text-gray-800.mb-2 (:content issue)]
            (when (:source issue)
              [:span.text-xs.text-gray-400.capitalize 
               (str "Detected via: " (:source issue))])])])]]))

;; --- Sentiment Visualization ---
(defn sentiment-breakdown []
  (let [sentiment-dist @(rf/subscribe [:sentiment-distribution])
        overall-sentiment @(rf/subscribe [:sentiment])
        overall-score (:score overall-sentiment 50)]
    [:div.bg-white.rounded-lg.shadow-sm.border
     [:div.px-6.py-4.border-b.bg-gray-50
      [:h2.text-lg.font-semibold.text-gray-900.flex.items-center
       [:span.mr-2 "📈"]
       "Sentiment Analysis"]]
     [:div.p-6
      ;; Overall sentiment gauge
      [:div.mb-6
       [:div.flex.justify-between.items-center.mb-2
        [:span.text-sm.font-medium.text-gray-700 "Overall Community Sentiment"]
        [:span.text-2xl.font-bold
         {:class @(rf/subscribe [:sentiment-color])}
         (str overall-score "%")]]
       [:div.w-full.bg-gray-200.rounded-full.h-3
        [:div.h-3.rounded-full.transition-all.duration-500
         {:class (cond
                   (>= overall-score 80) "bg-green-500"
                   (>= overall-score 60) "bg-green-400"
                   (<= overall-score 20) "bg-red-500"
                   (<= overall-score 40) "bg-red-400"
                   :else "bg-yellow-400")
          :style {:width (str overall-score "%")}}]]
       [:div.flex.justify-between.text-xs.text-gray-500.mt-1
        [:span "Very Negative"]
        [:span "Neutral"]
        [:span "Very Positive"]]]
      
      ;; Sentiment distribution
      (when sentiment-dist
        [:div
         [:h4.text-sm.font-medium.text-gray-700.mb-3 "Sentiment Distribution"]
         [:div.space-y-2
          (for [item sentiment-dist]
            ^{:key (:sentiment item)}
            [:div.flex.items-center.space-x-3
             [:div.w-16.text-xs.capitalize (:sentiment item)]
             [:div.flex-1.bg-gray-200.rounded-full.h-2
              [:div.h-2.rounded-full
               {:class (case (:sentiment item)
                         "very_positive" "bg-green-600"
                         "positive" "bg-green-400"
                         "neutral" "bg-gray-400"
                         "negative" "bg-red-400"
                         "very_negative" "bg-red-600"
                         "bg-blue-400")
                :style {:width (str (:percentage item) "%")}}]]
             [:span.text-xs.text-gray-600.w-8.text-right 
              (str (:percentage item) "%")]])]])]]))

;; --- System Status ---
(defn system-status []
  (let [pipeline-status @(rf/subscribe [:pipeline-status])
        cache-stats @(rf/subscribe [:cache-stats])
        active-channels @(rf/subscribe [:active-channels])]
    [:div.bg-white.rounded-lg.shadow-sm.border
     [:div.px-6.py-4.border-b.bg-gray-50
      [:h2.text-lg.font-semibold.text-gray-900.flex.items-center
       [:span.mr-2 "⚙️"]
       "System Status"]]
     [:div.p-6.space-y-4
      ;; Pipeline status
      [:div.flex.justify-between.items-center
       [:span.text-sm.text-gray-600 "Message Pipeline"]
       [status-indicator {:status (if @(rf/subscribe [:pipeline-running?]) :healthy :inactive)
                          :text (if @(rf/subscribe [:pipeline-running?]) "Running" "Stopped")}]]
      
      ;; Active channels
      (when (seq active-channels)
        [:div
         [:span.text-sm.text-gray-600 "Active Channels"]
         [:div.mt-2.flex.flex-wrap.gap-2
          (for [channel active-channels]
            ^{:key channel}
            [:span.px-2.py-1.bg-blue-100.text-blue-800.text-xs.rounded channel])]])
      
      ;; Cache performance
      (when cache-stats
        [:div.flex.justify-between.items-center
         [:span.text-sm.text-gray-600 "Cache Entries"]
         [:span.text-sm.font-medium.text-gray-900 (:size cache-stats 0)]])
      
      ;; Messages processed
      [:div.flex.justify-between.items-center
       [:span.text-sm.text-gray-600 "Messages Processed"]
       [:span.text-sm.font-medium.text-gray-900 @(rf/subscribe [:messages-processed])]]
      
      ;; Error count
      (let [errors @(rf/subscribe [:pipeline-errors])]
        [:div.flex.justify-between.items-center
         [:span.text-sm.text-gray-600 "Processing Errors"]
         [:span.text-sm.font-medium
          {:class (if (> errors 0) "text-red-600" "text-green-600")}
          errors]])]]))

;; --- Recommendations Panel ---
(defn recommendations-panel []
  (let [recommendations @(rf/subscribe [:recommendations])
        high-priority @(rf/subscribe [:high-priority-recommendations])]
    (when (seq recommendations)
      [:div.bg-white.rounded-lg.shadow-sm.border
       [:div.px-6.py-4.border-b.bg-gray-50
        [:h2.text-lg.font-semibold.text-gray-900.flex.items-center
         [:span.mr-2 "💡"]
         "Recommendations"
         (when (seq high-priority)
           [:span.ml-2.px-2.py-1.bg-red-100.text-red-800.text-xs.rounded-full
            (str (count high-priority) " urgent")])]]
       [:div.p-6.space-y-4
        (for [[idx rec] (map-indexed vector (take 5 recommendations))]
          ^{:key idx}
          [:div.border-l-4.pl-4.py-2
           {:class (case (:priority rec)
                     "high" "border-red-500 bg-red-50"
                     "medium" "border-yellow-500 bg-yellow-50"
                     "border-blue-500 bg-blue-50")}
           [:div.flex.justify-between.items-start.mb-2
            [:h4.font-medium.text-gray-900 (:title rec)]
            [:span.text-xs.px-2.py-1.rounded.capitalize
             {:class (case (:priority rec)
                       "high" "bg-red-100 text-red-800"
                       "medium" "bg-yellow-100 text-yellow-800"
                       "bg-blue-100 text-blue-800")}
             (str (:priority rec) " priority")]]
           [:p.text-sm.text-gray-700.mb-2 (:description rec)]
           (when (:themes rec)
             [:div.text-xs.text-gray-500
              "Themes: " (str/join ", " (:themes rec))])])]]))))