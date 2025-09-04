(ns player-intel.core
  (:require [reagent.dom :as rdom]
            [re-frame.core :as rf]
            [player-intel.events]  ; Load events
            [player-intel.subs]    ; Load subscriptions
            [player-intel.components :as c]))

;; --- Main Dashboard ---
(defn dashboard []
  (let [loading? @(rf/subscribe [:loading?])
        error @(rf/subscribe [:error])]
    [:div.min-h-screen.bg-gray-50
     ;; Header
     [c/dashboard-header]
     
     ;; Main content
     [:main.max-w-7xl.mx-auto.px-4.sm:px-6.lg:px-8.py-8
      ;; Error display
      (when error
        [c/error-message error])
      
      ;; Loading overlay
      (when loading?
        [:div.fixed.inset-0.bg-black.bg-opacity-25.flex.items-center.justify-center.z-50
         [c/loading-spinner]])
      
      ;; Metrics overview
      [c/metrics-overview]
      
      ;; Main grid layout
      [:div.grid.grid-cols-1.lg:grid-cols-3.gap-8
       ;; Left column - Main insights
       [:div.lg:col-span-2.space-y-8
        [c/theme-list]
        [c/urgent-issues]
        [c/recommendations-panel]]
       
       ;; Right column - Status and analytics
       [:div.space-y-8
        [c/sentiment-breakdown]
        [c/system-status]]]]]))

;; --- Settings Panel ---
(defn settings-panel []
  (let [auto-refresh? @(rf/subscribe [:auto-refresh?])
        selected-channel @(rf/subscribe [:selected-channel])]
    [:div.bg-white.rounded-lg.shadow-sm.border.p-6
     [:h3.text-lg.font-semibold.text-gray-900.mb-4 "Settings"]
     
     [:div.space-y-6
      ;; Channel selection
      [:div
       [:label.block.text-sm.font-medium.text-gray-700.mb-2 "Discord Channel"]
       [:input.w-full.px-3.py-2.border.border-gray-300.rounded-md.focus:outline-none.focus:ring-2.focus:ring-blue-500
        {:type "text"
         :placeholder "Enter channel ID"
         :value (or selected-channel "")
         :on-change #(rf/dispatch [:set-selected-channel (-> % .-target .-value)])}]]
      
      ;; Auto-refresh settings
      [:div
       [:label.block.text-sm.font-medium.text-gray-700.mb-2 "Auto-refresh"]
       [:div.flex.items-center.space-x-3
        [:input {:type "checkbox"
                 :checked auto-refresh?
                 :on-change #(rf/dispatch [:toggle-auto-refresh])}]
        [:span.text-sm.text-gray-600 "Automatically refresh data every 5 minutes"]]]
      
      ;; Actions
      [:div.flex.space-x-4
       [:button.px-4.py-2.bg-blue-600.text-white.rounded-md.hover:bg-blue-700.transition-colors
        {:on-click #(rf/dispatch [:fetch-digest selected-channel])}
        "Generate Digest"]
       [:button.px-4.py-2.bg-gray-600.text-white.rounded-md.hover:bg-gray-700.transition-colors
        {:on-click #(rf/dispatch [:invalidate-cache selected-channel])}
        "Clear Cache"]]]]))

;; --- App Shell ---
(defn app []
  (let [view-mode @(rf/subscribe [:view-mode])]
    [:div.app
     (case view-mode
       :settings [settings-panel]
       :dashboard [dashboard]
       [dashboard])]))

(defn ^:export init! []
  (rf/dispatch-sync [:initialize-db])
  (rf/dispatch [:start-refresh-timer])
  (rf/dispatch [:fetch-digest])
  (rf/dispatch [:fetch-pipeline-status])
  (rf/dispatch [:fetch-cache-stats])
  (rdom/render [app]
    (.getElementById js/document "app")))