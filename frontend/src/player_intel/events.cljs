(ns player-intel.events
  (:require [re-frame.core :as rf]
            [ajax.core :as ajax]
            [day8.re-frame.http-fx]
            [clojure.string :as str]))

;; --- Initial DB ---
(def initial-db
  {:loading? false
   :error nil
   :digest nil
   :digest-summary nil
   :pipeline-status nil
   :cache-stats nil
   :last-updated nil
   :auto-refresh? true
   :refresh-interval 300000  ; 5 minutes
   :selected-channel nil
   :view-mode :dashboard     ; :dashboard, :details, :settings
   :theme :light})

;; --- Events ---
(rf/reg-event-db
 :initialize-db
 (fn [_ _]
   initial-db))

;; Loading states
(rf/reg-event-db
 :set-loading
 (fn [db [_ loading?]]
   (assoc db :loading? loading?)))

(rf/reg-event-db
 :set-error
 (fn [db [_ error]]
   (assoc db 
          :error error
          :loading? false)))

;; Digest events
(rf/reg-event-fx
 :fetch-digest
 (fn [{:keys [db]} [_ channel-id]]
   (let [params (if channel-id 
                  {:channel_id channel-id}
                  {})]
     {:db (assoc db :loading? true :error nil)
      :http-xhrio {:method :get
                   :uri "/api/digest"
                   :params params
                   :response-format (ajax/json-response-format {:keywords? true})
                   :on-success [:digest-received]
                   :on-failure [:api-error]}})))

(rf/reg-event-db
 :digest-received
 (fn [db [_ digest]]
   (-> db
       (assoc :loading? false)
       (assoc :digest digest)
       (assoc :last-updated (js/Date.)))))

(rf/reg-event-fx
 :fetch-digest-summary
 (fn [{:keys [db]} [_ channel-id]]
   {:http-xhrio {:method :get
                 :uri "/api/digest-summary"
                 :params (when channel-id {:channel_id channel-id})
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success [:digest-summary-received]
                 :on-failure [:api-error]}}))

(rf/reg-event-db
 :digest-summary-received
 (fn [db [_ summary]]
   (assoc db :digest-summary summary)))

;; Pipeline events
(rf/reg-event-fx
 :fetch-pipeline-status
 (fn [_ _]
   {:http-xhrio {:method :get
                 :uri "/api/pipeline/status"
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success [:pipeline-status-received]
                 :on-failure [:api-error]}}))

(rf/reg-event-db
 :pipeline-status-received
 (fn [db [_ status]]
   (assoc db :pipeline-status status)))

(rf/reg-event-fx
 :start-pipeline
 (fn [_ [_ channel-id]]
   {:http-xhrio {:method :post
                 :uri "/api/pipeline/start"
                 :params {:channel_id channel-id}
                 :format (ajax/json-request-format)
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success [:pipeline-action-success]
                 :on-failure [:api-error]}}))

(rf/reg-event-fx
 :stop-pipeline
 (fn [_ [_ channel-id]]
   {:http-xhrio {:method :post
                 :uri "/api/pipeline/stop"
                 :params {:channel_id channel-id}
                 :format (ajax/json-request-format)
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success [:pipeline-action-success]
                 :on-failure [:api-error]}}))

(rf/reg-event-fx
 :pipeline-action-success
 (fn [_ [_ result]]
   {:dispatch [:fetch-pipeline-status]}))

;; Cache events
(rf/reg-event-fx
 :fetch-cache-stats
 (fn [_ _]
   {:http-xhrio {:method :get
                 :uri "/api/cache/stats"
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success [:cache-stats-received]
                 :on-failure [:api-error]}}))

(rf/reg-event-db
 :cache-stats-received
 (fn [db [_ stats]]
   (assoc db :cache-stats stats)))

(rf/reg-event-fx
 :invalidate-cache
 (fn [_ [_ channel-id]]
   {:http-xhrio {:method :post
                 :uri "/api/cache/invalidate"
                 :params {:channel_id channel-id}
                 :format (ajax/json-request-format)
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success [:cache-invalidated]
                 :on-failure [:api-error]}}))

(rf/reg-event-fx
 :cache-invalidated
 (fn [_ [_ result]]
   {:dispatch [:fetch-cache-stats]}))

;; UI events
(rf/reg-event-db
 :set-selected-channel
 (fn [db [_ channel-id]]
   (assoc db :selected-channel channel-id)))

(rf/reg-event-db
 :set-view-mode
 (fn [db [_ mode]]
   (assoc db :view-mode mode)))

(rf/reg-event-db
 :toggle-auto-refresh
 (fn [db _]
   (update db :auto-refresh? not)))

(rf/reg-event-db
 :set-refresh-interval
 (fn [db [_ interval]]
   (assoc db :refresh-interval interval)))

(rf/reg-event-db
 :set-theme
 (fn [db [_ theme]]
   (assoc db :theme theme)))

;; Error handling
(rf/reg-event-db
 :api-error
 (fn [db [_ error]]
   (-> db
       (assoc :loading? false)
       (assoc :error (if (map? error)
                       (or (:status-text error) 
                           (:response error)
                           "Unknown error")
                       (str error))))))

(rf/reg-event-db
 :clear-error
 (fn [db _]
   (assoc db :error nil)))

;; Auto-refresh logic
(defonce refresh-timer (atom nil))

(rf/reg-fx
 :start-auto-refresh
 (fn [interval]
   (when @refresh-timer
     (js/clearInterval @refresh-timer))
   (reset! refresh-timer
     (js/setInterval #(rf/dispatch [:refresh-data]) interval))))

(rf/reg-fx
 :stop-auto-refresh
 (fn []
   (when @refresh-timer
     (js/clearInterval @refresh-timer)
     (reset! refresh-timer nil))))

(rf/reg-event-fx
 :refresh-data
 (fn [{:keys [db]} _]
   (when (:auto-refresh? db)
     {:dispatch-n [[:fetch-digest (:selected-channel db)]
                   [:fetch-pipeline-status]
                   [:fetch-cache-stats]]})))

(rf/reg-event-fx
 :start-refresh-timer
 (fn [{:keys [db]} _]
   {:start-auto-refresh (:refresh-interval db)}))

(rf/reg-event-fx
 :stop-refresh-timer
 (fn [_ _]
   {:stop-auto-refresh nil}))