(ns player-intel.cache
  (:require [taoensso.timbre :as log]
            [clojure.data.json :as json])
  (:import [java.util.concurrent ConcurrentHashMap]
           [java.time Instant]))

;; Simple in-memory cache for development
;; In production, this would use Redis

(defonce cache-store (ConcurrentHashMap.))

(defn cache-key
  "Generate cache key for digest"
  [channel-id & [suffix]]
  (str "digest:" (or channel-id "global") (when suffix (str ":" suffix))))

(defn cache-get 
  "Get value from cache"
  [key]
  (when-let [cached-item (.get cache-store key)]
    (let [{:keys [value expires-at]} cached-item]
      (if (and expires-at (< (.toEpochMilli (Instant/now)) expires-at))
        value
        (do
          (.remove cache-store key)
          nil)))))

(defn cache-set! 
  "Set value in cache with TTL in seconds"
  [key value ttl-seconds]
  (let [expires-at (+ (.toEpochMilli (Instant/now)) (* ttl-seconds 1000))
        cache-item {:value value :expires-at expires-at}]
    (.put cache-store key cache-item)
    value))

(defn cache-del! 
  "Remove value from cache"
  [key]
  (.remove cache-store key))

(defn with-cache
  "Cache function results with TTL"
  [cache-key ttl-seconds f & args]
  (if-let [cached (cache-get cache-key)]
    (do
      (log/debug "Cache hit for key:" cache-key)
      cached)
    (do
      (log/debug "Cache miss for key:" cache-key)
      (let [result (apply f args)]
        (cache-set! cache-key result ttl-seconds)
        result))))

(defn get-cache-stats
  "Get cache statistics"
  []
  {:size (.size cache-store)
   :keys (vec (.keySet cache-store))})