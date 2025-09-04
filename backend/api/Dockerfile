FROM clojure:temurin-17-lein-alpine

WORKDIR /app

COPY deps.edn .
RUN clojure -P

COPY . .

EXPOSE 3000

CMD ["clojure", "-M:run"]