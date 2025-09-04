FROM node:18-alpine

WORKDIR /app

# Install shadow-cljs globally
RUN npm install -g shadow-cljs

COPY shadow-cljs.edn package.json* ./
RUN npm install

COPY . .

EXPOSE 8080

CMD ["shadow-cljs", "watch", "app"]