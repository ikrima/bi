.PHONY: run build stop clean health test

# Start all services
run:
	docker-compose up --build

# Build all images
build:
	docker-compose build

# Stop all services
stop:
	docker-compose down

# Stop and remove all containers, networks, and volumes
clean:
	docker-compose down -v --rmi all

# Check health of all services
health:
	@echo "Checking API health..."
	curl -f http://localhost:3000/health || echo "API not healthy"
	@echo "\nChecking ML service health..."
	curl -f http://localhost:8000/health || echo "ML service not healthy"
	@echo "\nChecking frontend..."
	curl -f http://localhost:8080 || echo "Frontend not accessible"

# Run in background
run-bg:
	docker-compose up --build -d

# View logs
logs:
	docker-compose logs -f

# Run specific service
run-api:
	docker-compose up --build api

run-ml:
	docker-compose up --build ml-service

run-frontend:
	docker-compose up --build frontend

# Development helpers
dev-api:
	cd backend/api && clojure -M:run

dev-ml:
	cd backend/ml && python main.py

dev-frontend:
	cd frontend && npx shadow-cljs watch app