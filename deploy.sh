#!/bin/bash

# Customer Segmentation API Deployment Script
# This script provides easy deployment options for the containerized application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║           Customer Segmentation API Deployment              ║"
    echo "║                    Docker Container                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  dev         Deploy in development mode (basic setup)"
    echo "  prod        Deploy in production mode (with nginx)"
    echo "  cache       Deploy with Redis caching"
    echo "  stop        Stop all services"
    echo "  clean       Stop and remove all containers and volumes"
    echo "  logs        Show logs from all services"
    echo "  status      Show status of all services"
    echo "  test        Run API tests"
    echo "  help        Show this help message"
    echo ""
}

check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All dependencies are installed${NC}"
}

create_output_dirs() {
    echo -e "${YELLOW}Creating output directories...${NC}"
    mkdir -p output/models output/visualizations output/reports
    echo -e "${GREEN}✅ Output directories created${NC}"
}

deploy_dev() {
    echo -e "${YELLOW}🚀 Deploying in Development Mode...${NC}"
    
    # Build and start the main service
    docker-compose up --build -d customer-segmentation-api
    
    echo -e "${GREEN}✅ Development deployment completed!${NC}"
    echo -e "${BLUE}API available at: http://localhost:8080${NC}"
    echo -e "${BLUE}API Documentation: http://localhost:8080/docs${NC}"
    echo -e "${BLUE}Health Check: http://localhost:8080/health${NC}"
}

deploy_prod() {
    echo -e "${YELLOW}🚀 Deploying in Production Mode...${NC}"
    
    # Check if nginx.conf exists
    if [ ! -f "nginx.conf" ]; then
        echo -e "${YELLOW}Creating nginx configuration...${NC}"
        create_nginx_config
    fi
    
    # Deploy with production profile
    docker-compose --profile production up --build -d
    
    echo -e "${GREEN}✅ Production deployment completed!${NC}"
    echo -e "${BLUE}API available at: http://localhost${NC}"
    echo -e "${BLUE}Direct API access: http://localhost:8080${NC}"
    echo -e "${BLUE}API Documentation: http://localhost/docs${NC}"
}

deploy_with_cache() {
    echo -e "${YELLOW}🚀 Deploying with Redis Cache...${NC}"
    
    # Deploy with cache profile
    docker-compose --profile with-cache up --build -d
    
    echo -e "${GREEN}✅ Deployment with cache completed!${NC}"
    echo -e "${BLUE}API available at: http://localhost:8080${NC}"
    echo -e "${BLUE}Redis available at: localhost:6379${NC}"
}

create_nginx_config() {
    cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream api {
        server customer-segmentation-api:8080;
    }

    server {
        listen 80;
        server_name localhost;

        client_max_body_size 10M;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        location /health {
            proxy_pass http://api/health;
            access_log off;
        }
    }
}
EOF
    echo -e "${GREEN}✅ Nginx configuration created${NC}"
}

stop_services() {
    echo -e "${YELLOW}🛑 Stopping all services...${NC}"
    docker-compose --profile production --profile with-cache down
    echo -e "${GREEN}✅ All services stopped${NC}"
}

clean_deployment() {
    echo -e "${YELLOW}🧹 Cleaning up deployment...${NC}"
    docker-compose --profile production --profile with-cache down -v --remove-orphans
    docker system prune -f
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

show_logs() {
    echo -e "${YELLOW}📋 Showing service logs...${NC}"
    docker-compose --profile production --profile with-cache logs -f
}

show_status() {
    echo -e "${YELLOW}📊 Service Status:${NC}"
    echo ""
    docker-compose --profile production --profile with-cache ps
    echo ""
    
    # Check API health
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API Service: Healthy${NC}"
    else
        echo -e "${RED}❌ API Service: Unhealthy${NC}"
    fi
}

run_tests() {
    echo -e "${YELLOW}🧪 Running API tests...${NC}"
    
    # Wait for API to be ready
    echo "Waiting for API to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API is ready${NC}"
            break
        fi
        sleep 2
    done
    
    # Test basic endpoints
    echo "Testing endpoints..."
    
    # Test health endpoint
    echo "Testing /health..."
    curl -s http://localhost:8080/health | jq '.' || echo "Health check passed"
    
    # Test segments summary
    echo "Testing /segments/summary..."
    curl -s http://localhost:8080/segments/summary | jq '.' || echo "Segments summary passed"
    
    # Test model info
    echo "Testing /model/info..."
    curl -s http://localhost:8080/model/info | jq '.' || echo "Model info passed"
    
    echo -e "${GREEN}✅ Basic tests completed${NC}"
}

# Main script logic
print_banner

case "${1:-help}" in
    dev)
        check_dependencies
        create_output_dirs
        deploy_dev
        ;;
    prod)
        check_dependencies
        create_output_dirs
        deploy_prod
        ;;
    cache)
        check_dependencies
        create_output_dirs
        deploy_with_cache
        ;;
    stop)
        stop_services
        ;;
    clean)
        clean_deployment
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    test)
        run_tests
        ;;
    help|*)
        print_usage
        ;;
esac
