# 🚀 Production Deployment Guide

This guide covers deploying the AI-Powered Data Analytics Tool in production environments.

## 📋 Pre-Deployment Checklist

### System Requirements
- **CPU**: 4+ cores (8+ recommended for heavy workloads)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ free space (models + data)
- **Network**: Stable internet for model downloads
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows Server

### Security Considerations
- [ ] Change default ports if needed
- [ ] Set up firewall rules
- [ ] Configure HTTPS/SSL certificates
- [ ] Set up user authentication (if required)
- [ ] Review data privacy requirements
- [ ] Configure backup strategies

---

## 🐳 Docker Deployment (Recommended)

### 1. Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  analytics-app:
    build: 
      context: .
      dockerfile: Dockerfile
    ports:
      - "80:8501"  # HTTP
      - "443:8501" # HTTPS (with reverse proxy)
    environment:
      - APP_ENV=production
      - OLLAMA_BASE_URL=http://ollama:11434
      - MAX_FILE_SIZE_MB=200
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_ENABLE_CORS=false
    volumes:
      - ./data:/app/data:ro
      - app_uploads:/app/uploads
      - ./logs:/app/logs
    depends_on:
      - ollama
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_MODELS=qwen2.5:8b,llama3.1:8b
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - analytics-app
    restart: unless-stopped

volumes:
  ollama_data:
  app_uploads:
  redis_data:
```

### 2. Nginx Configuration

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream analytics_app {
        server analytics-app:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        client_max_body_size 200M;

        location / {
            proxy_pass http://analytics_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Health check endpoint
        location /_stcore/health {
            proxy_pass http://analytics_app/_stcore/health;
        }
    }
}
```

### 3. Deploy with Docker

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Monitor logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale the application
docker-compose -f docker-compose.prod.yml up -d --scale analytics-app=3
```

---

## ☁️ Cloud Deployment Options

### AWS Deployment

#### Option A: AWS ECS with Fargate

1. **Build and push Docker image:**
```bash
# Build for AWS
docker build -t analytics-app .

# Tag for ECR
docker tag analytics-app:latest <account-id>.dkr.ecr.<region>.amazonaws.com/analytics-app:latest

# Push to ECR
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/analytics-app:latest
```

2. **Create ECS Task Definition:**
```json
{
  "family": "analytics-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "analytics-app",
      "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/analytics-app:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "APP_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/analytics-app",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Option B: AWS EC2 with Docker

```bash
# Launch EC2 instance (t3.large or larger)
# Install Docker and Docker Compose
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Deploy application
scp -r . ec2-user@your-instance:/home/ec2-user/analytics-app
ssh ec2-user@your-instance
cd analytics-app
docker-compose -f docker-compose.prod.yml up -d
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/analytics-app
gcloud run deploy analytics-app \
  --image gcr.io/PROJECT_ID/analytics-app \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
# Create resource group
az group create --name analytics-rg --location eastus

# Deploy container
az container create \
  --resource-group analytics-rg \
  --name analytics-app \
  --image your-registry/analytics-app:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8501 \
  --environment-variables APP_ENV=production
```

---

## 🔧 Environment Configuration

### Production Environment Variables

Create `.env.production`:

```bash
# Application
APP_ENV=production
DEBUG=false
LOG_LEVEL=INFO

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true

# File Upload
MAX_FILE_SIZE_MB=200
UPLOAD_DIR=/app/uploads

# OLLAMA
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:8b
OLLAMA_TIMEOUT=60

# Caching
ENABLE_CACHING=true
CACHE_TTL=3600

# Database (if using)
DATABASE_URL=postgresql://user:pass@localhost/analytics_db

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# Monitoring
SENTRY_DSN=your-sentry-dsn-here
```

### Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: analytics-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: analytics-app
  template:
    metadata:
      labels:
        app: analytics-app
    spec:
      containers:
      - name: analytics-app
        image: your-registry/analytics-app:latest
        ports:
        - containerPort: 8501
        env:
        - name: APP_ENV
          value: "production"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: analytics-app-service
spec:
  selector:
    app: analytics-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

---

## 📊 Monitoring and Logging

### Application Monitoring

1. **Health Checks:**
```python
# Add to app.py
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/app.log'),
        logging.StreamHandler()
    ]
)

# Add metrics endpoint
@st.cache_data
def get_app_metrics():
    return {
        'uptime': time.time() - start_time,
        'version': '1.0.0',
        'status': 'healthy'
    }
```

2. **Resource Monitoring:**
```bash
# Add resource monitoring with Docker stats
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
```

### Log Management

Configure centralized logging:

```yaml
# Add to docker-compose.prod.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
    labels: "app=analytics"
```

---

## 🔒 Security Hardening

### 1. Container Security

```dockerfile
# Add to Dockerfile
# Use non-root user
RUN addgroup -g 1001 -S appgroup && \
    adduser -S appuser -u 1001 -G appgroup

# Set proper permissions
RUN chown -R appuser:appgroup /app
USER appuser

# Security scanning
RUN apk add --no-cache dumb-init
ENTRYPOINT ["dumb-init", "--"]
```

### 2. Network Security

```bash
# Configure firewall
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# Docker network isolation
docker network create --driver bridge analytics-network
```

### 3. Data Security

- Encrypt data at rest
- Use HTTPS/TLS for data in transit
- Implement access controls
- Regular security updates
- Monitor for vulnerabilities

---

## 🔄 Backup and Recovery

### Automated Backups

```bash
#!/bin/bash
# backup_script.sh

# Backup volumes
docker run --rm -v analytics_ollama_data:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/ollama_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

# Backup application data
docker run --rm -v analytics_app_uploads:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/uploads_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

# Cleanup old backups (keep last 7 days)
find ./backups -name "*.tar.gz" -mtime +7 -delete
```

### Restore Process

```bash
#!/bin/bash
# restore_script.sh

BACKUP_FILE=$1

# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore data
docker run --rm -v analytics_ollama_data:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/$BACKUP_FILE -C /data

# Restart services
docker-compose -f docker-compose.prod.yml up -d
```

---

## 📈 Scaling Strategies

### Horizontal Scaling

1. **Load Balancer Configuration:**
```nginx
upstream analytics_backend {
    least_conn;
    server analytics-app-1:8501 weight=1 max_fails=3 fail_timeout=30s;
    server analytics-app-2:8501 weight=1 max_fails=3 fail_timeout=30s;
    server analytics-app-3:8501 weight=1 max_fails=3 fail_timeout=30s;
}
```

2. **Auto-scaling with Docker Swarm:**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml analytics

# Scale service
docker service scale analytics_analytics-app=5
```

### Vertical Scaling

- Increase CPU/Memory allocation
- Use larger instance types
- Optimize OLLAMA model size
- Implement caching strategies

---

## 🎯 Performance Optimization

### 1. Caching Strategy

```python
# Enhanced caching
import redis
import pickle

class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)
    
    def get_cached_analysis(self, data_hash):
        cached = self.redis_client.get(f"analysis:{data_hash}")
        return pickle.loads(cached) if cached else None
    
    def cache_analysis(self, data_hash, analysis, ttl=3600):
        self.redis_client.setex(f"analysis:{data_hash}", ttl, pickle.dumps(analysis))
```

### 2. Database Optimization

```python
# Add database for storing results
import sqlalchemy as sa

engine = sa.create_engine(DATABASE_URL)

def store_analysis_results(analysis_id, results):
    with engine.connect() as conn:
        conn.execute(
            "INSERT INTO analysis_results (id, results, created_at) VALUES (%s, %s, %s)",
            (analysis_id, json.dumps(results), datetime.utcnow())
        )
```

---

## 🚨 Troubleshooting Production Issues

### Common Production Problems

1. **High Memory Usage:**
```bash
# Monitor memory usage
docker stats --no-stream
free -h
```

2. **OLLAMA Model Loading Issues:**
```bash
# Check OLLAMA logs
docker-compose logs ollama

# Restart OLLAMA service
docker-compose restart ollama
```

3. **Application Crashes:**
```bash
# Check application logs
docker-compose logs analytics-app

# Check system resources
top
df -h
```

### Emergency Procedures

1. **Quick Rollback:**
```bash
# Rollback to previous version
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d --scale analytics-app=1
```

2. **Emergency Maintenance Mode:**
```nginx
# Add to nginx.conf
location / {
    return 503;
    add_header Content-Type text/plain;
    return "Service temporarily unavailable";
}
```

---

## 📞 Support and Maintenance

### Regular Maintenance Tasks

- [ ] **Weekly**: Check logs for errors
- [ ] **Weekly**: Monitor resource usage
- [ ] **Monthly**: Update base images
- [ ] **Monthly**: Review security patches
- [ ] **Quarterly**: Performance optimization review
- [ ] **Quarterly**: Backup testing

### Getting Production Support

1. **Log Collection:** Always collect relevant logs
2. **System Information:** Include system specs and resource usage
3. **Error Reproduction:** Steps to reproduce the issue
4. **Impact Assessment:** Business impact and urgency level

---

**🎉 Your production deployment is now ready to handle real-world analytics workloads!**