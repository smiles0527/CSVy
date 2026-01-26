# Docker Setup for Hockey ML Pipeline

## Quick Start

### Build and run both containers:
```bash
docker-compose up --build
```

### Access the services:
- **Streamlit Dashboard**: http://localhost:8501
- **MLflow UI**: http://localhost:5000

## Individual Container Commands

### Build containers:
```bash
docker-compose build
```

### Start in background:
```bash
docker-compose up -d
```

### View logs:
```bash
docker-compose logs -f
```

### Stop containers:
```bash
docker-compose down
```

### Stop and remove volumes (clean reset):
```bash
docker-compose down -v
```

## Container Details

### Streamlit (hockey-ml-dashboard)
- Port: 8501
- Mounts your local python/, data/, output/, config/ directories
- Auto-reloads when you edit code
- Connects to MLflow container for experiment tracking

### MLflow (hockey-ml-mlflow)
- Port: 5000
- Uses SQLite backend for metadata
- Persists experiments in Docker volume
- Accessible from Streamlit container via internal network

## Environment Variables

Set in docker-compose.yml or create a .env file:

```env
MLFLOW_TRACKING_URI=http://mlflow:5000
```

## Production Deployment

For production, consider:
1. Use a proper database (PostgreSQL) for MLflow
2. Add nginx reverse proxy
3. Enable SSL/TLS
4. Set resource limits
5. Use Docker secrets for credentials

Example production MLflow with PostgreSQL:
```yaml
mlflow:
  environment:
    - MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@db:5432/mlflow
```
