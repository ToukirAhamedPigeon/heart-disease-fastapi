# Heart Disease Prediction API

## Features
- FastAPI backend
- Random Forest model
- Dockerized for easy deployment
- Endpoints:
  - `/health` : Health check
  - `/info` : Model info & features
  - `/predict` : Predict heart disease

## Run Locally
```bash
docker-compose build
docker-compose up
