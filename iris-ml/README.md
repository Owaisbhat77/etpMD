# Iris ML Deployment (Simple Classification)

This folder shows the **complete ML deployment lifecycle**:

```
Train model → Save model → Build API → Docker → Run → Inference
```

## 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Train the model
```bash
python train.py
```
This creates `model.joblib`.

## 3) Run the API (without Docker)
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 4) Test inference
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## 5) Docker build & run
```bash
docker build -t iris-ml .
docker run -p 8000:8000 iris-ml
```
