# src/api.py
"""
XAItan FastAPI Wrapper
REST API для взаимодействия с модулем XAItan
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import structlog
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent))

from xaitan_core import XAItanCore, InferenceRequest, AuditRecord

log = structlog.get_logger()

app = FastAPI(
    title="XAItan API",
    description="eXplainable AI for Taming Adversarial Networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

module: Optional[XAItanCore] = None

# API ключи для аутентификации
API_KEY_ENV = os.getenv("XAI_API_KEYS", "dev-key-123")
VALID_API_KEYS = [key.strip() for key in API_KEY_ENV.split(",") if key.strip()]

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API ключ для доступа к модулю XAItan"
)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Проверка API ключа"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    return api_key

@app.on_event("startup")
async def startup_event():
    global module
    log.info("starting_xaitan_api")
    
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "attack_detector.onnx"
    baseline_path = base_dir / "data" / "baseline.npy"
    features_path = base_dir / "data" / "feature_names.txt"
    
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}")
    if not baseline_path.exists():
        raise RuntimeError(f"Baseline data not found at {baseline_path}")
    if not features_path.exists():
        raise RuntimeError(f"Feature names not found at {features_path}")
        
    baseline_data = np.load(baseline_path)
    feature_names = features_path.read_text().splitlines()
    
    module = XAItanCore(
        onnx_path=str(model_path),
        baseline_data=baseline_data,
        feature_names=feature_names,
        confidence_threshold=0.7,
        anomaly_threshold=0.4
    )
    log.info("xaitan_module_loaded")

@app.get("/health", tags=["System"])
async def health_check():
    """Проверка работоспособности API"""
    return {"status": "ok", "module": "loaded" if module else "uninitialized"}

@app.post("/predict", response_model=AuditRecord, tags=["Inference"])
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """
    Выполняет предсказание с объяснением и детекцией аномалий.
    
    Требуется API ключ в заголовке: X-API-Key
    """
    if not module:
        raise HTTPException(status_code=503, detail="Module not initialized")
    
    try:
        record = await module.process(request)
        return record
    except Exception as e:
        log.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = FastAPI.openapi(app)
    
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API ключ (передаётся в заголовке запроса)"
        }
    }
    
    if "/predict" in openapi_schema["paths"]:
        openapi_schema["paths"]["/predict"]["post"]["security"] = [{"ApiKeyAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
