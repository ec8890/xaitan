# src/xaitan_core.py
"""
XAItan Core Module
eXplainable AI for Taming Adversarial Networks
"""

import time
import hashlib
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import shap
import lime
import lime.lime_tabular
import structlog
from pydantic import BaseModel, Field

# Настройка логирования
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

class InferenceRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: f"req-{int(time.time())}")
    input_vector: List[float] = Field(..., min_length=1, max_length=128)
    risk_class: bool = False
    model_version: str = "1.0.0"

class AuditRecord(BaseModel):
    request_id: str
    input_hash: str
    timestamp: str
    prediction: int
    confidence: float
    explanation_method: str
    attribution_vector: Dict[str, float]
    anomaly_score: float
    anomaly_detected: bool
    latency_ms: float
    circuit_breaker_status: str = "closed"
    explanation_robustness: Optional[float] = None
    recommendations: List[str] = []

@dataclass
class ExplanationMetrics:
    method: str
    latency_ms: float
    stability_score: float = 0.0
    feature_count: int = 0
    max_attribution: float = 0.0
    entropy: float = 0.0

class CircuitBreaker:
    def __init__(self, max_failures: int = 3, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"
        self._lock = asyncio.Lock()

    async def record_failure(self):
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.max_failures:
                self.state = "open"

    def can_execute(self) -> bool:
        if self.state == "closed": return True
        if time.time() - self.last_failure_time > self.reset_timeout:
            self.state = "half-open"
            self.failures = 0
            return True
        return False

    async def record_success(self):
        async with self._lock:
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0

class XAItanCore:
    def __init__(self, onnx_path: str, baseline_data: np.ndarray, feature_names: List[str], 
                 confidence_threshold: float = 0.7, anomaly_threshold: float = 0.4):
        log.info("initializing_xaitan_core", onnx_path=onnx_path)
        
        if baseline_data.ndim != 2 or len(feature_names) != baseline_data.shape[1]:
            raise ValueError("Invalid baseline shape or feature count")
        
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.baseline = baseline_data.astype(np.float32)
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.confidence_threshold = confidence_threshold
        self.anomaly_threshold = anomaly_threshold
        self.circuit_breaker = CircuitBreaker()
        
        self.baseline_means = {f: float(np.mean(self.baseline[:, i])) for i, f in enumerate(self.feature_names)}
        self.baseline_stds = {f: float(np.std(self.baseline[:, i]) + 1e-8) for i, f in enumerate(self.feature_names)}
        
        self._init_explainers()
        log.info("xaitan_core_initialized", features=self.n_features)

    def _init_explainers(self):
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.baseline, mode="classification",
            feature_names=self.feature_names, discretize_continuous=False
        )
        self.shap_explainer = shap.KernelExplainer(self._predict_proba_wrapper, self.baseline[:200])

    def _predict_proba_wrapper(self, data):
        """Безопасная обертка для SHAP/LIME. Корректно обрабатывает батчи любого размера."""
        try:
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            input_name = self.session.get_inputs()[0].name
            out = self.session.run(None, {input_name: data.astype(np.float32)})[0]
            
            # Нормализация (softmax), если вышли логины
            if out.min() < 0 or out.max() > 1:
                e = np.exp(out - np.max(out, axis=1, keepdims=True))
                out = e / np.sum(e, axis=1, keepdims=True)
            
            # Приведение к 2D (n_samples, n_classes)
            if out.ndim == 1: 
                out = out.reshape(-1, 1)
            if out.shape[1] == 1:
                # Бинарный выход [p], превращаем в [1-p, p]
                p = out[:, 0]
                out = np.column_stack([1.0 - p, p])
            
            return out.astype(np.float32)
        except Exception as e:
            log.error("predict_proba_wrapper_error", error=str(e))
            n = data.shape[0] if data.ndim > 1 else 1
            return np.full((n, 2), 0.5, dtype=np.float32)

    def _select_explanation_method(self, confidence, risk_class, input_stats=None):
        if not self.circuit_breaker.can_execute(): return "none"
        if risk_class or confidence < self.confidence_threshold: return "SHAP"
        if input_stats and input_stats.get("input_anomaly_score", 0) > 0.5: return "SHAP"
        return "LIME"

    async def _generate_explanation(self, input_vector, method, pred_class):
        start = time.time()
        attribution = {}
        metrics = ExplanationMetrics(method=method, latency_ms=0.0)
        
        try:
            if method == "LIME":
                exp = await asyncio.to_thread(
                    self.lime_explainer.explain_instance, 
                    input_vector, self._predict_proba_wrapper, 
                    num_features=min(10, self.n_features), num_samples=30
                )
                attribution = {self.feature_names[idx]: float(w) for idx, w in exp.as_map().items() if idx < self.n_features}

            elif method == "SHAP":
                raw = await asyncio.to_thread(self.shap_explainer.shap_values, input_vector.reshape(1, -1))
                
                # Безопасное извлечение значений SHAP
                if isinstance(raw, list):
                    safe_class = pred_class if pred_class < len(raw) else 0
                    vals = raw[safe_class]
                else:
                    vals = raw
                
                # .flatten() гарантирует 1D массив без ошибок reshape
                vals = np.asarray(vals).flatten()
                
                for i, feat in enumerate(self.feature_names):
                    if i < len(vals):
                        attribution[feat] = float(vals[i])

            if attribution:
                vs = list(attribution.values())
                metrics.feature_count = sum(1 for v in vs if abs(v) > 1e-6)
                metrics.max_attribution = max(abs(v) for v in vs)
                
            metrics.latency_ms = (time.time() - start) * 1000
            return attribution, metrics
            
        except Exception as e:
            log.warning("explanation_failed", method=method, error=str(e))
            await self.circuit_breaker.record_failure()
            return {}, metrics

    def _calculate_input_anomaly_score(self, input_vector: List[float]) -> tuple[float, str]:
        """Возвращает (score, anomaly_type)"""
        if len(input_vector) != self.n_features: return 1.0, "invalid_length"
        
        z_scores = []
        max_z = 0.0
        for i, val in enumerate(input_vector):
            mean = self.baseline_means[self.feature_names[i]]
            std = self.baseline_stds[self.feature_names[i]]
            z = abs(val - mean) / std
            z_scores.append(min(z / 3.0, 1.0))
            max_z = max(max_z, z)
            
        score = float(np.mean(z_scores))
        
        if max_z > 4.5:
            return score, "adversarial_pattern"
        elif max_z > 3.0:
            return score, "feature_deviation"
        return score, "none"

    def _calculate_anomaly_score(self, attribution: Dict[str, float], confidence: float) -> tuple[float, List[str]]:
        if not attribution: return 0.0, []
        
        scores = []
        recs = []
        for feat in self.feature_names:
            attr_val = attribution.get(feat, 0.0)
            z = abs(attr_val) / self.baseline_stds[feat]
            scores.append(min(z / 3.0, 1.0))
            
        anomaly_score = float(np.mean(scores))
        
        if anomaly_score > 0.6:
            recs.append(" Обнаружена аномалия в пространстве атрибуций. Рекомендуется ручной аудит аналитиком SOC.")
        if confidence < 0.6:
            recs.append("⚠️ Низкая уверенность модели. Проверьте репрезентативность обучающей выборки.")
        if len([v for v in attribution.values() if abs(v) > 1.0]) > 3:
            recs.append("📉 Высокая дисперсия весов признаков. Возможна мультиколлинеарность или шум в данных.")
            
        return anomaly_score, recs

    def _generate_recommendations(self, anomaly_type: str, input_vector: List[float], 
                                  attribution: Dict[str, float], confidence: float) -> List[str]:
        recs = []
        if anomaly_type == "adversarial_pattern":
            recs.append("🚨 Выявлены паттерны, характерные для adversarial-атак. Рекомендуется активировать фильтры на уровне WAF/IPS.")
        elif anomaly_type == "feature_deviation":
            recs.append("📊 Значительное отклонение признаков от baseline. Проверьте источники данных на предмет искажений.")
            
        if confidence < self.confidence_threshold:
            recs.append("🔄 Низкая уверенность модели. Рассмотрите переобучение или ансамблирование.")
            
        if not recs and attribution:
            high_impact = [f for f, v in attribution.items() if abs(v) > 0.5]
            if high_impact:
                recs.append(f"💡 Ключевые факторы решения: {', '.join(high_impact[:3])}. Проверьте их валидность.")
                
        return recs if recs else ["✅ Аномалии не обнаружены. Решение модели стабильно."]

    async def process(self, request: InferenceRequest) -> AuditRecord:
        start = time.time()
        input_name = self.session.get_inputs()[0].name
        input_np = np.array(request.input_vector).reshape(1, -1).astype(np.float32)
        
        try:
            output = self.session.run(None, {input_name: input_np})
            probs = output[0].astype(np.float32)
            if probs.ndim == 1: probs = probs.reshape(1, -1)
            if probs.shape[1] == 1: probs = np.column_stack([1.0 - probs, probs])
            
            pred = int(np.argmax(probs, axis=1)[0])
            confidence = float(np.max(probs, axis=1)[0])
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")

        input_score, anomaly_type = self._calculate_input_anomaly_score(request.input_vector)
        method = self._select_explanation_method(confidence, request.risk_class, {"input_anomaly_score": input_score})
        
        attribution = {}
        robustness = None
        if method != "none" and (confidence < self.confidence_threshold or request.risk_class or anomaly_type != "none"):
            attribution, exp_metrics = await self._generate_explanation(input_np[0], method, pred)
            robustness = exp_metrics.stability_score

        attr_score, attr_recs = self._calculate_anomaly_score(attribution, confidence)
        final_score = max(input_score, attr_score)
        anomaly_detected = final_score > self.anomaly_threshold
        
        recommendations = self._generate_recommendations(anomaly_type, request.input_vector, attribution, confidence)
        if attr_recs:
            recommendations.extend(attr_recs)
            
        latency = (time.time() - start) * 1000
        
        return AuditRecord(
            request_id=request.request_id,
            input_hash=hashlib.sha256(json.dumps(request.input_vector).encode()).hexdigest(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            prediction=pred, confidence=round(confidence, 4),
            explanation_method=method,
            attribution_vector={k: round(v, 4) for k, v in attribution.items()},
            anomaly_score=round(final_score, 4), anomaly_detected=anomaly_detected,
            latency_ms=round(latency, 2), circuit_breaker_status=self.circuit_breaker.state,
            explanation_robustness=round(robustness, 4) if robustness else None,
            recommendations=recommendations
        )
