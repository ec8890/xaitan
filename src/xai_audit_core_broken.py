# src/xaitan_core.py
"""
XAItan Module: Explainable AI for ML Model Auditing
Модуль для объяснения предсказаний и детекции аномалий в ML-моделях
"""

import time
import hashlib
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from enum import Enum

import numpy as np
import onnxruntime as ort
import shap
import lime
import lime.lime_tabular
import structlog
from pydantic import BaseModel, Field

# Настройка структурированного логирования
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()


class RiskLevel(str, Enum):
    """Уровни риска для предсказаний"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InferenceRequest(BaseModel):
    """Модель запроса к модулю"""
    request_id: str = Field(default_factory=lambda: f"req-{int(time.time())}")
    input_vector: List[float] = Field(..., min_items=1, max_items=128)
    risk_class: bool = False
    model_version: str = "1.0.0"


class AuditRecord(BaseModel):
    """Модель аудиторской записи"""
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


class CircuitBreaker:
    """
    Circuit Breaker паттерн для защиты от каскадных сбоев
    Глава 3.1 диплома
    """
    def __init__(self, max_failures: int = 3, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open

    def record_failure(self):
        """Записывает ошибку и проверяет, нужно ли разомкнуть цепь"""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.max_failures:
            self.state = "open"
            log.warning("circuit_breaker_opened", failures=self.failures)

    def can_execute(self) -> bool:
        """Проверяет, можно ли выполнить операцию"""
        if self.state == "closed":
            return True

        # Проверяем, прошло ли время сброса
        if time.time() - self.last_failure_time > self.reset_timeout:
            self.state = "half-open"
            self.failures = 0
            log.info("circuit_breaker_half_open")
            return True

        return False

    def record_success(self):
        """Записывает успешное выполнение"""
        if self.state == "half-open":
            self.state = "closed"
            self.failures = 0
            log.info("circuit_breaker_closed")


class XAIAuditModule:
    """
    Основной модуль XAItan
    Объясняет предсказания ML-моделей и детектирует аномалии
    """

    def __init__(self, onnx_path: str, baseline_data: np.ndarray, feature_names: List[str]):
        """
        Инициализация модуля

        Args:
            onnx_path: Путь к ONNX-модели
            baseline_data: Базовые данные для SHAP
            feature_names: Имена признаков
        """
        log.info("initializing_xaitan_module", onnx_path=onnx_path)

        # Загрузка ONNX модели с оптимизацией
        sess_opts = ort.SessionOptions()
        sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        sess_opts.inter_op_num_threads = 2

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
            sess_options=sess_opts
        )

        self.baseline = baseline_data
        self.feature_names = feature_names
        self.circuit_breaker = CircuitBreaker(max_failures=3, reset_timeout=60)

        # Инициализация объяснителей
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.baseline,
            mode="classification",
            feature_names=self.feature_names,
            discretize_continuous=False
        )

        self.shap_explainer = shap.KernelExplainer(
            self._predict_proba,
            self.baseline[:200]  # Используем подмножество для скорости
        )

        self.confidence_threshold = 0.7
        log.info("xaitan_module_initialized")

    def _predict_proba(self, data: np.ndarray) -> np.ndarray:
        """
        Обёртка для SHAP: возвращает вероятности классов

        Args:
            data: Входные данные

        Returns:
            Вероятности классов
        """
        input_name = self.session.get_inputs()[0].name
        return self.session.run(None, {input_name: data.astype(np.float32)})[0]

    def _calculate_anomaly_score(self, attribution: Dict[str, float], baseline_means: Dict[str, float]) -> float:
        """
        Расчёт скоринга аномалии через KL-дивергенцию
        Глава 2.4 диплома

        Args:
            attribution: Вектор атрибуций
            baseline_means: Средние значения baseline

        Returns:
            Скоринг аномалии [0, 1]
        """
        kl_div = 0.0
        eps = 1e-8

        for feat, val in attribution.items():
            mean = baseline_means.get(feat, 0.0)
            if abs(mean) > eps:
                kl_div += abs(val) * np.log(abs(val) / (abs(mean) + eps) + eps)

        # Нормализуем в [0, 1]
        return min(kl_div / 0.5, 1.0)

    async def process(self, request: InferenceRequest) -> AuditRecord:
        """
        Основная функция обработки запроса

        Args:
            request: Запрос на обработку

        Returns:
            Аудиторская запись с объяснением
        """
        start = time.time()

        # 1. Инференс модели
        input_name = self.session.get_inputs()[0].name
        input_np = np.array(request.input_vector).reshape(1, -1).astype(np.float32)

    output = self.session.run(None, {input_name: input_np})
    output_data = output[0]

    # Обработка разных форматов вывода модели
    if output_data.ndim == 1:
            # Для 1D вывода (вероятности одного класса)
            pred = int(output_data > 0.5) if output_data.shape[0] == 1 else int(np.argmax(output_data))
            confidence = float(output_data[pred]) if output_data.ndim == 1 else float(np.max(output_data))
    else:
        # Для 2D вывода (стандартный случай)
     pred = int(np.argmax(output_data, axis=1)[0])
     confidence = float(np.max(output_data, axis=1)[0])

        # 2. Определение триггеров для XAI
        trigger = (
            confidence < self.confidence_threshold or
            request.risk_class
        )

        # 3. Генерация объяснений с Circuit Breaker
        attribution = {}
        method = "none"
        cb_status = self.circuit_breaker.state

        if trigger and self.circuit_breaker.can_execute():
            try:
                if confidence >= 0.5 and not request.risk_class:
                    # LIME: быстро для штатных случаев
                    exp = self.lime_explainer.explain_instance(
                        input_np[0],
                        self._predict_proba,
                        num_features=10,
                        num_samples=30
                    )
                    attribution = dict(exp.as_map())
                    method = "LIME"
                else:
                    # SHAP: точно для критичных случаев
                    shap_vals = self.shap_explainer.shap_values(input_np)
                    if isinstance(shap_vals, list):
                        shap_vals = shap_vals[pred]

                    attribution = {
                        feat: float(shap_vals[0, i])
                        for i, feat in enumerate(self.feature_names)
                    }
                    method = "SHAP"

                self.circuit_breaker.record_success()

            except Exception as e:
                self.circuit_breaker.record_failure()
                log.warning(
                    "explainer_failure",
                    error=str(e),
                    request_id=request.request_id
                )
                cb_status = "open"

        # 4. Детекция аномалий
        baseline_means = {
            f: float(np.mean(self.baseline[:, i]))
            for i, f in enumerate(self.feature_names)
        }

        anomaly_score = self._calculate_anomaly_score(attribution, baseline_means) if attribution else 0.0
        anomaly_detected = anomaly_score > 0.7

        # 5. Протоколирование
        input_hash = hashlib.sha256(
            json.dumps(request.input_vector).encode()
        ).hexdigest()

        latency = (time.time() - start) * 1000

        record = AuditRecord(
            request_id=request.request_id,
            input_hash=input_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            prediction=pred,
            confidence=confidence,
            explanation_method=method,
            attribution_vector={k: round(v, 4) for k, v in attribution.items()},
            anomaly_score=round(anomaly_score, 4),
            anomaly_detected=anomaly_detected,
            latency_ms=round(latency, 2),
            circuit_breaker_status=cb_status
        )

        # Асинхронное логирование
        asyncio.create_task(self._audit_log(record))

        return record

    async def _audit_log(self, record: AuditRecord):
        """Асинхронная запись аудиторского лога"""
        log.info("audit_record", **record.model_dump())
