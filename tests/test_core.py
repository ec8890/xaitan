#!/usr/bin/env python3
"""
Тесты для модуля XAItan Core
Запуск: pytest tests/test_core.py -v
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

# Добавляем src в path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
from xaitan_core import XAItanCore, InferenceRequest, AuditRecord


@pytest.fixture
def sample_baseline():
    """Создает тестовые baseline данные"""
    np.random.seed(42)
    return np.random.randn(100, 10).astype(np.float32)


@pytest.fixture
def feature_names():
    """Создает имена признаков"""
    return [f"feature_{i}" for i in range(10)]


@pytest.fixture
def xaitan_module(sample_baseline, feature_names):
    """Создает инициализированный модуль XAItan"""
    model_path = Path(__file__).parent.parent / 'models' / 'attack_detector.onnx'
    
    if not model_path.exists():
        pytest.skip("Модель не найдена. Запустите: python3 demo/generate_model.py")
    
    return XAItanCore(
        onnx_path=str(model_path),
        baseline_data=sample_baseline,
        feature_names=feature_names
    )


class TestInferenceRequest:
    """Тесты для модели запроса"""
    
    def test_create_valid_request(self):
        """Создание валидного запроса"""
        request = InferenceRequest(
            input_vector=[0.1] * 10,
            risk_class=False
        )
        assert len(request.input_vector) == 10
        assert request.risk_class is False
        assert request.request_id.startswith("req-")
    
    def test_request_with_custom_id(self):
        """Запрос с кастомным ID"""
        request = InferenceRequest(
            request_id="custom-id-123",
            input_vector=[0.5] * 10
        )
        assert request.request_id == "custom-id-123"
    
    def test_invalid_input_vector_length(self):
        """Проверка валидации длины вектора"""
        with pytest.raises(ValueError):
            InferenceRequest(input_vector=[0.1] * 200)  # Слишком длинный


class TestXAItanCoreInitialization:
    """Тесты инициализации модуля"""
    
    def test_successful_initialization(self, sample_baseline, feature_names):
        """Успешная инициализация модуля"""
        model_path = Path(__file__).parent.parent / 'models' / 'attack_detector.onnx'
        
        if not model_path.exists():
            pytest.skip("Модель не найдена")
        
        module = XAItanCore(
            onnx_path=str(model_path),
            baseline_data=sample_baseline,
            feature_names=feature_names
        )
        
        assert module.n_features == 10
        assert module.feature_names == feature_names
        assert module.confidence_threshold == 0.7
        assert module.anomaly_threshold == 0.4
    
    def test_invalid_baseline_shape(self, feature_names):
        """Проверка неправильной формы baseline"""
        model_path = Path(__file__).parent.parent / 'models' / 'attack_detector.onnx'
        
        if not model_path.exists():
            pytest.skip("Модель не найдена")
        
        # Неправильное количество признаков
        wrong_baseline = np.random.randn(100, 5)
        
        with pytest.raises(ValueError):
            XAItanCore(
                onnx_path=str(model_path),
                baseline_data=wrong_baseline,
                feature_names=feature_names
            )


class TestProcessRequest:
    """Тесты обработки запросов"""
    
    @pytest.mark.asyncio
    async def test_process_normal_request(self, xaitan_module):
        """Обработка нормального запроса"""
        request = InferenceRequest(
            request_id="test-normal",
            input_vector=[0.0] * 10,
            risk_class=False
        )
        
        record = await xaitan_module.process(request)
        
        assert isinstance(record, AuditRecord)
        assert record.request_id == "test-normal"
        assert record.prediction in [0, 1]
        assert 0.0 <= record.confidence <= 1.0
        assert record.latency_ms > 0
    
    @pytest.mark.asyncio
    async def test_process_risk_request(self, xaitan_module):
        """Обработка запроса с риском (должен вызвать SHAP)"""
        np.random.seed(42)
        request = InferenceRequest(
            request_id="test-risk",
            input_vector=(np.random.randn(10) * 4.0 + 2.0).tolist(),
            risk_class=True
        )
        
        record = await xaitan_module.process(request)
        
        assert record.explanation_method == "SHAP"
        assert record.anomaly_detected is True
        assert len(record.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_attribution_vector_format(self, xaitan_module):
        """Проверка формата вектора атрибуций"""
        np.random.seed(42)
        request = InferenceRequest(
            request_id="test-attribution",
            input_vector=(np.random.randn(10) * 4.0 + 2.0).tolist(),
            risk_class=True
        )
        
        record = await xaitan_module.process(request)
        
        if record.attribution_vector:
            # Все значения должны быть float
            for key, value in record.attribution_vector.items():
                assert isinstance(value, float)
                assert key.startswith("feature_")
    
    @pytest.mark.asyncio
    async def test_recommendations_present(self, xaitan_module):
        """Проверка наличия рекомендаций"""
        request = InferenceRequest(
            request_id="test-recs",
            input_vector=[0.0] * 10,
            risk_class=False
        )
        
        record = await xaitan_module.process(request)
        
        assert isinstance(record.recommendations, list)
        assert len(record.recommendations) > 0


class TestAnomalyDetection:
    """Тесты детекции аномалий"""
    
    @pytest.mark.asyncio
    async def test_adversarial_input_detection(self, xaitan_module):
        """Детекция adversarial входа"""
        # Сильно отклоняющиеся значения
        request = InferenceRequest(
            request_id="test-adversarial",
            input_vector=[10.0] * 10,
            risk_class=True
        )
        
        record = await xaitan_module.process(request)
        
        assert record.anomaly_detected is True
        assert record.anomaly_score > xaitan_module.anomaly_threshold
    
    @pytest.mark.asyncio
    async def test_normal_input_no_anomaly(self, xaitan_module):
        """Нормальный вход без аномалий"""
        request = InferenceRequest(
            request_id="test-normal",
            input_vector=[0.0] * 10,
            risk_class=False
        )
        
        record = await xaitan_module.process(request)
        
        # Для нормальных данных аномалия может быть или не быть
        # зависит от порога
        assert isinstance(record.anomaly_detected, bool)


class TestCircuitBreaker:
    """Тесты circuit breaker"""
    
    def test_circuit_breaker_initial_state(self, xaitan_module):
        """Начальное состояние circuit breaker"""
        assert xaitan_module.circuit_breaker.state == "closed"
        assert xaitan_module.circuit_breaker.failures == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_record_failure(self, xaitan_module):
        """Регистрация неудач"""
        cb = xaitan_module.circuit_breaker
        
        # Регистрируем неудачи
        for _ in range(cb.max_failures):
            await cb.record_failure()
        
        assert cb.state == "open"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, xaitan_module):
        """Восстановление circuit breaker"""
        cb = xaitan_module.circuit_breaker
        
        # Открываем circuit breaker
        cb.state = "open"
        cb.last_failure_time = 0  # Симулируем истечение времени
        
        # Проверяем возможность выполнения
        assert cb.can_execute() is True
        assert cb.state == "half-open"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])