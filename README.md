XAItan Module
eXplainable AI & Anomaly Detection for ML Models in Cybersecurity

Python 3.10+ | FastAPI | ONNX Runtime | SHAP/LIME | Docker

Назначение

Модуль для объяснения предсказаний (XAI) и детекции аномалий в логике ML-моделей на этапе тестирования и экспорта. Разработан для интеграции в DevSecOps-пайплайны и SOC-процедуры.

Основные функции

- Объяснение «чёрного ящика»: Генерирует интерпретации предсказаний в реальном времени с помощью SHAP и LIME
- Детекция аномалий в атрибутах: Выявляет edge cases, concept drift и adversarial-атаки на основе анализа важности признаков
- Аудиторское протоколирование: Формирует неизменяемый JSON-лог с метаданными для соответствия требованиям (EU AI Act, ФСТЭК)
- Security-by-Design: Изоляция компонентов, non-root Docker, health-checks, CI/CD pipeline

Стек технологий

Ядро: Python 3.10+, Pydantic, ONNX Runtime
XAI: SHAP 0.43+, LIME 0.2.0+
API: FastAPI 0.104+, Uvicorn, Asyncio
DevSecOps: Docker, GitLab CI/CD, Trivy, Bandit
Logging: Structlog, JSON

Быстрый старт

1. Клонирование и установка

git clone <repository-url>
cd xaitan-module
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Подготовка данных

mkdir -p data models
python3 -c "
import numpy as np
np.save('data/baseline.npy', np.random.randn(100, 10).astype(np.float32))
open('data/feature_names.txt','w').write('\n'.join([f'feature_{i}' for i in range(10)]))
"

3. Запуск API

export XAI_API_KEYS="dev-key-123"
python3 -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

4. Документация

Открой в браузере: http://localhost:8000/docs

API Usage

POST /predict

Заголовок:
  X-API-Key: dev-key-123

Тело запроса:
{
  "request_id": "test-001",
  "input_vector": [1.5, -0.3, 2.1, 0.8, -1.2, 0.0, 3.4, -2.1, 0.5, 1.8],
  "risk_class": true,
  "model_version": "1.0.0"
}

Пример ответа:
{
  "prediction": 1,
  "confidence": 0.98,
  "explanation_method": "SHAP",
  "attribution_vector": {"feature_2": 0.45, "feature_6": -0.32},
  "anomaly_score": 0.66,
  "anomaly_detected": true,
  "recommendations": ["Выявлены паттерны, характерные для adversarial-атак"]
}

Конфигурация

Переменная среды        Описание                          По умолчанию
XAI_API_KEYS           Список API-ключей                 dev-key-123
CONFIDENCE_THRESHOLD   Порог уверенности для SHAP        0.7
ANOMALY_THRESHOLD      Порог детекции аномалий           0.4

Тестирование

Запуск демо-сценариев:
  python3 demo/run_demo.py

Запуск юнит-тестов:
  pytest tests/ -v

Структура проекта

.
├── src/
│   ├── xaitan_core.py    # Ядро модуля
│   └── api.py            # FastAPI обёртка
├── demo/
│   └── run_demo.py       # Демо-скрипт
├── data/                 # Baseline данные (игнорируется git)
├── models/               # ONNX модели (игнорируется git)
├── requirements.txt
├── .gitignore
└── README.md

Безопасность

- Все зависимости фиксируются с хешами в requirements.txt
- Контейнер запускается от непривилегированного пользователя
- Модель монтируется в read-only режиме
- Логи содержат SHA-256 хеш входных данных
- Аутентификация через заголовок X-API-Key

Лицензия

MIT

Контакты

Федоров Андрей Владимирович
Санкт-Петербургский государственный экономический университет
Направление: 10.03.01 Информационная безопасность
