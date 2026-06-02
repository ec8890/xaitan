XAItan Module — Модуль объяснимого ИИ для кибербезопасности
============================================================

eXplainable AI & Anomaly Detection for ML Models in Cybersecurity

Python 3.10+ | FastAPI | ONNX Runtime | SHAP/LIME | Docker

📋 Описание

Модуль XAItan (eXplainable AI for Taming Adversarial Networks) предназначен для 
объяснения предсказаний ML-моделей и детекции аномалий в их логике на этапе тестирования.

**Проблема**: Современные нейросети работают как "чёрный ящик" — модель говорит 
"это DDoS-атака, блокируем IP", но не объясняет ПОЧЕМУ. В информационной безопасности 
нельзя слепо доверять модели — нужно понимать, какие именно признаки привели к решению.

**Решение**: XAItan подключается к готовой обученной модели и использует методы XAI 
(SHAP, LIME) для генерации человекочитаемых объяснений.

## Назначение

Модуль для объяснения предсказаний (XAI) и детекции аномалий в логике ML-моделей 
на этапе тестирования и экспорта. Разработан для интеграции в DevSecOps-пайплайны 
и SOC-процедуры.

## Основные функции

- **Объяснение «чёрного ящика»**: Генерирует интерпретации предсказаний в реальном 
  времени с помощью SHAP и LIME
- **Детекция аномалий в атрибутах**: Выявляет edge cases, concept drift и adversarial-атаки 
  на основе анализа важности признаков
- **Аудиторское протоколирование**: Формирует неизменяемый JSON-лог с метаданными для 
  соответствия требованиям (EU AI Act, ФСТЭК)
- **Security-by-Design**: Изоляция компонентов, non-root Docker, health-checks, CI/CD pipeline

## Стек технологий

- **Ядро**: Python 3.10+, Pydantic, ONNX Runtime
- **XAI**: SHAP 0.43+, LIME 0.2.0+
- **API**: FastAPI 0.104+, Uvicorn, Asyncio
- **DevSecOps**: Docker
- **Logging**: Structlog, JSON

## Быстрый старт

### 1. Клонирование и установка

```bash
git clone <repository-url>
cd xaitan-module
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Генерация тестовой модели

```bash
python3 demo/generate_model.py
```

### 3. Запуск демо-сценария

```bash
python3 demo/run_demo.py
```


## API Usage

### POST /predict


```

**Тело запроса:**
```json
{
  "request_id": "test-001",
  "input_vector": [1.5, -0.3, 2.1, 0.8, -1.2, 0.0, 3.4, -2.1, 0.5, 1.8],
  "risk_class": true,
  "model_version": "1.0.0"
}
```

**Пример ответа:**
```json
{
  "prediction": 1,
  "confidence": 0.98,
  "explanation_method": "SHAP",
  "attribution_vector": {"feature_2": 0.45, "feature_6": -0.32},
  "anomaly_score": 0.66,
  "anomaly_detected": true,
  "recommendations": ["Выявлены паттерны, характерные для adversarial-атак"]
}
```

## Конфигурация

| Переменная среды     | Описание                          | По умолчанию |
|---------------------|----------------------------------|--------------|
| CONFIDENCE_THRESHOLD| Порог уверенности для SHAP       | 0.7          |
| ANOMALY_THRESHOLD   | Порог детекции аномалий          | 0.4          |

## Тестирование

**Запуск демо-сценариев:**
```bash
python3 demo/run_demo.py
```

**Запуск юнит-тестов:**
```bash
pytest tests/ -v
```

## Структура проекта

```
.
├── src/
│   ├── xaitan_core.py    # Ядро модуля
│   └── api.py            # FastAPI обёртка
├── demo/
│   ├── generate_model.py # Генерация тестовой ONNX-модели
│   └── run_demo.py       # Демо-скрипт
├── tests/
│   └── test_core.py      # Юнит-тесты
├── data/                 # Baseline данные (игнорируется git)
├── models/               # ONNX модели (игнорируется git)
├── requirements.txt
├── .gitignore
└── README.md
```

## Безопасность

- Все зависимости фиксируются с хешами в requirements.lock.txt
- Контейнер запускается от непривилегированного пользователя
- Модель монтируется в read-only режиме
- Логи содержат SHA-256 хеш входных данных
- Аутентификация через заголовок X-API-Key

## Лицензия

MIT

## Контакты

Федоров Андрей Владимирович  
Санкт-Петербургский государственный экономический университет  
Направление: 10.03.01 Информационная безопасность

## Дипломная работа


**Год**: 2026
