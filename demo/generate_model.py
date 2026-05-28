#!/usr/bin/env python3
"""
Генерация тестовой ONNX-модели для демонстрации XAItan
Создаёт простую модель классификации на основе RandomForest
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from pathlib import Path

def generate_dummy_model():
    """Генерирует тестовую модель и сохраняет в ONNX формат"""
    print("🔄 Генерация тестовой модели RandomForest → ONNX...")
    
    # Генерируем синтетические данные для обучения
    # 500 образцов, 10 признаков
    np.random.seed(42)
    X = np.random.randn(500, 10).astype(np.float32)
    
    # Создаём простую логику классификации:
    # Класс 1, если сумма первых двух признаков > 0
    y = (X[:, 0] + X[:, 3] * 2 > 0).astype(np.int32)
    
    # Обучаем RandomForest
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Конвертируем в ONNX с zipmap=False для получения вероятностей вместо меток классов
    # Это критически важно для работы XAI методов (SHAP, LIME)
    initial_type = [('float_input', FloatTensorType([None, 10]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, options={type(model): {'zipmap': False}})
    
    # Сохраняем
    output_path = Path(__file__).parent.parent / 'models' / 'attack_detector.onnx'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    onnx.save_model(onnx_model, str(output_path))
    
    print(f"✅ Модель сохранена: {output_path}")
    print(f"📊 Точность на обучающей выборке: {model.score(X, y):.2%}")
    print(f"📁 Размер файла: {output_path.stat().st_size / 1024:.1f} KB")
    
    return output_path

if __name__ == "__main__":
    generate_dummy_model()
