#!/usr/bin/env python3
"""
Демонстрация работы модуля XAItan
Запуск: python3 demo/run_demo.py
"""

import sys
import asyncio
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from xaitan_core import XAItanCore, InferenceRequest

async def main():
    print("=" * 70)
    print("😈 XAItan: eXplainable AI for Taming Adversarial Networks")
    print("Объяснение предсказаний и детекция аномалий")
    print("=" * 70)
    
    print("\n[1/4] Загрузка модели и инициализация модуля...")
    model_path = Path(__file__).parent.parent / 'models' / 'attack_detector.onnx'
    
    if not model_path.exists():
        print(f"❌ Модель не найдена: {model_path}")
        print("💡 Запустите: python3 demo/generate_model.py")
        return
    
    np.random.seed(42)
    baseline = np.random.randn(100, 10).astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(10)]
    
    module = XAItanCore(
        onnx_path=str(model_path),
        baseline_data=baseline,
        feature_names=feature_names
    )
    print("✅ XAItan Module initialized")
    
    print("\n[2/4] Генерация тестовых запросов...")
    
    test_cases = [
        {
            "name": " Нормальный трафик",
            "input": (np.random.randn(10) * 0.8).astype(np.float32).tolist(),
            "risk_class": False
        },
        {
            "name": "🔴 Аномалия / Adversarial Input",
            "input": (np.random.randn(10) * 4.0 + 2.0).astype(np.float32).tolist(),
            "risk_class": True
        },
        {
            "name": "🟡 Пограничный случай",
            "input": (np.random.randn(10) * 0.5).astype(np.float32).tolist(),
            "risk_class": False
        }
    ]
    
    print(f"   Создано {len(test_cases)} сценария")
    
    print("\n[3/4] Обработка запросов:")
    print("-" * 70)
    
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\n📋 Тест {i}: {case['name']}")
        
        request = InferenceRequest(
            request_id=f"demo-{i:03d}",
            input_vector=case['input'],
            risk_class=case['risk_class']
        )
        
        record = await module.process(request)
        results.append(record)
        
        print(f"   Предсказание:          Класс {record.prediction}")
        print(f"   Уверенность:           {record.confidence:.2%}")
        print(f"   Метод объяснения:      {record.explanation_method}")
        print(f"   Аномалия:              {'🔥 ОБНАРУЖЕНА' if record.anomaly_detected else '✅ НОРМА'}")
        print(f"   Скоринг аномалии:      {record.anomaly_score:.4f}")
        print(f"   Latency:               {record.latency_ms:.2f} мс")
        
        if record.attribution_vector:
            print(f"\n   🔍 Влияние признаков (топ-3):")
            sorted_attrs = sorted(record.attribution_vector.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            for feat, val in sorted_attrs:
                if abs(val) > 0.0001:
                    direction = "⬆️" if val > 0 else "⬇️"
                    print(f"      {direction} {feat}: {val:+.4f}")
        else:
            print(f"\n   ⚠️  Атрибуты не сгенерированы (модель детерминирована)")
        
        if record.recommendations:
            print(f"\n   💡 Рекомендации модуля:")
            for rec in record.recommendations:
                print(f"      {rec}")
        
        print("-" * 70)
    
    print("\n[4/4] 📊 ИТОГИ:")
    anomalies = sum(1 for r in results if r.anomaly_detected)
    print(f"   Запросов: {len(results)}")
    print(f"   Аномалий: {anomalies}")
    print("   Логи сохранены в JSON")
    print("\n✅ ДЕМОНСТРАЦИЯ XAItan ЗАВЕРШЕНА!")

if __name__ == "__main__":
    asyncio.run(main())
