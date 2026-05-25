#!/usr/bin/env python3
"""运行 Ragas 评估

用法:
    python scripts/run_ragas_eval.py [--dataset path/to/dataset.json] [--count 5]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.ragas_evaluator import ragas_evaluator
from app.evaluation.dataset_generator import dataset_generator


async def main():
    parser = argparse.ArgumentParser(description="运行 Ragas 评估")
    parser.add_argument("--dataset", help="数据集 JSON 文件路径")
    parser.add_argument("--count", type=int, default=5, help="自动生成数量")
    args = parser.parse_args()

    dataset = None
    if args.dataset:
        with open(args.dataset, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"加载数据集: {args.dataset} ({len(dataset)} 条)")

    if not dataset:
        print(f"自动生成 {args.count} 条 QA 数据...")
        dataset = await dataset_generator.generate_dataset(
            source_dir="aiops-docs",
            count=args.count,
        )

    if not dataset:
        print("错误：没有可用数据集")
        return

    result = await ragas_evaluator.evaluate(dataset)
    print(f"\n评估结果 (run_id: {result['eval_run_id']})")
    print(f"时间: {result['timestamp']}")
    print(f"总数: {result['total_items']} 条")
    print(f"\n指标:")
    for metric, score in result["metrics"].items():
        print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
