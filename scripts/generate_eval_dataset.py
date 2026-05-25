#!/usr/bin/env python3
"""生成 RAG 评估数据集

用法:
    python scripts/generate_eval_dataset.py [--source aiops-docs] [--count 10]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.dataset_generator import dataset_generator


async def main():
    parser = argparse.ArgumentParser(description="生成 RAG 评估数据集")
    parser.add_argument("--source", default="aiops-docs", help="文档目录")
    parser.add_argument("--count", type=int, default=10, help="生成数量")
    args = parser.parse_args()

    dataset = await dataset_generator.generate_dataset(
        source_dir=args.source,
        count=args.count,
    )

    print(f"生成完成: {len(dataset)} 条 QA 对")
    for i, item in enumerate(dataset):
        print(f"\n[{i+1}] Q: {item['question'][:80]}...")
        print(f"    A: {item['ground_truth'][:80]}...")


if __name__ == "__main__":
    asyncio.run(main())
