#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.llm import OpenAIResponsesLLM
from src.pipelines.factory import create_pipeline

DEFAULT_QUESTIONS = ROOT / "data" / "raw" / "financebench" / "questions.jsonl"


def load_questions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Questions snapshot not found: {path}")
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for line_no,line in enumerate(f,start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError("No questions loaded.")
    return rows

def parse_temperature(raw: str):
    if raw.lower() in {"none","null","omit"}:
        return None
    return float(raw)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--architecture",required=True,choices=["P0","P1","P2","P3"])
    p.add_argument("--questions",type=Path,default=DEFAULT_QUESTIONS)
    p.add_argument("--limit",type=int,default=None)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--model",default=os.getenv("OPENAI_MODEL"))
    p.add_argument("--max-output-tokens",type=int,default=256)
    p.add_argument("--temperature",default="none")
    p.add_argument("--device",default=None)
    args=p.parse_args()

    if not args.model:
        raise ValueError("Supply --model MODEL or set OPENAI_MODEL.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}\n"
            "Use a new path or pass --overwrite deliberately."
        )

    questions=load_questions(args.questions)
    selected=questions if args.limit is None else questions[:args.limit]

    llm=OpenAIResponsesLLM(
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        temperature=parse_temperature(args.temperature),
    )
    pipeline=create_pipeline(
        args.architecture,project_root=ROOT,llm=llm,device=args.device
    )

    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix(args.output.suffix+".tmp")
    try:
        with temp.open("w",encoding="utf-8",newline="\n") as f:
            for row in selected:
                result=pipeline.answer(row)
                f.write(json.dumps(result,ensure_ascii=False)+"\n")
                f.flush()
        temp.replace(args.output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise

    print("\nRun complete.")
    print(f"Architecture: {args.architecture}")
    print(f"Questions:    {len(selected)}")
    print(f"Output:       {args.output}")

if __name__=="__main__":
    main()
