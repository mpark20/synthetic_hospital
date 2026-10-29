import json
import argparse
import os

def process_file(file_path: str):
    with open(file_path, "r") as f:
        records = [json.loads(line) for line in f]
    
    return [{"messages": r["messages"], "tools": r["tools"]} for r in records]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_files", nargs="+", type=str, help="Paths to jsonl output trace files")
    parser.add_argument("--out-dir", type=str, default="results/sft_data", help="Path to output directory (default: results/sft_data)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    for file_path in args.in_files:
        if not file_path.endswith(".jsonl"):
            raise ValueError(f"Input files must be in .jsonl format, got {file_path}")
        elif not os.path.exists(file_path):
            raise FileNotFoundError(f"No file with path {file_path} exists.")
        
        out_file = os.path.join(args.out_dir, os.path.basename(file_path).replace(".jsonl", ".json"))
        convos = process_file(file_path)
        with open(out_file, "w") as f:
            json.dump(convos, f, indent=2)
