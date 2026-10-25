"""Run in a download environment with huggingface_hub installed; resumable."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument('--revision', default='main')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
print(snapshot_download('Qwen/Qwen3.8-27B', revision=args.revision,
                        local_dir=root / 'checkpoints/Qwen3.8-27B', max_workers=4))
