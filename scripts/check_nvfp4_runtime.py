"""Report whether selected GPUs can enter the ModelOpt NVFP4 kernel path."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import torch


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    quant_path = args.model / 'hf_quant_config.json'
    index_path = args.model / 'model.safetensors.index.json'
    if not quant_path.is_file() or not index_path.is_file():
        raise SystemExit('checkpoint files are incomplete')
    quant = json.loads(quant_path.read_text())
    algorithms = sorted({layer.get('quant_algo') for layer in
                         quant['quantization']['quantized_layers'].values()})
    rows = []
    unsupported = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(index)
            rows.append({'index': index, 'name': torch.cuda.get_device_name(index),
                         'compute_capability': f'{capability[0]}.{capability[1]}',
                         'nvfp4_capability_ok': capability >= (8, 9)})
            if capability < (8, 9):
                unsupported.append(index)
    problems = []
    if 'NVFP4' not in algorithms:
        problems.append('checkpoint has no NVFP4 layers')
    if not rows:
        problems.append('CUDA is unavailable')
    if unsupported:
        problems.append(
            f'ModelOpt NVFP4 requires compute capability >= 8.9; unsupported GPUs: {unsupported}')
    report = {
        'status': 'ready_for_vllm_kernel_build' if not problems else 'blocked',
        'checkpoint': str(args.model),
        'quantization_algorithms': algorithms,
        'environment': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'vllm': package_version('vllm'),
            'flashinfer_python': package_version('flashinfer-python'),
            'compressed_tensors': package_version('compressed-tensors'),
        },
        'gpus': rows,
        'problems': problems,
    }
    rendered = json.dumps(report, indent=2) + '\n'
    print(rendered, end='', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    if problems:
        raise SystemExit('; '.join(problems))


if __name__ == '__main__':
    main()
