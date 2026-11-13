import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from .agent import MonitorAgent
from .backends import create_backend
from .config import load_config
from .server import make_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/monitor.toml')
    parser.add_argument('--backend', choices=['mock', 'openai'])
    parser.add_argument('--votes', type=int, choices=[1, 5])
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('replay')
    replay.add_argument('--input', default='examples/demo.jsonl')
    replay.add_argument('--output', default='outputs/demo.jsonl')
    replay.add_argument('--realtime', action='store_true')
    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.votes is not None:
        config = replace(config, votes=args.votes, temperature=0.7 if args.votes == 5 else 0.0)
    if args.backend:
        config = replace(config, backend=args.backend)
    agent = MonitorAgent(config, create_backend(config))
    if args.command == 'serve':
        with make_server(agent, args.host, args.port) as server:
            print(f'Monitor on http://{args.host}:{args.port}; backend={config.backend}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return

    source_path, output_path = Path(args.input), Path(args.output)
    if source_path.resolve() == output_path.resolve():
        parser.error('input and output must differ')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open() as source, output_path.open('w') as destination:
        for line in source:
            if not line.strip():
                continue
            started = time.perf_counter()
            item = json.loads(line)
            if agent.episode_id is not None and item['episode_id'] != agent.episode_id:
                agent.reset()
            result = agent.observe(item)
            encoded = json.dumps(result, ensure_ascii=False)
            print(encoded)
            destination.write(encoded + '\n')
            if args.realtime:
                time.sleep(max(0, 1 / config.assessment_hz
                               - (time.perf_counter() - started)))
