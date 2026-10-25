import argparse
import base64
from dataclasses import replace
import json
from pathlib import Path
import time

from .agent import ObserverAgent
from .evaluate import evaluate
from .backends import create_backend
from .config import load_config
from .server import make_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.toml')
    parser.add_argument('--backend', choices=['mock', 'openai'])
    commands = parser.add_subparsers(dest='command', required=True)
    replay = commands.add_parser('replay')
    replay.add_argument('--input', default='examples/demo.jsonl')
    replay.add_argument('--output', default='outputs/demo.jsonl')
    replay.add_argument('--realtime', action='store_true', help='Pace requests at target_hz (not a latency guarantee)')
    check = commands.add_parser('observe-image')
    check.add_argument('image', type=Path)
    score = commands.add_parser('evaluate')
    score.add_argument('--predictions', required=True, type=Path)
    score.add_argument('--labels', required=True, type=Path)
    serve = commands.add_parser('serve')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.backend:
        config = replace(config, backend=args.backend)
    if args.command == 'evaluate':
        def read_rows(path):
            with path.open() as source:
                return [json.loads(line) for line in source if line.strip()]
        print(json.dumps(evaluate(read_rows(args.predictions), read_rows(args.labels),
                                  config.phases), indent=2))
        return
    agent = ObserverAgent(config, create_backend(config))
    if args.command == 'observe-image':
        image = args.image.read_bytes()
        if image.startswith(b'\x89PNG\r\n\x1a\n'):
            mime = 'png'
        elif image.startswith(b'\xff\xd8\xff'):
            mime = 'jpeg'
        else:
            parser.error('image must be PNG or JPEG')
        result = agent.observe({'episode_id': 'image-smoke', 'frame_id': 'f1',
                                'timestamp_s': 0, 'images': [
                                    f'data:image/{mime};base64,' + base64.b64encode(image).decode()]})
        print(json.dumps(result, indent=2))
        if result['backend_error_type']:
            raise SystemExit(1)
        return
    if args.command == 'serve':
        with make_server(agent, args.host, args.port) as server:
            print(f'Observer on http://{args.host}:{args.port}; backend={config.backend}', flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    destination = Path(args.output)
    if Path(args.input).resolve() == destination.resolve():
        parser.error('input and output must differ')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input).open() as source, destination.open('w') as output:
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
            output.write(encoded + '\n')
            if args.realtime:
                time.sleep(max(0, 1 / config.target_hz - (time.perf_counter() - started)))
