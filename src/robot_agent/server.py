"""Local development FC adapter. Single agent/episode; not a production service."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json


def make_server(agent, host='127.0.0.1', port=8080):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json(200 if self.path == '/health' else 404,
                           {'status': 'ok', 'backend': agent.config.backend} if self.path == '/health' else {'error': 'not_found'})

        def do_POST(self):
            self.connection.settimeout(10)
            if self.path not in ('/observe', '/reset'):
                return self.send_json(404, {'error': 'not_found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8 * 1024 * 1024:
                    raise ValueError('body size must be between 1 byte and 8 MiB')
                data = json.loads(self.rfile.read(size))
                if self.path == '/reset':
                    if data != {}:
                        raise ValueError('reset expects {}')
                    agent.reset()
                    result = {'status': 'reset'}
                else:
                    result = agent.observe(data)
                self.send_json(200, result)
            except (ValueError, TypeError, TimeoutError) as exc:
                self.send_json(400, {'error': str(exc)})

    return HTTPServer((host, port), Handler)
