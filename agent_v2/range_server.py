"""Minimal HTTP server with Range (partial-content) support for video seeking."""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import re

_RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")

class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            super().do_GET()
            return

        fsize = os.path.getsize(path)
        range_header = self.headers.get("Range")

        if not range_header:
            super().do_GET()
            return

        m = _RANGE_RE.match(range_header)
        if not m:
            self.send_error(416)
            return

        start = int(m.group(1))
        end_str = m.group(2)
        end = int(end_str) if end_str else fsize - 1

        if start >= fsize or end >= fsize:
            self.send_error(416)
            return

        length = end - start + 1
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{fsize}")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            self.wfile.write(f.read(length))


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.chdir("/")
    server = HTTPServer(("", port), RangeHandler)
    print(f"Serving / on port {port} (Range support enabled)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
