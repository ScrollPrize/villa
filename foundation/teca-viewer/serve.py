import argparse
import functools
import http.server
import os
from pathlib import Path
import threading
from urllib.parse import unquote
import webbrowser


def runtime_path(parts):
    return (
        (len(parts) == 1 and parts[0] in ("index.html", "styles.css"))
        or (len(parts) > 1 and parts[0] in ("src", "vendor"))
    ) and all(part and not part.startswith(".") for part in parts)


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def send_head(self):
        hosts = self.headers.get_all("Host", [])
        local_hosts = {
            name + suffix
            for name in ("127.0.0.1", "localhost")
            for suffix in ("", f":{self.server.server_port}")
        }
        if len(hosts) != 1 or hosts[0].lower() not in local_hosts:
            self.send_error(403, "Host must be localhost or 127.0.0.1 on the viewer's port")
            return None
        try:
            path = unquote(self.path.split("?", 1)[0], errors="strict")
            parts = ("index.html",) if path == "/" else path.split("/")[1:]
            if not path.startswith("/") or "\\" in path or "\x00" in path or not runtime_path(parts):
                self.send_error(404, "Not a viewer runtime file")
                return None
            root = Path(self.directory).resolve()
            target = root.joinpath(*parts).resolve()
            if not runtime_path(target.relative_to(root).parts) or not target.is_file():
                self.send_error(404, "Not a viewer runtime file")
                return None
            stream = target.open("rb")
        except (OSError, RuntimeError, ValueError) as error:
            self.send_error(404, "Cannot read viewer runtime file", explain=type(error).__name__)
            return None
        info = os.fstat(stream.fileno())
        self.send_response(200)
        self.send_header("Content-type", self.guess_type(str(target)))
        self.send_header("Content-Length", str(info.st_size))
        self.send_header("Last-Modified", self.date_time_string(info.st_mtime))
        self.end_headers()
        return stream

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Serve the viewer's runtime files on loopback.")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open", action="store_true", help="Open the viewer in the default browser")
    args = parser.parse_args()
    handler = functools.partial(ViewerHandler, directory=str(Path(__file__).resolve().parent))
    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        parser.exit(1, f"Cannot start viewer on port {args.port}: {error}.\n")
    with server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Viewer: {url}", flush=True)
        if args.open:
            threading.Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nViewer host stopped.")


if __name__ == "__main__":
    main()
