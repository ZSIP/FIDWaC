#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIDWAC v2 - web GUI (Flask)
===========================
Starts the local web application from the webui/ package.
Open the printed address in a browser.

Usage:
    python3 app.py                  # http://127.0.0.1:5010
    python3 app.py --port 8080
    python3 app.py --host 0.0.0.0   # other machines can reach it; the app has no login, use on trusted networks only
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from webui import create_app  # noqa: E402  (needs the sys.path entry above)


def main(argv=None):
    parser = argparse.ArgumentParser(description="FIDWAC v2 web GUI")
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5010, help="port (default: 5010)")
    args = parser.parse_args(argv)

    # The page polls the job status about once per second; show only Werkzeug errors in the console.
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(f"FIDWAC v2 web GUI: http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    create_app().run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
