#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request


def check(url: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assistant", default="")
    parser.add_argument("--user", default="http://localhost:8001/v1")
    parser.add_argument("--teacher", default="")
    args = parser.parse_args()
    targets = [("user", args.user)]
    if args.assistant:
        targets.insert(0, ("assistant", args.assistant))
    if args.teacher:
        targets.append(("teacher", args.teacher))
    for name, url in targets:
        data = check(url)
        models = [item.get("id") for item in data.get("data", [])]
        print(f"{name}: OK {models}")


if __name__ == "__main__":
    main()
