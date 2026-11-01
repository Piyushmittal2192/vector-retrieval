#!/usr/bin/env python3
"""Render a ${VAR} template. Fails loudly on any missing variable.

  python3 ops/render.py deploy/k8s/retrieval.tmpl.yaml COLOR=blue INDEX_VERSION=v1 ... | kubectl apply -f -
"""
import re
import sys

template = open(sys.argv[1]).read()
values = dict(arg.split("=", 1) for arg in sys.argv[2:])
needed = set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", template))
missing = sorted(needed - set(values))
if missing:
    sys.exit(f"render.py: missing values for {', '.join(missing)}")
sys.stdout.write(re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: values[m.group(1)], template))
