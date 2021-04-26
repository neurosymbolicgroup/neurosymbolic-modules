 #!/usr/bin/env bash

# To run this script you need to install mypy first
mypy --namespace-packages --check-untyped-defs --follow-imports silent rl bidir tests experiments
