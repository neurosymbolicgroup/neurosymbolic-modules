 #!/usr/bin/env bash

# To run this script you need to install mypy first
mypy --namespace-packages --follow-imports skip \
$(find bidir -name "*.py")
