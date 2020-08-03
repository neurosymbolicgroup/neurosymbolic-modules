#!/bin/sh
singularity exec container.img python -u bin/arc2.py -t 16000 -g --no-consolidation -i 1 -c 19 &> arc_7_30.out
