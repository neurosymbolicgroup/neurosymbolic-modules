#!/bin/sh
singularity exec container.img python -u bin/arc2.py -t 64000 -g --no-consolidation -i 1 -c 9 &> arc_7_30.out
