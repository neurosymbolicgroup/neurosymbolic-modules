#!/bin/sh
singularity exec container.img python -u bin/arc2.py -t 6000 -R 4000 -i 20 -c 22 > arc_7_18_2.out 2>&1
