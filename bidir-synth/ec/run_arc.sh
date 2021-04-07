#!/bin/sh
singularity exec container.img python -u bin/arc_simon.py -t 100 -i 5 -R 3000
