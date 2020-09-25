#!/bin/sh
singularity exec container.img python -u bin/arc_simon.py -t 2000 -i 5 -R 3000 &> arc_9_7_with_rec.out
