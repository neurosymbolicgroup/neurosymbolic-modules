#!/bin/sh
srun --time=08:00:00 --output=output11-1.out --ntasks=40 singularity exec container.img python -u bin/arc_simon.py -t 240 -i 10 -R 3600 &> rect_run.out
