#!/bin/sh
singularity exec container.img python -u bin/arc2.py -t 16000 -i 20 -R 4000
