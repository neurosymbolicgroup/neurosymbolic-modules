#!/bin/sh
singularity exec container.img python -u bin/arc_simon.py -t 250 -i 10 -R 5000 -c 100 &> arc_9_7_with_rec_2.out
# singularity exec container.img python -u bin/arc_simon.py -t 1 -i 1 -R -1 &> arc_9_7_with_rec.out
