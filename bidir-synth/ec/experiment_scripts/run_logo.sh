#!/bin/sh
# singularity exec ../ec/container.img python -u ../ec/bin/logo.py -t 500
# nohup singularity exec ../ec/container.img python -u ../ec/bin/logo.py -t 500 --structurePenalty 1.5 --pseudoCounts 30.0 --biasOptimal --contextual --split 0.5 --testingTimeout 1500  --storeTaskMetrics --taskReranker randomShuffle --taskBatchSize 10 -i 20  -R 1800 --reuseRecognition -c 10 > logo1.out &
singularity exec container.img python -u bin/logo.py -t 1 --structurePenalty 1.5 --pseudoCounts 30.0 --biasOptimal --contextual --split 0.5 --testingTimeout 1  --storeTaskMetrics --taskReranker randomShuffle --taskBatchSize 10 -i 2  -R 1 --reuseRecognition -c 1 --solver python
