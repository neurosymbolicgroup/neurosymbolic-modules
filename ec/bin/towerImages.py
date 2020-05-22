import datetime
import os
import random

import binutil

from pylab import imshow,show
from dreamcoder.domains.tower.makeTowerTasks import makeSupervisedTasks
from dreamcoder.domains.tower.makeTowerTasks import dSLDemo
from dreamcoder.domains.tower.makeTowerTasks import SupervisedTower
from dreamcoder.domains.tower.tower_common import *

bricks = [SupervisedTower("brickwall, %dx%d"%(w,h), """(for j %d
                          (embed (for i %d h (r 6)))
                          (embed (r 0) (for i %d h (r 6))))"""%(h,w,w))
          for w in range(1,9)
          for h in range(1,9) ]

for j,t in enumerate(bricks):
    t.exportImage("/home/salford/bricks/tower_%d.png"%j,
                  drawHand=False)

