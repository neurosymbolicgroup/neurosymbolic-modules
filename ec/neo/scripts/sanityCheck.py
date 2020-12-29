import os
import re

myDir = 'sketches/morpheus_solutions/'
benchFiles = os.listdir(myDir)
for f in benchFiles:
    with open(myDir + f, 'r') as fdes:
        loc = myDir + f
        nums = re.findall(r'\d+', f)
        idx = nums[0]
        print 'Running benchmark===========================' + loc
        first_line = fdes.readline().strip()
        print first_line
        comps = first_line.split(" ")
        print len(comps)
        cmd = 'timeout 120 ant neoMorpheus -Dapp=./problem/Morpheus/r'+idx+'.json -Ddepth='+ str(len(comps)) +' -Dlearn=true -Dstat=false -Dfile='+loc
        #if len(comps) == 3:
        #cmd = 'ant neoMorpheus -Dapp=./problem/Morpheus/r'+idx+'.json -Ddepth='+ str(len(comps)) +' -Dlearn=true -Dstat=false -Dfile='+loc
        print cmd
        os.system(cmd)
