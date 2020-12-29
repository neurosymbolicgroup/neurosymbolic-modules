import os
import re

myDir = 'sketches/morpheus_pldi/'
ngramLoc = 'sketches/'
benchFiles = os.listdir(myDir)
for f in benchFiles:
    with open(myDir + f, 'r') as fdes:
        nums = re.findall(r'\d+', f)
        idx = nums[0]
        print 'Running benchmark===========================' + idx
        first_line = fdes.readline().strip()
        print first_line
        comps = first_line.split(" ")
        print len(comps)
        compLen = str(len(comps))
        loc = ngramLoc + 'ngram-size' + compLen + '.txt'
        out = 'output/morpheus-learn-11-10/' + compLen + '/r' + idx + '.txt'
        cmd = 'gtimeout 300 ant neoMorpheus -Dapp=./problem/Morpheus-PLDI/r'+idx+'.json -Ddepth='+ compLen +' -Dlearn=true -Dstat=false -Dfile=' + loc + ' > '  +  out
        print cmd
        #os.system(cmd)
