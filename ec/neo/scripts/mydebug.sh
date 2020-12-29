for f in problem/DeepCoder-Debug-T3/*.json ; do
	y=$(basename $f .json)
	echo $y
    ant neoDeep -Dapp=./problem/DeepCoder-Debug-T3/$y.json -Ddepth=3 -Dlearn=true -Dstat=false -Dfile="" > output/debug-learn/$y.txt
done
