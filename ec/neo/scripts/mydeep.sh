for f in problem/DeepCoder-New/*.json ; do
	y=$(basename $f .json)
	echo $y
	ant neoDeep -Dapp=./problem/DeepCoder-New/$y.json -Ddepth=3 -Dlearn=false -Dstat=false -Dfile="" > output/mydeep/$y.txt
done
