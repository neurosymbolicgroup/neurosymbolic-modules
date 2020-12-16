size=4
echo "DeepCode experiments====================================size=4"
for dir in /Users/yufeng/research/genesys/problem/DeepCoder/*; do
    echo "running benchmark $dir learn=off stat=off" 
    gtimeout 600 ant neodeep -Dapp=$dir -Ddepth=$size -Dlearn=false -Dstat=false
    echo "running benchmark $dir learn=on stat=off" 
    gtimeout 600 ant neodeep -Dapp=$dir -Ddepth=$size -Dlearn=true -Dstat=false
    echo "running benchmark $dir learn=off stat=on" 
    gtimeout 600 ant neodeep -Dapp=$dir -Ddepth=$size -Dlearn=false -Dstat=true
    echo "running benchmark $dir learn=on stat=on" 
    gtimeout 600 ant neodeep -Dapp=$dir -Ddepth=$size -Dlearn=true -Dstat=true
done
