# Program Synthesis using Conflict-Driven Learning

 Yu Feng, Ruben Martins, Osbert Bastani, Isil Dillig. Program Synthesis using Conflict-Driven Learning. PLDI'18.

# Command-line options

- app: source of the benchmark in json.
- depth: size of the sketch
- learn: enable conflict-driven learning
- stat: enable statistical model
- file: source of the ngram ranking provided by Morpheus
- spec: abstract semantics of the DSL constructs (e.g., gather, spread, mutate, etc).

# Neo for DeepCode 
Original deepCode: no learning + statistical model:

ant neoDeep -Dapp=./problem/DeepCoder-New/prog13.json -Ddepth=3 -Dlearn=false -Dstat=false -Dfile=""


# Neo for Morpheus

Without n-gram information:

ant neoMorpheus -Dapp=./problem/Morpheus/r4.json -Ddepth=3 -Dlearn=false -Dstat=false -Dfile="" -Dspec=specs/Morpheus/

With n-gram information using a file:

ant neoMorpheus -Dapp=./problem/Morpheus/r1.json -Ddepth=3 -Dlearn=true -Dstat=false -Dfile=sketches/ngram-size3.txt -Dspec=specs/Morpheus/

# Set up neural net model

 requires:
 - Python 2.7
 - NumPy and Tensorflow

 The latter can be installed using the following commands:

pip install numpy
pip install tensorflow

 Then, run org.genesys.clients.DeepCoderDeciderMain to test the Python decider.

 If a python interpreter other than the default should be used, then create
 a text file ./model/tmp/python_path.txt and include the path. For example,
 to use /usr/local/bin/python, include "/usr/local/bin/" in this file.
 
