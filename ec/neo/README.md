# Program Synthesis using Conflict-Driven Learning
original repo: [https://github.com/utopia-group/neo][1]

# To run

1) Install ant.  On Mac this would look like:
	brew install ant
And then add the symlinks indicated by the end of the brew instructions.

2) Run:
	ant neoMorpheus -Dapp=./problem/Morpheus/r4.json -Ddepth=3 -Dlearn=false -Dstat=false -Dfile="" -Dspec=specs/Morpheus/

# Troubleshooting:

	“libz3java.dylib” cannot be opened because the developer cannot be verified.
	- Go to Finder, Control-Click the .dylib file, Click „Open.“  Now your Mac should have saved the app as an exception to your security settings.

[1]:	https://github.com/utopia-group/neo