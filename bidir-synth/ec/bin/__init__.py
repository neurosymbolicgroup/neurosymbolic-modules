"""
EC codebase scripts and executables

These scripts should be run from the root of the repository, e.g.

    python bin/<script>

For more usage examples, see the official_figures and official_experiments documents.

Module mapping details:

TODO: remove module mapping code when backwards-compatibility is no longer required.

The below module mapping is required for backwards-compatibility with old pickle files
generated from before the EC codebase refactor. New files added to the codebase do not
need to be added to the mapping, but if the existing modules are moved, then this the
mapping needs to be updated to reflect the move or rename.

The mapping uses the following pattern:

    sys.modules[<old module path>] = <new module reference>

This is because the previous structure of the codebase was completely flat, and when refactoring
to a hierarchical files, loading previous pickle files no longer works properly. It is important
to retain the ability to read old pickle files generated from official experiments. As a workaround,
the old module paths are included below. A preferable alternative would be to export program state
into JSON files instead of pickle files to avoid issues where the underlying classes change, so that
could be a future improvement to this project. Until then, we use the module mapping workaround.

For more info, see this StackOverflow answer: https://stackoverflow.com/a/2121918/2573242
"""
import sys

from ec.bin import analyzeDepth
from ec.bin import compiledDriver
from ec.bin import examineFrontier
from ec.bin import graphs
from ec.bin import launch
from ec.bin import logReports
from ec.bin import physics
from ec.bin import rational
from ec.bin import scientificLaws
from ec.bin import symmetryBreaking
from ec.bin import taskRankGraphs
from ec.bin.deprecated import compressionGraph, evolution, extractDeepcoderDataset, python_server, symbolicRegression

sys.modules['analyzeDepth'] = analyzeDepth
sys.modules['compiledDriver'] = compiledDriver
sys.modules['compressionGraph'] = compressionGraph
sys.modules['evolution'] = evolution
sys.modules['examineFrontier'] = examineFrontier
sys.modules['extractDeepcoderDataset'] = extractDeepcoderDataset
sys.modules['graphs'] = graphs
sys.modules['launch'] = launch
sys.modules['logReports'] = logReports
sys.modules['physics'] = physics
sys.modules['python_server'] = python_server
sys.modules['rational'] = rational
sys.modules['scientificLaws'] = scientificLaws
sys.modules['symbolicRegression'] = symbolicRegression
sys.modules['symmetryBreaking'] = symmetryBreaking
sys.modules['taskRankGraphs'] = taskRankGraphs
