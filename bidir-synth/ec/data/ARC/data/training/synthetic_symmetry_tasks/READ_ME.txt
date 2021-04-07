###########################
HOW TO USE TASK_CREATION.PY
###########################

Task creation is set up to generate three different types of symmetric tasks,
1. Whole patterns: grids that have symmetry about their horizontal, vertical, and both diagonals
2. Repeated Units: essentially tesselations of whole grids. The whole grid is the repeated unit
3. Diagonals: grid composed of diagonals

Additionally, the input/output pairs come in two different types
1. Whole grid: the input has a bunch of occlusions totaling no more than 15% of the area of the grid
		the output is the whole grid with the occlusions filled in
2. Occlusion: the input has a single occlusion that takes up less than 15% of the area of the grid
		the output is the occluded area filled in

The naming structure for the files is
(tasktype)_(i/otype)_(height)_(width)_(number)_(input or output).(jpg or json)
it actually might be width_height I cant remember


###########################
HOW TO RUN TASK_CREATION.PY
###########################
Essentially you just have to run the generate_occlusion function

For all of the below, if gen_occlusion is true, i/o will be an occlusion, otherwise will be a whole grid
n is the number of each size grid you want to generate. the actual sizes of grids created is slightly arbitrary.
									you can find the grid sizes in the gen_occlusion function or generate_diagonal_pattern function

to generate diagonals:
	generate_occlusion(n,generate_diagonal_pattern,'diagonal_patterns',gen_occlusion)
	if n=15, 105 patterns generated for grid sizes 3, 5-30 in intervals of 5
generate whole patterns
	generate_occlusion(n,generate_whole_pattern,'whole_patterns', gen_occlusion)
	if n=15, 105 patterns generated for grid sizes 3, 5-30 in intervals of 5
generate repeatable units:
	generate_occlusion(n,generate_repeated_pattern, 'repeated_units',gen_occlusion)
	if n=7, 98 patterns generated for grids of a bunch of different sizes, because the tesselation makes randomly sized grids
	

NOTE: the file structure currently set up in generate_occlusion will save each of the outputs of these functions 
into their respective directories. you can change this pretty easily by altering the function's code
all output files have distinct names so saving them all in one directory should be ok.

ADDITIONALLY the current json file structure might not be ideal, you can change how the output files will look in
save_grid_jpg_json


