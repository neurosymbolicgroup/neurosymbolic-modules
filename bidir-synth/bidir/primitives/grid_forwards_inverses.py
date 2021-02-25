def grid_split_color(grid_split):
	"""
	Returns the color used to split the grid.
	"""
	color, columns, rows, shape = grid_split
	return color

def pixels(g):
	"""
	returns array of pixels in the array
	"""
	return np.copy(g.grid)

def split(g, intval):
	"""
	Returns an array of subgrids splitting by int. Ex if 
	evaluated on the output grids of task 316, would return a 3x3 of grids,
	each of which is either a 3x3 blue grid or 3x3 black grid.
	"""
	subgrids = np.copy(g.grid).reshape(intval, intval)
	outputgrids = [Grid(minigrid) for minigrid in subgrids]
	return outputgrids


def undo_split(gridarray):
	"""
	exact inverse of split
	"""
	rawgrids = [np.copy(g.grid) for grid in gridarray]
	return Grid(rawgrid)
