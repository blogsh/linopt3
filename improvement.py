import numpy as np
import numpy.linalg as la
import scipy.io
import heuristic
from itertools import product

def improve2opt(c, initial):
	N, route = c.shape[0], initial[:]
	J, Jn = heuristic.compute_distance(c, route), 0

	while True:
		for i, k in product(range(N), range(N)):
			if k >= i:
				routen = route[0:i] + route[i:k][::-1] + route[k:]
				routen[-1] = routen[0]
				Jn = heuristic.compute_distance(c, routen)
				if Jn < J:
					J, route = Jn, routen
					break
		else: break

	return route

if __name__ == '__main__':
	c = scipy.io.loadmat('TSP_huge.mat')['c']
	initial = heuristic.nearest_neighbor(c)
	route = improve2opt(c, initial)
	print(
		heuristic.compute_distance(c, initial),
		' --> ',
		heuristic.compute_distance(c, route)
	)

	J, X = heuristic.make_output(c, route)
	scipy.io.savemat('solution.mat', {'X':X, 'J':J})
