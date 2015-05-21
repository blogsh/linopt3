import numpy as np
import numpy.linalg as la
import scipy.io

def nearest_neighbor(c, initial = None):
	N = c.shape[0]
	available, route = list(range(1, N)), [0]

	if initial is not None:
		route = initial[:]
		available = list(set(range(N)) - set(route))

	for i in range(len(available)):
		route.append(available[np.argmin(c[route[-1], available])])
		available.remove(route[-1])
	
	return route + [0]

def compute_distance(c, route):
	N = c.shape[0]
	return sum(c[route[:-1], route[1:]])

def make_output(c, route):
	N = c.shape[0]
	J = compute_distance(c, route)
	X = np.zeros((N, N))
	for i in range(N): X[min(route[i:i+2]), max(route[i:i+2])] = 1
	return J, X

if __name__ == '__main__':
	c = scipy.io.loadmat('TSP_small.mat')['c']
	J, X = make_output(c, nearest_neighbor(c))
	print(J)
	scipy.io.savemat('solution.mat', {'X':X, 'J':J})
