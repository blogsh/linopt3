import numpy as np
import numpy.linalg as la
import scipy.io
import heuristic
import numpy.random as rand
import matplotlib.pyplot as plt
import multiprocessing as mp

def ant(tau, eta, a, b):
	N = tau.shape[0]
	available = list(range(N))
	route = [rand.randint(N)]
	available.remove(route[0])

	while len(available) > 0:
		tauc = tau[route[-1], available]**a
		etac = eta[route[-1], available]**b
		cdf = np.cumsum((tauc * etac) / sum(tauc * etac))
		index = cdf.searchsorted(rand.uniform(0, cdf[-1]))
		route.append(available[index])
		available.remove(route[-1])

	return route + [route[0]]

def aco(c, a, b, r, K, I, Q = 1):
	N = c.shape[0]
	best_route = list(range(N)) + [0]
	best_distance = heuristic.compute_distance(c, best_route)

	tau = np.ones((N, N)) / best_distance
	eta = 1 / c

	pool = mp.Pool()

	for i in range(I): # Iterations
		routes = []

		queue = [pool.apply_async(ant, (tau, eta, a, b)) for k in range(K)]
		routes = [result.get() for result in queue]

		dtau = np.zeros((N,N))
		distances = []

		for route in routes:
			distance = heuristic.compute_distance(c, route)
			for j in range(1,N):
				dtau[route[j-1], route[j]] += Q / distance
				dtau[route[j], route[j-1]] += Q / distance
			distances.append(distance)

		tau = (1 - r) * tau + dtau

		if best_distance is None: 
			best_distance = distances[0]
			best_route = routes[0]

		if min(distances) < best_distance:
			best_distance = min(distances)
			best_route = routes[np.argmin(distances)]

		if i % 1 == 0:
			#print(sum(abs(np.array(distances) - best_distance)), best_distance)
			print(i, best_distance)

	return best_route

if __name__ == '__main__':
	c = scipy.io.loadmat('TSP_huge.mat')['c']
	route = aco(c, 1, 2, 0.5, 500, 1000)

	J, X = heuristic.make_output(c, route)
	scipy.io.savemat('solution.mat', {'X':X, 'J':J})
