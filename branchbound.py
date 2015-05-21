import numpy as np
import heuristic, lagrange
import scipy.io, scipy.optimize as opt
import ampl
from collections import deque

dataset = 'TSP_medium'
c = scipy.io.loadmat('%s.mat' % dataset)['c']
N = c.shape[0]

nn_route = heuristic.nearest_neighbor(c)
g_pessimistic = heuristic.compute_distance(c, nn_route)
g_route = None

branches = deque()
for node in range(1,N):
	branches.append([0, node])

def objective(x):
	return np.dot(c, x.reshape((N,N)))

branchCount = 0
dropCount = 0

Ks, BCs, DCs, Ls = [], [], [], []

k = 0
while len(branches) > 0:
	current = branches.popleft()
	l_optimistic = ampl.solve_lp('%s.dat' % dataset, current)

	if l_optimistic is not None and l_optimistic < g_pessimistic:
		branchCount += 1
		nn_route = heuristic.nearest_neighbor(c, current)
		l_pessimistic, X = heuristic.make_output(c, nn_route)

		if l_pessimistic < g_pessimistic:
			g_pessimistic = l_pessimistic
			g_route = X

		destinations = list(set(range(N)) - set(current))
		for destination in destinations:
			branches.append(current + [destination])
	else:
		dropCount += 1

	Ks.append(k)
	BCs.append(branchCount)
	DCs.append(dropCount)
	Ls.append(len(branches))

	k += 1
	if k % 100 == 0:
		print(g_pessimistic, len(branches), branchCount, dropCount)

print(g_pessimistic, len(branches), branchCount, dropCount)
print(g_route)

import matplotlib.pyplot as plt

plt.figure()
plt.title('Breadth First')
plt.plot(Ks, BCs, 'b')
plt.plot(Ks, DCs, 'r')
plt.plot(Ks, Ls, 'k')
plt.xlim([min(Ks), max(Ks)])
plt.grid()
plt.xlabel('Iterations')
plt.legend(['Branched', 'Dropped', 'Left'], 'upper left')
plt.show()
