import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from itertools import product
import improvement, heuristic

def mst_prim(c):
	N = c.shape[0]
	S, Sn, T = [1], list(range(2,N)), []
	X = np.zeros((N,N))
	
	while len(T) < N - 2:
		cc = c[S][:,Sn]
		index = np.unravel_index(np.argmin(cc), cc.shape)
		T.append((S[index[0]], Sn[index[1]]))
		indices = (S[index[0]], Sn[index[1]])
		X[min(indices), max(indices)] = 1
		X[max(indices), min(indices)] = 1
		S.append(Sn[index[1]])
		Sn.remove(Sn[index[1]])

	return X

def plot(X, X1 = None):
	pos = scipy.io.loadmat('TSP_small.mat')['pos']
	N = X.shape[0]
	plt.figure()

	for (i,j) in product(range(N), range(N)):
		if X[i,j]:
			plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 'b')
		if X1 is not None and X1[i,j]:
			plt.plot([pos[i, 0]+1, pos[j, 0]+1], [pos[i, 1]+1, pos[j, 1]+1], 'r')

	for i in range(N):
		plt.plot(pos[i,0], pos[i,1], 'ob')
		plt.annotate(str(i+1), [pos[i,0], pos[i,1]])

	plt.show()

def q1mst(c, pi):
	N = c.shape[0]

	ct = np.zeros((N, N))
	for (i, j) in product(range(N), range(N)): 
		ct[i,j] = c[i,j] - pi[i] - pi[j]

	# Create MST
	X = mst_prim(ct)

	# Create 1-MST
	indices = np.argsort(ct[0,:])
	X[0, indices[0]] = 1
	X[0, indices[1]] = 1
	X[indices[0], 0] = 1
	X[indices[1], 0] = 1

	obj = np.sum(ct * np.triu(X)) + 2 * sum(pi)
	return obj, X

def eulerian_tour(X):
	# Based on doubling the 1-MST

	X = np.copy(X)
	X = 2 * X
	N = X.shape[0]
	tour = [0]

	while np.sum(X) > 0:
		available = np.where(np.sum(X,0) > 0)[0]
		possible = list(set(tour).intersection(available))
		subtour = [possible[0]]

		for i in range(N):
			if X[0,i] > 0 and i in tour:
				subtour = [i]
				break

		assert(len(subtour) > 0)

		while len(subtour) < 2 or subtour[0] != subtour[-1]:
			current = subtour[-1]
			target = np.where(X[current,:] > 0)[0][0]
			X[current, target] -= 1
			X[target, current] -= 1
			subtour.append(target)

		index = tour.index(subtour[0])
		tour[index:index+1] = subtour
		
	return tour

def shortcut_tsp(T):
	N = max(T) + 1
	X = np.zeros((N, N))
	path = []

	for k in T:
		if not k in path:
			path.append(k)

			if len(path) > 1:
				X[path[-1], path[-2]] = 1
				X[path[-2], path[-1]] = 1

	X[path[0], path[-1]] = 1
	X[path[-1], path[0]] = 1

	return X, path

def lagrangian(c, pos):
	N = c.shape[0]

	#q, X1 = q1mst(c, np.zeros((N)))
	#X = greedy_matching(c, X1)

	#T = eulerian_tour(X1)
	#X, route = shortcut_tsp(T)
	#plot(X, X1)
	#exit()

	#T = eulerian_tour(X1)
	#print(T)
	#X = shortcut_tsp(T)

	#plot(X)

	#print(np.sum(X,0))
	#plot(X1, X)


	#pi = np.zeros((N))
	#q, X = q1mst(c, pi)

	#pi = np.ones((N))
	#q1, X1 = q1mst(c, [0, 30, 0, 0, 0, 0, 0, 0, 0, 0])

	#print(q, q1)
	#plot(X, X1)


	#exit()
	N = c.shape[0]
	
	xi = 2
	xif = 1.1
	pi = np.zeros((N))
	qbest = 0
	qhbest = None
	worsecount = 0
	worsek = 100
	K = 500

	qs = []
	qhs = []
	qbests = []
	qhbests = []

	Xbest = np.zeros((N, N))
	Xhbest = np.zeros((N,N))
	q0, X0 = q1mst(c, np.zeros((N)))

	for k in range(K):
		# 1-MST
		q, X = q1mst(c, pi)

		# Subgradient
		h = 2 - np.sum(X, 0)
		a = xi * (qbest - q) / np.sum(h**2)
		pi = pi + a * h

		# Feasibility heuristic
		T = eulerian_tour(X)
		Xh, route = shortcut_tsp(T)
		#route = improvement.improve2opt(c, route + [route[0]])
		#qh = heuristic.compute_distance(c, route)
		qh = np.sum(c * np.triu(Xh))

		if q > qbest:
			qbest = q
			#qhbest = qh
			#Xhbest = Xh
			Xbest = X
		else:
			worsecount += 1

		if qhbest is None or qh < qhbest:
			qhbest = qh
			Xhbest = Xh

		if worsecount > worsek:
			xi = xi / xif
			worsecount = 0

		print(qbest, qhbest)
		qbests.append(qbest)
		qhbests.append(qhbest)
		qs.append(q)
		qhs.append(qh)

	#plot(Xbest, Xhbest)
	ks = list(range(K))

	plt.figure(figsize=(12,6))

	plt.subplot(1,2,1)
	plt.plot(qs, 'b')
	plt.plot(qhs, 'r')
	plt.plot(np.array(qhs) / 2, 'r--')
	plt.grid()
	plt.ylim([0, max(max(qhs), max(qs))])

	plt.subplot(1,2,2)
	plt.plot(qbests, 'b')
	plt.plot(qhbests, 'r')
	plt.plot(np.array(qhbests) / 2, 'r--')
	plt.grid()
	plt.ylim([0, max(max(qhs), max(qs))])

	plt.show()

if __name__ == '__main__':
	c = scipy.io.loadmat('TSP_small.mat')['c']
	pos = scipy.io.loadmat('TSP_small.mat')['pos']
	lagrangian(c, pos)
	#q(c, np.ones((c.shape[0])) * 1, pos)
	#q(c, [1, 0, 0, 0, 500, 0, 0, 0, 0, 0], pos)
	#route = aco(c, 1, 2, 0.5, 500, 1000)

	#J, X = heuristic.make_output(c, route)
	#scipy.io.savemat('solution.mat', {'X':X, 'J':J})

