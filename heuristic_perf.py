import heuristic
import scipy.io
import subprocess as sp
import os, time, pickle

N = []
T = []

devnull = open(os.devnull, 'w')

for Ni in range(100, 10001, 100):
	print(Ni)
	
	os.environ['PATH'] += ":/home/sebastian/.matlab/bin"
	args = ('matlab', '-r', 'create_TSP(%d, \'heuristic\');exit;' % Ni, '-nodisplay')
	sp.call(args, env=os.environ, stdout=devnull, stderr=devnull)

	c = scipy.io.loadmat('heuristic.mat')['c']

	start = time.time()
	heuristic.neirest_neighbor(c)
	end = time.time()

	N.append(Ni)
	T.append(end - start)

	with open('perf.dat', 'wb+') as f: 
		pickle.dump((N, T), f)
