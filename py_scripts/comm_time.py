from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# num_features = 7532
num_readings = 100

for num_features in range(1000, 200000, 1000):
  if rank != 0:
    grad = 0.001*np.random.randn(num_features, 1)
    for i in range(num_readings):
      payload = {
        'rank': rank,
        'grad': grad,
        'loss': time.time(),
        'iter': i,
      }
      comm.send(payload, dest=0)
  elif rank == 0:
    time_taken = []
    for i in range(num_readings):
      payload = comm.recv(source=1)
      total_time = time.time()-payload['loss']
      time_taken.append(total_time)
    print('Num Features: ' + str(num_features) + ' Time: ' + str(sum(time_taken)/num_readings))