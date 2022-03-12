from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
  data = {'a': 7, 'b': 3.14}
  for i in range(1, 13):
    data['a'] = i+1
    comm.send(data, dest=i, tag=11)
    print('Sending ' + str(data['a']))

elif rank != 0:
  data = comm.recv(source=0, tag=11)
  print(data['a'])
