#!/usr/bin/env python
# coding: utf-8

# Import libraries
from mpi4py import MPI
import numpy as np
import threading
import param_beta
import time
import random
import sys

# COMM_CHANNEL_DELAY = 0.042321488
COMM_CHANNEL_DELAY = param_beta.COMM_CHANNEL_DELAY
# Buffer size = (Size of W + size of b + 100 (extra))
BUFFER_SIZE = 2000*1024
BATCH_SIZE = 50
NUM_WORKERS = 12

# Tags
CODED_TAG = 2
WEIGHT_TAG = 3

# MPI4Py constants
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
name = MPI.Get_processor_name()
p = comm.Get_size()

# Linear Reg params
NUM_ITER = 50  # Number of Iterations
NUM_DATA_POINTS = 6240
NUM_FEATURES = 7532  # num of features
learning_rate = 0.003  # learning rate for batch gradient descent

# TGC params -> 3*3
decoding_matrix_A = [[0, 1, 2],
                     [1, 0, 1],
                     [2, -1, 0]]

encoding_coefs = {
    '1':  [0.5, 1],  # (1, 1)
    '2': [1, -1],  # (1, 2)
    '3': [0.5, 1],  # (1, 3)
    # Childern of (1, 1)
    '4': [0.5, 0.5, 1, 1],  # (2, 1)
    '5': [1, 1, -1, -1],  # (2, 2)
    '6': [0.5, 0.5, 1, 1],  # (2, 3)
    # Childern of (1, 2)
    '7': [-0.5, -0.5, -1, -1],  # (2, 4)
    '8': [-1, -1, 1, 1],  # (2, 5)
    '9': [-0.5, -0.5, -1, -1],  # (2, 6)
    # Childern of (1, 3)
    '10': [0.5, 0.5, 1, 1],  # (2, 7)
    '11': [1, 1, -1, -1],  # (2, 8)
    '12': [0.5, 0.5, 1, 1],  # (2, 9)
}

NodeTimeStamps = {'rank': my_rank}


def linear_reg_loss(X, Y, W):
  return np.sum((X.dot(W)-Y) ** 2)/(2*NUM_DATA_POINTS)


def linear_reg_gradient(X, Y, W):
  # print(X.shape, Y.shape, W.shape)
  return (X.dot(W)-Y).T.dot(X)/NUM_DATA_POINTS


def update_weights(W, grad, learning_rate=0.001):
  return W - learning_rate*grad['dW']


def get_decoding_row(ranks_org):
  ranks = np.copy(ranks_org)
  ranks.sort()
  for idx, rank in enumerate(ranks):
    if rank != idx:
      return idx
  return ranks.shape[0]


def get_encoding_coef(my_rank):
  return encoding_coefs[str(my_rank)]


def read_datasets(my_rank):
  datasets = []

  if my_rank <= 3:
    # dataset1 = np.load('/home/ec2-user/TGC/Data_' +
    #                    str(my_rank)+'/data_'+str(my_rank)+'_0.npy')
    # dataset2 = np.load('/home/ec2-user/TGC/Data_' +
    #                    str(my_rank)+'/data_'+str(my_rank)+'_1.npy')
    dataset1 = np.random.randn(832, NUM_FEATURES+1)
    dataset2 = np.random.randn(832, NUM_FEATURES+1)
    datasets.append(dataset1)
    datasets.append(dataset2)
  else:
    # dataset1 = np.load('/home/ec2-user/TGC/Data_' +
    #                    str(my_rank)+'/data_'+str(my_rank)+'_0.npy')
    # dataset2 = np.load('/home/ec2-user/TGC/Data_' +
    #                    str(my_rank)+'/data_'+str(my_rank)+'_1.npy')
    # dataset3 = np.load('/home/ec2-user/TGC/Data_' +
    #                    str(my_rank)+'/data_'+str(my_rank)+'_2.npy')
    # dataset4 = np.load('/home/ec2-user/TGC/Data_' +
    #                    str(my_rank)+'/data_'+str(my_rank)+'_3.npy')
    dataset1 = np.random.randn(416, NUM_FEATURES+1)
    dataset2 = np.random.randn(416, NUM_FEATURES+1)
    dataset3 = np.random.randn(416, NUM_FEATURES+1)
    dataset4 = np.random.randn(416, NUM_FEATURES+1)
    datasets.append(dataset1)
    datasets.append(dataset2)
    datasets.append(dataset3)
    datasets.append(dataset4)
  return datasets


def weight_send(W, my_rank):
  data = {'W': W}
  random_id = random.randint(3*my_rank+1, 3*my_rank+3)
  for dest_id in range(3*my_rank, 3*my_rank+3):
    data['is_partial_straggler'] = (random_id == dest_id+1)
    comm.send(data, dest=dest_id+1, tag=WEIGHT_TAG)


def collect_grad_coded(my_rank, iter_no):
  received_payload = []
  received_rank = []
  num_workers = 0
  while num_workers < 2:
    payload = comm.recv(source=MPI.ANY_SOURCE, tag=CODED_TAG)
    worker_iter = payload['iter']
    if worker_iter != iter_no:
      # Reject prev iter's results
      # Handles racing conditions
      continue

    # TRANSFORM RANK
    worker_rank = payload['rank'] - 3*my_rank - 1
    num_workers = num_workers+1
    received_payload.append(payload)
    received_rank.append(worker_rank)
  # print('Received from: ' + str(np.array(received_rank)+3*my_rank+1))
  decoding_row = get_decoding_row(received_rank)
  full_grad = {'dW': np.zeros(W.shape)}
  full_loss = 0
  for i in range(len(received_payload)):
    worker_rank = received_rank[i]
    grad = received_payload[i]['grad']
    loss = received_payload[i]['loss']
    full_grad['dW'] += decoding_matrix_A[decoding_row][worker_rank]*grad['dW']
    full_loss += decoding_matrix_A[decoding_row][worker_rank]*loss

  return {'grad': full_grad, 'loss': full_loss}


# === Master ===
if my_rank == 0:
  # Initialize weights
  W = 0.0001*np.random.randn(NUM_FEATURES, 1)
  # W = np.load('/home/ec2-user/TGC/weights.npy')
  iter_infos = []
  start_time = time.time()
  for i in range(0, NUM_WORKERS):
    comm.send(start_time, dest=i+1)
  for iter_no in range(NUM_ITER):
    print('#'+str(iter_no))
    weight_send_start_time = time.time()
    weight_send(W, my_rank)
    payload_coded = collect_grad_coded(my_rank, iter_no)

    full_grad = {
        'dW': payload_coded['grad']['dW']
    }

    iter_infos.append({
        'iter_no': iter_no,
        'time': time.time()-weight_send_start_time,
        'loss': payload_coded['loss'],
    })
    W = update_weights(
        W, full_grad, learning_rate/(np.sqrt(iter_no+1)))
  end_time = time.time()
  print('Difference: ' + str(end_time-start_time))

  for iter_info in iter_infos:
    print('iter_no: ' + str(iter_info['iter_no']) + ' loss: ' +
          str(iter_info['loss']) + ' time: ' + str(iter_info['time']))

  # ============ GradCheck Code ===========
  # print('Coded Loss: ' + str(payload_coded['loss']))
  # gradW = np.load('/home/ec2-user/TGC/dW.npy')
  # gradb = np.load('/home/ec2-user/TGC_1/Init/gradb.npy')
  # print('=====================================================')
  # print('Error: ' + str(np.sum(abs(gradW-payload_coded['grad']['dW']))))
  # print('Grad Computed: ')
  # print(payload_coded['grad']['dW'][:10, :])
  # print('Grad Org: ')
  # print(gradW[:10, :])

  # print(np.sum(abs(gradb-grad['db'])))

# === Worker nodes ===
else:
  node_start_time = comm.recv(source=0)
  parent_rank = (my_rank-1)/3
  datasets = read_datasets(my_rank)
  encoding_coefs = get_encoding_coef(my_rank)
  buf = bytearray(BUFFER_SIZE)
  payload_req = comm.irecv(buf, source=parent_rank, tag=WEIGHT_TAG)
  interrupt = False

  for iter_no in range(NUM_ITER):
    if interrupt == False:
      payload = payload_req.wait()

    W = payload['W']
    is_partial_straggler = payload['is_partial_straggler']
    # if is_partial_straggler:
    #   print('Rank-' + str(my_rank) + ' is a partial straggler')
    if COMM_CHANNEL_DELAY > 0:
      time.sleep(COMM_CHANNEL_DELAY)
    if my_rank <= 3:
      weight_send(W, my_rank)

    if iter_no < NUM_ITER-1:
      # Init req
      payload_req = comm.irecv(buf, source=parent_rank, tag=WEIGHT_TAG)

    local_grad = {'dW': np.zeros(W.shape)}
    local_loss = 0
    interrupt = False

    #### Coded computation ####
    # Process each of the assigned data set
    for i in range(len(datasets)):
      X = datasets[i][:, :-1]
      y = datasets[i][:, -1]
      M = X.shape[0]
      loss = 0.0
      grad = {
          'dW': np.zeros_like(W),
      }
      for j in range(M/BATCH_SIZE):
        NodeTimeStamps['coded_start_time'] = time.time()
        # Check for interrupts here
        if iter_no < NUM_ITER-1:
          isPayloadAval, payload = payload_req.test()
          if isPayloadAval:
            interrupt = True
            break

        # tmp_loss, tmp_grad = softmax_gradient_scalar(W, b, X[j, :], y[j])
        tmp_grad = linear_reg_gradient(
            X[BATCH_SIZE*j:min(M, BATCH_SIZE*(j+1)), :],
            y[BATCH_SIZE*j:min(M, BATCH_SIZE*(j+1)), np.newaxis],
            W
        )
        tmp_loss = linear_reg_loss(
            X[BATCH_SIZE*j:min(M, BATCH_SIZE*(j+1)), :],
            y[BATCH_SIZE*j:min(M, BATCH_SIZE*(j+1)), np.newaxis],
            W
        )
        tmp_grad = np.reshape(tmp_grad, grad['dW'].shape)
        # print(tmp_grad.shape, grad['dW'].shape)
        grad['dW'] = grad['dW'] + tmp_grad
        loss = loss+tmp_loss

        NodeTimeStamps['coded_end_time'] = time.time()
        NodeTimeStamps['coded_duration'] = NodeTimeStamps['coded_end_time'] - \
            NodeTimeStamps['coded_start_time']
        if is_partial_straggler:
          time.sleep(NodeTimeStamps['coded_duration'])
      if interrupt:
        break

      local_grad['dW'] += encoding_coefs[i]*grad['dW']
      local_loss += encoding_coefs[i]*loss

    if interrupt == False:
      if my_rank <= 3:
        # receive data
        child_payload = collect_grad_coded(my_rank, iter_no)
        local_grad['dW'] = child_payload['grad']['dW'] + local_grad['dW']
        local_loss = child_payload['loss'] + local_loss
      send_data = {
          'rank': my_rank,
          'grad': local_grad,
          'loss': local_loss,
          'iter': iter_no,
      }
      grad_send_start_time = time.time()
      if COMM_CHANNEL_DELAY > 0:
        time.sleep(COMM_CHANNEL_DELAY)
      comm.send(send_data, dest=parent_rank, tag=CODED_TAG)
