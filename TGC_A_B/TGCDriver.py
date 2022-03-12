#!/usr/bin/env python
# coding: utf-8

# Import libraries
from mpi4py import MPI
import numpy as np
import params
import threading
import time
import random
import sys
import math

# COMM_CHANNEL_DELAY=0.042321488
COMM_CHANNEL_DELAY = params.COMM_CHANNEL_DELAY
DATA_POINT_INCREMENT_FACTOR = params.DATA_POINT_INCREMENT_FACTOR
# Buffer size = (Size of W + size of b + 100 (extra))
BUFFER_SIZE = params.BUFFER_SIZE
BATCH_SIZE = params.BATCH_SIZE
NUM_WORKERS = params.NUM_WORKERS
NUM_ITER = params.NUM_ITER  # Number of Iterations
STRAGGLING_FACTOR_ALPHA = params.STRAGGLING_FACTOR_ALPHA

# Tags
NAIVE_TAG = 1
CODED_TAG = 2
WEIGHT_TAG = 3

# MPI4Py constants
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
name = MPI.Get_processor_name()
p = comm.Get_size()

tgc_params = None
if sys.argv[1] == 'alpha':
  tgc_params = params.TGCAlpha(my_rank, DATA_POINT_INCREMENT_FACTOR)
elif sys.argv[1] == 'alpha-beta':
  tgc_params = params.TGCAlphaBeta(my_rank, DATA_POINT_INCREMENT_FACTOR)
elif sys.argv[1] == 'beta':
  tgc_params = params.TGCBeta(my_rank, DATA_POINT_INCREMENT_FACTOR)
elif sys.argv[1] == 'plain':
  tgc_params = params.TGC(my_rank, DATA_POINT_INCREMENT_FACTOR)
else:
  print("Invalid argument")
  exit()

# Linear Reg params
NUM_DATA_POINTS = 6240 * DATA_POINT_INCREMENT_FACTOR
NUM_FEATURES = 7532  # num of features
learning_rate = 0.003  # learning rate for batch gradient descent

NodeTimeStamps = {'rank': my_rank}

D_naive = None
if tgc_params.is_alpha:
  D_naive = np.random.randn(tgc_params.get_m_naive(), NUM_FEATURES + 1)

D_coded = []
num_D_coded = 2 if (my_rank <= 3) else 4
for ind in range(num_D_coded):
  D_coded.append(np.random.randn(
      tgc_params.get_m_coded(ind), NUM_FEATURES + 1))


def linear_reg_loss(X, Y, W):
  return np.sum((X.dot(W)-Y) ** 2)/(2*NUM_DATA_POINTS)


def linear_reg_gradient(X, Y, W):
  Y = np.reshape(Y, (Y.shape[0], 1))
  return (X.dot(W) - Y).T.dot(X) / NUM_DATA_POINTS


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
  return tgc_params.encoding_coefs[str(my_rank)]


def get_num_coded_datasets():
  if my_rank <= 3:
    return 2
  else:
    return 4


def read_batch_dataset(index, dataset_ind):
  num_examples = tgc_params.get_m_coded(dataset_ind)
  endIndex = min(num_examples, BATCH_SIZE * (index + 1))
  startIndex = BATCH_SIZE * index
  return D_coded[dataset_ind][startIndex:endIndex, :]


def read_batch_dataset_naive(index):
  num_examples = tgc_params.get_m_naive()
  endIndex = min(num_examples, BATCH_SIZE * (index + 1))
  startIndex = BATCH_SIZE * index
  # return np.random.randn(endIndex - startIndex, NUM_FEATURES + 1)
  return D_naive[startIndex:endIndex, :]


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
    full_grad['dW'] += tgc_params.decoding_matrix_A[decoding_row][worker_rank]*grad['dW']
    full_loss += tgc_params.decoding_matrix_A[decoding_row][worker_rank]*loss

  return {'grad': full_grad, 'loss': full_loss}


def collect_grad_naive(iter_num):
  num_workers = 0
  aggreagated_grad = {
      'dW': np.zeros((NUM_FEATURES, 1)),
      # 'db': 0,
  }
  aggreagated_loss = 0
  while num_workers < 3:
    payload = comm.recv(source=MPI.ANY_SOURCE, tag=NAIVE_TAG)
    if payload['iter'] != iter_num:
      # Reject the naive gradient
      # Should not happen in normal case!
      print(
          '[Communication Error] Received naive partition from the prev iteration',
          str(iter_num) + '!=' + str(payload['iter']))
      continue
    num_workers += 1
    grad = payload['grad']
    loss = payload['loss']
    aggreagated_loss += loss
    aggreagated_grad['dW'] += grad['dW']
    # aggreagated_grad['db'] += grad['db']
  return {
      'grad': aggreagated_grad,
      'loss': aggreagated_loss
  }


def collect_and_send_naive_grad(naive_payload, my_rank, iter_no):
  full_payload = naive_payload
  parent_rank = (my_rank-1)/3
  if my_rank <= 3:
    num_workers = 0
    # Aggregate children's naive grads
    buf = bytearray(BUFFER_SIZE)
    while num_workers < 3:
      payload_req = comm.irecv(buf, source=MPI.ANY_SOURCE, tag=NAIVE_TAG)
      isPayloadAval, payload = payload_req.test()
      while not isPayloadAval:
        isPayloadAval, payload = payload_req.test()
        time.sleep(0.01)
      if payload['iter'] != iter_no:
        # Reject the naive gradient
        # Should not happen in normal case!
        print(
            '[Communication Error] Received naive partition from the prev iteration',
            str(iter_no) + '!=' + str(payload['iter']))
        continue
      num_workers += 1
      grad = payload['grad']
      loss = payload['loss']
      full_payload['loss'] += loss
      full_payload['grad']['dW'] += grad['dW']
  # Send the aggregated naive grad to the parent
  if COMM_CHANNEL_DELAY > 0:
    time.sleep(COMM_CHANNEL_DELAY)
  naive_grad_send_start_time = time.time()
  comm.send(full_payload, dest=parent_rank, tag=NAIVE_TAG)
  NodeTimeStamps['naive_comm_time_wo_delay'] = time.time() - \
      naive_grad_send_start_time

  # debug_print("Naive Sent", node_start_time, iter_no)
  # print('Worker ' + str(my_rank) + ' Naive grad sent')


def check_child_naive_grad(req, nums):
  isPayloadAval, payload = req.test()
  if not isPayloadAval:
    return False, None
  if payload['iter'] != iter_no:
    # Reject the naive gradient
    # Should not happen in normal case!
    print('rank: ' + str(my_rank) +
          '[Communication Error] Received naive partition from the prev iteration',
          str(iter_no) + '!=' + str(payload['iter']) + ' num_rcvd: ' + str(nums))
  return True, payload


def safe_dict_key_delete(dict, key):
  if key in dict:
    del dict[key]


def debug_print(st, start_time, iter_no):
  print('[Rank: '+str(my_rank) + ' Iter: ' + str(iter_no) +
        ' Time: ' + str(time.time()-start_time) + '] ' + st)


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
    # if tgc_params.is_alpha:
    #   payload_naive = collect_grad_naive(iter_no)
    payload_coded = collect_grad_coded(my_rank, iter_no)

    full_grad = {}
    full_loss = 0
    full_grad['dW'] = payload_coded['grad']['dW']
    full_loss = payload_coded['loss']
    # if tgc_params.is_alpha:
    #   full_grad['dW'] += payload_naive['grad']['dW']
    #   full_loss += payload_naive['loss']

    iter_infos.append({
        'iter_no': iter_no,
        'time': time.time()-weight_send_start_time,
        'loss': full_loss,
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
  # datasets = read_datasets(my_rank)
  # naive_dataset = read_naive_dataset()
  tgc_params.encoding_coefs = get_encoding_coef(my_rank)
  buf = bytearray(BUFFER_SIZE)
  naive_buf = bytearray(BUFFER_SIZE)
  payload_req = comm.irecv(buf, source=parent_rank, tag=WEIGHT_TAG)
  interrupt = False
  payload = {}

  for iter_no in range(NUM_ITER):
    if interrupt == False:
      payload = payload_req.wait()

    W = payload['W']
    is_partial_straggler = payload['is_partial_straggler']
    # if is_partial_straggler:
    #   print('Rank-' + str(my_rank) + ' is a partial straggler')
    if COMM_CHANNEL_DELAY > 0:
      time.sleep(COMM_CHANNEL_DELAY)

    # debug_print("Weight received", node_start_time, iter_no)

    if my_rank <= 3:
      weight_send(W, my_rank)

    # X_naive = naive_dataset[:, :-1]
    # y_naive = naive_dataset[:, -1]
    # M_naive = X_naive.shape[0]
    if tgc_params.is_alpha:
      M_naive_val = tgc_params.get_m_naive()
      naive_grad = {'dW': np.zeros(W.shape)}
      naive_loss = 0
      NodeTimeStamps['naive_start_time'] = time.time()
      for j in range(int(math.ceil(float(M_naive_val)/BATCH_SIZE))):
        naive_dataset_batch = read_batch_dataset_naive(j)
        X_naive_batch = naive_dataset_batch[:, :-1]
        Y_naive_batch = naive_dataset_batch[:, -1]
        tmp_grad = linear_reg_gradient(
            X_naive_batch,
            Y_naive_batch,
            W
        )
        tmp_loss = linear_reg_loss(
            X_naive_batch,
            Y_naive_batch,
            W
        )
        tmp_grad = np.reshape(tmp_grad, naive_grad['dW'].shape)
        naive_grad['dW'] = naive_grad['dW'] + tmp_grad
        naive_loss = naive_loss+tmp_loss

      naive_payload = {
          'rank': my_rank,
          'grad': naive_grad,
          'loss': naive_loss,
          'iter': iter_no,
      }
      NodeTimeStamps['naive_end_time'] = time.time()
      NodeTimeStamps['naive_duration'] = NodeTimeStamps['naive_end_time'] - \
          NodeTimeStamps['naive_start_time']

      if is_partial_straggler:
        time.sleep((STRAGGLING_FACTOR_ALPHA-1) *
                   NodeTimeStamps['naive_duration'])

      # comm.send(naive_payload = check_child_naive_grad(naive_payload_req), dest=parent_rank, tag=NAIVE_TAG)
      # Naive computatio is_naive_avln
      # Thread -> wait for children's
      #  naive grad -> collect and send to parent
      # naive_grad_collection_thread = threading.Thread(
      #     target=collect_and_send_naive_grad, name='naive_grad_collection_thread',
      #     args=(naive_payload = check_child_naive_grad(naive_payload_req), my_rank, iter_no))
      # # debug_print("Naive Computed", node_start_time, iter_no is_naive_avl)
      # naive_grad_collection_thread.start()

    if iter_no < NUM_ITER-1:
      # Init req
      payload_req = comm.irecv(buf, source=parent_rank, tag=WEIGHT_TAG)

    # NAIVE RCVD
    if tgc_params.is_alpha:
      is_naive_grad_sent = False
      all_child_naive_rcvd = False
      if my_rank <= 3:
        num_naive_rcvd = 0
        naive_payload_req = comm.irecv(
            naive_buf, source=MPI.ANY_SOURCE, tag=NAIVE_TAG)
      else:
        naive_grad_rcvd_at = time.time()

    local_grad = {'dW': np.zeros(W.shape)}
    local_loss = 0
    interrupt = False

    #### Coded computation ####

    # Process each of the assigned data set
    num_coded_datasets = get_num_coded_datasets()
    debug_calc_examples = 0
    avg_coded_batch_time = 0
    count_of_batches = 0
    NodeTimeStamps['full_coded_start_time'] = time.time()
    for i in range(num_coded_datasets):
      M = tgc_params.get_m_coded(i)
      loss = 0.0
      grad = {
          'dW': np.zeros_like(W),
      }
      count_of_batches += math.ceil(float(M)/BATCH_SIZE)
      # print(str(my_rank) + " " + "["+str(iter_no)+"]-> DT_IND: " + str(i) + "num_ex: " + str(M),".....J="+str(math.ceil(M/BATCH_SIZE)))
      for j in range(int(math.ceil(float(M)/BATCH_SIZE))):
        # Check for interrupts here
        if iter_no < NUM_ITER-1:
          isPayloadAval, payload = payload_req.test()
          if isPayloadAval:
            interrupt = True
            break

        if tgc_params.is_alpha:
          if my_rank <= 3 and num_naive_rcvd < 3:
            is_naive_avl, child_naive_payload = check_child_naive_grad(
                naive_payload_req, num_naive_rcvd)
            if is_naive_avl:
              num_naive_rcvd += 1
              naive_payload['grad']['dW'] += child_naive_payload['grad']['dW']
              naive_payload['loss'] += child_naive_payload['loss']

              if num_naive_rcvd < 3:
                naive_payload_req = comm.irecv(
                    naive_buf, source=MPI.ANY_SOURCE, tag=NAIVE_TAG)
              else:
                all_child_naive_rcvd = True
                naive_grad_rcvd_at = time.time()
          elif not is_naive_grad_sent and (time.time() - naive_grad_rcvd_at >= COMM_CHANNEL_DELAY):
            if (my_rank <= 3 and all_child_naive_rcvd) or (my_rank > 3):
              comm.send(naive_payload, dest=parent_rank, tag=NAIVE_TAG)
              is_naive_grad_sent = True

        NodeTimeStamps['coded_start_time'] = time.time()
        coded_dataset_batch = read_batch_dataset(j, i)
        debug_calc_examples += coded_dataset_batch.shape[0]
        X_coded_batch = coded_dataset_batch[:, :-1]
        Y_coded_batch = coded_dataset_batch[:, -1]

        # tmp_loss, tmp_grad = softmax_gradient_scalar(W, b, X[j, :], y[j])
        tmp_grad = linear_reg_gradient(
            X_coded_batch,
            Y_coded_batch,
            W
        )
        tmp_loss = linear_reg_loss(
            X_coded_batch,
            Y_coded_batch,
            W
        )

        tmp_grad = np.reshape(tmp_grad, grad['dW'].shape)
        # print(tmp_grad.shape, grad['dW'].shape)
        grad['dW'] = grad['dW'] + tmp_grad
        loss = loss+tmp_loss

        NodeTimeStamps['coded_end_time'] = time.time()
        NodeTimeStamps['coded_duration'] = NodeTimeStamps['coded_end_time'] - \
            NodeTimeStamps['coded_start_time']
        avg_coded_batch_time += NodeTimeStamps['coded_duration']
        if is_partial_straggler:
          time.sleep((STRAGGLING_FACTOR_ALPHA-1) *
                     NodeTimeStamps['coded_duration'])
      if interrupt:
        break

      local_grad['dW'] += tgc_params.encoding_coefs[i]*grad['dW']
      local_loss += tgc_params.encoding_coefs[i]*loss

    # debug_print("Coded Computed", node_start_time, iter_no)
    NodeTimeStamps['full_coded_duration'] = time.time(
    ) - NodeTimeStamps['full_coded_start_time']
    NodeTimeStamps["Tot_batchtime"] = avg_coded_batch_time
    # NodeTimeStamps["Avg_batchtime"] = avg_coded_batch_time/count_of_batches
    NodeTimeStamps["count_batches"] = count_of_batches

    if tgc_params.is_alpha and my_rank <= 3:
      while num_naive_rcvd < 3:
        is_naive_avl, child_naive_payload = check_child_naive_grad(
            naive_payload_req, num_naive_rcvd)
        if is_naive_avl:
          num_naive_rcvd += 1
          naive_payload['grad']['dW'] += child_naive_payload['grad']['dW']
          naive_payload['loss'] += child_naive_payload['loss']

          if num_naive_rcvd < 3:
            naive_payload_req = comm.irecv(
                naive_buf, source=MPI.ANY_SOURCE, tag=NAIVE_TAG)
          else:
            naive_grad_rcvd_at = time.time()

    if tgc_params.is_alpha and not is_naive_grad_sent:
      remaining_comm_delay = time.time()-naive_grad_rcvd_at
      if remaining_comm_delay < COMM_CHANNEL_DELAY:
        time.sleep(COMM_CHANNEL_DELAY-remaining_comm_delay)
      comm.send(naive_payload, dest=parent_rank, tag=NAIVE_TAG)
      is_naive_grad_sent = True

    # if tgc_params.is_alpha:
    #   naive_grad_collection_thread.join()
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
      if COMM_CHANNEL_DELAY > 0:
        time.sleep(COMM_CHANNEL_DELAY)
      grad_send_start_time = time.time()
      comm.send(send_data, dest=parent_rank, tag=CODED_TAG)
      NodeTimeStamps['coded_comm_time_wo_delay'] = time.time() - \
          grad_send_start_time
      # debug_print("Coded Sent", node_start_time, iter_no)
      NodeTimeStamps['was_interrupted'] = False
    else:
      NodeTimeStamps['was_interrupted'] = True
    safe_dict_key_delete(NodeTimeStamps, 'full_coded_start_time')
    safe_dict_key_delete(NodeTimeStamps, 'naive_start_time')
    safe_dict_key_delete(NodeTimeStamps, 'coded_start_time')
    safe_dict_key_delete(NodeTimeStamps, 'naive_end_time')
    safe_dict_key_delete(NodeTimeStamps, 'coded_end_time')
    NodeTimeStamps['iter_no'] = iter_no
    NodeTimeStamps['num_coded_datapoints'] = debug_calc_examples
    # print(NodeTimeStamps)
