import numpy as np
import time

# num_features = 7532
num_data_points = 250
num_readings = 20

for num_features in range(1000, 200000, 1000):
  time_taken = []
  for i in range(num_readings):
    dataset = np.random.randn(num_data_points, num_features+1)
    X = dataset[:, :-1]
    Y = dataset[:, -1]
    W = 0.001*np.random.randn(num_features, 1)

    start_comp_time = time.time()
    grad = (X.dot(W)-Y).T.dot(X)/num_data_points
    comp_duration = time.time()-start_comp_time
    end_comp_time = time.time()
    # print('Dry run: ' + str(end_comp_time-start_comp_time)
    W = 0.001*np.random.randn(num_features, 1)
    start_comp_time = time.time()

    X = X[i*num_data_points:(i+1):num_data_points]
    Y = Y[i*num_data_points:(i+1):num_data_points]
    grad = (X.dot(W)-Y).T.dot(X)/num_data_points

    # comp_duration = time.time()-start_comp_time
    end_comp_time = time.time()
    time_taken.append(end_comp_time-start_comp_time)

  avgTime = sum(time_taken)/num_readings
  # print(time_taken)
  # print(avgTime)
  print('Num Features: ' + str(num_features) + ' Time: ' + str(sum(time_taken)/num_readings))
  with open('data_comp_time.txt', 'a') as myfile:
    myfile.write('Num Features: ' + str(num_features) + ' Time: ' + str(sum(time_taken)/num_readings))
  # print(avgTime*6240/num_data_points)