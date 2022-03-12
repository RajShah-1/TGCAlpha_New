import numpy as np
import time

num_features = 7532
num_data_points = 100
num_readings = 10000

time_taken = []

for i in range(num_readings):
  W = 0.001*np.random.randn(num_features, 1)
  dataset = np.random.randn(num_data_points, num_features+1)
  start_comp_time = time.time()
  W = 0.001*np.random.randn(num_features, 1)

  X_batch = dataset[:, :-1]
  y_batch = dataset[:, -1]
  grad = (X_batch.dot(W)-y_batch).T.dot(X_batch)/num_data_points

  comp_duration = time.time()-start_comp_time
  time_taken.append(comp_duration)
  
  with open('data_comp_straggle.txt', 'a') as myfile:
    myfile.write(str(comp_duration))

avgTime = sum(time_taken)/num_readings
print(time_taken)
print(avgTime)
print(avgTime*6240/num_data_points)