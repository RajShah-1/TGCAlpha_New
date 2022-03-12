COMM_CHANNEL_DELAY = 0.120059625*0.4
# COMM_CHANNEL_DELAY = 0.1
# beta*d = delay => delay/d -> must be constant
DATA_POINT_INCREMENT_FACTOR = 1
BUFFER_SIZE = 2000 * 1024
BATCH_SIZE = 50
NUM_ITER = 100
NUM_WORKERS = 12
STRAGGLING_FACTOR_ALPHA = 3


class TGCAlphaBeta:
  def __init__(self, rank, data_point_inc_fact):
    self.my_rank = rank
    self.DATA_POINT_INCREMENT_FACTOR = data_point_inc_fact

  is_alpha = True
  decoding_matrix_A = [[0, 1, 2],
                       [1, 0, 1],
                       [2, -1, 0]]

  encoding_coefs = {
      '1': [0.5, 1],  # (1, 1)
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

  def get_m_naive(self):
    if self.my_rank <= 3:
      return 608 * self.DATA_POINT_INCREMENT_FACTOR
    else:
      return 192 * self.DATA_POINT_INCREMENT_FACTOR

  def get_m_coded(self, dataset_ind):
    if self.my_rank <= 3:
      dataset_size = [608, 608]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR
    else:
      dataset_size = [96, 96, 96, 96]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR

  def get_num_coded_datasets(self):
    if self.my_rank <= 3:
      return 2
    else:
      return 4


class TGCBeta:
  def __init__(self, rank, data_point_inc_fact):
    self.my_rank = rank
    self.DATA_POINT_INCREMENT_FACTOR = data_point_inc_fact

  is_alpha = False

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

  def get_m_coded(self, dataset_ind):
    if self.my_rank <= 3:
      dataset_size = [1768, 1768]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR
    else:
      dataset_size = [104, 104, 104, 104]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR

  def get_num_coded_datasets(self):
    if self.my_rank <= 3:
      return 2
    else:
      return 4


class TGCAlpha:
  def __init__(self, rank, data_point_inc_fact):
    self.my_rank = rank
    self.DATA_POINT_INCREMENT_FACTOR = data_point_inc_fact

  is_alpha = True
  decoding_matrix_A = [[0, 1, 2],
                       [1, 0, 1],
                       [2, -1, 0]]

  encoding_coefs = {
      '1':  [0.5, 0.5],  # (1, 1)
      '2': [1, 1],  # (1, 2)
      '3': [0.5, 0.5],  # (1, 3)
      # Childern of (1, 1)
      '4': [0.25, 0.5, 1, 1],  # (2, 1)
      '5': [1, 1, -1, -1],  # (2, 2)
      '6': [0.25, 0.5, 1, 1],  # (2, 3)
      # Childern of (1, 2)
      '7': [0.5, 0.5, -1, -1],  # (2, 4)
      '8': [-1, -1, 1, 1],  # (2, 5)
      '9': [0.5, -0.5, -1, -1],  # (2, 6)
      # Childern of (1, 3)
      '10': [0.25, 0.5, 1, 1],  # (2, 7)
      '11': [1, 1, -1, -1],  # (2, 8)
      '12': [0.25, 0.5, 1, 1],  # (2, 9)
  }

  def get_m_naive(self):
    if self.my_rank <= 3:
      return 320 * self.DATA_POINT_INCREMENT_FACTOR
    else:
      return 320 * self.DATA_POINT_INCREMENT_FACTOR

  def get_m_coded(self, dataset_ind):
    if self.my_rank <= 3:
      dataset_size = [320, 320]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR
    else:
      dataset_size = [160, 160, 160, 160]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR

  def get_num_coded_datasets(self):
    if self.my_rank <= 3:
      return 2
    else:
      return 4


class TGC:
  def __init__(self, rank, data_point_inc_fact):
    self.my_rank = rank
    self.DATA_POINT_INCREMENT_FACTOR = data_point_inc_fact

  is_alpha = False
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

  def get_m_coded(self, dataset_ind):
    if self.my_rank <= 3:
      dataset_size = [832, 832]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR
    else:
      dataset_size = [416, 416, 416, 416]
      return dataset_size[dataset_ind] * self.DATA_POINT_INCREMENT_FACTOR

  def get_num_coded_datasets(self):
    if self.my_rank <= 3:
      return 2
    else:
      return 4
