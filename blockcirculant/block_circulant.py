from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

def calculate_indx(input_dim, output_dim, p_size=8):
  num_filters_out = output_dim
  num_filters_in = input_dim

  def block_indx(k, rc, cc):
    rc = int((rc + k - 1) // k) * k
    cc = int((cc + k - 1) // k) * k
    i = np.arange(0, k, 1).reshape([1, k])
    j = np.arange(0, -k, -1).reshape([k, 1])
    indx = i + j
    indx = (indx + k) % k
    m = np.tile(indx, [int(rc // k), int(cc // k)])
    offset = np.arange(0, rc * cc)
    i = (offset // cc) // k
    j = (offset % cc) // k
    offset = (i * cc + j * k).reshape([rc, cc])
    return m + offset

  if p_size and p_size <= np.min([num_filters_out, num_filters_in]):
    indx = block_indx(p_size, num_filters_in, num_filters_out)
    target_c = num_filters_in * num_filters_out // p_size
    # print("you are using BlockCirc", p_size)
  else:
    # print("sorry, not enough size for partitoning", num_filters_in, num_filters_out)
    target_c = np.max([num_filters_in, num_filters_out])
    a, b = np.ogrid[0:target_c, 0:-target_c:-1]
    indx = a + b
  # print('num_filters_in:{}'.format(num_filters_in))
  # print('num_filters_out:{}'.format(num_filters_out))
  # print('target_c:{}'.format(target_c))
  indx = (indx + target_c) % target_c
  # print(target_c)
  indx = indx[:input_dim, :output_dim]
  # print(indx)
  return target_c, indx


def generate_shape_and_indices(input_dim,
                               output_dim,
                               kernel_h=None,
                               kernel_w=None,
                               block_size=8):
  num_of_weights, indices = calculate_indx(input_dim, output_dim, block_size)
  if kernel_h and kernel_w:
    indices = np.tile(indices, (kernel_h, kernel_w, 1, 1))
    offset = 0
    for i in range(kernel_h):
      for j in range(kernel_w):
        indices[i, j, :, :] += offset * num_of_weights
        offset += 1
    num_of_weights *= kernel_h * kernel_w
  indices = np.reshape(indices, [-1])
  return num_of_weights, indices


def make_block_circulant(W, num_of_weights, indices):
  """calculate Z based on W"""
  Z = np.reshape(W, [-1])
  sums = np.zeros([num_of_weights])
  counts = np.zeros([num_of_weights])
  # calculate mean

  for i in range(len(Z)):
    label = indices[i]
    sums[label] = Z[i]
    counts[label] += 1

  for i in range(num_of_weights):
    if counts[i] == 0:
      continue
    sums[i] /= counts[i]
  # update mean
  # Z = tf.gather(sums, indices)
  for i in range(len(Z)):
    label = indices[i]
    Z[i] = sums[label]
  return np.reshape(Z, W.shape)
