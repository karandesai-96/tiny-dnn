/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/fully_params.h"

namespace tiny_dnn {
namespace kernels {

/**
 * Forward propogation for fully connected layer with internal backend
 * @param in_data
 * @param weights
 * @param bias
 * @param layer_parallelize
 */
template <typename S1, typename S2>
inline void fully_connected_op_internal(const Tensor<float_t, S1> &in_data,
                                        const Parameter &weights,
                                        const Parameter &bias,
                                        Tensor<float_t, S2> &out_data,
                                        const bool layer_parallelize) {
  size_t out_size = out_data.shape()[1], in_size = in_data.shape()[1];
  for_i(layer_parallelize, in_data.shape()[0], [&](int sample) {
    for (size_t i = 0; i < out_size; i++) {
      out_data.host_at(sample, i) = float_t{0};
      for (size_t c = 0; c < in_size; c++) {
        out_data.host_at(sample, i) +=
          weights.data_at(c, i) * in_data.host_at(sample, c);
      }

      if (bias.size() >= out_size) {
        out_data.host_at(sample, i) += bias.data_at(i);
      }
    }
  });
}

/**
 * Backward propogation for fully connected layer with internal backend
 * @param prev_out
 * @param weights
 * @param bias
 * @param curr_delta
 * @param prev_delta
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3>
inline void fully_connected_op_internal(const Tensor<float_t, S1> &prev_out,
                                        Parameter &weights,
                                        Parameter &bias,
                                        Tensor<float_t, S2> &curr_delta,
                                        Tensor<float_t, S3> &prev_delta,
                                        const bool layer_parallelize) {
  size_t out_size = curr_delta.shape()[1], in_size = prev_delta.shape()[1],
         sample_num = prev_out.shape()[0];
  for (size_t sample = 0; sample < sample_num; sample++) {
    for (size_t c = 0; c < in_size; c++) {
      // propagate delta to previous layer
      // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
      for (size_t i = 0; i < out_size; i++) {
        prev_delta.host_at(sample, c) +=
          curr_delta.host_at(sample, i) * weights.data_at(c, i);
      }
    }

    for_(layer_parallelize, 0, size_t(out_size), [&](const blocked_range &r) {
      // accumulate weight-step using delta
      // dW[c * out_size + i] += current_delta[i] * prev_out[c]
      for (size_t c = 0; c < in_size; c++) {
        for (size_t i = r.begin(); i < r.end(); i++) {
          weights.grad_at(sample, c, i) +=
            prev_out.host_at(sample, c) * curr_delta.host_at(sample, i);
        }
      }

      if (bias.grad()->size() >= out_size) {
        // vec_t& db = *in_grad[2];
        for (size_t i = r.begin(); i < r.end(); i++) {
          bias.grad_at(sample, i) += curr_delta.host_at(sample, i);
        }
      }
    });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
