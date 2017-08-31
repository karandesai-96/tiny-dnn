/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tiny_dnn/core/kernels/fully_connected_op_internal.h"

namespace tiny_dnn {
namespace kernels {

#ifdef CNN_USE_AVX

/**
 * Kernel for forward propogation for fully connected layer with AVX backend
 * (single precision)
 * @param in_data
 * @param weights
 * @param bias
 * @param out_data
 * @param layer_parallelize
 */
template <typename S1, typename S2>
inline void avx_fully_connected_forward_kernel(const Tensor<float, S1> &in_data,
                                               const Parameter &weights,
                                               const Parameter &bias,
                                               Tensor<float, S2> &out_data,
                                               const bool layer_parallelize) {
  size_t out_size = out_data.shape()[1], in_size = in_data.shape()[1],
         sample_size = in_data.shape()[0];
  if (bias.size() >= out_size) {
    size_t nblocks  = out_size / 8;
    size_t nremains = out_size & 7;
    if (nremains) {
      int32_t mask_src[] = {
        -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0,
      };
      __m256i imask =
        _mm256_loadu_si256((__m256i const *)(mask_src + 8 - nremains));
      for_i(layer_parallelize, sample_size, [&](size_t sample) {
        auto in  = in_data.host_pointer(sample, 0);
        auto out = out_data.host_pointer(sample, 0);
        {
          auto bias_pointer = bias.data_pointer(0);
          for (size_t i = 0; i < nblocks; ++i) {
            __m256 b = _mm256_loadu_ps(&bias_pointer[8 * i]);
            _mm256_storeu_ps(&*std::next(out, 8 * i), b);
          }
          auto b = _mm256_maskload_ps(bias.data_pointer(8 * nblocks), imask);
          _mm256_maskstore_ps(&out[8 * nblocks], imask, b);
        }
        auto W_pointer = weights.data_pointer(0, 0);
        for (size_t c = 0; c < in_size; c++) {
          auto in_val = _mm256_set1_ps(in[c]);
          auto pW     = &W_pointer[c * out_size];
          for (size_t i = 0; i < nblocks / 2; ++i) {
            __m256 sum0 = _mm256_loadu_ps(&out[16 * i]);
            __m256 sum1 = _mm256_loadu_ps(&out[16 * i + 8]);
            __m256 w0   = _mm256_loadu_ps(&pW[16 * i]);
            __m256 w1   = _mm256_loadu_ps(&pW[16 * i + 8]);
            sum0        = madd256_ps(w0, in_val, sum0);
            sum1        = madd256_ps(w1, in_val, sum1);
            _mm256_storeu_ps(&out[16 * i], sum0);
            _mm256_storeu_ps(&out[16 * i + 8], sum1);
          }
          if (nblocks & 1) {
            __m256 sum0 = _mm256_loadu_ps(&out[nblocks / 2 * 16]);
            __m256 w0   = _mm256_loadu_ps(&pW[nblocks / 2 * 16]);
            sum0        = madd256_ps(w0, in_val, sum0);
            _mm256_storeu_ps(&out[nblocks / 2 * 16], sum0);
          }
          __m256 sum = _mm256_maskload_ps(&out[8 * nblocks], imask);
          __m256 w   = _mm256_maskload_ps(&pW[8 * nblocks], imask);
          sum        = madd256_ps(w, in_val, sum);
          _mm256_maskstore_ps(&out[8 * nblocks], imask, sum);
        }
      });
    } else {
      for_i(layer_parallelize, in_data.shape()[0], [&](size_t sample) {
        auto in        = in_data.host_pointer(sample, 0);
        auto out       = out_data.host_pointer(sample, 0);
        auto b_pointer = bias.data_pointer(0);
        for (size_t i = 0; i < nblocks; ++i) {
          __m256 b = _mm256_loadu_ps(&b_pointer[8 * i]);
          _mm256_storeu_ps(&out[8 * i], b);
        }
        auto W_pointer = weights.data_pointer(0, 0);
        for (size_t c = 0; c < in_size; c++) {
          auto in_val = _mm256_set1_ps(in[c]);
          auto pW     = &W_pointer[c * out_size];
          for (size_t i = 0; i < nblocks; ++i) {
            __m256 sum = _mm256_loadu_ps(&out[8 * i]);
            __m256 w   = _mm256_loadu_ps(&pW[8 * i]);
            sum        = madd256_ps(w, in_val, sum);
            _mm256_storeu_ps(&out[8 * i], sum);
          }
        }
      });
    }
  } else {
    for_i(layer_parallelize, sample_size, [&](size_t sample) {
      auto in  = in_data.host_iter(sample, 0);
      auto out = out_data.host_iter(sample, 0);
      for (size_t i = 0; i < out_size; i++) {
        float sum = 0.0f;
        for (size_t c = 0; c < in_size; c++) {
          sum += weights.data_at(c, i) * *std::next(in, c);
        }
        *std::next(out, i) = sum;
      }
    });
  }
}

/**
 * Kernel for forward propogation for fully connected layer with AVX backend
 * (double precision). Currently calls for internal backend
 * @param in_data
 * @param weights
 * @param bias
 * @param out_data
 * @param layer_parallelize
 */
template <typename S1, typename S2>
inline void avx_fully_connected_forward_kernel(
  const Tensor<double, S1> &in_data,
  const Parameter &weights,
  const Parameter &bias,
  Tensor<double, S2> &out_data,
  const bool layer_parallelize) {
  // fallback to tiny-backend when float_t is double
  fully_connected_op_internal(in_data, weights, bias, out_data,
                              layer_parallelize);
}

/**
 * Kernel for backward propogation for fully connected layer with AVX backend
 * (single precision).
 * @param prev_out
 * @param weights
 * @param bias
 * @param curr_delta
 * @param prev_delta
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3>
inline void avx_fully_connected_back_kernel(const Tensor<float, S1> &prev_out,
                                            Parameter &weights,
                                            Parameter &bias,
                                            Tensor<float, S2> &curr_delta,
                                            Tensor<float, S3> &prev_delta,
                                            const bool layer_parallelize) {
  size_t out_size = curr_delta.shape()[1], in_size = prev_delta.shape()[1],
         sample_size = prev_out.shape()[0];
  if (bias.grad()->size() >= out_size) {
    for (size_t sample = 0; sample < sample_size; sample++) {
      for (size_t c = 0; c < in_size; c++) {
        // propagate delta to previous layer
        // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
        prev_delta.host_at(sample, c) +=
          vectorize::dot(curr_delta.host_pointer(sample, 0),
                         weights.data_pointer(c, 0), out_size);
      }
      for_(layer_parallelize, 0, size_t(out_size), [&](const blocked_range &r) {
        // accumulate weight-step using delta
        // dW[c * out_size + i] += current_delta[i] * prev_out[c]
        size_t len = r.end() - r.begin();
        for (size_t c = 0; c < in_size; c++) {
          vectorize::muladd(curr_delta.host_pointer(sample, r.begin()),
                            prev_out.host_at(sample, c), len,
                            weights.grad_pointer(sample, c, r.begin()));
        }
        // vec_t& db = *in_grad[2];
        vectorize::reduce(curr_delta.host_pointer(sample, r.begin()), len,
                          bias.grad_pointer(sample, r.begin()));
      });
    }
  } else {
    for (size_t sample = 0; sample < sample_size; sample++) {
      for (size_t c = 0; c < in_size; c++) {
        // propagate delta to previous layer
        // prev_delta[c] += current_delta[r] * W_[c * out_size_ + r]
        prev_delta.host_at(sample, c) +=
          vectorize::dot(curr_delta.host_pointer(sample, 0),
                         weights.data_pointer(c, 0), out_size);
      }
      for_(layer_parallelize, 0, size_t(out_size), [&](const blocked_range &r) {
        // accumulate weight-step using delta
        // dW[c * out_size + i] += current_delta[i] * prev_out[c]
        size_t len = r.end() - r.begin();
        for (size_t c = 0; c < in_size; c++) {
          vectorize::muladd(curr_delta.host_pointer(sample, r.begin()),
                            prev_out.host_at(sample, c), len,
                            weights.grad_pointer(sample, c, r.begin()));
        }
      });
    }
  }
}

/**
 * Kernel for backward propogation for fully connected layer with AVX backend
 * (double precision). Currently calls for internal backend
 * @param prev_out
 * @param weights
 * @param bias
 * @param curr_delta
 * @param prev_delta
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3>
inline void avx_fully_connected_back_kernel(const Tensor<double, S1> &prev_out,
                                            Parameter &weights,
                                            Parameter &bias,
                                            Tensor<double, S2> &curr_delta,
                                            Tensor<double, S3> &prev_delta,
                                            const bool layer_parallelize) {
  // fallback to tiny-backend when float_t is double
  fully_connected_op_internal(prev_out, weights, bias, curr_delta, prev_delta,
                              layer_parallelize);
}

#endif  // CNN_USE_AVX

/**
 * Forward propogation for fully connected layer with internal backend
 * @param in_data
 * @param weights
 * @param bias
 * @param out_data
 * @param layer_parallelize
 */
template <typename S1, typename S2>
inline void fully_connected_op_avx(const Tensor<float_t, S1> &in_data,
                                   const Parameter &weights,
                                   const Parameter &bias,
                                   Tensor<float_t, S2> &out_data,
                                   const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  avx_fully_connected_forward_kernel(in_data, weights, bias, out_data,
                                     layer_parallelize);
#else
  CNN_UNREFERENCED_PARAMETER(in_data);
  CNN_UNREFERENCED_PARAMETER(weights);
  CNN_UNREFERENCED_PARAMETER(bias);
  CNN_UNREFERENCED_PARAMETER(out_data);
  CNN_UNREFERENCED_PARAMETER(layer_parallelize);
  throw nn_error("tiny-dnn has not been compiled with AVX support.");
#endif
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
inline void fully_connected_op_avx(const Tensor<float_t, S1> &prev_out,
                                   Parameter &weights,
                                   Parameter &bias,
                                   Tensor<float_t, S2> &curr_delta,
                                   Tensor<float_t, S3> &prev_delta,
                                   const bool layer_parallelize) {
#ifdef CNN_USE_AVX
  avx_fully_connected_back_kernel(prev_out, weights, bias, curr_delta,
                                  prev_delta, layer_parallelize);
#else
  CNN_UNREFERENCED_PARAMETER(prev_out);
  CNN_UNREFERENCED_PARAMETER(weights);
  CNN_UNREFERENCED_PARAMETER(bias);
  CNN_UNREFERENCED_PARAMETER(curr_delta);
  CNN_UNREFERENCED_PARAMETER(prev_delta);
  CNN_UNREFERENCED_PARAMETER(layer_parallelize);
  throw nn_error("tiny-dnn has not been compiled with AVX support.");
#endif
}

}  // namespace kernels
}  // namespace tiny_dnn
