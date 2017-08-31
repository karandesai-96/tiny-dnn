/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "tiny_dnn/core/framework/tensor.h"
#include "tiny_dnn/util/parameter_init.h"
#include "tiny_dnn/util/util.h"

namespace tiny_dnn {

enum class parameter_type : int8_t { weight = 0x1, bias = 0x2 };

class Parameter : public std::enable_shared_from_this<Parameter> {
 public:
  /**
   * Initializes an empty parameter taking in the dimensions of weights and
   * biases of a layer. Currently supported for maximum 4-dimensions, and
   * stored as a flat ``Tensor``. Parameters are flat and represented in NCHW
   * format.
   *
   * todo (karandesai) : generalize to n-dimensions
   * todo (karandesai) : add an n-dimensional view for easy indexing
   *
   * @param out_channels number of feature maps in next layer
   * @param in_channels  filter depth / input channels
   * @param height       filter height
   * @param width        filter width
   * @param type         whether parameter is a weight or a bias
   * @param trainable    whether parameter will be updated while training
   */
  // TODO(karan): what is preferred order? probably, height and width first
  Parameter(size_t out_channels,
            size_t in_channels,
            size_t height,
            size_t width,
            parameter_type type,
            bool trainable = true)
    : type_(type),
      shape_(width, height, in_channels * out_channels),
      trainable_(trainable),
      initialized_(false),
      data_({shape_.size()}, 0),
      grad_({1, shape_.size()}, 0) {}

  /**
   * Initializes an empty parameter taking in the dimensions.
   *
   * @param shape     shape of the parameter
   * @param type      whether parameter is a weight or a bias
   * @param trainable whether parameter will be updated while training
   */
  Parameter(const std::vector<size_t> &shape,
            parameter_type type,
            bool trainable = true)
    : type_(type),
      shape_(0, 0, 0),  // todo (karandesai) : remove after generalization
      trainable_(trainable),
      initialized_(false),
      data_(shape),
      grad_(shape) {
    data_.fill(0);
    grad_.repeat(1, data_);
  }

  // copy constructor
  Parameter(const Parameter &other)
    : type_(other.type()),
      shape_(other.shape()),
      trainable_(other.is_trainable()),
      initialized_(other.initialized()),
      data_(*(other.data())),
      grad_(*(other.grad())) {}

  shape3d shape() const { return shape_; }

  size_t size() const { return data_.size(); }

  parameter_type type() const { return type_; }

  bool is_trainable() const { return trainable_; }

  void set_trainable(bool trainable = true) { trainable_ = trainable; }

  bool initialized() const { return initialized_; }

  void set_initialized(bool initialized = true) { initialized_ = initialized; }

  void initialize(parameter_init::function &f,
                  const size_t &fan_in  = 1,
                  const size_t &fan_out = 1) {
    f.fill(data_, fan_in, fan_out);
    clear_grads();
  }

  Tensor<> *data() { return &data_; }

  const Tensor<> *data() const { return &data_; }

  void set_data(const Tensor<> &data) {
    data_        = data;
    initialized_ = true;
  }

  Tensor<> *grad() { return &grad_; }

  const Tensor<> *grad() const { return &grad_; }

  void set_grad(const Tensor<> &grad) { grad_ = grad; }

  void resize_grad(size_t sample_count) {
    grad_.reshape({sample_count, size()});
    clear_grads();
  }

  void merge_grads(Tensor<> *dst) {
    tensor_t grad_t = grad_.toTensor();
    vec_t dst_t{0};
    const auto &grad_head = grad_t[0];
    size_t sz             = grad_head.size();
    dst_t.resize(sz, 0);
    std::copy(grad_head.begin(), grad_head.end(), dst_t.begin());
    // @todo consider adding parallelism
    for (size_t sample = 1, sample_count = grad_.shape()[0];
         sample < sample_count; ++sample) {
      vectorize::reduce<float_t>(&grad_t[sample][0], sz, &dst_t[0]);
    }
    *dst = Tensor<>(dst_t);
  }

  void clear_grads() { grad_.fill(float_t{0}); }

  /**
   * @name Data and Grad Accessing Utilities
   * @{
   */
  template <typename... Args>
  float_t &data_at(const Args... args) {
    return data_.host_at(args...);
  }

  template <typename... Args>
  float_t &grad_at(const size_t &sample, const Args... args) {
    return grad_.host_at(sample, args...);
  }

  template <typename... Args>
  float_t *data_pointer(const Args... args) {
    return data_.host_pointer(args...);
  }

  template <typename... Args>
  float_t *grad_pointer(const size_t &sample, const Args... args) {
    return grad_.host_pointer(sample, args...);
  }

  template <typename... Args>
  float_t data_at(const Args... args) const {
    return data_.host_at(args...);
  }

  template <typename... Args>
  float_t grad_at(const size_t &sample, const Args... args) const {
    return grad_.host_at(sample, args...);
  }

  template <typename... Args>
  const float_t *data_pointer(const Args... args) const {
    return data_.host_pointer(args...);
  }

  template <typename... Args>
  const float_t *grad_pointer(const size_t &sample, const Args... args) const {
    return grad_.host_pointer(sample, args...);
  }
  /** @} */  // Data and Grad Accessing Utilities

  // todo (karandesai) : introduce support for HDF
  /**
   * @name Serialization - Deserialization Utilities
   * @{
   */
  void save(std::ostream &os,
            const int precision = std::numeric_limits<float_t>::digits10 + 2) {
    os << std::setprecision(precision);
    for (size_t i = 0; i < data_.size(); i++) {
      os << data_at(i) << " ";
    }
  }

  void load(std::istream &is,
            const int precision = std::numeric_limits<float_t>::digits10 + 2) {
    is >> std::setprecision(precision);
    for (size_t i = 0; i < data_.size(); i++) {
      is >> data_at(i);
    }
  }
  /** @} */  // Serialization - Deserialization Utilities

 private:
  parameter_type type_;

  // todo (karandesai) : replace with vector<size_t> for n-dimensional
  // parameters
  shape3d shape_;

  bool trainable_;
  bool initialized_;

  Tensor<> data_;
  Tensor<> grad_;
};  // class parameter

// todo (karandesai) : analyze performance between raw pointer and shared_ptr
// after fc parameter integration

using Parameters      = std::vector<Parameter *>;
using ConstParameters = std::vector<const Parameter *>;

}  // namespace tiny_dnn
