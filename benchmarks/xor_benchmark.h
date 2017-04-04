#pragma once

#include "benchmark/benchmark.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {
namespace tiny_benchmark {

void XOR_train(benchmark::State& state) {
    network<sequential> nn;
    adam opt;
    nn << fully_connected_layer(2, 3) << tanh_layer(3)
       << fully_connected_layer(3, 1) << softmax_layer(1);

    std::vector<vec_t> input_data  { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<vec_t> desired_out {    {0},    {1},    {1},    {0} };
    size_t batch_size = 1;
    int epochs = 30;

    while (state.KeepRunning()) {
        nn.fit<mse, adam>(opt, input_data, desired_out, batch_size, epochs);
    }
}

// Register the function as a benchmarks
BENCHMARK(XOR_train);
}
}  // namespace tiny_dnn
