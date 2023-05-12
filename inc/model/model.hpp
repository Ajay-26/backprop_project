#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cassert>
class LinearLayer;
#include "layer.hpp"
#include "dataset.hpp"


class Model{
private:
    LinearLayer* layers;
    int *output_sizes; //sizes of each layer's output. This will be given by the user
    int n_layers;

    float *inputs;
    int input_size;

    float *outputs;
    int output_size;

    int epochs;
    float lr;
    int batch_size;
    DataSetHandler* dataset_handler;

public:
    friend class LinearLayer;
    Model(int n_layers, int input_size, int *output_sizes, int epochs, float learning_rate, int batch_size);
    void forward(float *inputs, int input_len); //for assertion on length of input
    void backward();
    void gradient_descent_step();
    int get_num_layers();
    void train(DataSetHandler* dataset);
    float *get_outputs();
    float* predict(float *inputs, int input_len);
    void prepare_dataset(DataSetHandler* dataset);
    void deinit();
    
};