#pragma once
#include <cstdio>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
class Model;
#include "model.hpp"

typedef void (*activation_function)(float*,float*, int);

void sigmoid(float *input, float *output, int length);

void sigmoid_diff(float *input, float *output, int length);

class LinearLayer{
    /*
        This class implements a linear layer, with inputs X. 
        The idea is to compute Z = W[X:1] and then compute
        A = activation_func(Z)
        The gradients will get computed while the forward prop is happening
        We will require a pointer to the previous layer to compute the gradients which we will have
        The backward prop will perform gradient descent with a learning rate that will be gained from the model
    */
private:
    float* inputs;
    int input_size;
    float** weights;

    float** gradients;
    float* linear_outputs; //we will try to store the outputs here and not the sigmoid outputs since we just need to apply the function for that. Hopefully doesn't take as long
    float* outputs; //sigmoid outputs
    float* error; //buffer to make computation of gradients much easier
    int output_size;
    activation_function activation_func;

public:
    void init(int layer_size, int output_size, const char *activation_type);
    void load_layer(float *inputs); //copy the inputs array into the layer's array
    void feed_forward(Model *model, int layer_num); //does a forward pass
    void backward_propagate(Model *model, int layer_num);   //does a backward pass and computes the errors which will be used to compute the gradients
    void compute_gradients();    //computes the gradients using the error
    float* get_outputs(); //helper function to get a pointer to the output array
    void descend_grads(float lr);
    void deinit();
};