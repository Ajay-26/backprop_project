#include "layer.hpp"

void sigmoid(float *input, float *output, int length){
    int i;
    for(i=0;i<length;i++){
        output[i] = 1.0/(1 + exp(-input[i]));
    }
    return;
}

void sigmoid_diff(float *input, float *output, int length){
    //compute in place in the output
    int i;
    sigmoid(input,output, length);
    for(i=0;i<length;i++){
        output[i] = output[i] * (1 - output[i]);
    }
}

void LinearLayer::init(int input_size, int output_size, const char *activation_type){
    int i,j;
    this->input_size = input_size;
    this->output_size = output_size;
    //initialise the activation func
    if(std::strcmp(activation_type, "sigmoid") == 0){
        this->activation_func = sigmoid;
    }
    this->inputs = new float[this->input_size];
    if(this->inputs == nullptr){
        printf("Bad alloc");
        return;
    }
    for(i=0;i<this->input_size;i++){
        this->inputs[i] = 0;
    }
    //initialise outputs to 0
    this->outputs = new float[this->output_size];
    if(this->outputs == nullptr){
        printf("Bad alloc");
        return;
    }
    for(i=0;i<this->output_size; i++){
        this->outputs[i] = 0;
    }

    this->linear_outputs = new float[this->output_size];
    if(this->linear_outputs == nullptr){
        printf("Bad alloc");
        return;
    }
    for(i=0;i<this->output_size; i++){
        this->linear_outputs[i] = 0;
    }

    this->error = new float[this->output_size];
    if(this->error == nullptr){
        printf("Bad alloc");
        return;
    }
    for(i=0;i<this->output_size; i++){
        this->error[i] = 0;
    }

    //initialise weights to 0
    this->weights = new float*[this->output_size];
    if(this->weights == nullptr){
        printf("Bad alloc");
        return;
    }
    for(i=0;i<this->output_size;i++){
        weights[i] = new float[this->input_size];
        if(this->weights[i] == nullptr){
            printf("Bad alloc");
            return;
        }
        for(j=0;j<this->input_size;j++){
            weights[i][j] = 0;
        }
    }

    //initialise gradients to 0
    this->gradients = new float*[this->output_size];
    if(this->gradients == nullptr){
        printf("Bad alloc");
        return;
    }
    for(i=0;i<this->output_size;i++){
        gradients[i] = new float[this->input_size];
        if(this->gradients[i] == nullptr){
        printf("Bad alloc");
        return;
    }
        for(j=0;j<this->input_size;j++){
            gradients[i][j] = 0;
        }
    }
    return;
}

void LinearLayer::load_layer(float* inputs){
    int i;
    for(i=0;i<this->input_size;i++){
        this->inputs[i] = inputs[i];
    }
    return;
}

void LinearLayer::feed_forward(Model *model, int layer_num){
    int i,j,k;
    for(i=0;i<this->output_size;i++){
        this->linear_outputs[i] = 0;
        for(j=0;j<this->input_size;j++){
            this->linear_outputs[i] += this->weights[i][j] * this->inputs[j];
        }
    }
    sigmoid(this->linear_outputs, this->outputs, this->output_size);
    return;
}

void LinearLayer::backward_propagate(Model *model, int layer_num){
    //first compute the error of the layer
    int i,j;
    int num_layers = model->get_num_layers();
    float *output_ptr;
    if(layer_num == num_layers - 1){
        //if last layer, then compute the error using the cost function
        int i;
        //first initialise the error to be gradient of sigmoid
        sigmoid_diff(this->linear_outputs, this->error, this->output_size);
        
        //then multiply by cost - pred to compute it in place
        output_ptr = model->get_outputs();
        for(i=0;i<this->output_size;i++){
            this->error[i] = (this->outputs[i] - output_ptr[i]) * this->error[i];
        }
    }else{
        //if not last layer, then compute using the error of layer indexed layer_num + 1        
        float* next_error = model->layers[layer_num + 1].error;
        int next_size = model->layers[layer_num + 1].output_size;
        float **next_weights = model->layers[layer_num + 1].weights;
        sigmoid_diff(this->linear_outputs, this->error, this->output_size);
        float val;
        for(i=0;i<this->output_size;i++){
            val = this->error[i];
            this->error[i] = 0;
            for(j=0;j<next_size; j++){
                this->error[i] += next_weights[j][i] * next_error[j];
            }
            this->error[i] = this->error[i] * val;
        }
    }
    return;
}

void LinearLayer::compute_gradients(){
    //compute the gradients using the errors and inputs to the model
    int i,j;
    for(i=0;i<this->output_size;i++){
        for(j=0;j<this->input_size;j++){
            this->gradients[i][j] = this->error[i] * this->inputs[j];
        }
    }
    return;
}

float* LinearLayer::get_outputs(){
    return this->outputs;
}

void LinearLayer::descend_grads(float lr){
    int i,j;
    for(i=0;i<this->output_size;i++){
        for(j=0;j<this->input_size;j++){
            //decrease the weights
            this->weights[i][j] = this->weights[i][j] - lr * this->gradients[i][j];
            
            //refresh gradients as well
            this->gradients[i][j] = 0;
        }
    }
    return;
}

void LinearLayer::deinit(){
    int i,j;
    delete[] this->inputs;
    delete[] this->outputs;
    delete[] this->error;
    delete[] this->linear_outputs;
    for(i=0;i<this->output_size;i++){
        delete[] this->weights[i];
        delete[] this->gradients[i];
    }
    delete[] this->weights;
    delete[] this->gradients;
}