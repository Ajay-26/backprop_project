#include "model.hpp"

Model::Model(int n_layers, int input_size, int *output_sizes, int epochs, float learning_rate, int batch_size){
    int i;
    int in_size, out_size;
    this->n_layers = n_layers;
    this->input_size = input_size;
    this->output_sizes = output_sizes;
    this->output_size = output_sizes[n_layers-1];
    this->epochs = epochs;
    this->lr = learning_rate;
    this->batch_size = batch_size;

    //allocate memory for each layer
    this->layers = new LinearLayer[n_layers];
    in_size = input_size;
    for(i=0;i<this->n_layers;i++){
        out_size = this->output_sizes[i];
        this->layers[i].init(in_size, out_size, "sigmoid");
        in_size = out_size;
    }

    //allocate buffers for input and output
}

void Model::forward(float *inputs, int input_len){
    int i,output_size;
    float* layer_input;
    float* layer_output;
    layer_input = inputs;
    
    //feed forward through each of the layers
    for(i=0;i<this->n_layers; i++){
        this->layers[i].load_layer(layer_input);
        this->layers[i].feed_forward(this, i);
        layer_input = this->layers[i].get_outputs();
    }
    
    //copy the output of the last layer into the output array in this layer
    for(i=0;i<this->output_size;i++){
        this->outputs[i] = layer_input[i];
    }
    //forward propagation is done
    return;
}

int Model::get_num_layers(){
    return this->n_layers;
}

void Model::backward(){
    int i;
    for(i=this->n_layers-1; i >= 0; i--){
        this->layers[i].backward_propagate(this,i);
        this->layers[i].compute_gradients(); //can possibly spawn off another thread for this, since other tasks are independent
    }
}

void Model::gradient_descent_step(){
    //can do this in a multi-threaded way as well
    int i;
    for(i=0;i<this->n_layers;i++){
        this->layers[i].descend_grads(this->lr);
    }
    return;
}

float* Model::get_outputs(){
    return this->outputs;
}

void Model::prepare_dataset(DataSetHandler* dataset){
    this->dataset_handler = dataset;
}

void Model::train(DataSetHandler *dataset){
    printf("Here\n");
    int epoch,batch_idx;
    this->prepare_dataset(dataset);
    for(epoch=0; epoch<this->epochs; epoch++){
        for(batch_idx=0;batch_idx < this->batch_size; batch_idx++){
            this->dataset_handler->get_random_sample(&(this->inputs), &(this->outputs));
            this->forward(this->inputs,this->input_size);
            this->backward();
            this->gradient_descent_step();
        }
    }
    return;
}

float* Model::predict(float *inputs, int input_len){
    assert(input_len == this->input_size);
    if(inputs == nullptr){
        return nullptr;
    }
    this->forward(inputs, input_len);
    return this->outputs;
}
