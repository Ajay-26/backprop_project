#include "dataset.hpp"
#include "model.hpp"

int main(int argc, char **argv){
    DataSetHandler d("../mnist/train-images-idx3-ubyte","../mnist/train-labels-idx1-ubyte");
    d.parse_data();
    int layer_size[2] = {8,1};
    Model m(2,784,layer_size,1,0.0025,64);
    m.train(&d);
    m.deinit();
    printf("Done\n");
    return -1;
}