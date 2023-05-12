#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstring>
#include <cerrno>
#include <cstdint>
#include <cstdlib>

class DataSetHandler{
    private:
        //Store the path to the file as well in a C-style string
        const char* filename;
        const char* labels_filename;
        //all samples are in a flattened array
        float** values;
        //all labels are also in a flattened array
        float* labels;
        //Each sample is of length vector_len, and there are n_samples samples
        int n_samples;
        int sample_len;
        int vector_len;
        bool is_label_file;
        bool is_train;
        int magic_number;
        int batch_size;

    public:
        DataSetHandler(const char *name, const char *label_name);
        int read_file(const char *fn);
        void deinit();
        void parse_data();
        void get_random_sample(float** random_sample_ptr, float** random_label_ptr); //This function will return a random sample and label
};
