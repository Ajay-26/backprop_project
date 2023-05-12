#include "dataset.hpp"

DataSetHandler::DataSetHandler(const char *name, const char *labels_name){
    this->filename = name; //this should be a C-Style string
    this->labels_filename = labels_name; //this should be a C-Style string
    int latest_delim = -1;
    int len = strlen(this->filename);

    int i;
    this->vector_len = 1;
    while(i < len){
        if(this->filename[i] == '/'){
            latest_delim = i;
        }
        i++;
    }
    //set values for magic number, n_samples and sample_len to be expected values. Values read will be matched with this later
    int gap = 6;
    this->n_samples = 10000;
    //the file would be either train_images-... or train_labels-... or t10k...
    if(this->filename[latest_delim + 2] == 'r'){
        this->is_train = true;
        gap = 7;
        this->n_samples = 60000;
    }
    if(this->filename[latest_delim+gap] == 'i'){
        this->is_label_file = false;
        this->magic_number = 2051;
        this->sample_len = 28;
    }else if(this->filename[latest_delim+gap] == 'l'){
        this->is_label_file = true;
        this->magic_number = 2049;
        this->sample_len = 1;
    }else{
        printf("Error: Wrong file format specified\n");
    }
}

int DataSetHandler::read_file(const char *fn){
    printf("%s\n", fn);
    FILE *fp = fopen(fn,"r");
    if(fp == nullptr){
        printf("Could not open file! Value of errno = %d\n",errno);
        return -1;
    }
    int c,i;
    int idx = 0;
    int ret;
    uint8_t byte_group[4] = {0,0,0,0};
    uint32_t magic_num,n,length,width;
    //first read the magic number
    ret = fread((void*)byte_group,1,4,fp);
    if(ret != 4){
        printf("Read did not work\n");
        fclose(fp);
        return -1;
    }
    magic_num = byte_group[3] | (byte_group[2] << 8) | (byte_group[1] << 16) | (byte_group[0] << 24);
    if(this->is_label_file == false && magic_num != this->magic_number){
        printf("Incorrect magic number! Given %u and expected 2051\n",magic_num);
        fclose(fp);
        return -1;
    }

    if(this->is_label_file == true && magic_num != this->magic_number){
        printf("Incorrect magic number! Given %u and expected 2049\n",magic_num);
        fclose(fp);
        return -1;
    }

    //read the number of samples
    ret = fread((void*)byte_group,1,4,fp);
    if(ret != 4){
        printf("Read did not work\n");
        fclose(fp);
        return -1;
    }
    n = byte_group[3] | (byte_group[2] << 8) | (byte_group[1] << 16) | (byte_group[0] << 24);
    if(this->is_label_file == false){
        this->n_samples = n;
    }
    if(this->is_label_file == false && n != this->n_samples){
        printf("Incorrect num samples! Given %u and expected 60000 here\n",n);
        printf("%x,%x,%x,%x\n", byte_group[0],byte_group[1],byte_group[2],byte_group[3]);
        fclose(fp);
        return -1;
    }

    if(this->is_label_file == true && n != this->n_samples){
        printf("Incorrect num samples! Given %u and expected %u\n",n,this->n_samples);
        fclose(fp);
        return -1;
    }

    //read the length and width, only if it is not the label file because the last byte of the magic number is 3. For the test file it is 1, so that means there are no dims to parse, only n_samples
    if(this->is_label_file == false){
        //read the length
        ret = fread((void*)byte_group,1,4,fp);
        if(ret != 4){
            printf("Read did not work\n");
            fclose(fp);
            return -1;
        }
        length = byte_group[3] | (byte_group[2] << 8) | (byte_group[1] << 16) | (byte_group[0] << 24);
        if(this->is_label_file == false){
            this->sample_len = length;
            this->vector_len = (this->vector_len)*length;
        }
        if(length != this->sample_len){
            printf("Incorrect value of length! Given %u and expected 28\n",length);
            fclose(fp);
            return -1;
        }

        //read the width
        ret = fread((void*)byte_group,1,4,fp);
        if(ret != 4){
            printf("Read did not work\n");
            fclose(fp);
            return -1;
        }
        width = byte_group[3] | (byte_group[2] << 8) | (byte_group[1] << 16) | (byte_group[0] << 24);
        if(this->is_label_file == false){
            this->vector_len = (this->vector_len)*width;
        }
        if(this->is_label_file == false && width != this->sample_len){
            printf("Incorrect value of width! Given %u and expected 28\n",width);
            fclose(fp);
            return -1;
        }
    }

    //Prepare arrays to store the data
    
    if(this->is_label_file == false){
        float** buffer;
        int buf_len;
        this->values = new float*[this->n_samples];
        for(i=0;i<this->n_samples;i++){
            this->values[i] = new float[this->vector_len];
        }
        buffer = this->values;
        //Now just read the data
        int data_byte;
        idx = 0;
        uint8_t tmp;
        while(1){
            data_byte = fgetc(fp);
            if(feof(fp)){
                break;
            }   
            else{
                tmp = (uint8_t)data_byte;
                buffer[idx/(this->vector_len)][idx%(this->vector_len)] = ((float)tmp)/255.0;
                idx++;
            }
        }
        if(idx != (this->n_samples)*(this->vector_len)){
            printf("Deleting buffer, did not get as many bytes as expected, got %d, expected %d\n",buf_len, idx);
            delete[] buffer;
        }
    }else{
        float* buffer;
        int buf_len;
        buf_len = this->n_samples;
        this->labels = new float[buf_len];
        buffer = this->labels;
        //Now just read the data
        int data_byte;
        uint8_t tmp;
        idx = 0;
        while(1){
            data_byte = fgetc(fp);
            if(feof(fp)){
                break;
            }   
            else{
                tmp =  (uint8_t)data_byte;
                buffer[idx] = (float)tmp;
                idx++;
            }
        }
        if(idx != (this->n_samples)){
            printf("Deleting buffer, did not get as many bytes as expected, got %d, expected %d\n",buf_len, idx);
            delete[] buffer;
        }
    }
    
    fclose(fp);
    return 0;
}

void DataSetHandler::deinit(){
    delete[] this->labels;
    delete[] this->values;
    return;
}

void DataSetHandler::parse_data(){
    int ret;
    this->is_label_file = false;
    this->magic_number = 2051;
    this->sample_len = 28;
    ret = this->read_file(this->filename);
    if(ret == -1){
        return;
    }
    
    printf("Read through the images, now going to read through the labels\n");
    
    this->is_label_file = true;
    this->magic_number = 2049;
    this->sample_len = 1;
    ret = this->read_file(this->labels_filename);
    if(ret == -1){
        return;
    }
    printf("Parsed files\n");
    return;
}

void DataSetHandler::get_random_sample(float** random_sample_ptr, float** random_label_ptr){
    int random_idx;
    random_idx = rand() % (this->n_samples);
    *random_sample_ptr = this->values[random_idx];
    *random_label_ptr = &(this->labels[random_idx]);
    return;
}