#include "data_handler.hpp"


data_handler::data_handler(){
    data_array = new std::vector<data * >;
    test_data = new std::vector<data * >;
    training_data = new std::vector<data * >;
    validation_data = new std::vector<data * >;

}
data_handler::~data_handler(){
    // FREE Dynamically Allocated Stuff
}

void data_handler::read_features_vector(std::string path){
    uint32_t header[4]; // |MAGIC|NUM IMAGES|ROWSIZE|COLSIZE|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f){
        for(int i = 0; i < 4; i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Input File Header.\n");
        int image_size = header[2]*header[3];
        for(int i = 0; i < header[1]; i++){
            data *d = new data();
            uint8_t element[1];
            for(int j = 0; j < image_size; j++){
                if(fread(element, sizeof(element), 1, f)){
                    d->append_to_feature_vector(element[0]);
                }else{
                    printf("Error Reading from File.\n");
                    exit(1);
                }
            }
            data_array->push_back(d);
        }
        printf("Successfully read and stored %lu feature vectors.\n", data_array->size());
    }else{
        printf("Could not find file.\n");
        exit(1);
    }
}
void data_handler::read_feature_labels(std::string path){
    uint32_t header[2]; // |MAGIC|NUM IMAGES|
    unsigned char bytes[4];
    FILE *f = fopen(path.c_str(), "r");
    if (f){
        for(int i = 0; i < 2; i++){
            if(fread(bytes, sizeof(bytes), 1, f)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Done getting Label File Header.\n");
        for(int i = 0; i < header[1]; i++){
            uint8_t element[1];
            if(fread(element, sizeof(element), 1, f)){
                data_array->at(i)->set_label(element[0]);
            }else{
                printf("Error Reading from File.\n");
                exit(1);
            }
        }
        printf("Successfully read and stored Label.\n");
    }else{
        printf("Could not find file.\n");
        exit(1);
    }
}
void data_handler::set_splitted_data(std::vector<data *> *& d, int new_size, std::unordered_set<int> & used_indexes){
    int cnt = 0;
    while(cnt < new_size){
        int rand_index = rand() % data_array->size();
        if(used_indexes.find(rand_index) == used_indexes.end()){
            d->push_back(data_array->at(rand_index));
            used_indexes.insert(rand_index);
            cnt++;
        }
    }
}
void data_handler::split_data(const double train_size_prc=0.75, const double test_size_prc=0.2, const double valid_size_prc=0.05){
    std::unordered_set<int> used_indexes;
    int train_size = data_array->size() * train_size_prc;
    int valid_size = data_array->size() * test_size_prc;
    int test_size = data_array->size() * valid_size_prc;

    // Train Data
    set_splitted_data(training_data, train_size, used_indexes);
    printf("Training Data Size: %lu.\n", training_data->size());

    // Valid Data
    set_splitted_data(validation_data, valid_size, used_indexes);
    printf("Validation Data Size: %lu.\n", validation_data->size());

    // Test Data
    set_splitted_data(test_data, test_size, used_indexes);
    printf("Test Data Size: %lu.\n", test_data->size());
}
void data_handler::count_classes(){
    int cnt = 0;
    for(unsigned i = 0; i < data_array->size(); i++){
        if(class_map.find(data_array->at(i)->get_label()) == class_map.end()){
            class_map[data_array->at(i)->get_label()] = cnt;
            data_array->at(i)->set_enumerated_label(cnt);
            cnt++;
        }
    }
    num_classes = cnt;
    printf("Successfully Extracted %d Unique Classes.\n", num_classes);
}

uint32_t data_handler::convert_to_little_endian(const unsigned char* bytes){
    return (uint32_t) ((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

std::vector<data *> * data_handler::get_training_data(){
    return training_data;
}
std::vector<data *> * data_handler::get_test_data(){
    return test_data;
}
std::vector<data *> * data_handler::get_validation_data(){
    return validation_data;
}


int main(){
    data_handler *dh = new data_handler();
    dh->read_features_vector("./data/train-images.idx3-ubyte");
    dh->read_feature_labels("./data/train-labels.idx1-ubyte");
    dh->split_data(0.75, 0.2, 0.05);
    dh->count_classes();
}