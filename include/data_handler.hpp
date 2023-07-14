#ifndef __DATA_HANDLER_H
#define __DATA_HANDLER_H

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>


class data_handler{
    std::vector<data *> *data_array; // all of the data (pre-split)
    std::vector<data *> *training_data;
    std::vector<data *> *test_data;
    std::vector<data *> *validation_data;

    int num_classes;
    int feature_vector_size;
    std::map<uint8_t, int> class_map;

    const double TRAIN_SET_PERCENT = 0.75;
    const double TEST_SET_PERCENT = 0.20;
    const double VALIDATION_PRESENT = 0.05;

    public:
    data_handler();
    ~data_handler();

    void read_features_vector(std::string path);
    void read_feature_labels(std::string path);
    void split_data(const double train_size_prc, const double test_size_prc, const double valid_size_prc);
    void count_classes();

    uint32_t convert_to_little_endian(const unsigned char* bytes);

    std::vector<data *> * get_training_data();
    std::vector<data *> * get_test_data();
    std::vector<data *> * get_validation_data();

    private:
    void set_splitted_data(std::vector<data *> *& d, int new_size, std::unordered_set<int> & used_indexes);
};

#endif