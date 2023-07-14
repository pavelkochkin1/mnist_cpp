#ifndef __KNN_H
#define __KNN_H

#include <vector>
#include "data.hpp"


class knn{
    int k;
    std::vector<data *> * neighbours;
    std::vector<data *> * train_data;
    std::vector<data *> * valid_data;
    std::vector<data *> * test_data;

    public:
    knn();
    knn(int);
    ~knn();

    void find_knearest(data *query_point);
    void set_training_data(std::vector<data *> *vec);
    void set_test_data(std::vector<data *> *vec);
    void set_valid_data(std::vector<data *> *vec);
    void set_k(int val);

    int predict();
    double calculate_distance(data* query_point, data* input);
    double validate_performance();
    double test_performance();
};

#endif