#include "knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data_handler.hpp"


knn::knn(){}
knn::knn(int val){
    k = val;
}
knn::~knn(){}

void knn::find_knearest(data *query_point){
    neighbours = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();
    double previous_min = min;
    int idx = 0;

    for (int i = 0; i < k; i++){
        if (i == 0){
            for (int j = 0; j < train_data->size(); i++){
                double distance = calculate_distance(query_point, train_data->at(i));
                train_data->at(j)->set_distance(distance);
                if (distance < min){
                    min = distance;
                    idx = j;
                }
            }
            neighbours->push_back(train_data->at(idx));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        } else {
            for (int j = 0; j < train_data->size(); j++){
                double distance = calculate_distance(query_point, train_data->at(i));
                if (distance > previous_min && distance < min){
                    min = distance;
                    idx = j;
                }
            }
            neighbours->push_back(train_data->at(idx));
            previous_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}
void knn::set_training_data(std::vector<data *> *vec){
    train_data = vec;
}
void knn::set_test_data(std::vector<data *> *vec){
    test_data = vec;
}
void knn::set_valid_data(std::vector<data *> *vec){
    valid_data = vec;
}
void knn::set_k(int val){
    k = val;
}

int knn::predict();

double knn::calculate_distance(data* query_point, data* input){
    double distance = 0.0;
    if(query_point->get_feature_vector_size() != input->get_feature_vector_size()){
        printf("Error Vector Size Mismatch.\n");
        exit(1);
    }
#ifndef EUCLID
    for(unsigned i = 0; i < query_point->get_feature_vector_size(); i++){
        distance = pow(query_point->get_feature_vector()->at(i) - input->get_feature_vector()->at(i), 2);
    }
    distance = sqrt(distance);
#elif defined MANHATTAN
    // PUT MANHATTAN IMPLEMENTATION HERE
#endif
    return distance;
}
double knn::validate_performance();
double knn::test_performance();