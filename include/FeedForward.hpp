#ifndef FEED_FORWARD_H

#define FEED_FORWARD_H

#include<vector>
#include<iostream>
#include<cmath>
#include<random>

class FeedForward
{
    public:
        explicit FeedFoward(int model_dim);

        std::vector<double> forward(const std::vector<double> &input) const;
    
    private:
        int model_dim;

        const int hidden_dim = 4 * model_dim;

        std::vector<std::vector<double>> W1;
        std::vector<double> b1;

        std::vector<std::vector<double>> W2;
        std::vector<double> b2;

        void initialize_weights(std::vector<std::vector<double>> &weights, int rows, int cols);
        void initialize_bias(std::vector<double> &bias, int size);

        std::vector<double> relu(const std::vector<double> &x) const;
};

#endif