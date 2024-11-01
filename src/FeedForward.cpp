#include "../include/FeedForward.hpp"

FeedForward::FeedFoward(int model_dim) : model_dim(model_dim) {
    W1.resize(hidden_dim, std::vector<double>(model_dim));
    b1.resize(hidden_dim);

    W2.resize(model_dim, std::vector<double>(hidden_dim));
    b2.resize(hidden_dim);

    initialize_weights(W1, hidden_dim, model_dim);
    initialize_bias(b1, hidden_dim);

    initialize_weights(W2, hidden_dim, model_dim);
    initialize_bias(b2, model_dim);
}

void FeedForward::initialize_weights(std::vector<std::vector<double>> &weights, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            weights[i][j] = dis(gen);
        }
    }
}

void FeedForward::initialize_bias(std::vector<double> &bias, int size) {
    std::fill(bias.begin(), bias.end(), 0.0);
}

std::vector<double> FeedForward::relu(const std::vector<double> &x) const {
    std::vector<double> output(x.size());
    for(size_t i = 0; i < x.size(); ++i)
    {
        output[i] = std::max(0.0, x[i]);
    }
    return output;
}

std::vector<double> FeedForward::forward(const std::vector<double> &input) const {
    std::vector<double> hidden_layer(hidden_dim, 0.0);
    for(int i = 0; i < hidden_dim; ++i) 
    {
        for(int j = 0; j < model_dim; ++j)
        {
            hidden_layer[i] += input[j] * W1[i][j];
        }
        hidden_layer[i] += b1[i]; 
    }

    hidden_layer = relu(hidden_layer);

    std::vector<double> output_layer(model_dim, 0.0);
    for(int i = 0; i < model_dim, ++i)
    {
        for(int j = 0; j < hidden_dim; ++j)
        {
            output_layer[i] += hidden_layer[j] * W2[i][j];
        }
        output_layer[i] += b2[i];
    }
    return output_layer;
}
