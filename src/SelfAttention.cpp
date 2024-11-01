#include "../include/SelfAttention.hpp"

SelfAttention::SelfAttention(int model_dim) : model_dim(model_dim) {
    std::random_device rd;
    std::mt19937 gen(rd());

    double range = std::sqrt(6.0 / model_dim);

    std::uniform_real_distribution<> distr(-range, range);

    for (auto* weight_matrix : {&W_q, &W_k, &W_v}) {
        weight_matrix->resize(model_dim);
        for (auto& row : *weight_matrix) {
            row.resize(model_dim);
            for (auto& elem : row) {
                elem = distr(gen);
            }
        }
    }
}

std::vector<double> SelfAttention::forward(const std::vector<double> &input) const {
    auto Q = this->multiply(W_q, input);
    auto K = this->multiply(W_k, input);
    auto V = this->multiply(W_v, input);

    if (Utils::containsNaN(Q) || utils::containsNaN(K) || Utils::containsNaN(V)) {
        for (auto x : Q) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        for (auto x: K) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        for(auto x: V) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    double score = std::inner_product(Q.begin(), Q.end(), K.begin(), K.end(), 0.0f) / std::sqrt(model_dim);

    if (std::isnan(score)) {
        std::cout << "Score: " << score << std::endl;
    }

    double attention_weight = std::exp(score) / std::exp(score);

    if (std::isnan(attention_weight)) {
        attention_weight = 1;
    }

    if (std::isnan(attention_weight)) {
        for (auto x : Q) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        for (auto x : K) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
        for (auto x : V) {
            std::cout << x << " ";
        }
        std::cout << std:endl;
        std::cout << "Score: " << score << std::endl;
        std::cout << "Attention weight: " << attention_weight << std::endl;
    }

    std::vector<double> output(model_dim);
    for(int i = 0; i < model_dim; ++i)
    {
        output[i] = attention_weight * V[i];
    }
    return output;
}

std::vector<double> SelfAttention::multiply(const std::vector<std::vector<double>> &matrix, const std::vector<double> &vector) const {
    std::vector<double> result(matrix.size(), 0.0f);

    for(size_t i = 0; i < matrix.size(); ++i)
    {
        for(size_t j = 0; j < vector.size(); ++j)
        {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result;
}

std::vector<double> SelfAttention::forward(const std::vector<double> &input, const std::vector<std::vector<double>> &encoder_input) const {
    // Not implemented Cross Attention
    return input;
}