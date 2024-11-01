#include "../include/LayerNorm.hpp"

LayerNorm::LayerNorm(int model_dim) : model_dim(model_dim) {
    gamma.resize(model_dim, 0.001f);

    beta.resize(model_dim, 0.0f);
}

std::vector<double> LayerNorm::normalize(const std::vector<double>& input) const {
    double mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();

    double variance = 0.0f;

    for (double val : input) 
    {
        variance += (val - mean) * (val - mean);
    }

    variance = variance / input.size();

    const double epsilon = 1e-5;

    std::vector<double> normalized(input.size());

    for (size_t i = 0; i < input.size(); ++i)
    {
        normalized[i] = (input[i] - mean) / std::sqrt(variance + epsilon);

        normalized[i] = gamma[0] * normalized[i] + beta[0];
    }
}