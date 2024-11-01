#ifndef SELF_ATTENTION_H

#define SELF_ATTENTION_H

#include "HelpFunc.hpp"

#include<vector>
#include<iostream>
#include<cmath>
#include<algorithm>
#include<numeric>
#include<stdexcept>
#include<random>

class SelfAttention
{
    public:
        explicit SelfAttention(int model_dim);

        // Computes self attention
        std::vector<double> forward(const std::vector<double>& input) const;

        // multiplies a matrix by a vector, used in attention calculations
        std::vector<double> multiply(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector) const;

        // Uses cross attention between input and encoder outputs
        std::vector<double> forward(const std::vector<double> &input, const std::vector<std::vector<double>> &encoder_input) const;

        // Computes Attention Scores of Q & K
        std::vector<double> computeAttentionScores(const std::vector<double> &Q, const std::vector<std::vector<double>> &K);
    
    private:
        int model_dim;

        std::vector<std::vector<double>> W_q, W_k, W_v;
};

#endif