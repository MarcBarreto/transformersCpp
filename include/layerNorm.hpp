#ifndef LAYERNORM_H

#define LAYERNORM_H

#include<vector>
#include<iostream>
#include<cmath>
#include<algorithm>
#include<numeric>

class LayerNorm
{
    public:
        explicit LayerNorm(int model_dim);

        std::vector<double> normalize(const std::vector<double>& input) const;

    private:
        int model_dim;

        std::vector<double> gamma, beta;
};

#endif