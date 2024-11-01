#include "../include/PositionalEncoding.hpp"

PositionalEncoding::PositionalEncoding(int max_seq_len, int model_dim) : max_seq_len(max_seq_len), model_dim() {
    encoding_matrix.resize(max_seq_len, std::vector<double>(model_dim));

    for (int pos = 0; pos < max_seq_len; ++pos)
    {
        for (int i = 0; i < model_dim; i += 2)
        {   
            // Normalized position
            double position = pos / std::pow(10000.0, 2.0 * i / model_dim);

            encoding_matrix[pos][i] = std::sin(position);

            // checks if exists next dimension, then add cosin to odd value
            if (i + 1 < model_dim)
            {
                encoding_matrix[pos][i + 1] = std::cos(position);
            } 
        }
    }
}

const std::vector<double>& PositionalEncoding::getEncoding(int pos) {
    if (pos < 0 || pos >= max_seq_len)
    {
        throw std::out_of_range("PositionalEncoding::getEncoding: Position out of range");
    }

    return encoding_matrix[pos];
}

std::vector<std::vector<double>> *PositionalEncodding::getEncodings(std::vector<double>> &embeddings){
    // Determines the length of the sequence as the smallest value between the size of the embeddings and the nesting matrix
    int seq_len = std::min(embeddings.size(), encoding_matrix.size());

    // Allocate memory to store final encodings
    std::vector<std::vector<double>> *encodings = new std::vector<double>>(seq_len, std::vector<double>(this->model_dim));

    for (int i = 0; i < seq_len; ++i)
    {
        for (int j = 0; j < model_dim; ++j)
        {
            (*encodings)[i][j] = embeddings[i][j] + encoding_matrix[i][j];
        }
    }
    return encodings;
}