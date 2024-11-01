#ifndef EMBEDDING_H

#define EMBEDDING_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include<iomanip>
#include <random>
#include <fstream>
#include <sstream>

class Embedding 
{
    public:
        Embedding(int vocab_size, int embed_dim);

        // Return array from a specific token
        std::vector<double> getEmbedding(int token_id);

        // Get a token sequence and convert to embedding sequence
        std::vector<std::vector<double>> *tokenToEmbeddings(std::vector<int> tokens);

        void saveEmbeddingMatrix(const std::string &filename);

        std::vector<std::vector<double>> *generateRandomEmbeddingMatrix(int embed_dim, int vocab_size);
    
        void printEmbeddingMatrix();

        ~Embedding();
        
    private:
        int vocab_size;

        int embed_dim;

        int max_seq_len;

        std::vector<std::vector<double>> *embedding_matrix;
};

#endif