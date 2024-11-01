#ifndef SELF_TOKENIZER_H

#define SELF_TOKENIZER_H

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map)
#include <fstream>

class Tokenizer
{
    public:
        Tokenizer();

        // Transform text to tokens ids
        std::vector<int> tokenize(std::string text);

        int getVocabSize();

        void printWordToTokenIdMap() const;

        // Transform tokens ids to text
        std::string detokenize(std::vector<int> tokens);

        void saveTokenMap(std::string filename);

        void loadTokenMap(std::string filename);

        std::vector<std::vector<double>> oneHotEncode(int token_id);

        ~Tokenizer();
    
    private:
        std::unordered_map<std::string, int> word_to_token_id;

        std::unordered_map<int, std::string> token_id_to_word;

        int vocab_size;
};

#endif