#ifndef GENERATE_FEATURES_H
#define GENERATE_FEATURES_H

#include <Python.h>
extern "C" {
    #define NO_IMPORT_ARRAY
    #define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
    #include "numpy/arrayobject.h"
}
#include <deque>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "models.h"

typedef uint32_t pos_index_t;

constexpr int dimensions[] = {30, 90}; 
constexpr int dimensions2[] = {5, 90}; // dimensions for second matrix
constexpr int WINDOW = dimensions[1] / 3;
constexpr int REF_ROWS = 1; // ref_rows=1 to include draft in the feature
//constexpr float threshold_prop = 0; // need this proportion of reads to support a base(ACTG) in the position to include it
//constexpr unsigned int align_len_threshold = 0; // need avg ins len >= this at the position to align it 

constexpr float UNCERTAIN_POSITION_THRESHOLD = 0.15;
constexpr float NON_GAP_THRESHOLD = 0.01;
constexpr uint64_t LABEL_SEQ_ID = -1;

struct Data{
    std::vector<std::vector<std::pair<pos_index_t, pos_index_t>>> positions;
    std::vector<PyObject*> X;
    std::vector<PyObject*> Y;
    std::vector<PyObject*> X2;
};

struct PosInfo{
    Bases base;

    PosInfo(Bases b) : base(b) {};
};

struct PosStats {
    uint16_t n_total = 0;
    uint16_t n_GAP = 0;
    uint16_t n_A = 0;
    uint16_t n_C = 0;
    uint16_t n_G = 0;
    uint16_t n_T = 0;

    uint16_t n_bq = 0;
    uint16_t n_mq = 0;
    float avg_bq = 0;
    float avg_mq = 0;
    uint16_t largest_diff = 0; // frequency of the most common alternative nucleotide, by default it is 0 so there is no base disagreeing with draft
    // which means that at any position, if there are bases that do not agree with the draft 
    // the disagreeing base with the highest frequency will be the most common alternative nucleotide
    // 2 conditions must be met: the base disagrees with the draft, it has the highest frequency among the alternative bases

    //PosStats() : avg_mq(0), n_mq(0), avg_pq(0), n_pq(0) {};
    
    
};

struct EnumClassHash
{
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

extern std::unordered_map<Bases, uint8_t, EnumClassHash> ENCODED_BASES;

struct pair_hash {
    template <class T1, class T2> // p is a pointer pointing to pair, which is the key to be hashed
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2; // Bitwise XOR (exclusive or), 1 when both bits are different, 0 when they are the same
    }
};

class FeatureGenerator {

    private:
        // unchanged after construction
        std::unique_ptr<BAMFile> bam;
        std::unique_ptr<PositionIterator> pileup_iter;
        const char* draft;

        bool has_labels;
        uint16_t counter = 0;
        
        // store progress
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, uint8_t, pair_hash> labels;
        std::deque<std::pair<pos_index_t, pos_index_t>> pos_queue; // double ended queue
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, std::unordered_map<uint32_t, PosInfo>, pair_hash> align_info;
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, uint8_t, pair_hash> labels_info;
        std::unordered_map<uint32_t, std::pair<pos_index_t, pos_index_t>> align_bounds;
        std::unordered_map<uint32_t, bool> strand;
        std::unordered_map<std::pair<pos_index_t, pos_index_t>, PosStats, pair_hash> stats_info;
        // how far away is each uncertain position away from the previous one
        // or the start of the queue
        std::queue<uint16_t> distances; 
        // Deque is short for "Double ended queue" -> Queue: you can insert only in one end and remove from the other.
        // Deque: you can insert and remove from both ends.

        struct segment {
            std::string sequence;
            uint64_t index;
            uint8_t mq;
            std::vector<uint8_t> bqs; // bqs of all bases in this segment: the 1st is the bq of the base before ins segment, the last is after
            segment(std::string seq, int id, uint8_t mq, std::vector<uint8_t> bqs) : sequence(seq), index(id), mq(mq), bqs(bqs) {};
            segment(std::string seq, int id) : sequence(seq), index(id) {};
        };

        Bases char_to_base(char c);
        char base_to_char(Bases b);
        char forward_int_to_char(uint8_t i);
        uint8_t char_to_forward_int(char c);
            
        void align_center_star(pos_index_t base_index, std::vector<segment>& segments, int star_index,
            std::vector<segment>& no_ins_reads);

        void align_ins_longest_star(pos_index_t base_index, std::vector<segment>& ins_segments, 
            std::vector<segment>& no_ins_reads);

        void align_ins_center_star(pos_index_t base_index, std::vector<segment>& ins_segments,
            std::vector<segment>& no_ins_reads);

        int find_center(std::vector<segment>& segments);

        int find_longest(std::vector<segment>& segments);

        void convert_py_labels_dict(PyObject *dict);

        void increment_base_count(std::pair<pos_index_t, pos_index_t>& index, Bases b);

        void add_bq_sample(std::pair<pos_index_t, pos_index_t>& index, float bq);

        void add_mq_sample(std::pair<pos_index_t, pos_index_t>& index, uint8_t mq);

        void pos_queue_push(std::pair<pos_index_t, pos_index_t>& index);

        void pos_queue_pop(uint16_t num);

    public:
        FeatureGenerator(const char* filename, const char* ref, const char* region, PyObject* dict);   

        std::unique_ptr<Data> generate_features();
};


#endif //GENERATE_FEATURES_H
