#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <utility>
#include <random>
#include <set>
#include <string>
#include <cmath>
#include <iostream>

#include "edlib.h"
#include "generate_features.h"

// For reverse strand add +6
std::unordered_map<Bases, uint8_t, EnumClassHash> ENCODED_BASES = {
    {Bases::A, 0},
    {Bases::C, 1},
    {Bases::G, 2},
    {Bases::T, 3},
    {Bases::GAP, 4},
    {Bases::UNKNOWN, 5}
};

// Changes char to int, where 0,1,2,3,4,5 represent ACGT*N
static constexpr uint8_t CHAR_TO_FORWARD_INT_MAP[] = {
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 4, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 0, 5, 1, 5, 5,
    5, 2, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 3, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5
};

// changes int values (forward strand, 0-5) to char
static constexpr char FORWARD_INT_TO_CHAR_MAP[] = {'A', 'C', 'G', 'T', '*', 'N'};

FeatureGenerator::FeatureGenerator(const char* filename, const char* ref, const char* region, PyObject* dict): draft(ref) {
    
    bam = readBAM(filename);
    pileup_iter = bam->pileup(region);
    if (dict == Py_None) { 
        has_labels = false;
    } else {
        convert_py_labels_dict(dict);
        has_labels = true;
    }
    
}

void FeatureGenerator::convert_py_labels_dict(PyObject *dict) {

    Py_ssize_t pos = 0;
    PyObject *key = NULL;
    PyObject *value = NULL;    

    if (! PyDict_Check(dict)) {
        PyErr_Format(PyExc_TypeError, 
                "Argument \"dict\" to %s must be dict not \"%s\"", 
                __FUNCTION__, Py_TYPE(dict)->tp_name);	       
    }
    while (PyDict_Next(dict, &pos, &key, &value)) {
        if (! PyTuple_Check(key)) {
            PyErr_SetString(PyExc_TypeError, "A key of dict is not a tuple!");
            labels.clear();
        } 
        if (PyTuple_Size(key) != static_cast<Py_ssize_t>(2)) {
            PyErr_SetString(PyExc_TypeError, "A tuple of dict is not a pair!");
            labels.clear();
        }
        PyObject *pair_item0 = PyTuple_GetItem(key, 0);
        PyObject *pair_item1 = PyTuple_GetItem(key, 1);
        if ((!PyLong_Check(pair_item0)) || (!PyLong_Check(pair_item1))) {
            PyErr_SetString(PyExc_TypeError, "A tuple of dict does contain two longs!");
            labels.clear();
        }
        if (! PyLong_Check(value)) {
            PyErr_SetString(PyExc_TypeError, "A value of dict is not of long type!");
            labels.clear();
        }
        pos_index_t pair_item0_c = PyLong_AsUnsignedLong(pair_item0);
        pos_index_t pair_item1_c = PyLong_AsUnsignedLong(pair_item1);
        uint8_t value_c = PyLong_AsUnsignedLong(value);
        if (PyErr_Occurred()) {
            labels.clear();
        }
        labels.emplace(std::make_pair(pair_item0_c, pair_item1_c), value_c);
    }
}

void FeatureGenerator::add_bq_sample(std::pair<pos_index_t, pos_index_t>& index, float bq) {
    auto& info = stats_info[index];
    info.avg_bq = info.avg_bq + (bq - info.avg_bq)/ ++info.n_bq;
}

void FeatureGenerator::add_mq_sample(std::pair<pos_index_t, pos_index_t>& index, uint8_t mq) {
    auto& info = stats_info[index];
    info.avg_mq = info.avg_mq + (float) (mq - info.avg_mq)/ ++info.n_mq;
}

void FeatureGenerator::increment_base_count(std::pair<pos_index_t, pos_index_t>& index, Bases b) {
    auto& s = stats_info[index];
    s.n_total++;
    Bases draft_base = Bases::GAP;
    // index here is a pair of indices: index.first = draft_base index, index.second = insertion index
    if (index.second == 0) {
        draft_base = char_to_base(draft[index.first]);
    }
    bool diff = draft_base != b;
    switch(b) {
        case Bases::A:
            s.n_A++;
            if (diff && s.n_A > s.largest_diff) {
               s.largest_diff = s.n_A;
            }
            break;
        case Bases::C:
            s.n_C++;
            if (diff && s.n_C > s.largest_diff) {
                s.largest_diff = s.n_C;
            }
            break;
        case Bases::G:
            s.n_G++;
            if (diff && s.n_G > s.largest_diff) {
                s.largest_diff = s.n_G;
            }
            break;
        case Bases::T:
            s.n_T++;
            if (diff && s.n_T > s.largest_diff) {
                s.largest_diff = s.n_T;
            }
            break;
        case Bases::GAP:
            s.n_GAP++;
            if (diff && s.n_GAP > s.largest_diff) {
                s.largest_diff = s.n_GAP;
            }
            break;
        case Bases::UNKNOWN:
            std::cout << "Unknown base!" << std::endl;
    }
}


Bases FeatureGenerator::char_to_base(char c) {

    return static_cast<Bases>(static_cast<int>(CHAR_TO_FORWARD_INT_MAP[static_cast<uint8_t>(c)]));
}

char FeatureGenerator::base_to_char(Bases b) {

    return FORWARD_INT_TO_CHAR_MAP[static_cast<int>(b)];
}

char FeatureGenerator::forward_int_to_char(uint8_t i) {

    return FORWARD_INT_TO_CHAR_MAP[i];
}

uint8_t FeatureGenerator::char_to_forward_int(char c) {

    return CHAR_TO_FORWARD_INT_MAP[static_cast<uint8_t>(c)];
}


// this function will be called for each index
void FeatureGenerator::pos_queue_push(std::pair<pos_index_t, pos_index_t>& index) {
    //std::cout << "counter in " << counter << std::endl;
    bool is_uncertain = false;
    auto& s = stats_info[index];
    //char draft_base = draft[index.first];
    uint16_t num_total = s.n_total;
    //uint16_t num_same; //same as draft
    if (index.second != 0) { // if this is an inserted base
        //num_same = s.n_GAP;
        //uint16_t num_not_gap = num_total - s.n_GAP;
        //std::cout << "index " << index.first << ", " << index.second << " " << (float) s.largest_diff/num_total << std::endl;
        if ((float) s.largest_diff/ num_total < NON_GAP_THRESHOLD) { // NON_GAP_THRESHOLD = 0.01
            // result feature's insertion index might not be continuous
            // e.g. if (45,3) does not have enough supporting reads but (45,4) does then (45,3) is skipped
            //num_filter++;
            align_info.erase(index);
            return;
        }
    } 
    if ((float) s.largest_diff/num_total >= UNCERTAIN_POSITION_THRESHOLD) { // UNCERTAIN_POSITION_THRESHOLD = 0.15
        is_uncertain = true;
    } 
   // std::cout << index.first << ", " << index.second << " : " << (float) s.largest_diff/num_total << std::endl;
    if (is_uncertain) {
        pos_queue.push_back(index);
        distances.push(counter); // records all numbers of certain positions between uncertain positions
        counter = 0;

    } else {
        pos_queue.push_back(index);
        counter++; // counts the number of certain positions between any 2 uncertain positions
    }
    //std::cout << "counter out " << counter << std::endl;
}

// should have at least num elements in the deque container
void FeatureGenerator::pos_queue_pop(uint16_t num) {
    for (auto it = pos_queue.begin(), end = pos_queue.begin() + num; it != end; ++it) {
        align_info.erase(*it);
    }
    
    pos_queue.erase(pos_queue.begin(), pos_queue.begin() + num);
    while (distances.size() > 0 && num >=(distances.front()+1)) {
        num -= distances.front() + 1; 
        distances.pop();
    }
    if (distances.empty()) { 
        counter -= num; 
    } else {
        // since distances is not empty, the second condition of the while loop must be false (i.e. num <(distances.front()+1))
        distances.front() -= num;
    }

}

// base_index = rpos, segments = ins_segments (includes one truth segment), star_pos_index = best segment index, no_ins_reads = no_ins_reads
void FeatureGenerator::align_center_star(pos_index_t base_index, std::vector<segment>& segments, int star_pos_index, 
        std::vector<segment>& no_ins_reads) {

    std::vector<segment*> non_label_seqs;
    non_label_seqs.reserve(segments.size());

    segment star = segments[star_pos_index]; // get the best insertion segment

    std::unordered_map<uint32_t, PosInfo> star_positions[star.sequence.size()];
    
    std::vector<std::unordered_map<uint32_t, PosInfo>> ins_positions[star.sequence.size()+1];
    
    // stores labels of the star sequence. By default all gaps
    uint8_t star_positions_labels[star.sequence.size()];  
    for (unsigned int i = 0; i < star.sequence.size(); i ++) {
        star_positions_labels[i] = ENCODED_BASES[Bases::GAP];
    }
    
    // stores the labels aligned to the positions where the star has gaps after aligning
    std::vector<uint8_t> ins_positions_labels[star.sequence.size()+1];    
    
    int total_ins_pos = 0;
    for (auto& s : segments) {
        if (s.index != LABEL_SEQ_ID) non_label_seqs.push_back(&s); // if it is not ground truth
        if (s.index != star.index) { // if it is not the best insertion segment (might be ground truth)
            // calculate the edit distance between this segment and the best insertion segment (align this segment to the best insertion segment)
            EdlibAlignResult result = edlibAlign(s.sequence.c_str(), s.sequence.size(), star.sequence.c_str(),
                    star.sequence.size(), edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_PATH, NULL, 0));

            int ref_pos = -1; // pointing to before next to read ref base
            int query_pos = -1; // pointing to before next to read query base
            unsigned int ins_index = 0; // index of next insertion, 0-based
            char char_at_pos;
            Bases base_at_pos;
            for (int i = 0; i < result.alignmentLength; i++) {
                switch (result.alignment[i]) {
                    case 0: // match
                        ins_index = 0;	      
                        char_at_pos = s.sequence[++query_pos]; // s is the segment being aligned to star
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++;
                        if (s.index == LABEL_SEQ_ID) { // if this segment is the ground truth, add it to the label
                            star_positions_labels[ref_pos] = char_to_forward_int(char_at_pos);
                        } else { // else it is just an insertion segment (not star)
                            star_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos)); 
                            // insertion segment index = read_id, base
                        }
                        break;
                    case 1: // insertion [has base on insertion segment, del on star]
                        char_at_pos = s.sequence[++query_pos]; // get the segment's base
                        base_at_pos = char_to_base(char_at_pos);
                        if (ins_positions[ref_pos+1].size() < ins_index + 1) { // if not enough maps to record bases in that position
                            ins_positions[ref_pos+1].push_back(std::unordered_map<uint32_t, PosInfo>{});
                            ins_positions_labels[ref_pos+1].push_back(ENCODED_BASES[Bases::GAP]);
                            total_ins_pos++;
                        }
                        if (s.index == LABEL_SEQ_ID) { // if this segment is the ground truth
                            ins_positions_labels[ref_pos+1][ins_index] = char_to_forward_int(char_at_pos);
                        } else {
                            ins_positions[ref_pos+1][ins_index].emplace(s.index, PosInfo(base_at_pos));
                        }
                        ins_index++;
                        break;
                    case 2: // deletion has base on star, del on insertion segment]
                        ins_index = 0;
                        ref_pos++;
                        break;
                    case 3: // mismatch
                        ins_index = 0; 
                        char_at_pos = s.sequence[++query_pos];
                        base_at_pos = char_to_base(char_at_pos);
                        ref_pos++;
                        if (s.index == LABEL_SEQ_ID) { // if this segment is the ground truth
                            star_positions_labels[ref_pos] = char_to_forward_int(char_at_pos);
                        } else {
                            star_positions[ref_pos].emplace(s.index, PosInfo(base_at_pos));
                        }
                        break;
                    default:
                        std::cout << "Uknown alignment result!\n";
                }
            }
            edlibFreeAlignResult(result); // cleaning
        } else { // if it is star, also might be the ground truth
            // record bases on the star
            for (unsigned int i = 0; i < s.sequence.size(); i++) {
                const char char_at_pos = s.sequence[i];
                Bases base_at_pos = char_to_base(char_at_pos);               
                if (s.index == LABEL_SEQ_ID) {
                    star_positions_labels[i] = char_to_forward_int(char_at_pos);
                } else {
                    star_positions[i].emplace(s.index, PosInfo(base_at_pos));
                }
            }
        }
    }
  
    uint16_t pos_counts[non_label_seqs.size()] = {0}; 
    
    pos_index_t count = 1;
    // correspond to positions before the first position of star before aligning
    for (unsigned int i = 0; i < ins_positions[0].size(); i++) {
        auto& map = ins_positions[0][i];
        auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);
        
        count++;
        
        for (uint16_t k = 0; k < non_label_seqs.size(); k++) {
            auto& s = non_label_seqs[k];
            if (map.find(s->index) == map.end()) {
                map.emplace(s->index, PosInfo(Bases::GAP));                
                add_bq_sample(index, ( (float) s->bqs[pos_counts[k]] + s->bqs[pos_counts[k] + 1]) /2 );
                add_mq_sample(index, s->mq);

            } else {
               
                add_bq_sample(index, s->bqs[++pos_counts[k]]);
                add_mq_sample(index, s->mq);
            }
        }
        for (auto& s: no_ins_reads) { 
            map.emplace(s.index, PosInfo(Bases::GAP));
            add_bq_sample(index, ((float) s.bqs[0] + s.bqs[1]) /2);
            add_mq_sample(index, s.mq);

             
        }
        for (auto& pair: map) {
            auto b = pair.second.base;
            increment_base_count(index, b);  

        }

        align_info[index] = map;
        if (has_labels) {
            labels_info[index] = ins_positions_labels[0][i]; 
        }
        pos_queue_push(index);
    }

    // correspond to positions on star before aligning, and insertions after them(the inner loop)
    for (unsigned int i = 0; i < star.sequence.size(); i++) {
        auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);

        count++;
        
        for (uint16_t k = 0 ; k < non_label_seqs.size(); k++) {
            auto& s = non_label_seqs[k];
            if (star_positions[i].find(s->index) == star_positions[i].end()) {
                star_positions[i].emplace(s->index, PosInfo(Bases::GAP));
                add_bq_sample(index, ((float) s->bqs[pos_counts[k]] + s->bqs[pos_counts[k] + 1])/2);
                add_mq_sample(index, s->mq);

            } else {
                add_bq_sample(index, s->bqs[++pos_counts[k]]);
                add_mq_sample(index, s->mq);

            }
        }
        for (auto& s: no_ins_reads) {
            star_positions[i].emplace(s.index, PosInfo(Bases::GAP));
            add_bq_sample(index, ((float) s.bqs[0] + s.bqs[1]) /2);
            add_mq_sample(index, s.mq);
        }
        for (auto& pair: star_positions[i]) {
            auto b = pair.second.base;
            increment_base_count(index, b);  

        }


        align_info[index] = star_positions[i];
        pos_queue_push(index); 
        if (has_labels) {
            labels_info[index] = star_positions_labels[i];
        }

        for (unsigned int j = 0; j < ins_positions[i+1].size(); j++) {
            auto& map = ins_positions[i+1][j];
            auto index = std::pair<pos_index_t, pos_index_t>(base_index, count);
          
            count++;
            
            for (uint16_t k = 0; k< non_label_seqs.size(); k++) {
                auto& s = non_label_seqs[k];
                if (map.find(s->index) == map.end()) {
                    map.emplace(s->index, PosInfo(Bases::GAP));
                   
                    add_bq_sample(index, ((float) s->bqs[pos_counts[k]] + s->bqs[pos_counts[k] + 1])/2  );
                    add_mq_sample(index, s->mq);

                } else {
                   
                    add_bq_sample(index, s->bqs[++pos_counts[k]]);
                    add_mq_sample(index, s->mq);
                }
            }
            for (auto& s: no_ins_reads) {
                map.emplace(s.index, PosInfo(Bases::GAP));	
                add_bq_sample(index, ((float) s.bqs[0] + s.bqs[1]) /2);
                add_mq_sample(index, s.mq);

            }
            for (auto& pair: map) {
                auto b = pair.second.base;
                increment_base_count(index, b);  

            }

            align_info[index] = map;
            if (has_labels) {
                labels_info[index] = ins_positions_labels[i+1][j];
            }
            pos_queue_push(index);
        }
    } 
}

int FeatureGenerator::find_center(std::vector<segment>& segments) { // find center of insertion segments
    int dists[segments.size()]{0}; 
    for (unsigned int i = 0; i < segments.size(); i++) { 
        for (unsigned int j = i + 1; j < segments.size(); j++) { 
            EdlibAlignResult result = edlibAlign(segments[i].sequence.c_str(), segments[i].sequence.size(), segments[j].sequence.c_str(),
                    segments[j].sequence.size(), edlibNewAlignConfig(-1, EDLIB_MODE_NW, EDLIB_TASK_DISTANCE, NULL, 0));
            dists[i] += result.editDistance;
            dists[j] += result.editDistance;
            
            edlibFreeAlignResult(result); // cleaning
        }
    }
    int best_pos_index = 0;
    for (unsigned int i = 0; i < segments.size(); i++) {
        if (dists[i] < dists[best_pos_index]) { // get the smallest sum of edit distance, the "center"
            best_pos_index = i;
        }
    }
    return best_pos_index;    

}

int FeatureGenerator::find_longest(std::vector<segment>& segments) { // finds the longest segment
    int best_index = 0;
    int highest_len = 0;
    for (int i = 0; i < segments.size(); i++) {
        int len = segments[i].sequence.size();
        if (len > highest_len) {
            best_index = i;
            highest_len = len;
        }
    }
    return best_index;
}


// base_index = rpos
void FeatureGenerator::align_ins_longest_star(pos_index_t base_index, std::vector<segment>& ins_segments,
        std::vector<segment>& no_ins_reads) {
    int longest_index = find_longest(ins_segments); // the index of the longest read in ins_segments
    align_center_star(base_index, ins_segments, longest_index, no_ins_reads);

}

void FeatureGenerator::align_ins_center_star(pos_index_t base_index, std::vector<segment>& ins_segments,
        std::vector<segment>& no_ins_reads) {
    int center_index = find_center(ins_segments);
    align_center_star(base_index, ins_segments, center_index, no_ins_reads);

}

std::unique_ptr<Data> FeatureGenerator::generate_features() {   
    npy_intp dims[2]; // dimensions of X1 (2d)
    npy_intp dims2[2]; // dimensions of X2 (2d)
    npy_intp labels_dim[1]; // dimensions of labels (1d)
    srand(49);
    labels_dim[0] = dimensions[1]; // labels_dim[0] = S
    for (int i = 0; i < 2; i++) {
        dims[i] = dimensions[i]; // dimensions[0] = R, dimensions[1] = S
        dims2[i] = dimensions2[i]; // dimensions2[0] = 5, dimensions2[1] = S
    }
 
    auto data = std::unique_ptr<Data>(new Data()); // positions, X, Y, X2
    
    // for each position in draft
    while (pileup_iter->has_next()) {
        auto column = pileup_iter->next(); // get the column
        long rpos = column->position;
        if (rpos < pileup_iter->start()) continue;
        if (rpos >= pileup_iter->end()) break;
        std::vector<segment> ins_segments;
        std::vector<segment> no_ins_reads;
        
        // if labels are provided by truth2draft: 
        std::string s;
        if (has_labels) {
            std::pair<pos_index_t, pos_index_t> index {rpos, 0};
            labels_info[index] = labels[index];
            pos_index_t ins_count = 1; // check if there is any insertion (bases present in truth but not in draft)
            index = std::make_pair(rpos, ins_count); // first insertion index: (rpos, 1)
            auto found = labels.find(index);
            while (found != labels.end()) { 
                // while the consecutive items can be found -> there are insertions in truth thats not present in draft
                char c = forward_int_to_char(labels[index]);
                // put it into the insertion segment s
                s.push_back(c); 
                ins_count++;
                index = std::make_pair(rpos, ins_count);
                found = labels.find(index);
            }
        }


        if (s.size() > 0) { // the draft is wrong, truth has a segment not present in draft
            ins_segments.emplace_back(std::move(s), LABEL_SEQ_ID); // LABEL_SEQ_ID = -1
        } // not gonna happen in inference mode, only training

        // now start from (rpos,0) again
        std::pair<pos_index_t, pos_index_t> base_index(rpos, 0);

        // time to check each read at this position
        while(column->has_next()) { // a column is made up of bases from many reads at one position
            auto r = column->next(); // column -> next() goes down the column of bases, r is one of the reads in this column 
            if (r->is_refskip()) continue;
            if (align_bounds.find(r->query_id()) == align_bounds.end()) {
                align_bounds.emplace(r->query_id(), std::make_pair(r->ref_start(), r->ref_end()));
            }
            strand.emplace(r->query_id(), !r->rev()); // strand information of this read
            
            if (r->is_del()) {
                // DELETION
                align_info[base_index].emplace(r->query_id(), PosInfo(Bases::GAP));
                // maps an index (key A) to another map B (value A), which maps the query id (key B) to the base (value B).
                increment_base_count(base_index, Bases::GAP);
                add_mq_sample(base_index, r->mqual());
                add_bq_sample(base_index, ((float) r->qqual(-1) + r->qqual(0)) /2);
                // base quality of deletion: (quality of the previous base + quality of the next base)/2
            } else {
                // POSITION
                auto qbase = r->qbase(0); // get query base
                align_info[base_index].emplace(r->query_id(), PosInfo(qbase));
                increment_base_count(base_index, qbase);
                add_mq_sample(base_index, r->mqual());
                add_bq_sample(base_index,  r->qqual(0));
                // INSERTION
                if (r-> indel() > 0) {
                    std::string s;
                    s.reserve(r->indel());
                    std::vector<uint8_t> segment_bqs;
                    segment_bqs.push_back(r->qqual(0));
                    for (int i = 1, n = r->indel(); i <= n; ++i) { // store the insertion segment and inserted base qualities
                        qbase = r->qbase(i);
                        s.push_back(base_to_char(qbase));
                        segment_bqs.push_back(r->qqual(i));
                    }
                    segment_bqs.push_back(r->qqual(r->indel() + 1)); // store the quality of the next base after the insertion segment
                    // take the next base quality because if there's a gap after the insertion, the gap's quality needs both the left and right bases to be calculated
                    ins_segments.emplace_back(std::move(s), r->query_id(), r->mqual(), std::move(segment_bqs));
                } else {
                    no_ins_reads.emplace_back("", r->query_id(), r->mqual(),std::initializer_list<uint8_t>{r->qqual(0), r->qqual(1)});
                }
           }
        } // done with all the reads at one position

        pos_queue_push(base_index); // this base_index is (rpos,0)

        if (ins_segments.size() > 0) { // at this position, there exists at least one read with insertion
            //align_ins_longest_star(rpos, ins_segments, no_ins_reads); // use align_ins_center_star
            align_ins_center_star(rpos, ins_segments, no_ins_reads);
        }

        //BUILD FEATURE MATRIX
        while (pos_queue.size() >= dimensions[1]) {
            if (distances.empty())  {
                // if all are certain, remove most of positions in pos_queue 
                // only keep 75% of window size
                pos_queue_pop(pos_queue.size() - dimensions[1]/4 * 3);
                continue;
                
            } else if (distances.front() >= dimensions[1]) { // the first certain segment is larger than window size
                uint16_t a = distances.front() - dimensions[1]/4 * 3; // certain segment length - 75% window size
                uint16_t b = pos_queue.size();
                pos_queue_pop(std::min(a, b));
                continue;
            } 

            std::set<uint32_t> valid_aligns;
            const auto it = pos_queue.begin();

            for (auto s = 0; s < dimensions[1]; s++) {    
                auto curr = it + s;
                
                for (auto& align : align_info[*curr]) {
                    if (align.second.base != Bases::UNKNOWN) {
                        valid_aligns.emplace(align.first);
                    }
                }
            }
            std::vector<uint32_t> valid(valid_aligns.begin(), valid_aligns.end());
           
            int valid_size = valid.size();
            // when number of valid reads < threshold, then add draft base as reads for x times (comment this block off later)
            //int num_draft = REF_ROWS;
            //if(valid_size < 3) num_draft = 15;

            auto X = PyArray_SimpleNew(2, dims, NPY_UINT8);
            auto X2 = PyArray_SimpleNew(2, dims2, NPY_UINT16);
            auto Y = PyArray_SimpleNew(1, labels_dim, NPY_UINT8);
            
            uint8_t* value_ptr;
            uint16_t *value_ptr_16;

            // First handle assembly (REF_ROWS)
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s; uint8_t value;

                if (curr->second != 0) value = ENCODED_BASES[Bases::GAP];
                else value = ENCODED_BASES[get_base(draft[curr->first])];

                // change this part back to 'for (int r = 0; r < REF_ROWS; r++)' later
                for (int r = 0; r < REF_ROWS; r++) {
                    value_ptr = (uint8_t*) PyArray_GETPTR2(X, r, s);
                    *value_ptr = value; // Forward strand - no +6
                }
            }
            //fill up X2 
            for (auto s = 0; s < dimensions[1]; s++) {
                auto curr = it + s;
                auto pos_stats = stats_info[*curr];

                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 0, s);
                *value_ptr_16 = pos_stats.n_GAP;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 1, s);
                *value_ptr_16 = pos_stats.n_A;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 2, s);
                *value_ptr_16 = pos_stats.n_C;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 3, s);
                *value_ptr_16 = pos_stats.n_G;
                value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 4, s);
                *value_ptr_16 = pos_stats.n_T;
                // value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 5, s);
                //*value_ptr_16 = static_cast<uint16_t>(pos_stats.avg_mq);
                //value_ptr_16 = (uint16_t*) PyArray_GETPTR2(X2, 6, s);
                //*value_ptr_16 = static_cast<uint16_t>(pos_stats.avg_bq);
                
            }

            for (int r = REF_ROWS; r < dimensions[0]; r++) {

                uint8_t base;
                auto random_n = rand();
                auto random_num = random_n  % valid_size;

                uint32_t query_id = valid[random_num];

                auto& fwd = strand[query_id];


                auto it = pos_queue.begin();
                for (auto s = 0; s < dimensions[1]; s++) {
                    auto curr = it + s;

                    auto pos_itr = align_info[*curr].find(query_id);
                    auto& bounds = align_bounds[query_id];
                    if (pos_itr == align_info[*curr].end()) {
                        if (curr->first < bounds.first || curr->first > bounds.second) {
                            base = ENCODED_BASES[Bases::UNKNOWN];
                        } else {
                            base = ENCODED_BASES[Bases::GAP];
                        }
                    } else {
                        base = ENCODED_BASES[pos_itr->second.base];
                    }

                    value_ptr = (uint8_t*) PyArray_GETPTR2(X, r, s);
                    *value_ptr = fwd ? base : (base + 6);

                }

            }

            if (has_labels) {
                for (auto s = 0; s < dimensions[1]; s++) {
                    auto curr = it + s;
                    uint8_t value = labels_info[*curr];
                    value_ptr = (uint8_t*) PyArray_GETPTR1(Y, s);
                    *value_ptr = value;
                }
            }
            data->X.push_back(X);
            data->X2.push_back(X2);
            data->Y.push_back(Y);
            data->positions.emplace_back(pos_queue.begin(), pos_queue.begin() + dimensions[1]);
            pos_queue_pop(WINDOW);
        }
    } 
    return data;
}