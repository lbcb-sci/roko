//
// Created by dominik on 10. 10. 2019..
//

#ifndef MODELS_H
#define MODELS_H


#include <memory>
#include <string>
#include <stdexcept>

extern "C" {
    #include "sam.h"

    typedef struct {
        htsFile* file;
        bam_hdr_t* header;
        hts_itr_t* iter;
    } PileupData;

    int iter_bam(void* data, bam1_t* b);

    constexpr uint8_t min_mapping_quality = 1; //change it back to 1 later
    constexpr uint16_t filter_flag = BAM_FUNMAP | BAM_FDUP | BAM_FQCFAIL | BAM_FSUPPLEMENTARY | BAM_FSECONDARY;
}

	
enum class Bases {A = 0, C = 1, G = 2, T = 3, GAP = 4, UNKNOWN= 5};

Bases get_base(char b);

class BAMFile;
class PositionIterator;
class Position;
class Alignment;

typedef struct RegionInfo {
    const std::string name;
    const int start;
    const int end;

    RegionInfo(std::string, int, int);
} RegionInfo;

std::unique_ptr<RegionInfo> get_region(const std::string&);
std::unique_ptr<BAMFile> readBAM(const char*);

class BAMFile {
public:
    friend std::unique_ptr<BAMFile> readBAM(const char*);
    std::unique_ptr<PositionIterator> pileup(const std::string&);

protected:
    std::unique_ptr<htsFile, decltype(&hts_close)> bam_;
    std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)> bam_idx_;
    std::unique_ptr<bam_hdr_t, decltype(&bam_hdr_destroy)> header_;

    BAMFile(std::unique_ptr<htsFile, decltype(&hts_close)>,
            std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)>,
            std::unique_ptr<bam_hdr_t, decltype(&bam_hdr_destroy)>);
};

class PositionIterator {
public:
    friend std::unique_ptr<PositionIterator> BAMFile::pileup(const std::string &);
    std::unique_ptr<Position> next();
    bool has_next();
    int start() {return region_->start;};
    int end() {return region_->end;};

protected:
    std::unique_ptr<PileupData> pileup_data_;
    std::unique_ptr<__bam_mplp_t, decltype(&bam_mplp_destroy)> mplp_iter_;
    std::shared_ptr<const bam_pileup1_t*> pileup_;
    std::unique_ptr<RegionInfo> region_;

    int pos_, tid_, count_, current_next_;
    bool processed_ = true;

    PositionIterator(std::unique_ptr<PileupData>,
                    std::unique_ptr<__bam_mplp_t, decltype(&bam_mplp_destroy)>,
                    std::shared_ptr<const bam_pileup1_t*>, std::unique_ptr<RegionInfo>);
};

class Position {
public:
    int position;

    friend std::unique_ptr<Position> PositionIterator::next();
    std::unique_ptr<Alignment> next();
    bool has_next();
    int count() {return count_;};

protected:
    std::string contig_;
    int count_;
    int current_ = 0;
    std::shared_ptr<const bam_pileup1_t*> data_;

    Position(std::string , int, int, std::shared_ptr<const bam_pileup1_t*>);
};

class Alignment {
public:
    friend std::unique_ptr<Alignment> Position::next();
    int is_refskip() {return read_->is_refskip;};
    int is_del() {return read_->is_del;};
    uint32_t query_id() {return read_->b->id;};
    int indel() { return read_->indel;};
    Bases qbase(int32_t);
    uint8_t qqual(int);
    long ref_start() {return read_->b->core.pos;};
    long ref_end() {return bam_endpos(read_->b);};
    bool rev() {return bam_is_rev(read_->b);};
    uint8_t mqual() {return read_->b->core.qual;};

protected:
    const bam_pileup1_t* read_;

    explicit Alignment(const bam_pileup1_t* read);
};

inline Bases Alignment::qbase(int32_t offset) {
    auto seq = bam_get_seq(read_->b);
    int base = bam_seqi(seq, read_->qpos + offset);

    switch (base) {
        case 1:
            return Bases::A;
        case 2:
            return Bases::C;
        case 4:
            return Bases::G;
        case 8:
            return Bases::T;
        case 15:
            return Bases::UNKNOWN;
        default:
            throw std::runtime_error("No base for given integer");
    }
}

inline uint8_t Alignment::qqual(int offset) {
    auto qual_string = bam_get_qual(read_->b);
    if (qual_string) {
        return qual_string[read_->qpos + offset];
    }

    return 10; // not important
}

inline Bases get_base(char b) {
    switch (b) {
        case 'A':
        case 'a':
            return Bases::A;
        case 'C':
        case 'c':
            return Bases::C;
        case 'G':
        case 'g':
            return Bases::G;
        case 'T':
        case 't':
            return Bases::T;
        case 'N':
        case '-':
            return Bases::UNKNOWN;
        case '*':
            return Bases::GAP;
        default:
            throw std::runtime_error("No base for given integer");
    }
}


#endif //FEATURES_GENERATION_MODELS_H
