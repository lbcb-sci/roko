//
// Created by dominik on 10. 10. 2019..
//
#include "models.h"
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cassert>
#include <utility>


extern "C" {
    int iter_bam(void* data, bam1_t* b) {
        int status;
        PileupData* plp_data = (PileupData*) data;

        //std::cout << "min_mapping_quality: (should be 1 now) " << int(min_mapping_quality) << "\n";

        while (1) {
            if (plp_data->iter) {
                status = sam_itr_next(plp_data->file, plp_data->iter, b);
            } else {
                status = sam_read1(plp_data->file, plp_data->header, b);
            }
            if (status < 0) break;

            if (b->core.flag & filter_flag) continue;
            if (b->core.flag & BAM_FPAIRED && ((b->core.flag & BAM_FPROPER_PAIR) == 0)) continue;
            if (b->core.qual < min_mapping_quality) continue;
            break;
        }
        
        return status;
    }
}

RegionInfo::RegionInfo(std::string n, int s, int e): name(std::move(n)), start(s), end(e) {}

std::unique_ptr<BAMFile> readBAM(const char* filename) {
    std::unique_ptr<htsFile, decltype(&hts_close)> bam(hts_open(filename, "rb"), hts_close);
    if (!bam) {
        throw std::runtime_error("Cannot open BAM file.");
    }

    std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)> idx(sam_index_load(bam.get(), filename),
            hts_idx_destroy);
    if (!idx) {
        throw std::runtime_error("Cannot open BAM index.");
    }

    std::unique_ptr<bam_hdr_t, decltype(&bam_hdr_destroy)> header(sam_hdr_read(bam.get()), bam_hdr_destroy);
    if (!header) {
        throw std::runtime_error("Cannot read BAM header.");
    }

    return std::unique_ptr<BAMFile>(new BAMFile(std::move(bam), std::move(idx), std::move(header)));
}

BAMFile::BAMFile(std::unique_ptr<htsFile, decltype(&hts_close)> bam,
        std::unique_ptr<hts_idx_t, decltype(&hts_idx_destroy)> idx,
        std::unique_ptr<bam_hdr_t, decltype(&bam_hdr_destroy)> header)
        : bam_(std::move(bam)), bam_idx_(std::move(idx)), header_(std::move(header)) {
}

std::unique_ptr<RegionInfo> get_region(const std::string& region) {
    int start, end;
    char* end_name = (char*) hts_parse_reg(region.c_str(), &start, &end);

    long len = end_name - region.c_str();
    std::string contig(region, 0, len);

    return std::unique_ptr<RegionInfo>(new RegionInfo(std::move(contig), start, end));
}

std::unique_ptr<PositionIterator> BAMFile::pileup(const std::string& region) {
    std::unique_ptr<PileupData> data(new PileupData);

    data->file = this->bam_.get(); data->header = this->header_.get();
    data->iter = bam_itr_querys(this->bam_idx_.get(), this->header_.get(), region.c_str());

    // Creating multi-iterator
    auto data_raw = data.get();
    bam_mplp_t mplp = bam_mplp_init(1, iter_bam, (void **) &data_raw);
    std::unique_ptr<__bam_mplp_t, decltype(&bam_mplp_destroy)> mplp_iter(mplp, bam_mplp_destroy);

    // Pointer to data array for one position
    std::shared_ptr<const bam_pileup1_t*> pileup(const_cast<const bam_pileup1_t**>(new bam_pileup1_t*));

    // Region info
    auto region_info = get_region(region);

    return std::unique_ptr<PositionIterator>(new PositionIterator(std::move(data), std::move(mplp_iter),
            std::move(pileup), std::move(region_info)));
}

PositionIterator::PositionIterator(std::unique_ptr<PileupData> pileup_data,
                                   std::unique_ptr<__bam_mplp_t, decltype(&bam_mplp_destroy)> mplp_iter,
                                   std::shared_ptr<const bam_pileup1_t*> pileup, std::unique_ptr<RegionInfo> region)
                                   : pileup_data_(std::move(pileup_data)), mplp_iter_(std::move(mplp_iter)),
                                   pileup_(std::move(pileup)), region_(std::move(region)){

}

bool PositionIterator::has_next() {
    if (processed_) {
        current_next_ = bam_mplp_auto(mplp_iter_.get(), &tid_, &pos_, &count_, pileup_.get());
        processed_ = false;
    }

    return current_next_ > 0;
}

std::unique_ptr<Position> PositionIterator::next() {
    if(!has_next()) {
        throw std::runtime_error("No more positions to iterate.");
    }

    const char* contig_name = pileup_data_->header->target_name[tid_];
    assert (region_->name == contig_name);
    assert(pos_ >= region_->start);
    assert(pos_ < region_->end);

    processed_ = true;
    return std::unique_ptr<Position>(new Position(region_->name, pos_, count_, pileup_));
}

Position::Position(std::string contig,
        int pos, int count, std::shared_ptr<const bam_pileup1_t *> data) :
        position(pos), contig_(std::move(contig)), count_(count), data_(std::move(data)) {}

bool Position::has_next() {
    return current_ < count_;
}

std::unique_ptr<Alignment> Position::next() {
    if (current_ >= count_) {
        throw std::runtime_error("No more reads in position to iterate.");
    }

    const bam_pileup1_t* read = *data_ + current_;
    current_++;

    return std::unique_ptr<Alignment>(new Alignment(read));
}

Alignment::Alignment(const bam_pileup1_t *read) : read_(read) {}
