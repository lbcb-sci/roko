#ifndef GENERATE_H
#define GENERATE_H

#include <Python.h>
extern "C" {
    #define NO_IMPORT_ARRAY
    #define PY_ARRAY_UNIQUE_SYMBOL gen_ARRAY_API
    #include "numpy/arrayobject.h"
}

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "models.h"


constexpr int dimensions[] = {200, 90};
constexpr int CENTER = dimensions[1] / 2;
constexpr int WINDOW = dimensions[1] / 3;
constexpr int MAX_INS = 3;
constexpr int REF_ROWS = 0;

struct Data{
    std::vector<std::vector<std::pair<long, long>>> positions;
    std::vector<PyObject*> X;
};

struct PosInfo{
    Bases base;

    PosInfo(Bases b) : base(b) {};
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
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Mainly for demonstration purposes, i.e. works but is overly simple
        // In the real world, use sth. like boost.hash_combine
        return h1 ^ h2;  
    }
};

std::unique_ptr<Data> generate_features(const char*, const char*, const char*);


#endif //GENERATE_H
