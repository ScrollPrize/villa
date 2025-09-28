#pragma once

#include <opencv2/core.hpp>

// Provide operator== for cv::Vec2i to be used by default by std::unordered_set
static inline bool operator==(const cv::Vec2i& a, const cv::Vec2i& b) {
    return a[0] == b[0] && a[1] == b[1];
}

namespace std {
    template <>
    struct hash<cv::Vec2i> {
        std::size_t operator()(const cv::Vec2i& p) const {
            auto h1 = std::hash<int>{}(p[0]);
            auto h2 = std::hash<int>{}(p[1]);
            // A common way to combine hashes.
            return h1 ^ (h2 << 1);
        }
    };

    struct vec2i_hash {
        std::size_t operator()(const cv::Vec2i& v) const {
            // A simple hash combination
            std::size_t h1 = std::hash<int>()(v[0]);
            std::size_t h2 = std::hash<int>()(v[1]);
            return h1 ^ (h2 << 1);
        }
    };
}

struct vec3i_hash {
    static inline size_t operator()(cv::Vec3i p)
    {
        size_t hash1 = std::hash<int>{}(p[0]);
        size_t hash2 = std::hash<int>{}(p[1]);
        size_t hash3 = std::hash<int>{}(p[2]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        return hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

struct vec3f_hash {
    static inline size_t operator()(cv::Vec3f p)
    {
        size_t hash1 = std::hash<float>{}(p[0]);
        size_t hash2 = std::hash<float>{}(p[1]);
        size_t hash3 = std::hash<float>{}(p[2]);

        //magvec4i_hashic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        return hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
    }
};

struct vec4i_hash {
    static inline size_t operator()(cv::Vec4i p)
    {
        size_t hash1 = std::hash<int>{}(p[0]);
        size_t hash2 = std::hash<int>{}(p[1]);
        size_t hash3 = std::hash<int>{}(p[2]);
        size_t hash4 = std::hash<int>{}(p[3]);

        //magic numbers from boost. should be good enough
        size_t hash = hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
        hash =  hash  ^ (hash3 + 0x9e3779b9 + (hash << 6) + (hash >> 2));
        hash =  hash  ^ (hash4 + 0x9e3779b9 + (hash << 6) + (hash >> 2));

        return hash;
    }
};

struct string_pair_hash {
    static inline size_t operator()(const std::pair<std::string,std::string>& p)
    {
        size_t hash1 = std::hash<std::string>{}(p.first);
        size_t hash2 = std::hash<std::string>{}(p.second);

        //magic numbers from boost. should be good enough
        return hash1  ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    }
};