#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pace26::core {

class RandomError : public std::runtime_error {
public:
    explicit RandomError(const std::string& message)
        : std::runtime_error("Random error: " + message) {}
};

class Random {
public:
    explicit Random(std::uint64_t seed = default_seed())
        : state_(seed) {
        if (state_ == 0) {
            state_ = 0x9e3779b97f4a7c15ULL;
        }
    }

    static std::uint64_t default_seed() {
        const auto now = std::chrono::high_resolution_clock::now()
                             .time_since_epoch()
                             .count();

        std::uint64_t seed = static_cast<std::uint64_t>(now);
        seed ^= 0x9e3779b97f4a7c15ULL;
        return seed;
    }

    std::uint64_t seed() const noexcept {
        return state_;
    }

    std::uint64_t next_u64() {
        std::uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    std::uint32_t next_u32() {
        return static_cast<std::uint32_t>(next_u64() >> 32);
    }

    std::uint64_t uniform_u64(std::uint64_t lo, std::uint64_t hi) {
        if (lo > hi) {
            throw RandomError("invalid uniform_u64 range");
        }

        const std::uint64_t range = hi - lo + 1;

        if (range == 0) {
            return next_u64();
        }

        return lo + (next_u64() % range);
    }

    int uniform_int(int lo, int hi) {
        if (lo > hi) {
            throw RandomError("invalid uniform_int range");
        }

        return static_cast<int>(
            uniform_u64(
                static_cast<std::uint64_t>(lo),
                static_cast<std::uint64_t>(hi)
            )
        );
    }

    std::size_t uniform_index(std::size_t size) {
        if (size == 0) {
            throw RandomError("cannot sample index from empty range");
        }

        return static_cast<std::size_t>(
            uniform_u64(0, static_cast<std::uint64_t>(size - 1))
        );
    }

    double uniform_double() {
        constexpr double inv = 1.0 / static_cast<double>(UINT64_MAX);
        return static_cast<double>(next_u64()) * inv;
    }

    bool coin(double probability = 0.5) {
        if (probability < 0.0 || probability > 1.0) {
            throw RandomError("coin probability must be in [0,1]");
        }

        return uniform_double() < probability;
    }

    template <class T>
    T& choice(std::vector<T>& values) {
        if (values.empty()) {
            throw RandomError("cannot choose from empty vector");
        }

        return values[uniform_index(values.size())];
    }

    template <class T>
    const T& choice(const std::vector<T>& values) {
        if (values.empty()) {
            throw RandomError("cannot choose from empty vector");
        }

        return values[uniform_index(values.size())];
    }

    template <class T>
    void shuffle(std::vector<T>& values) {
        if (values.size() <= 1) {
            return;
        }

        for (std::size_t i = values.size() - 1; i > 0; --i) {
            const std::size_t j = uniform_index(i + 1);
            std::swap(values[i], values[j]);
        }
    }

private:
    std::uint64_t state_;
};

}  // namespace pace26::core