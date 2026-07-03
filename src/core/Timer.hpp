#pragma once

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

namespace pace26::core {

class TimerError : public std::runtime_error {
public:
    explicit TimerError(const std::string& message)
        : std::runtime_error("Timer error: " + message) {}
};

class Timer {
public:
    using Clock = std::chrono::steady_clock;

    explicit Timer(double limit_seconds = 300.0)
        : start_(Clock::now()),
          limit_seconds_(limit_seconds) {
        if (limit_seconds_ <= 0.0) {
            throw TimerError("time limit must be positive");
        }
    }

    Timer(double limit_seconds, Clock::time_point start)
        : start_(start),
          limit_seconds_(limit_seconds) {
        if (limit_seconds_ <= 0.0) {
            throw TimerError("time limit must be positive");
        }
    }

    void reset(double new_limit_seconds) {
        if (new_limit_seconds <= 0.0) {
            throw TimerError("time limit must be positive");
        }

        start_ = Clock::now();
        limit_seconds_ = new_limit_seconds;
    }

    double elapsed_seconds() const {
        const auto now = Clock::now();
        const std::chrono::duration<double> elapsed = now - start_;
        return elapsed.count();
    }

    double limit_seconds() const noexcept {
        return limit_seconds_;
    }

    Clock::time_point start_time() const noexcept {
        return start_;
    }

    double remaining_seconds() const {
        return std::max(0.0, limit_seconds_ - elapsed_seconds());
    }

    bool expired() const {
        return elapsed_seconds() >= limit_seconds_;
    }

    bool should_stop(double guard_seconds = 0.0) const {
        return elapsed_seconds() + guard_seconds >= limit_seconds_;
    }

    double fraction_used() const {
        return std::min(1.0, elapsed_seconds() / limit_seconds_);
    }

private:
    Clock::time_point start_;
    double limit_seconds_ = 300.0;
};

class TimeBudget {
public:
    TimeBudget(const Timer& global_timer, double local_seconds)
        : global_timer_(global_timer),
          local_seconds_(std::max(0.0, local_seconds)),
          local_start_(Timer::Clock::now()) {}

    double local_elapsed_seconds() const {
        const auto now = Timer::Clock::now();
        const std::chrono::duration<double> elapsed = now - local_start_;
        return elapsed.count();
    }

    double local_remaining_seconds() const {
        return std::max(0.0, local_seconds_ - local_elapsed_seconds());
    }

    double remaining_seconds() const {
        return std::min(
            global_timer_.remaining_seconds(),
            local_remaining_seconds()
        );
    }

    bool expired() const {
        return global_timer_.expired() || local_elapsed_seconds() >= local_seconds_;
    }

    bool should_stop(double guard_seconds = 0.0) const {
        return remaining_seconds() <= guard_seconds;
    }

private:
    const Timer& global_timer_;
    double local_seconds_ = 0.0;
    Timer::Clock::time_point local_start_;
};

}  // namespace pace26::core
