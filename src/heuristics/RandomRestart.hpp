#pragma once

#include "core/Forest.hpp"
#include "core/Random.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"

#include "FastCherryApprox.hpp"
#include "LocalImprove.hpp"
#include "TwoApprox.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class RandomRestartError : public std::runtime_error {
public:
    explicit RandomRestartError(const std::string& message)
        : std::runtime_error("RandomRestart error: " + message) {}
};

/*
 * Randomized portfolio controller.
 *
 * This is not a separate approximation algorithm. It is a repeated-restart
 * metaheuristic:
 *
 *   candidate generator
 *      -> perturb
 *      -> LocalImprove
 *      -> keep best
 *
 * It is designed to be safe:
 * - all candidate forests are partitions;
 * - LocalImprove only accepts feasible merges;
 * - on any failure, the incumbent is retained.
 */
class RandomRestart {
private:
    using Tree = pace26::core::Tree;
    using LabelComponent = pace26::core::LabelComponent;
    using LabelForest = pace26::core::LabelForest;
    using Timer = pace26::core::Timer;
    using Random = pace26::core::Random;

public:
    struct Stats {
        std::size_t initial_components = 0;
        std::size_t final_components = 0;
        std::size_t best_updates = 0;

        bool fast_attempted = false;
        bool fast_improved = false;
        bool two_attempted = false;
        bool two_improved = false;
        bool singleton_attempted = false;
        bool singleton_improved = false;

        std::size_t restarts_attempted = 0;
        std::size_t restarts_completed = 0;
        std::size_t restart_improvements = 0;
        std::size_t restart_exceptions = 0;
        std::array<std::size_t, 6> restart_mode_counts{};
        bool stopped = false;
    };

    struct Options {
        int max_restarts = 128;
        int min_restarts = 1;

        double guard_seconds = 0.50;

        bool use_fast_cherry = true;
        bool use_two_approx = true;
        bool use_singleton_start = true;
        bool use_perturbed_incumbent = true;

        /*
         * Do not run expensive generators on huge full instances.
         * They are meant for reduced/clustered instances.
         */
        std::size_t two_approx_max_leaves = 3000;
        std::size_t singleton_start_max_leaves = 1500;

        /*
         * Perturbation:
         * randomly detach a few labels from existing components as singletons,
         * then LocalImprove tries to merge them back differently.
         */
        double perturb_component_probability = 0.30;
        int max_components_to_perturb = 128;
        int max_detached_labels_per_component = 3;

        /*
         * If non-zero, do not perturb components larger than this.
         * Usually leave as zero.
         */
        std::size_t max_component_size_to_perturb = 0;

        FastCherryApprox::Options fast_options;
        TwoApprox::Options two_options;
        LocalImprove::Options improve_options;
        Stats* stats = nullptr;
    };

    RandomRestart() : RandomRestart(Options{}) {}

    explicit RandomRestart(Options options)
        : options_(options) {
        if (options_.max_restarts < 0) {
            throw RandomRestartError("max_restarts cannot be negative");
        }

        if (options_.min_restarts < 0) {
            throw RandomRestartError("min_restarts cannot be negative");
        }

        if (options_.perturb_component_probability < 0.0 ||
            options_.perturb_component_probability > 1.0) {
            throw RandomRestartError("perturb probability must be in [0,1]");
        }

        if (options_.max_detached_labels_per_component < 1) {
            options_.max_detached_labels_per_component = 1;
        }
    }

    LabelForest solve(
        const Tree& t1,
        const Tree& t2,
        const Timer* timer = nullptr,
        Random* rng = nullptr,
        const LabelForest* incumbent = nullptr
    ) const {
        require_same_leaf_set(t1, t2);
        Stats* stats = options_.stats;
        if (stats != nullptr) {
            *stats = Stats{};
        }

        Random local_rng;
        Random& random = (rng != nullptr) ? *rng : local_rng;

        LabelForest best = incumbent != nullptr
            ? *incumbent
            : LabelForest::singleton_forest_from_tree(t1);
        best.normalize();
        best.validate_partition_of(t1.leaf_labels);
        if (stats != nullptr) {
            stats->initial_components = best.component_count();
        }

        auto consider = [&](LabelForest candidate) {
            candidate.normalize();
            candidate.validate_partition_of(t1.leaf_labels);

            if (is_better(candidate, best)) {
                best = std::move(candidate);
                if (stats != nullptr) {
                    ++stats->best_updates;
                }
                return true;
            }
            return false;
        };

        /*
         * 1. Deterministic fast baseline.
         */
        if (options_.use_fast_cherry && !should_stop(timer)) {
            if (stats != nullptr) {
                stats->fast_attempted = true;
            }
            try {
                const std::size_t before_components = best.component_count();
                FastCherryApprox fast(options_.fast_options);
                LabelForest candidate = fast.solve(t1, t2, timer, &random);
                candidate = improve_candidate(t1, t2, std::move(candidate), timer, &random, false);
                const bool improved = consider(std::move(candidate));
                if (stats != nullptr && improved && best.component_count() < before_components) {
                    stats->fast_improved = true;
                }
            } catch (const std::exception&) {
                // Keep incumbent.
            }
        }

        /*
         * 2. One TwoApprox attempt, if small enough.
         */
        if (options_.use_two_approx &&
            static_cast<std::size_t>(t1.leaf_count()) <= options_.two_approx_max_leaves &&
            !should_stop(timer)) {
            if (stats != nullptr) {
                stats->two_attempted = true;
            }
            try {
                const std::size_t before_components = best.component_count();
                TwoApprox::Options two_opts = options_.two_options;
                two_opts.max_full_leaves = std::min(
                    two_opts.max_full_leaves,
                    options_.two_approx_max_leaves
                );

                TwoApprox two(two_opts);
                LabelForest candidate = two.solve(t1, t2, timer, &random);
                candidate = improve_candidate(t1, t2, std::move(candidate), timer, &random, false);
                const bool improved = consider(std::move(candidate));
                if (stats != nullptr && improved && best.component_count() < before_components) {
                    stats->two_improved = true;
                }
            } catch (const std::exception&) {
                // Keep incumbent.
            }
        }

        /*
         * 3. Singleton-start local improvement, useful on small reduced clusters.
         */
        if (options_.use_singleton_start &&
            static_cast<std::size_t>(t1.leaf_count()) <= options_.singleton_start_max_leaves &&
            !should_stop(timer)) {
            if (stats != nullptr) {
                stats->singleton_attempted = true;
            }
            try {
                const std::size_t before_components = best.component_count();
                LabelForest candidate = LabelForest::singleton_forest_from_tree(t1);
                candidate = improve_candidate(t1, t2, std::move(candidate), timer, &random, true);
                const bool improved = consider(std::move(candidate));
                if (stats != nullptr && improved && best.component_count() < before_components) {
                    stats->singleton_improved = true;
                }
            } catch (const std::exception&) {
                // Keep incumbent.
            }
        }

        /*
         * 4. Random restarts.
         */
        int restart = 0;

        while (restart < options_.max_restarts) {
            if (restart >= options_.min_restarts && should_stop(timer)) {
                break;
            }

            ++restart;
            if (stats != nullptr) {
                ++stats->restarts_attempted;
                ++stats->restart_mode_counts[static_cast<std::size_t>(restart % 6)];
            }

            try {
                const std::size_t before_components = best.component_count();
                LabelForest candidate = generate_candidate(t1, t2, best, timer, &random, restart);

                if (options_.use_perturbed_incumbent && random.coin(0.70)) {
                    candidate = perturb_forest(std::move(candidate), random);
                }

                candidate = improve_candidate(t1, t2, std::move(candidate), timer, &random, true);
                const bool improved = consider(std::move(candidate));
                if (stats != nullptr) {
                    ++stats->restarts_completed;
                    if (improved && best.component_count() < before_components) {
                        ++stats->restart_improvements;
                    }
                }
            } catch (const std::exception&) {
                // Random restart failed; ignore and continue.
                if (stats != nullptr) {
                    ++stats->restart_exceptions;
                }
            }
        }

        best.normalize();
        best.validate_partition_of(t1.leaf_labels);
        if (stats != nullptr) {
            stats->final_components = best.component_count();
            stats->stopped = should_stop(timer);
        }
        return best;
    }

private:
    Options options_;

    static void require_same_leaf_set(const Tree& t1, const Tree& t2) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw RandomRestartError("input trees do not have the same leaf set");
        }
    }

    bool should_stop(const Timer* timer) const {
        return timer != nullptr && timer->should_stop(options_.guard_seconds);
    }

    static std::uint64_t forest_tiebreak_score(const LabelForest& forest) {
        /*
         * Larger components are often more useful if component count ties.
         * Sum of squares rewards larger blocks.
         */
        std::uint64_t score = 0;

        for (const LabelComponent& component : forest.components) {
            const std::uint64_t s = static_cast<std::uint64_t>(component.labels.size());
            score += s * s;
        }

        return score;
    }

    static bool is_better(const LabelForest& candidate, const LabelForest& incumbent) {
        if (candidate.component_count() != incumbent.component_count()) {
            return candidate.component_count() < incumbent.component_count();
        }

        return forest_tiebreak_score(candidate) > forest_tiebreak_score(incumbent);
    }

    LabelForest improve_candidate(
        const Tree& t1,
        const Tree& t2,
        LabelForest candidate,
        const Timer* timer,
        Random* rng,
        bool randomized
    ) const {
        candidate.normalize();
        candidate.validate_partition_of(t1.leaf_labels);

        if (should_stop(timer)) {
            return candidate;
        }

        LocalImprove::Options improve_opts = options_.improve_options;
        improve_opts.deterministic = !randomized;

        LocalImprove improver(improve_opts);

        return improver.improve(
            t1,
            t2,
            std::move(candidate),
            timer,
            rng
        );
    }

    LabelForest generate_candidate(
        const Tree& t1,
        const Tree& t2,
        const LabelForest& best,
        const Timer* timer,
        Random* rng,
        int restart
    ) const {
        const std::size_t n = static_cast<std::size_t>(t1.leaf_count());

        /*
         * Cycle through generators. The random choice is deliberately biased
         * toward perturbing the incumbent because it is usually already decent.
         */
        const int mode = restart % 6;

        if (mode == 0 && options_.use_fast_cherry) {
            FastCherryApprox fast(options_.fast_options);
            return fast.solve(t1, t2, timer, rng);
        }

        if (mode == 1 &&
            options_.use_two_approx &&
            n <= options_.two_approx_max_leaves) {
            TwoApprox::Options two_opts = options_.two_options;
            two_opts.max_full_leaves = std::min(
                two_opts.max_full_leaves,
                options_.two_approx_max_leaves
            );

            TwoApprox two(two_opts);
            return two.solve(t1, t2, timer, rng);
        }

        if (mode == 2 &&
            options_.use_singleton_start &&
            n <= options_.singleton_start_max_leaves) {
            return LabelForest::singleton_forest_from_tree(t1);
        }

        /*
         * Default: restart from incumbent, perturb it, and improve again.
         */
        return best;
    }

    LabelForest perturb_forest(LabelForest forest, Random& rng) const {
        forest.normalize();

        LabelForest perturbed;
        perturbed.components.reserve(forest.components.size() * 2);

        std::vector<int> component_ids;
        component_ids.reserve(forest.components.size());

        for (int i = 0; i < static_cast<int>(forest.components.size()); ++i) {
            const LabelComponent& component = forest.components[static_cast<std::size_t>(i)];

            if (component.labels.size() > 1) {
                component_ids.push_back(i);
            }
        }

        rng.shuffle(component_ids);

        std::vector<char> should_perturb(forest.components.size(), 0);

        int marked = 0;

        for (int id : component_ids) {
            if (marked >= options_.max_components_to_perturb) {
                break;
            }

            const LabelComponent& component = forest.components[static_cast<std::size_t>(id)];

            if (options_.max_component_size_to_perturb != 0 &&
                component.labels.size() > options_.max_component_size_to_perturb) {
                continue;
            }

            if (rng.coin(options_.perturb_component_probability)) {
                should_perturb[static_cast<std::size_t>(id)] = 1;
                ++marked;
            }
        }

        for (int i = 0; i < static_cast<int>(forest.components.size()); ++i) {
            LabelComponent component = forest.components[static_cast<std::size_t>(i)];

            if (!should_perturb[static_cast<std::size_t>(i)] ||
                component.labels.size() <= 1) {
                perturbed.add_component(std::move(component));
                continue;
            }

            std::vector<std::uint32_t> labels = component.labels;
            rng.shuffle(labels);

            const int max_detach = std::min<int>(
                static_cast<int>(labels.size()) - 1,
                options_.max_detached_labels_per_component
            );

            if (max_detach <= 0) {
                perturbed.add_component(std::move(component));
                continue;
            }

            const int detach_count = rng.uniform_int(1, max_detach);

            std::vector<std::uint32_t> remaining;
            remaining.reserve(labels.size() - static_cast<std::size_t>(detach_count));

            for (int j = 0; j < static_cast<int>(labels.size()); ++j) {
                if (j < detach_count) {
                    perturbed.components.emplace_back(labels[static_cast<std::size_t>(j)]);
                } else {
                    remaining.push_back(labels[static_cast<std::size_t>(j)]);
                }
            }

            if (!remaining.empty()) {
                perturbed.add_component(LabelComponent(std::move(remaining)));
            }
        }

        perturbed.normalize();
        return perturbed;
    }
};

}  // namespace pace26::heuristics
