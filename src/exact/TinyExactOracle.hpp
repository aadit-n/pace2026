#pragma once

#include "core/Forest.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pace26::exact {

class TinyExactOracle {
public:
    using LabelForest = pace26::core::LabelForest;
    using LabelComponent = pace26::core::LabelComponent;
    using Timer = pace26::core::Timer;
    using Tree = pace26::core::Tree;

    struct Options {
        std::size_t max_leaves = 12;
        std::uint64_t max_search_nodes = 4000000ULL;
        double guard_seconds = 0.05;
        std::size_t max_cache_entries = 65536;
    };

    struct Stats {
        bool cache_hit = false;
        bool timed_out = false;
        bool node_limit_hit = false;
        std::size_t leaves = 0;
        std::size_t subsets_tested = 0;
        std::size_t valid_components = 0;
        std::uint64_t search_nodes = 0;
        std::size_t result_components = 0;
        std::size_t cache_entries = 0;
    };

    TinyExactOracle() = default;

    explicit TinyExactOracle(Options options)
        : options_(options) {}

    std::optional<LabelForest> solve(
        const Tree& t1,
        const Tree& t2,
        Timer* timer = nullptr,
        Stats* stats_out = nullptr
    ) const {
        Stats stats;
        stats.leaves = static_cast<std::size_t>(t1.leaf_count());

        auto finish = [&](std::optional<LabelForest> result) {
            stats.cache_entries = cache().size();
            if (result.has_value()) {
                stats.result_components = result->component_count();
            }
            if (stats_out != nullptr) {
                *stats_out = stats;
            }
            return result;
        };

        if (t1.leaf_labels != t2.leaf_labels ||
            stats.leaves == 0 ||
            stats.leaves > options_.max_leaves ||
            stats.leaves > 30) {
            return finish(std::nullopt);
        }

        if (t1.node_count() > 63 || t2.node_count() > 63) {
            return finish(std::nullopt);
        }

        if (timer != nullptr && timer->should_stop(options_.guard_seconds)) {
            stats.timed_out = true;
            return finish(std::nullopt);
        }

        const std::vector<std::uint32_t>& labels = t1.leaf_labels;
        const std::unordered_map<std::uint32_t, int> label_to_bit =
            build_label_to_bit(labels);
        const std::string key = canonical_pair_key(t1, t2, label_to_bit);

        auto& memo = cache();
        auto it = memo.find(key);
        if (it != memo.end()) {
            stats.cache_hit = true;
            return finish(forest_from_masks(labels, it->second.component_masks));
        }

        const int n = static_cast<int>(labels.size());
        const std::uint32_t all_mask =
            n == 32 ? std::numeric_limits<std::uint32_t>::max()
                    : ((std::uint32_t{1} << n) - 1U);

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<std::size_t>(all_mask));

        std::vector<std::vector<int>> candidates_by_bit(
            static_cast<std::size_t>(n)
        );

        for (std::uint32_t mask = 1; mask <= all_mask; ++mask) {
            ++stats.subsets_tested;

            if (!same_restricted_topology(t1, t2, label_to_bit, mask)) {
                continue;
            }

            Candidate candidate;
            candidate.mask = mask;
            candidate.size = popcount(mask);
            candidate.edges1 = induced_edge_mask(t1, labels, mask);
            candidate.edges2 = induced_edge_mask(t2, labels, mask);

            const int index = static_cast<int>(candidates.size());
            candidates.push_back(candidate);

            for (int bit = 0; bit < n; ++bit) {
                if ((mask & (std::uint32_t{1} << bit)) != 0) {
                    candidates_by_bit[static_cast<std::size_t>(bit)].push_back(index);
                }
            }
        }

        stats.valid_components = candidates.size();

        for (std::vector<int>& bucket : candidates_by_bit) {
            std::sort(
                bucket.begin(),
                bucket.end(),
                [&](int a, int b) {
                    const Candidate& ca = candidates[static_cast<std::size_t>(a)];
                    const Candidate& cb = candidates[static_cast<std::size_t>(b)];
                    if (ca.size != cb.size) {
                        return ca.size > cb.size;
                    }
                    return ca.mask < cb.mask;
                }
            );
        }

        SearchState state;
        state.candidates = &candidates;
        state.candidates_by_bit = &candidates_by_bit;
        state.timer = timer;
        state.options = &options_;
        state.stats = &stats;
        state.all_mask = all_mask;
        state.max_candidate_size = 1;
        state.best_count = static_cast<int>(labels.size());

        for (const Candidate& candidate : candidates) {
            state.max_candidate_size =
                std::max(state.max_candidate_size, candidate.size);
        }

        state.best_masks.reserve(labels.size());
        for (int bit = 0; bit < n; ++bit) {
            state.best_masks.push_back(std::uint32_t{1} << bit);
        }

        dfs(
            state,
            all_mask,
            0,
            0,
            0
        );

        if (stats.timed_out || stats.node_limit_hit) {
            return finish(std::nullopt);
        }

        LabelForest result = forest_from_masks(labels, state.best_masks);
        result.normalize();

        if (memo.size() >= options_.max_cache_entries) {
            memo.clear();
        }
        memo.emplace(key, CacheEntry{state.best_masks});

        return finish(std::move(result));
    }

private:
    struct SignatureKey {
        int a = 0;
        int b = 0;

        bool operator==(const SignatureKey& other) const noexcept {
            return a == other.a && b == other.b;
        }
    };

    struct SignatureKeyHash {
        std::size_t operator()(const SignatureKey& key) const noexcept {
            std::uint64_t x = static_cast<std::uint64_t>(key.a) + 0x9e3779b97f4a7c15ULL;
            x ^= static_cast<std::uint64_t>(key.b) + 0xbf58476d1ce4e5b9ULL + (x << 6) + (x >> 2);
            return static_cast<std::size_t>(x);
        }
    };

    class SignatureInterner {
    public:
        int leaf(int bit) {
            return intern({0, bit + 1});
        }

        int internal(int left, int right) {
            if (left <= 0) return right;
            if (right <= 0) return left;
            if (right < left) {
                std::swap(left, right);
            }
            return intern({left, right});
        }

    private:
        std::unordered_map<SignatureKey, int, SignatureKeyHash> ids_;
        int next_id_ = 1;

        int intern(const SignatureKey& key) {
            auto it = ids_.find(key);
            if (it != ids_.end()) {
                return it->second;
            }
            const int id = next_id_++;
            ids_.emplace(key, id);
            return id;
        }
    };

    struct Candidate {
        std::uint32_t mask = 0;
        std::uint64_t edges1 = 0;
        std::uint64_t edges2 = 0;
        int size = 0;
    };

    struct CacheEntry {
        std::vector<std::uint32_t> component_masks;
    };

    struct SearchState {
        const std::vector<Candidate>* candidates = nullptr;
        const std::vector<std::vector<int>>* candidates_by_bit = nullptr;
        Timer* timer = nullptr;
        const Options* options = nullptr;
        Stats* stats = nullptr;
        std::uint32_t all_mask = 0;
        int max_candidate_size = 1;
        int best_count = 0;
        std::vector<std::uint32_t> chosen_masks;
        std::vector<std::uint32_t> best_masks;
    };

    Options options_;

    static std::unordered_map<std::string, CacheEntry>& cache() {
        static std::unordered_map<std::string, CacheEntry> memo;
        return memo;
    }

    static int popcount(std::uint32_t mask) {
#if defined(__GNUC__) || defined(__clang__)
        return __builtin_popcount(mask);
#else
        int count = 0;
        while (mask != 0) {
            mask &= mask - 1;
            ++count;
        }
        return count;
#endif
    }

    static std::unordered_map<std::uint32_t, int> build_label_to_bit(
        const std::vector<std::uint32_t>& labels
    ) {
        std::unordered_map<std::uint32_t, int> label_to_bit;
        label_to_bit.reserve(labels.size() * 2);

        for (std::size_t i = 0; i < labels.size(); ++i) {
            label_to_bit.emplace(labels[i], static_cast<int>(i));
        }

        return label_to_bit;
    }

    static std::string canonical_subtree(
        const Tree& tree,
        int node,
        const std::unordered_map<std::uint32_t, int>& label_to_bit
    ) {
        const auto& tree_node = tree.nodes[static_cast<std::size_t>(node)];
        if (tree_node.is_leaf()) {
            auto it = label_to_bit.find(tree_node.label);
            if (it == label_to_bit.end()) {
                throw std::runtime_error("tiny exact label missing from map");
            }
            return "L" + std::to_string(it->second);
        }

        std::string left = canonical_subtree(tree, tree_node.left, label_to_bit);
        std::string right = canonical_subtree(tree, tree_node.right, label_to_bit);
        if (right < left) {
            std::swap(left, right);
        }
        return "(" + left + "," + right + ")";
    }

    static std::string canonical_pair_key(
        const Tree& t1,
        const Tree& t2,
        const std::unordered_map<std::uint32_t, int>& label_to_bit
    ) {
        return std::to_string(t1.leaf_count()) +
               "|" +
               canonical_subtree(t1, t1.root, label_to_bit) +
               "|" +
               canonical_subtree(t2, t2.root, label_to_bit);
    }

    static int restricted_signature(
        const Tree& tree,
        const std::unordered_map<std::uint32_t, int>& label_to_bit,
        std::uint32_t mask,
        SignatureInterner& interner
    ) {
        std::vector<int> signature(static_cast<std::size_t>(tree.node_count()), 0);

        for (int node_index : tree.postorder) {
            const auto& node = tree.nodes[static_cast<std::size_t>(node_index)];
            if (node.is_leaf()) {
                auto it = label_to_bit.find(node.label);
                if (it != label_to_bit.end() &&
                    (mask & (std::uint32_t{1} << it->second)) != 0) {
                    signature[static_cast<std::size_t>(node_index)] =
                        interner.leaf(it->second);
                }
                continue;
            }

            const int left_sig = signature[static_cast<std::size_t>(node.left)];
            const int right_sig = signature[static_cast<std::size_t>(node.right)];
            signature[static_cast<std::size_t>(node_index)] =
                interner.internal(left_sig, right_sig);
        }

        return signature[static_cast<std::size_t>(tree.root)];
    }

    static bool same_restricted_topology(
        const Tree& t1,
        const Tree& t2,
        const std::unordered_map<std::uint32_t, int>& label_to_bit,
        std::uint32_t mask
    ) {
        if ((mask & (mask - 1U)) == 0) {
            return true;
        }

        SignatureInterner interner;
        const int sig1 = restricted_signature(t1, label_to_bit, mask, interner);
        const int sig2 = restricted_signature(t2, label_to_bit, mask, interner);
        return sig1 > 0 && sig1 == sig2;
    }

    static std::uint64_t induced_edge_mask(
        const Tree& tree,
        const std::vector<std::uint32_t>& labels,
        std::uint32_t mask
    ) {
        if ((mask & (mask - 1U)) == 0) {
            return 0;
        }

        int root = -1;
        for (std::size_t bit = 0; bit < labels.size(); ++bit) {
            if ((mask & (std::uint32_t{1} << bit)) == 0) {
                continue;
            }
            const int node = tree.node_of_label(labels[bit]);
            root = root < 0 ? node : tree.lca(root, node);
        }

        std::uint64_t edges = 0;
        for (std::size_t bit = 0; bit < labels.size(); ++bit) {
            if ((mask & (std::uint32_t{1} << bit)) == 0) {
                continue;
            }

            int node = tree.node_of_label(labels[bit]);
            while (node != root) {
                if (node < 0 || node >= 63) {
                    throw std::runtime_error("tiny exact tree edge index out of range");
                }
                edges |= (std::uint64_t{1} << node);
                node = tree.parent[static_cast<std::size_t>(node)];
            }
        }

        return edges;
    }

    static LabelForest forest_from_masks(
        const std::vector<std::uint32_t>& labels,
        const std::vector<std::uint32_t>& masks
    ) {
        LabelForest forest;
        forest.components.reserve(masks.size());

        for (std::uint32_t mask : masks) {
            std::vector<std::uint32_t> component_labels;
            component_labels.reserve(static_cast<std::size_t>(popcount(mask)));
            for (std::size_t bit = 0; bit < labels.size(); ++bit) {
                if ((mask & (std::uint32_t{1} << bit)) != 0) {
                    component_labels.push_back(labels[bit]);
                }
            }
            forest.add_component(LabelComponent(std::move(component_labels)));
        }

        forest.normalize();
        return forest;
    }

    static void maybe_stop(SearchState& state) {
        if ((state.stats->search_nodes & 2047ULL) != 0) {
            return;
        }

        if (state.stats->search_nodes >= state.options->max_search_nodes) {
            state.stats->node_limit_hit = true;
            return;
        }

        if (state.timer != nullptr &&
            state.timer->should_stop(state.options->guard_seconds)) {
            state.stats->timed_out = true;
        }
    }

    static int choose_branch_bit(
        const SearchState& state,
        std::uint32_t uncovered,
        std::uint64_t used_edges1,
        std::uint64_t used_edges2
    ) {
        int best_bit = -1;
        std::size_t best_count = std::numeric_limits<std::size_t>::max();

        int n = 0;
        for (std::uint32_t tmp = uncovered; tmp != 0; tmp >>= 1U) {
            ++n;
        }
        for (int bit = 0; bit < n; ++bit) {
            if ((uncovered & (std::uint32_t{1} << bit)) == 0) {
                continue;
            }

            std::size_t count = 0;
            const std::vector<int>& bucket =
                (*state.candidates_by_bit)[static_cast<std::size_t>(bit)];
            for (int index : bucket) {
                const Candidate& candidate =
                    (*state.candidates)[static_cast<std::size_t>(index)];
                if ((candidate.mask & uncovered) == candidate.mask &&
                    (candidate.edges1 & used_edges1) == 0 &&
                    (candidate.edges2 & used_edges2) == 0) {
                    ++count;
                }
            }

            if (count < best_count) {
                best_count = count;
                best_bit = bit;
                if (count <= 1) {
                    break;
                }
            }
        }

        return best_bit;
    }

    static void dfs(
        SearchState& state,
        std::uint32_t uncovered,
        std::uint64_t used_edges1,
        std::uint64_t used_edges2,
        int chosen_count
    ) {
        if (state.stats->timed_out || state.stats->node_limit_hit) {
            return;
        }

        ++state.stats->search_nodes;
        maybe_stop(state);

        if (state.stats->timed_out || state.stats->node_limit_hit) {
            return;
        }

        if (uncovered == 0) {
            if (chosen_count < state.best_count) {
                state.best_count = chosen_count;
                state.best_masks = state.chosen_masks;
            }
            return;
        }

        if (chosen_count >= state.best_count) {
            return;
        }

        const int remaining = popcount(uncovered);
        const int optimistic =
            (remaining + state.max_candidate_size - 1) / state.max_candidate_size;
        if (chosen_count + optimistic >= state.best_count) {
            return;
        }

        const int branch_bit =
            choose_branch_bit(state, uncovered, used_edges1, used_edges2);
        if (branch_bit < 0) {
            return;
        }

        const std::vector<int>& bucket =
            (*state.candidates_by_bit)[static_cast<std::size_t>(branch_bit)];

        for (int index : bucket) {
            const Candidate& candidate =
                (*state.candidates)[static_cast<std::size_t>(index)];

            if ((candidate.mask & uncovered) != candidate.mask ||
                (candidate.edges1 & used_edges1) != 0 ||
                (candidate.edges2 & used_edges2) != 0) {
                continue;
            }

            state.chosen_masks.push_back(candidate.mask);
            dfs(
                state,
                uncovered & ~candidate.mask,
                used_edges1 | candidate.edges1,
                used_edges2 | candidate.edges2,
                chosen_count + 1
            );
            state.chosen_masks.pop_back();

            if (state.stats->timed_out || state.stats->node_limit_hit) {
                return;
            }
        }
    }
};

}  // namespace pace26::exact
