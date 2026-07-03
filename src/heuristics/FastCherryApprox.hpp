#pragma once

#include "core/Forest.hpp"
#include "core/Tree.hpp"
#include "core/Timer.hpp"
#include "core/Random.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class FastCherryApproxError : public std::runtime_error {
public:
    explicit FastCherryApproxError(const std::string& message)
        : std::runtime_error("FastCherryApprox error: " + message) {}
};

/*
 * Fast safe baseline heuristic.
 *
 * It computes canonical rooted-subtree signatures in both trees.
 * If a rooted labelled subtree appears in both trees, it is a valid agreement
 * forest component. We then select maximal such subtrees and output their
 * leaf sets as components.
 *
 * This is effectively repeated common-cherry/common-subtree contraction,
 * but implemented directly on the current trees.
 *
 * Guarantees:
 * - returns a partition of the leaf set;
 * - each component has identical rooted topology in T1 and T2;
 * - components are disjoint by construction;
 * - runtime is roughly O(n log n), practically linear for PACE-sized trees.
 */
class FastCherryApprox {
private:
    using Tree = pace26::core::Tree;
    using LabelForest = pace26::core::LabelForest;
    using LabelComponent = pace26::core::LabelComponent;
    using Timer = pace26::core::Timer;
    using Random = pace26::core::Random;

    struct SignatureKey {
        std::uint8_t type = 0; // 0 = leaf, 1 = internal
        std::uint64_t a = 0;
        std::uint64_t b = 0;

        bool operator==(const SignatureKey& other) const noexcept {
            return type == other.type && a == other.a && b == other.b;
        }
    };

    struct SignatureKeyHash {
        std::size_t operator()(const SignatureKey& key) const noexcept {
            std::uint64_t x = 0x9e3779b97f4a7c15ULL;

            x ^= static_cast<std::uint64_t>(key.type) +
                 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);

            x ^= key.a +
                 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);

            x ^= key.b +
                 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);

            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;

            return static_cast<std::size_t>(x);
        }
    };

    class SignatureInterner {
    private:
        std::unordered_map<SignatureKey, int, SignatureKeyHash> id_of_;
        int next_id_ = 1;

    public:
        int leaf(std::uint32_t label) {
            SignatureKey key;
            key.type = 0;
            key.a = label;
            key.b = 0;

            auto it = id_of_.find(key);
            if (it != id_of_.end()) {
                return it->second;
            }

            const int id = next_id_++;
            id_of_.emplace(key, id);
            return id;
        }

        int internal(int left_sig, int right_sig) {
            if (left_sig <= 0 || right_sig <= 0) {
                throw FastCherryApproxError("invalid child signature");
            }

            std::uint64_t a = static_cast<std::uint64_t>(left_sig);
            std::uint64_t b = static_cast<std::uint64_t>(right_sig);

            // Rooted but unordered binary tree: child order does not matter.
            if (a > b) {
                std::swap(a, b);
            }

            SignatureKey key;
            key.type = 1;
            key.a = a;
            key.b = b;

            auto it = id_of_.find(key);
            if (it != id_of_.end()) {
                return it->second;
            }

            const int id = next_id_++;
            id_of_.emplace(key, id);
            return id;
        }
    };

    struct SignatureInfo {
        std::vector<int> sig;
        std::vector<std::uint32_t> leaf_count;
    };

    struct Candidate {
        std::uint32_t leaf_count = 0;
        int sig = 0;
        int node1 = -1;
        int node2 = -1;
    };

    static void require_same_leaf_set(const Tree& t1, const Tree& t2) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw FastCherryApproxError("input trees do not have the same leaf set");
        }
    }

    static SignatureInfo compute_signatures(
        const Tree& tree,
        SignatureInterner& interner
    ) {
        SignatureInfo info;
        info.sig.assign(static_cast<std::size_t>(tree.node_count()), 0);
        info.leaf_count.assign(static_cast<std::size_t>(tree.node_count()), 0);

        for (int u : tree.postorder) {
            if (tree.is_leaf(u)) {
                info.sig[static_cast<std::size_t>(u)] =
                    interner.leaf(tree.label(u));

                info.leaf_count[static_cast<std::size_t>(u)] = 1;
            } else {
                const int l = tree.left(u);
                const int r = tree.right(u);

                const int ls = info.sig[static_cast<std::size_t>(l)];
                const int rs = info.sig[static_cast<std::size_t>(r)];

                info.sig[static_cast<std::size_t>(u)] =
                    interner.internal(ls, rs);

                info.leaf_count[static_cast<std::size_t>(u)] =
                    info.leaf_count[static_cast<std::size_t>(l)] +
                    info.leaf_count[static_cast<std::size_t>(r)];
            }
        }

        return info;
    }

    static void mark_subtree_blocked(
        const Tree& tree,
        int root,
        std::vector<char>& blocked
    ) {
        std::vector<int> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();

            if (blocked[static_cast<std::size_t>(u)]) {
                continue;
            }

            blocked[static_cast<std::size_t>(u)] = 1;

            if (!tree.is_leaf(u)) {
                stack.push_back(tree.left(u));
                stack.push_back(tree.right(u));
            }
        }
    }

    static std::vector<std::uint32_t> collect_labels(
        const Tree& tree,
        int root
    ) {
        return tree.labels_under(root);
    }

    static LabelForest build_forest_from_candidates(
        const Tree& t1,
        const Tree& t2,
        const std::vector<Candidate>& raw_candidates
    ) {
        std::vector<Candidate> candidates = raw_candidates;

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                if (a.leaf_count != b.leaf_count) {
                    return a.leaf_count > b.leaf_count;
                }

                return a.sig < b.sig;
            }
        );

        std::vector<char> blocked1(static_cast<std::size_t>(t1.node_count()), 0);
        std::vector<char> blocked2(static_cast<std::size_t>(t2.node_count()), 0);
        std::unordered_set<std::uint32_t> covered;
        covered.reserve(t1.leaf_labels.size() * 2);

        LabelForest forest;

        for (const Candidate& c : candidates) {
            if (c.leaf_count <= 1) {
                continue;
            }

            if (blocked1[static_cast<std::size_t>(c.node1)] ||
                blocked2[static_cast<std::size_t>(c.node2)]) {
                continue;
            }

            LabelComponent component;
            component.labels = collect_labels(t1, c.node1);
            component.normalize();

            forest.add_component(std::move(component));

            const LabelComponent& added = forest.components.back();
            for (std::uint32_t label : added.labels) {
                covered.insert(label);
            }

            mark_subtree_blocked(t1, c.node1, blocked1);
            mark_subtree_blocked(t2, c.node2, blocked2);
        }

        /*
         * Add every original label not already covered.
         *
         * These singleton components are always valid.
         */
        for (std::uint32_t label : t1.leaf_labels) {
            if (covered.find(label) == covered.end()) {
                forest.components.emplace_back(label);
            }
        }

        forest.normalize();
        forest.validate_partition_of(t1.leaf_labels);

        return forest;
    }

public:
    struct Options {
        /*
         * Usually 2.
         *
         * min_component_size = 2 means cherries/subtrees are merged.
         * min_component_size = 1 degenerates toward singleton handling and is not useful.
         */
        std::uint32_t min_component_size = 2;

        /*
         * If true, use the whole common subtree rooted at the root if both trees
         * are identical. This can produce a forest with one component.
         */
        bool allow_whole_tree_component = true;
    };
    FastCherryApprox() : FastCherryApprox(Options{}) {}

    explicit FastCherryApprox(Options options)
        : options_(options) {
        if (options_.min_component_size == 0) {
            throw FastCherryApproxError("min_component_size must be positive");
        }
    }

    LabelForest solve(
        const Tree& t1,
        const Tree& t2,
        const Timer* timer = nullptr,
        Random* rng = nullptr
    ) const {
        (void)rng;

        if (timer != nullptr && timer->should_stop(0.05)) {
            return LabelForest::singleton_forest_from_tree(t1);
        }

        require_same_leaf_set(t1, t2);

        SignatureInterner interner;

        SignatureInfo s1 = compute_signatures(t1, interner);
        SignatureInfo s2 = compute_signatures(t2, interner);

        /*
         * Because PACE leaf labels are unique, a rooted labelled subtree signature
         * can occur at most once in a tree.
         */
        std::unordered_map<int, int> node_of_signature_t2;
        node_of_signature_t2.reserve(static_cast<std::size_t>(t2.node_count()) * 2);

        for (int v : t2.postorder) {
            const std::uint32_t count =
                s2.leaf_count[static_cast<std::size_t>(v)];

            if (count < options_.min_component_size) {
                continue;
            }

            if (!options_.allow_whole_tree_component &&
                v == t2.root) {
                continue;
            }

            const int sig = s2.sig[static_cast<std::size_t>(v)];
            node_of_signature_t2.emplace(sig, v);
        }

        std::vector<Candidate> candidates;
        candidates.reserve(static_cast<std::size_t>(t1.node_count()));

        for (int u : t1.postorder) {
            if (timer != nullptr && timer->should_stop(0.05)) {
                break;
            }

            const std::uint32_t count =
                s1.leaf_count[static_cast<std::size_t>(u)];

            if (count < options_.min_component_size) {
                continue;
            }

            if (!options_.allow_whole_tree_component &&
                u == t1.root) {
                continue;
            }

            const int sig = s1.sig[static_cast<std::size_t>(u)];

            auto it = node_of_signature_t2.find(sig);
            if (it == node_of_signature_t2.end()) {
                continue;
            }

            const int v = it->second;

            const std::uint32_t count2 =
                s2.leaf_count[static_cast<std::size_t>(v)];

            if (count != count2) {
                throw FastCherryApproxError("signature leaf-count mismatch");
            }

            Candidate c;
            c.leaf_count = count;
            c.sig = sig;
            c.node1 = u;
            c.node2 = v;
            candidates.push_back(c);
        }

        if (timer != nullptr && timer->should_stop(0.05)) {
            return LabelForest::singleton_forest_from_tree(t1);
        }

        return build_forest_from_candidates(t1, t2, candidates);
    }

private:
    Options options_;
};

}  // namespace pace26::heuristics
