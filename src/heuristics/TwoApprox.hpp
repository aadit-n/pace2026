#pragma once

#include "core/Forest.hpp"
#include "core/Random.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"
#include "FastCherryApprox.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class TwoApproxError : public std::runtime_error {
public:
    explicit TwoApproxError(const std::string& message)
        : std::runtime_error("TwoApprox error: " + message) {}
};

/*
 * Optimised two-tree rooted MAF 2-approx-style implementation.
 *
 * It follows the active-leaf algorithm structure:
 * - Preprocess common active siblings.
 * - Find a minimal incompatible active sibling set in T1.
 * - ResolveSet / ResolvePair.
 * - Cut edges in both active forests.
 * - Return the components induced by deleted edges in T1.
 *
 * Practical warning:
 * This is much heavier than FastCherryApprox. Use it on reduced/clustered
 * subinstances, not blindly on every 17k-leaf instance.
 */
class TwoApprox {
private:
    using Tree = pace26::core::Tree;
    using LabelForest = pace26::core::LabelForest;
    using LabelComponent = pace26::core::LabelComponent;
    using Timer = pace26::core::Timer;
    using TimeBudget = pace26::core::TimeBudget;
    using Random = pace26::core::Random;

public:
    struct Options {
        std::size_t max_full_leaves = 5000;
        double local_time_limit_seconds = 20.0;
        double guard_seconds = 0.05;

        bool run_fast_cherry_fallback = true;
        bool run_safe_greedy_merge = true;

        /*
         * The internal algorithm can be deterministic.
         * Random is only used for optional tie-breaking / future extensions.
         */
        bool deterministic = true;
    };
    TwoApprox() : TwoApprox(Options{}) {}

    explicit TwoApprox(Options options)
        : options_(options) {}

    LabelForest solve(
        const Tree& t1,
        const Tree& t2,
        const Timer* global_timer = nullptr,
        Random* rng = nullptr
    ) const {
        require_same_leaf_set(t1, t2);

        LabelForest fallback =
            LabelForest::singleton_forest_from_tree(t1);

        if (options_.run_fast_cherry_fallback) {
            FastCherryApprox fast;
            fallback = fast.solve(t1, t2, global_timer, rng);
        }

        if (global_timer != nullptr && global_timer->should_stop(options_.guard_seconds)) {
            return fallback;
        }

        if (static_cast<std::size_t>(t1.leaf_count()) > options_.max_full_leaves) {
            return fallback;
        }

        try {
            Solver solver(t1, t2, options_, global_timer, rng);
            LabelForest candidate = solver.run();

            if (options_.run_safe_greedy_merge &&
                (global_timer == nullptr || !global_timer->should_stop(options_.guard_seconds))) {
                solver.safe_greedy_merge(candidate);
            }

            candidate.normalize();
            candidate.validate_partition_of(t1.leaf_labels);

            if (candidate.component_count() < fallback.component_count()) {
                return candidate;
            }

            return fallback;
        } catch (const std::exception&) {
            /*
             * In PACE, a safe mediocre answer beats a clever crash.
             */
            return fallback;
        }
    }

private:
    struct SignatureKey {
        std::uint8_t type = 0;
        std::uint64_t a = 0;
        std::uint64_t b = 0;

        bool operator==(const SignatureKey& other) const noexcept {
            return type == other.type && a == other.a && b == other.b;
        }
    };

    struct SignatureKeyHash {
        std::size_t operator()(const SignatureKey& k) const noexcept {
            std::uint64_t x = 0x9e3779b97f4a7c15ULL;
            x ^= static_cast<std::uint64_t>(k.type) + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
            x ^= k.a + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
            x ^= k.b + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);

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

            int id = next_id_++;
            id_of_.emplace(key, id);
            return id;
        }

        int internal(int a, int b) {
            if (a <= 0 || b <= 0) {
                throw TwoApproxError("invalid restricted signature");
            }

            std::uint64_t x = static_cast<std::uint64_t>(a);
            std::uint64_t y = static_cast<std::uint64_t>(b);

            if (x > y) {
                std::swap(x, y);
            }

            SignatureKey key;
            key.type = 1;
            key.a = x;
            key.b = y;

            auto it = id_of_.find(key);
            if (it != id_of_.end()) {
                return it->second;
            }

            int id = next_id_++;
            id_of_.emplace(key, id);
            return id;
        }
    };

    struct ActiveState {
        std::vector<std::uint32_t> labels;
        std::unordered_map<std::uint32_t, int> index_of_label;

        std::vector<char> active;
        std::vector<int> t1_leaf_node;
        std::vector<int> t2_leaf_node;

        /*
         * Used by retroactive merge.
         * neighbor[i] stores the index of the active leaf u that caused i to be cut
         * as part of W.
         */
        std::vector<int> neighbor;

        explicit ActiveState(const Tree& t1, const Tree& t2) {
            labels = t1.leaf_labels;
            std::sort(labels.begin(), labels.end());

            index_of_label.reserve(labels.size() * 2);

            const std::size_t n = labels.size();
            active.assign(n, 1);
            t1_leaf_node.assign(n, -1);
            t2_leaf_node.assign(n, -1);
            neighbor.assign(n, -1);

            for (std::size_t i = 0; i < n; ++i) {
                std::uint32_t x = labels[i];
                index_of_label.emplace(x, static_cast<int>(i));
                t1_leaf_node[i] = t1.node_of_label(x);
                t2_leaf_node[i] = t2.node_of_label(x);
            }
        }

        int index(std::uint32_t label) const {
            auto it = index_of_label.find(label);
            if (it == index_of_label.end()) {
                throw TwoApproxError("unknown leaf label");
            }
            return it->second;
        }

        std::uint32_t label(int idx) const {
            return labels[static_cast<std::size_t>(idx)];
        }

        bool is_active(int idx) const {
            return active[static_cast<std::size_t>(idx)] != 0;
        }

        void deactivate(int idx) {
            active[static_cast<std::size_t>(idx)] = 0;
        }

        int active_count() const {
            int count = 0;
            for (char x : active) {
                if (x) {
                    ++count;
                }
            }
            return count;
        }
    };

    class MutableTree {
    public:
        const Tree& base;
        ActiveState& state;
        bool is_t1 = true;

        /*
         * edge_deleted_above[u] means the edge from parent[u] to u is cut.
         * For root this is ignored.
         */
        std::vector<char> edge_deleted_above;

        explicit MutableTree(
            const Tree& tree,
            ActiveState& active_state,
            bool tree1
        )
            : base(tree),
              state(active_state),
              is_t1(tree1),
              edge_deleted_above(static_cast<std::size_t>(tree.node_count()), 0) {}

        int node_of_index(int idx) const {
            return is_t1
                ? state.t1_leaf_node[static_cast<std::size_t>(idx)]
                : state.t2_leaf_node[static_cast<std::size_t>(idx)];
        }

        int index_of_leaf_node(int node) const {
            std::uint32_t x = base.label(node);
            return state.index(x);
        }

        bool edge_above_deleted(int node) const {
            if (node == base.root) {
                return false;
            }

            return edge_deleted_above[static_cast<std::size_t>(node)] != 0;
        }

        void delete_edge_above(int node) {
            if (node == base.root || node < 0) {
                return;
            }

            edge_deleted_above[static_cast<std::size_t>(node)] = 1;
        }

        void undelete_edge_above(int node) {
            if (node == base.root || node < 0) {
                return;
            }

            edge_deleted_above[static_cast<std::size_t>(node)] = 0;
        }

        int sibling(int node) const {
            int p = base.parent[static_cast<std::size_t>(node)];
            if (p < 0) {
                return -1;
            }

            if (base.left(p) == node) {
                return base.right(p);
            }

            if (base.right(p) == node) {
                return base.left(p);
            }

            throw TwoApproxError("parent pointer inconsistency");
        }

        std::vector<int> active_indices_under(int root, bool respect_deleted_edges = true) const {
            std::vector<int> result;

            std::vector<int> stack;
            stack.push_back(root);

            while (!stack.empty()) {
                int u = stack.back();
                stack.pop_back();

                if (base.is_leaf(u)) {
                    int idx = index_of_leaf_node(u);
                    if (state.is_active(idx)) {
                        result.push_back(idx);
                    }
                    continue;
                }

                int l = base.left(u);
                int r = base.right(u);

                if (!respect_deleted_edges || !edge_above_deleted(l)) {
                    stack.push_back(l);
                }

                if (!respect_deleted_edges || !edge_above_deleted(r)) {
                    stack.push_back(r);
                }
            }

            return result;
        }

        int active_count_under(int node, bool respect_deleted_edges = true) const {
            return static_cast<int>(active_indices_under(node, respect_deleted_edges).size());
        }

        bool subtree_contains_active_index(int root, int idx) const {
            std::vector<int> active = active_indices_under(root, true);
            return std::find(active.begin(), active.end(), idx) != active.end();
        }

        int root_of_subtree(int node) const {
            int u = node;

            while (u != base.root) {
                int p = base.parent[static_cast<std::size_t>(u)];
                if (p < 0 || edge_above_deleted(u)) {
                    return u;
                }
                u = p;
            }

            return u;
        }

        int root_of_subtree_by_index(int idx) const {
            return root_of_subtree(node_of_index(idx));
        }

        /*
         * Java lcaInactiveTree() equivalent.
         */
        int lca_inactive_tree_from_leaf_node(int node) const {
            if (node == base.root) {
                return node;
            }

            int p = base.parent[static_cast<std::size_t>(node)];

            if (active_count_under(p, true) > 1) {
                return node;
            }

            if (!edge_above_deleted(node)) {
                return lca_inactive_tree_from_leaf_node(p);
            }

            return node;
        }

        int p_active(int node) const {
            if (node == base.root) {
                return node;
            }

            if (active_count_under(node, true) == 1) {
                int p = base.parent[static_cast<std::size_t>(node)];
                if (p < 0) {
                    return node;
                }
                return p_active(p);
            }

            return node;
        }

        int move_up_to_keep_invariant(int node) const {
            if (node == base.root) {
                return node;
            }

            int p = base.parent[static_cast<std::size_t>(node)];
            int sib = sibling(node);

            if (!edge_above_deleted(node) && sib >= 0 && edge_above_deleted(sib)) {
                return move_up_to_keep_invariant(p);
            }

            return node;
        }

        void make_path_by_flipping_edges(int node, int ancestor) {
            if (node == base.root || node == ancestor) {
                return;
            }

            int p = base.parent[static_cast<std::size_t>(node)];
            int sib = sibling(node);

            if (edge_above_deleted(node)) {
                if (p != ancestor && sib >= 0) {
                    edge_deleted_above[static_cast<std::size_t>(sib)] = 1;
                }

                edge_deleted_above[static_cast<std::size_t>(node)] = 0;
            }

            if (p != ancestor) {
                make_path_by_flipping_edges(p, ancestor);
            }
        }

        bool active_sibling_with(int a_idx, int b_idx) const {
            if (!state.is_active(a_idx) || !state.is_active(b_idx)) {
                return false;
            }

            int a = node_of_index(a_idx);
            int b = node_of_index(b_idx);

            if (root_of_subtree(a) != root_of_subtree(b)) {
                return false;
            }

            int l = base.lca(a, b);
            return active_count_under(l, true) == 2;
        }

        std::pair<int, int> first_active_sibling_pair(int root) const {
            int count = active_count_under(root, true);

            if (count < 2) {
                return {-1, -1};
            }

            if (count == 2) {
                std::vector<int> leaves = active_indices_under(root, true);
                return {leaves[0], leaves[1]};
            }

            if (base.is_leaf(root)) {
                return {-1, -1};
            }

            int l = base.left(root);
            int r = base.right(root);

            if (!edge_above_deleted(l) && active_count_under(l, true) >= 2) {
                auto p = first_active_sibling_pair(l);
                if (p.first >= 0) {
                    return p;
                }
            }

            if (!edge_above_deleted(r) && active_count_under(r, true) >= 2) {
                auto p = first_active_sibling_pair(r);
                if (p.first >= 0) {
                    return p;
                }
            }

            return {-1, -1};
        }

        std::vector<std::uint32_t> component_labels_from_root(int root) const {
            std::vector<std::uint32_t> result;

            std::vector<int> stack;
            stack.push_back(root);

            while (!stack.empty()) {
                int u = stack.back();
                stack.pop_back();

                if (base.is_leaf(u)) {
                    result.push_back(base.label(u));
                    continue;
                }

                int l = base.left(u);
                int r = base.right(u);

                if (!edge_above_deleted(l)) {
                    stack.push_back(l);
                }

                if (!edge_above_deleted(r)) {
                    stack.push_back(r);
                }
            }

            std::sort(result.begin(), result.end());
            return result;
        }

        std::vector<int> component_roots() const {
            std::vector<int> roots;
            roots.push_back(base.root);

            for (int u = 0; u < base.node_count(); ++u) {
                if (u == base.root) {
                    continue;
                }

                if (edge_above_deleted(u)) {
                    roots.push_back(u);
                }
            }

            return roots;
        }
    };

    class Solver {
    public:
        Solver(
            const Tree& input_t1,
            const Tree& input_t2,
            const Options& input_options,
            const Timer* input_timer,
            Random* input_rng
        )
            : t1_ref(input_t1),
              t2_ref(input_t2),
              options(input_options),
              timer(input_timer),
              rng(input_rng),
              state(input_t1, input_t2),
              t1(input_t1, state, true),
              t2(input_t2, state, false) {}

        LabelForest run() {
            TimeBudget local_budget(dummy_or_global_timer(), options.local_time_limit_seconds);

            preprocess();

            while (state.active_count() > 0) {
                if (should_stop()) {
                    break;
                }

                if (compatible_active_leaves_of_node(t1, t1.base.root)) {
                    /*
                     * If all remaining active leaves are compatible, merge them by
                     * deactivating until one representative remains, then preprocess.
                     */
                    std::vector<int> active = t1.active_indices_under(t1.base.root, true);
                    for (std::size_t i = 1; i < active.size(); ++i) {
                        state.deactivate(active[i]);
                    }
                    preprocess();
                    continue;
                }

                int miass = find_minimal_incompatible_active_sibling_set(t1.base.root);

                if (miass < 0) {
                    break;
                }

                int child_a = t1.base.left(miass);
                int child_b = t1.base.right(miass);

                std::vector<int> active_a = t1.active_indices_under(child_a, true);
                std::vector<int> active_b = t1.active_indices_under(child_b, true);
                std::vector<int> active_all = t1.active_indices_under(miass, true);

                if (active_a.empty() || active_b.empty()) {
                    preprocess();
                    continue;
                }

                int node_to_resolve = child_a;
                int other_node = child_b;
                std::vector<int> set_to_resolve = active_a;
                std::vector<int> other_set = active_b;

                int lca_resolve_t2 = lca_in_tree_by_indices(t2, set_to_resolve);
                int lca_all_t2 = lca_in_tree_by_indices(t2, active_all);

                if (lca_resolve_t2 != lca_all_t2) {
                    node_to_resolve = child_b;
                    other_node = child_a;
                    set_to_resolve = active_b;
                    other_set = active_a;
                }

                bool resolved = resolve_set(node_to_resolve);

                if (!resolved) {
                    handle_resolve_set_failure(miass, node_to_resolve, other_node);
                }

                preprocess();
            }

            return build_forest_from_t1_cuts();
        }

        void safe_greedy_merge(LabelForest& forest) const {
            forest.normalize();
            forest.validate_partition_of(t1_ref.leaf_labels);

            bool changed = true;
            int rounds = 0;

            while (changed && rounds < 4) {
                changed = false;
                ++rounds;

                for (std::size_t i = 0; i < forest.components.size(); ++i) {
                    if (should_stop_const()) {
                        return;
                    }

                    for (std::size_t j = i + 1; j < forest.components.size(); ++j) {
                        std::vector<std::uint32_t> merged =
                            forest.components[i].labels;

                        merged.insert(
                            merged.end(),
                            forest.components[j].labels.begin(),
                            forest.components[j].labels.end()
                        );

                        std::sort(merged.begin(), merged.end());

                        if (!compatible_label_set(merged)) {
                            continue;
                        }

                        forest.components[i].labels = std::move(merged);
                        forest.components[i].normalize();

                        forest.components.erase(forest.components.begin() + static_cast<long>(j));

                        forest.normalize();
                        changed = true;
                        break;
                    }

                    if (changed) {
                        break;
                    }
                }
            }

            forest.validate_partition_of(t1_ref.leaf_labels);
        }

    private:
        const Tree& t1_ref;
        const Tree& t2_ref;
        const Options& options;
        const Timer* timer = nullptr;
        Random* rng = nullptr;

        ActiveState state;
        MutableTree t1;
        MutableTree t2;

        Timer fallback_timer{options.local_time_limit_seconds};

        const Timer& dummy_or_global_timer() const {
            if (timer != nullptr) {
                return *timer;
            }
            return fallback_timer;
        }

        bool should_stop() const {
            return timer != nullptr && timer->should_stop(options.guard_seconds);
        }

        bool should_stop_const() const {
            return timer != nullptr && timer->should_stop(options.guard_seconds);
        }

        int lca_in_tree_by_indices(const MutableTree& tree, const std::vector<int>& indices) const {
            if (indices.empty()) {
                return -1;
            }

            int l = tree.node_of_index(indices.front());

            for (std::size_t i = 1; i < indices.size(); ++i) {
                l = tree.base.lca(l, tree.node_of_index(indices[i]));
            }

            return l;
        }

        std::vector<int> convert_indices_to_nodes(
            const MutableTree& tree,
            const std::vector<int>& indices
        ) const {
            std::vector<int> nodes;
            nodes.reserve(indices.size());

            for (int idx : indices) {
                nodes.push_back(tree.node_of_index(idx));
            }

            return nodes;
        }

        int restricted_signature(
            const Tree& tree,
            const std::vector<char>& selected,
            SignatureInterner& interner
        ) const {
            std::vector<int> sig(static_cast<std::size_t>(tree.node_count()), 0);

            for (int u : tree.postorder) {
                if (tree.is_leaf(u)) {
                    int idx = state.index(tree.label(u));
                    if (selected[static_cast<std::size_t>(idx)]) {
                        sig[static_cast<std::size_t>(u)] =
                            interner.leaf(tree.label(u));
                    }
                } else {
                    int a = sig[static_cast<std::size_t>(tree.left(u))];
                    int b = sig[static_cast<std::size_t>(tree.right(u))];

                    if (a == 0 && b == 0) {
                        sig[static_cast<std::size_t>(u)] = 0;
                    } else if (a == 0) {
                        sig[static_cast<std::size_t>(u)] = b;
                    } else if (b == 0) {
                        sig[static_cast<std::size_t>(u)] = a;
                    } else {
                        sig[static_cast<std::size_t>(u)] =
                            interner.internal(a, b);
                    }
                }
            }

            return sig[static_cast<std::size_t>(tree.root)];
        }

        bool compatible_indices(const std::vector<int>& indices) const {
            if (indices.size() < 3) {
                return true;
            }

            std::vector<char> selected(state.labels.size(), 0);

            for (int idx : indices) {
                selected[static_cast<std::size_t>(idx)] = 1;
            }

            SignatureInterner interner;

            int s1 = restricted_signature(t1_ref, selected, interner);
            int s2 = restricted_signature(t2_ref, selected, interner);

            return s1 == s2;
        }

        bool compatible_label_set(const std::vector<std::uint32_t>& labels) const {
            if (labels.size() < 3) {
                return true;
            }

            std::vector<int> indices;
            indices.reserve(labels.size());

            for (std::uint32_t x : labels) {
                indices.push_back(state.index(x));
            }

            return compatible_indices(indices);
        }

        bool compatible_active_leaves_of_node(const MutableTree& tree, int node) const {
            std::vector<int> active = tree.active_indices_under(node, true);
            return compatible_indices(active);
        }

        void deactivate(int idx) {
            state.deactivate(idx);
        }

        void cut_label_in_t1_only(int idx) {
            int n1 = t1.lca_inactive_tree_from_leaf_node(t1.node_of_index(idx));
            t1.delete_edge_above(n1);
            deactivate(idx);
        }

        void cut_label_in_both(int idx) {
            int n1 = t1.lca_inactive_tree_from_leaf_node(t1.node_of_index(idx));
            int n2 = t2.lca_inactive_tree_from_leaf_node(t2.node_of_index(idx));

            t1.delete_edge_above(n1);
            t2.delete_edge_above(n2);

            deactivate(idx);
        }

        void preprocess() {
            bool changed = true;

            while (changed) {
                if (should_stop()) {
                    return;
                }

                changed = false;

                /*
                 * Merge common active siblings.
                 */
                while (true) {
                    auto pair = t1.first_active_sibling_pair(t1.base.root);
                    if (pair.first < 0) {
                        break;
                    }

                    bool did_merge = false;

                    /*
                     * Find all active sibling pairs in T1 by scanning nodes.
                     */
                    for (int u = 0; u < t1.base.node_count(); ++u) {
                        if (t1.base.is_leaf(u)) {
                            continue;
                        }

                        if (t1.edge_above_deleted(u)) {
                            continue;
                        }

                        if (t1.active_count_under(u, true) != 2) {
                            continue;
                        }

                        std::vector<int> leaves = t1.active_indices_under(u, true);
                        if (leaves.size() != 2) {
                            continue;
                        }

                        int a = leaves[0];
                        int b = leaves[1];

                        if (t2.active_sibling_with(a, b)) {
                            deactivate(a);
                            changed = true;
                            did_merge = true;
                            break;
                        }
                    }

                    if (!did_merge) {
                        break;
                    }
                }

                /*
                 * If a T2 active component contains only one active leaf, cut
                 * the corresponding inactive-tree edge in T1 and deactivate.
                 */
                std::vector<int> active_all = t2.active_indices_under(t2.base.root, false);

                for (int idx : active_all) {
                    if (!state.is_active(idx)) {
                        continue;
                    }

                    int root = t2.root_of_subtree_by_index(idx);

                    if (t2.active_count_under(root, true) == 1) {
                        int n1 = t1.lca_inactive_tree_from_leaf_node(t1.node_of_index(idx));
                        t1.delete_edge_above(n1);
                        deactivate(idx);
                        changed = true;
                    }

                    if (should_stop()) {
                        return;
                    }
                }
            }
        }

        int find_minimal_incompatible_active_sibling_set(int node) const {
            if (t1.base.is_leaf(node)) {
                return -1;
            }

            if (compatible_active_leaves_of_node(t1, node)) {
                return -1;
            }

            int l = t1.base.left(node);
            int r = t1.base.right(node);

            int lc = t1.edge_above_deleted(l) ? 0 : t1.active_count_under(l, true);
            int rc = t1.edge_above_deleted(r) ? 0 : t1.active_count_under(r, true);

            if (lc == 0 && rc > 0) {
                return find_minimal_incompatible_active_sibling_set(r);
            }

            if (rc == 0 && lc > 0) {
                return find_minimal_incompatible_active_sibling_set(l);
            }

            bool left_compatible = lc <= 1 || compatible_active_leaves_of_node(t1, l);
            bool right_compatible = rc <= 1 || compatible_active_leaves_of_node(t1, r);

            if (left_compatible && right_compatible) {
                return node;
            }

            if (!left_compatible) {
                return find_minimal_incompatible_active_sibling_set(l);
            }

            return find_minimal_incompatible_active_sibling_set(r);
        }

        bool resolve_pair(int u_idx, int v_idx) {
            if (should_stop()) {
                return false;
            }

            int u_t2 = t2.node_of_index(u_idx);
            int v_t2 = t2.node_of_index(v_idx);

            int root_u = t2.root_of_subtree(u_t2);
            int root_v = t2.root_of_subtree(v_t2);

            if (root_u != root_v) {
                int l = t2.base.lca(u_t2, v_t2);

                if (t2.root_of_subtree(l) == root_u) {
                    std::swap(u_idx, v_idx);
                    std::swap(u_t2, v_t2);
                    std::swap(root_u, root_v);
                }

                if (t2.active_count_under(root_u, true) > 1) {
                    cut_label_in_both(u_idx);
                    return true;
                }

                cut_label_in_t1_only(u_idx);
                return false;
            }

            int lca_uv = t2.base.lca(u_t2, v_t2);

            if (t2.active_count_under(lca_uv, true) == 2) {
                deactivate(u_idx);
                return false;
            }

            /*
             * Ensure u is not the direct active-parent side.
             */
            int p_u = t2.p_active(u_t2);
            if (p_u == lca_uv) {
                std::swap(u_idx, v_idx);
                std::swap(u_t2, v_t2);
            }

            int p = t2.p_active(u_t2);

            if (t2.base.is_leaf(p)) {
                cut_label_in_both(u_idx);
                return false;
            }

            int c1 = t2.base.left(p);
            int c2 = t2.base.right(p);

            bool u_in_c1 = t2.subtree_contains_active_index(c1, u_idx);
            int w = u_in_c1 ? c2 : c1;

            t2.delete_edge_above(w);

            std::vector<int> w_active = t2.active_indices_under(w, true);
            for (int x : w_active) {
                state.neighbor[static_cast<std::size_t>(x)] = u_idx;
            }

            if (t2.active_count_under(lca_uv, true) == 2) {
                deactivate(u_idx);
                return true;
            }

            cut_label_in_both(u_idx);
            return false;
        }

        bool resolve_set(int node_in_t1) {
            bool had_final_cut_or_merge_after_cut = false;

            while (t1.active_count_under(node_in_t1, true) >= 3) {
                if (should_stop()) {
                    return true;
                }

                auto pair = t1.first_active_sibling_pair(node_in_t1);

                if (pair.first < 0 || pair.second < 0) {
                    break;
                }

                had_final_cut_or_merge_after_cut |=
                    resolve_pair(pair.first, pair.second);
            }

            std::vector<int> remaining = t1.active_indices_under(node_in_t1, true);

            if (remaining.size() != 2) {
                return true;
            }

            int a = remaining[0];
            int b = remaining[1];

            int a_t2 = t2.node_of_index(a);
            int b_t2 = t2.node_of_index(b);

            int root_a = t2.root_of_subtree(a_t2);
            int root_b = t2.root_of_subtree(b_t2);
            int lca_ab = t2.base.lca(a_t2, b_t2);

            if (root_a == root_b && t2.active_count_under(lca_ab, true) == 2) {
                deactivate(a);
                return true;
            }

            if (root_a == root_b &&
                t2.active_count_under(root_a, true) != t2.active_count_under(lca_ab, true)) {
                if (had_final_cut_or_merge_after_cut) {
                    resolve_pair(a, b);

                    if (!state.is_active(b)) {
                        b = a;
                    }

                    cut_label_in_both(b);
                    return true;
                }

                return false;
            }

            cut_label_in_t1_only(a);
            if (t2.active_count_under(root_a, true) > 1) {
                int n2 = t2.lca_inactive_tree_from_leaf_node(t2.node_of_index(a));
                t2.delete_edge_above(n2);
            }
            deactivate(a);

            cut_label_in_t1_only(b);
            if (t2.active_count_under(root_b, true) > 1) {
                int n2 = t2.lca_inactive_tree_from_leaf_node(t2.node_of_index(b));
                t2.delete_edge_above(n2);
            }
            deactivate(b);

            return true;
        }

        void retroactive_merge_if_possible(int idx, int miass_node) {
            int neigh = state.neighbor[static_cast<std::size_t>(idx)];

            if (neigh < 0 || !state.index_of_label.size()) {
                return;
            }

            int idx_t1 = t1.node_of_index(idx);
            int neigh_t1 = t1.node_of_index(neigh);

            t1.make_path_by_flipping_edges(idx_t1, miass_node);
            t1.make_path_by_flipping_edges(neigh_t1, miass_node);

            int idx_t2 = t2.node_of_index(idx);
            int neigh_t2 = t2.node_of_index(neigh);
            int lca = t2.base.lca(idx_t2, neigh_t2);

            t2.make_path_by_flipping_edges(idx_t2, lca);
            t2.make_path_by_flipping_edges(neigh_t2, lca);
        }

        void handle_resolve_set_failure(
            int miass_node,
            int resolved_node,
            int other_node
        ) {
            if (should_stop()) {
                return;
            }

            bool had_final_cut_or_merge_after_cut = false;

            /*
             * Resolve the other child until one active leaf remains.
             */
            while (t1.active_count_under(other_node, true) >= 2) {
                auto pair = t1.first_active_sibling_pair(other_node);
                if (pair.first < 0 || pair.second < 0) {
                    break;
                }

                had_final_cut_or_merge_after_cut |=
                    resolve_pair(pair.first, pair.second);

                if (should_stop()) {
                    return;
                }
            }

            std::vector<int> two = t1.active_indices_under(resolved_node, true);
            std::vector<int> one = t1.active_indices_under(other_node, true);

            if (two.size() != 2 || one.size() != 1) {
                return;
            }

            int a = two[0];
            int b = two[1];
            int c = one[0];

            int root_a = t2.root_of_subtree_by_index(a);
            int root_b = t2.root_of_subtree_by_index(b);
            int root_c = t2.root_of_subtree_by_index(c);

            /*
             * Case 1: last 3 active leaves are in one T2 component.
             */
            if (root_a == root_b && root_b == root_c) {
                int lca_ac = t2.base.lca(t2.node_of_index(a), t2.node_of_index(c));
                int lca_bc = t2.base.lca(t2.node_of_index(b), t2.node_of_index(c));

                if (t2.base.depth[static_cast<std::size_t>(lca_ac)] <
                    t2.base.depth[static_cast<std::size_t>(lca_bc)]) {
                    std::swap(a, b);
                    std::swap(lca_ac, lca_bc);
                }

                int moved = t2.move_up_to_keep_invariant(lca_ac);
                t2.delete_edge_above(moved);

                cut_label_in_both(b);

                if (t2.active_count_under(moved, true) == 2) {
                    deactivate(c);
                } else {
                    cut_label_in_both(c);
                    cut_label_in_both(a);
                }

                return;
            }

            /*
             * Case 2: last 3 active leaves are in three T2 components.
             */
            if (root_a != root_b && root_a != root_c && root_b != root_c) {
                cut_label_in_both(c);

                bool retroactive_possible =
                    !had_final_cut_or_merge_after_cut &&
                    t2.active_count_under(root_a, true) == 1 &&
                    t2.active_count_under(root_b, true) == 1;

                if (t2.active_count_under(root_a, true) > 1) {
                    cut_label_in_both(a);
                } else {
                    cut_label_in_t1_only(a);
                }

                if (t2.active_count_under(root_b, true) > 1) {
                    cut_label_in_both(b);
                } else {
                    cut_label_in_t1_only(b);
                }

                if (retroactive_possible) {
                    retroactive_merge_if_possible(b, miass_node);
                }

                return;
            }

            /*
             * Case 3: last 3 active leaves are in two T2 components.
             */
            int p = b;
            int q = c;
            int isolated = a;

            if (root_a == root_b) {
                p = a;
                q = b;
                isolated = c;
            } else if (root_a == root_c) {
                p = a;
                q = c;
                isolated = b;
            }

            bool isolated_was_singleton_component = false;
            int isolated_root = t2.root_of_subtree_by_index(isolated);

            if (t2.active_count_under(isolated_root, true) == 1) {
                cut_label_in_t1_only(isolated);
                isolated_was_singleton_component = true;
            } else {
                cut_label_in_both(isolated);
            }

            int lca_pq = t2.base.lca(t2.node_of_index(p), t2.node_of_index(q));

            if (t2.active_count_under(lca_pq, true) == 2) {
                deactivate(p);
            } else {
                had_final_cut_or_merge_after_cut |= resolve_pair(p, q);

                if (!state.is_active(q)) {
                    q = p;
                }

                cut_label_in_both(q);

                if (!had_final_cut_or_merge_after_cut && isolated_was_singleton_component) {
                    retroactive_merge_if_possible(isolated, miass_node);
                }
            }
        }

        LabelForest build_forest_from_t1_cuts() const {
            LabelForest forest;
            std::unordered_set<std::uint32_t> seen_labels;
            seen_labels.reserve(t1_ref.leaf_labels.size() * 2);

            for (int root : t1.component_roots()) {
                std::vector<std::uint32_t> labels =
                    t1.component_labels_from_root(root);

                if (labels.empty()) {
                    continue;
                }

                for (std::uint32_t label : labels) {
                    seen_labels.insert(label);
                }
                forest.add_component(LabelComponent(std::move(labels)));
            }

            /*
             * Defensive fallback: any missed label becomes singleton.
             */
            for (std::uint32_t x : t1_ref.leaf_labels) {
                if (seen_labels.find(x) == seen_labels.end()) {
                    forest.components.emplace_back(x);
                }
            }

            forest.normalize();
            forest.validate_partition_of(t1_ref.leaf_labels);
            return forest;
        }
    };

    static void require_same_leaf_set(const Tree& t1, const Tree& t2) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw TwoApproxError("input trees do not have the same leaf set");
        }
    }

    Options options_;
};

}  // namespace pace26::heuristics
