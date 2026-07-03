#pragma once

#include "core/Forest.hpp"
#include "core/Random.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class LocalImproveError : public std::runtime_error {
public:
    explicit LocalImproveError(const std::string& message)
        : std::runtime_error("LocalImprove error: " + message) {}
};

class LocalImprove {
private:
    using Tree = pace26::core::Tree;
    using LabelComponent = pace26::core::LabelComponent;
    using LabelForest = pace26::core::LabelForest;
    using Timer = pace26::core::Timer;
    using Random = pace26::core::Random;

public:
    struct Options {
        int max_rounds = 6;

        bool try_singleton_reinsertion = true;
        bool try_pair_merges = true;
        bool try_structural_pair_merges = false;

        /*
         * Safety valve.
         *
         * Compatibility checking is O(n) in the current implementation.
         * Keep these finite for large instances.
         */
        std::size_t max_candidates_per_singleton = 256;
        std::size_t max_pair_tests_per_round = 20000;
        std::size_t max_structural_pair_tests_per_round = 128;
        std::size_t structural_neighbor_window = 4;

        /*
         * Avoid spending time trying to create giant components in the local improver.
         * Use 0 to disable this cap.
         */
        std::size_t max_merged_component_size = 0;

        double guard_seconds = 0.10;

        bool deterministic = true;
    };

    LocalImprove() : LocalImprove(Options{}) {}

    explicit LocalImprove(Options options)
        : options_(options) {}

    LabelForest improve(
        const Tree& t1,
        const Tree& t2,
        LabelForest forest,
        const Timer* timer = nullptr,
        Random* rng = nullptr
    ) const {
        require_same_leaf_set(t1, t2);

        forest.normalize();
        forest.validate_partition_of(t1.leaf_labels);

        if (timer != nullptr && timer->should_stop(options_.guard_seconds)) {
            return forest;
        }

        try {
            State state(t1, t2, std::move(forest), options_, timer, rng);

            for (int round = 0; round < options_.max_rounds; ++round) {
                if (state.should_stop()) {
                    break;
                }

                bool changed = false;

                if (options_.try_structural_pair_merges) {
                    changed |= state.try_structural_pair_merges();
                }

                if (state.should_stop()) {
                    break;
                }

                if (options_.try_singleton_reinsertion) {
                    changed |= state.try_singleton_reinsertion();
                }

                if (state.should_stop()) {
                    break;
                }

                if (options_.try_pair_merges) {
                    changed |= state.try_pair_merges();
                }

                if (!changed) {
                    break;
                }
            }

            LabelForest improved = state.to_forest();
            improved.normalize();
            improved.validate_partition_of(t1.leaf_labels);
            return improved;
        } catch (const std::exception&) {
            /*
             * Local improvement should never risk the whole solver.
             * Return the original valid forest if anything goes wrong.
             */
            forest.normalize();
            forest.validate_partition_of(t1.leaf_labels);
            return forest;
        }
    }

private:
    struct ComponentState {
        LabelComponent component;

        std::vector<int> edges1;
        std::vector<int> edges2;

        bool active = true;

        std::size_t size() const noexcept {
            return component.labels.size();
        }

        bool is_singleton() const noexcept {
            return active && component.labels.size() == 1;
        }
    };

    struct MergeCandidate {
        std::vector<std::uint32_t> labels;
        std::vector<int> edges1;
        std::vector<int> edges2;
    };

    class State {
    public:
        State(
            const Tree& input_t1,
            const Tree& input_t2,
            LabelForest input_forest,
            const Options& input_options,
            const Timer* input_timer,
            Random* input_rng
        )
            : t1(input_t1),
              t2(input_t2),
              options(input_options),
              timer(input_timer),
              rng(input_rng) {
            input_forest.normalize();
            input_forest.validate_partition_of(t1.leaf_labels);

            const std::size_t scratch_nodes = static_cast<std::size_t>(
                std::max(t1.node_count(), t2.node_count())
            );
            sig_hash1.assign(scratch_nodes, 0);
            sig_hash2.assign(scratch_nodes, 0);
            virtual_left_child.assign(scratch_nodes, -1);
            virtual_right_child.assign(scratch_nodes, -1);
            signature_nodes.reserve(static_cast<std::size_t>(t1.leaf_count()) * 2);
            signature_stack.reserve(static_cast<std::size_t>(t1.leaf_count()));
            leaf_nodes1.reserve(static_cast<std::size_t>(t1.leaf_count()));
            leaf_nodes2.reserve(static_cast<std::size_t>(t1.leaf_count()));
            initialise_components(std::move(input_forest));
        }

        bool should_stop() const {
            return timer != nullptr && timer->should_stop(options.guard_seconds);
        }

        bool try_singleton_reinsertion() {
            std::vector<int> singletons;
            std::vector<int> candidates;

            for (int id = 0; id < static_cast<int>(components.size()); ++id) {
                if (!components[static_cast<std::size_t>(id)].active) {
                    continue;
                }

                if (components[static_cast<std::size_t>(id)].is_singleton()) {
                    singletons.push_back(id);
                } else {
                    candidates.push_back(id);
                }
            }

            /*
             * If all components are singletons, allow singleton-singleton merges.
             */
            if (candidates.empty()) {
                candidates = singletons;
            }

            if (!options.deterministic && rng != nullptr) {
                rng->shuffle(singletons);
                rng->shuffle(candidates);
            } else {
                std::sort(
                    candidates.begin(),
                    candidates.end(),
                    [&](int a, int b) {
                        return components[static_cast<std::size_t>(a)].size() >
                               components[static_cast<std::size_t>(b)].size();
                    }
                );
            }

            bool changed = false;

            for (int singleton_id : singletons) {
                if (should_stop()) {
                    break;
                }

                if (!is_active(singleton_id) || !components[static_cast<std::size_t>(singleton_id)].is_singleton()) {
                    continue;
                }

                std::size_t tested = 0;

                for (int target_id : candidates) {
                    if (should_stop()) {
                        break;
                    }

                    if (singleton_id == target_id) {
                        continue;
                    }

                    if (!is_active(singleton_id) || !is_active(target_id)) {
                        continue;
                    }

                    if (tested >= options.max_candidates_per_singleton) {
                        break;
                    }

                    ++tested;

                    MergeCandidate candidate;
                    if (!can_merge(singleton_id, target_id, candidate)) {
                        continue;
                    }

                    apply_merge(singleton_id, target_id, std::move(candidate));
                    changed = true;
                    break;
                }
            }

            return changed;
        }

        bool try_pair_merges() {
            std::vector<int> ids = active_component_ids();

            if (!options.deterministic && rng != nullptr) {
                rng->shuffle(ids);
            } else {
                std::sort(
                    ids.begin(),
                    ids.end(),
                    [&](int a, int b) {
                        return components[static_cast<std::size_t>(a)].size() <
                               components[static_cast<std::size_t>(b)].size();
                    }
                );
            }

            bool changed = false;
            std::size_t tests = 0;

            for (std::size_t ai = 0; ai < ids.size(); ++ai) {
                if (should_stop() || tests >= options.max_pair_tests_per_round) {
                    break;
                }

                const int a = ids[ai];

                if (!is_active(a)) {
                    continue;
                }

                for (std::size_t bi = ai + 1; bi < ids.size(); ++bi) {
                    if (should_stop() || tests >= options.max_pair_tests_per_round) {
                        break;
                    }

                    const int b = ids[bi];

                    if (!is_active(b) || a == b) {
                        continue;
                    }

                    ++tests;

                    MergeCandidate candidate;
                    if (!can_merge(a, b, candidate)) {
                        continue;
                    }

                    apply_merge(a, b, std::move(candidate));
                    changed = true;
                    break;
                }
            }

            return changed;
        }

        bool try_structural_pair_merges() {
            struct Summary {
                int id = -1;
                int min1 = std::numeric_limits<int>::max();
                int max1 = std::numeric_limits<int>::min();
                int min2 = std::numeric_limits<int>::max();
                int max2 = std::numeric_limits<int>::min();
                std::size_t size = 0;
            };

            struct PairCandidate {
                int a = -1;
                int b = -1;
                int score = 0;
                std::size_t union_size = 0;
            };

            const std::vector<int> ids = active_component_ids();
            if (ids.size() < 2) {
                return false;
            }

            std::unordered_map<std::uint32_t, int> component_of_label;
            component_of_label.reserve(t1.leaf_labels.size() * 2);

            std::vector<Summary> summaries(components.size());
            for (int id : ids) {
                const ComponentState& component =
                    components[static_cast<std::size_t>(id)];
                Summary& summary = summaries[static_cast<std::size_t>(id)];
                summary.id = id;
                summary.size = component.size();

                for (std::uint32_t label : component.component.labels) {
                    component_of_label[label] = id;

                    const int node1 = t1.node_of_label(label);
                    const int node2 = t2.node_of_label(label);
                    const int pos1 = t1.tin[static_cast<std::size_t>(node1)];
                    const int pos2 = t2.tin[static_cast<std::size_t>(node2)];
                    summary.min1 = std::min(summary.min1, pos1);
                    summary.max1 = std::max(summary.max1, pos1);
                    summary.min2 = std::min(summary.min2, pos2);
                    summary.max2 = std::max(summary.max2, pos2);
                }
            }

            std::vector<PairCandidate> candidates;
            candidates.reserve(ids.size() * 2);
            std::unordered_set<std::uint64_t> seen;
            seen.reserve(ids.size() * 4);

            auto add_pair = [&](int a, int b, int score_bonus) {
                if (a == b || !is_active(a) || !is_active(b)) {
                    return;
                }
                if (a > b) {
                    std::swap(a, b);
                }

                const std::uint64_t key =
                    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
                    static_cast<std::uint32_t>(b);
                if (!seen.insert(key).second) {
                    return;
                }

                const Summary& sa = summaries[static_cast<std::size_t>(a)];
                const Summary& sb = summaries[static_cast<std::size_t>(b)];
                const std::size_t union_size = sa.size + sb.size;
                if (options.max_merged_component_size != 0 &&
                    union_size > options.max_merged_component_size) {
                    return;
                }

                const int min1 = std::min(sa.min1, sb.min1);
                const int max1 = std::max(sa.max1, sb.max1);
                const int min2 = std::min(sa.min2, sb.min2);
                const int max2 = std::max(sa.max2, sb.max2);
                const int gap1 = std::max(0, max1 - min1 + 1 - static_cast<int>(union_size));
                const int gap2 = std::max(0, max2 - min2 + 1 - static_cast<int>(union_size));

                candidates.push_back({
                    a,
                    b,
                    static_cast<int>(union_size) + 2 * (gap1 + gap2) + score_bonus,
                    union_size
                });
            };

            auto add_neighbor_pairs = [&](int tree_index) {
                std::vector<int> order = ids;
                std::sort(order.begin(), order.end(), [&](int a, int b) {
                    const Summary& sa = summaries[static_cast<std::size_t>(a)];
                    const Summary& sb = summaries[static_cast<std::size_t>(b)];
                    const int posa = tree_index == 1 ? sa.min1 : sa.min2;
                    const int posb = tree_index == 1 ? sb.min1 : sb.min2;
                    if (posa != posb) {
                        return posa < posb;
                    }
                    if (sa.size != sb.size) {
                        return sa.size < sb.size;
                    }
                    return a < b;
                });

                const std::size_t window =
                    std::max<std::size_t>(1, options.structural_neighbor_window);
                for (std::size_t i = 0; i < order.size(); ++i) {
                    const std::size_t end = std::min(order.size(), i + window + 1);
                    for (std::size_t j = i + 1; j < end; ++j) {
                        add_pair(order[i], order[j], 0);
                    }
                }
            };

            add_neighbor_pairs(1);
            add_neighbor_pairs(2);

            auto add_tree_witness_pairs = [&](const Tree& tree) {
                struct NodeSummary {
                    int leaf_count = 0;
                    int pure_component = -1;
                    std::vector<int> component_ids;
                };

                std::vector<NodeSummary> node_summary(
                    static_cast<std::size_t>(tree.node_count())
                );

                auto trim = [&](std::vector<int>& component_ids) {
                    std::sort(component_ids.begin(), component_ids.end());
                    component_ids.erase(
                        std::unique(component_ids.begin(), component_ids.end()),
                        component_ids.end()
                    );
                    std::sort(
                        component_ids.begin(),
                        component_ids.end(),
                        [&](int a, int b) {
                            const std::size_t sa =
                                summaries[static_cast<std::size_t>(a)].size;
                            const std::size_t sb =
                                summaries[static_cast<std::size_t>(b)].size;
                            if (sa != sb) {
                                return sa < sb;
                            }
                            return a < b;
                        }
                    );
                    if (component_ids.size() > 10) {
                        component_ids.resize(10);
                    }
                };

                for (int u : tree.postorder) {
                    NodeSummary& current =
                        node_summary[static_cast<std::size_t>(u)];

                    if (tree.is_leaf(u)) {
                        auto it = component_of_label.find(tree.label(u));
                        if (it != component_of_label.end()) {
                            current.leaf_count = 1;
                            current.pure_component = it->second;
                            current.component_ids.push_back(it->second);
                        }
                        continue;
                    }

                    const NodeSummary& left =
                        node_summary[static_cast<std::size_t>(tree.left(u))];
                    const NodeSummary& right =
                        node_summary[static_cast<std::size_t>(tree.right(u))];

                    current.leaf_count = left.leaf_count + right.leaf_count;
                    current.component_ids = left.component_ids;
                    current.component_ids.insert(
                        current.component_ids.end(),
                        right.component_ids.begin(),
                        right.component_ids.end()
                    );
                    trim(current.component_ids);
                    if (current.component_ids.size() == 1) {
                        current.pure_component = current.component_ids.front();
                    }

                    if (left.pure_component >= 0) {
                        for (int neighbor : right.component_ids) {
                            const int bonus = -5 - std::min(4, left.leaf_count) +
                                std::min(6, std::max(0, static_cast<int>(right.component_ids.size()) - 1)) -
                                (right.pure_component >= 0 ? 2 : 0);
                            add_pair(left.pure_component, neighbor, bonus);
                        }
                    }

                    if (right.pure_component >= 0) {
                        for (int neighbor : left.component_ids) {
                            const int bonus = -5 - std::min(4, right.leaf_count) +
                                std::min(6, std::max(0, static_cast<int>(left.component_ids.size()) - 1)) -
                                (left.pure_component >= 0 ? 2 : 0);
                            add_pair(right.pure_component, neighbor, bonus);
                        }
                    }
                }
            };

            add_tree_witness_pairs(t1);
            add_tree_witness_pairs(t2);

            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const PairCandidate& a, const PairCandidate& b) {
                    if (a.score != b.score) {
                        return a.score < b.score;
                    }
                    if (a.union_size != b.union_size) {
                        return a.union_size < b.union_size;
                    }
                    if (a.a != b.a) {
                        return a.a < b.a;
                    }
                    return a.b < b.b;
                }
            );

            std::size_t tests = 0;
            for (const PairCandidate& pair : candidates) {
                if (should_stop() ||
                    tests >= options.max_structural_pair_tests_per_round) {
                    break;
                }
                if (!is_active(pair.a) || !is_active(pair.b)) {
                    continue;
                }

                ++tests;
                MergeCandidate candidate;
                if (!can_merge(pair.a, pair.b, candidate)) {
                    continue;
                }

                apply_merge(pair.a, pair.b, std::move(candidate));
                return true;
            }

            return false;
        }

        LabelForest to_forest() const {
            LabelForest forest;

            for (const ComponentState& state : components) {
                if (!state.active) {
                    continue;
                }

                forest.add_component(state.component);
            }

            forest.normalize();
            forest.validate_partition_of(t1.leaf_labels);
            return forest;
        }

    private:
        const Tree& t1;
        const Tree& t2;
        const Options& options;
        const Timer* timer = nullptr;
        Random* rng = nullptr;

        std::vector<ComponentState> components;

        /*
         * edge_owner[child_node] = component id using edge parent(child_node)->child_node.
         * -1 means unused.
         *
         * Root edge is ignored.
         */
        std::vector<int> edge_owner1;
        std::vector<int> edge_owner2;

        struct SignatureValue {
            std::uint64_t first = 0;
            std::uint64_t second = 0;

            bool empty() const noexcept {
                return first == 0 && second == 0;
            }

            bool operator==(const SignatureValue& other) const noexcept {
                return first == other.first && second == other.second;
            }
        };

        mutable std::vector<std::uint64_t> sig_hash1;
        mutable std::vector<std::uint64_t> sig_hash2;
        mutable std::vector<int> virtual_left_child;
        mutable std::vector<int> virtual_right_child;
        mutable std::vector<int> signature_nodes;
        mutable std::vector<int> signature_stack;
        mutable std::vector<int> leaf_nodes1;
        mutable std::vector<int> leaf_nodes2;

        void initialise_components(LabelForest forest) {
            edge_owner1.assign(static_cast<std::size_t>(t1.node_count()), -1);
            edge_owner2.assign(static_cast<std::size_t>(t2.node_count()), -1);

            components.clear();
            components.reserve(forest.components.size());

            for (LabelComponent& component : forest.components) {
                component.normalize();

                if (component.empty()) {
                    continue;
                }

                if (!compatible(component.labels)) {
                    throw LocalImproveError("initial forest contains incompatible component");
                }

                ComponentState state;
                state.component = std::move(component);
                state.edges1 = induced_edges(t1, state.component.labels);
                state.edges2 = induced_edges(t2, state.component.labels);
                state.active = true;

                components.push_back(std::move(state));
            }

            for (int id = 0; id < static_cast<int>(components.size()); ++id) {
                mark_edges(id, components[static_cast<std::size_t>(id)].edges1, edge_owner1);
                mark_edges(id, components[static_cast<std::size_t>(id)].edges2, edge_owner2);
            }
        }

        bool is_active(int id) const {
            return id >= 0 &&
                   id < static_cast<int>(components.size()) &&
                   components[static_cast<std::size_t>(id)].active;
        }

        std::vector<int> active_component_ids() const {
            std::vector<int> ids;

            for (int id = 0; id < static_cast<int>(components.size()); ++id) {
                if (components[static_cast<std::size_t>(id)].active) {
                    ids.push_back(id);
                }
            }

            return ids;
        }

        void mark_edges(
            int component_id,
            const std::vector<int>& edges,
            std::vector<int>& owner
        ) {
            for (int edge_child : edges) {
                if (edge_child == t1.root || edge_child == t2.root) {
                    continue;
                }

                int& slot = owner[static_cast<std::size_t>(edge_child)];

                if (slot != -1 && slot != component_id) {
                    throw LocalImproveError("initial forest has overlapping induced edges");
                }

                slot = component_id;
            }
        }

        void unmark_edges(
            int component_id,
            const std::vector<int>& edges,
            std::vector<int>& owner
        ) {
            for (int edge_child : edges) {
                int& slot = owner[static_cast<std::size_t>(edge_child)];

                if (slot == component_id) {
                    slot = -1;
                }
            }
        }

        bool edges_available_for_merge(
            const std::vector<int>& edges,
            const std::vector<int>& owner,
            int a,
            int b
        ) const {
            for (int edge_child : edges) {
                const int slot = owner[static_cast<std::size_t>(edge_child)];

                if (slot != -1 && slot != a && slot != b) {
                    return false;
                }
            }

            return true;
        }

        static void sort_unique(std::vector<std::uint32_t>& labels) {
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        }

        bool can_merge(int a, int b, MergeCandidate& out) const {
            if (!is_active(a) || !is_active(b) || a == b) {
                return false;
            }

            const ComponentState& ca = components[static_cast<std::size_t>(a)];
            const ComponentState& cb = components[static_cast<std::size_t>(b)];

            out.labels = ca.component.labels;
            out.labels.insert(
                out.labels.end(),
                cb.component.labels.begin(),
                cb.component.labels.end()
            );

            sort_unique(out.labels);

            if (out.labels.size() != ca.component.labels.size() + cb.component.labels.size()) {
                return false;
            }

            if (options.max_merged_component_size != 0 &&
                out.labels.size() > options.max_merged_component_size) {
                return false;
            }

            if (!compatible(out.labels)) {
                return false;
            }

            out.edges1 = induced_edges(t1, out.labels);
            out.edges2 = induced_edges(t2, out.labels);

            if (!edges_available_for_merge(out.edges1, edge_owner1, a, b)) {
                return false;
            }

            if (!edges_available_for_merge(out.edges2, edge_owner2, a, b)) {
                return false;
            }

            return true;
        }

        void apply_merge(int keep_id, int remove_id, MergeCandidate candidate) {
            if (!is_active(keep_id) || !is_active(remove_id)) {
                throw LocalImproveError("attempted to merge inactive component");
            }

            ComponentState& keep = components[static_cast<std::size_t>(keep_id)];
            ComponentState& remove = components[static_cast<std::size_t>(remove_id)];

            unmark_edges(keep_id, keep.edges1, edge_owner1);
            unmark_edges(keep_id, keep.edges2, edge_owner2);
            unmark_edges(remove_id, remove.edges1, edge_owner1);
            unmark_edges(remove_id, remove.edges2, edge_owner2);

            keep.component.labels = std::move(candidate.labels);
            keep.component.normalize();
            keep.edges1 = std::move(candidate.edges1);
            keep.edges2 = std::move(candidate.edges2);

            remove.component.labels.clear();
            remove.edges1.clear();
            remove.edges2.clear();
            remove.active = false;

            mark_edges(keep_id, keep.edges1, edge_owner1);
            mark_edges(keep_id, keep.edges2, edge_owner2);
        }

        std::vector<int> induced_edges(
            const Tree& tree,
            const std::vector<std::uint32_t>& labels
        ) const {
            std::vector<int> edges;

            if (labels.size() <= 1) {
                return edges;
            }

            int root = tree.node_of_label(labels.front());

            for (std::size_t i = 1; i < labels.size(); ++i) {
                root = tree.lca(root, tree.node_of_label(labels[i]));
            }

            for (std::uint32_t label : labels) {
                int u = tree.node_of_label(label);

                while (u != root) {
                    edges.push_back(u);
                    u = tree.parent[static_cast<std::size_t>(u)];

                    if (u < 0) {
                        throw LocalImproveError("invalid parent path while computing induced edges");
                    }
                }
            }

            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

            return edges;
        }

        static std::uint64_t mix64(std::uint64_t x) noexcept {
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ULL;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebULL;
            x ^= x >> 31;
            return x;
        }

        static SignatureValue leaf_signature(std::uint32_t label) noexcept {
            const std::uint64_t x = static_cast<std::uint64_t>(label);
            return {
                mix64(0x51ed270bULL ^ x),
                mix64(0xd1b54a32d192ed03ULL ^ (x + 0x9e3779b97f4a7c15ULL))
            };
        }

        static SignatureValue internal_signature(
            SignatureValue a,
            SignatureValue b
        ) noexcept {
            if (b.first < a.first ||
                (b.first == a.first && b.second < a.second)) {
                std::swap(a, b);
            }
            return {
                mix64(0xa0761d6478bd642fULL ^ a.first ^ mix64(b.first)),
                mix64(0xe7037ed1a0b428dbULL ^ a.second ^ mix64(b.second))
            };
        }

        SignatureValue restricted_signature(
            const Tree& tree,
            const std::vector<int>& leaf_nodes
        ) const {
            if (leaf_nodes.empty()) {
                return {};
            }

            signature_nodes = leaf_nodes;
            for (std::size_t i = 0; i + 1 < leaf_nodes.size(); ++i) {
                signature_nodes.push_back(tree.lca(leaf_nodes[i], leaf_nodes[i + 1]));
            }

            std::sort(signature_nodes.begin(), signature_nodes.end(), [&](int a, int b) {
                return tree.tin[static_cast<std::size_t>(a)] < tree.tin[static_cast<std::size_t>(b)];
            });
            signature_nodes.erase(
                std::unique(signature_nodes.begin(), signature_nodes.end()),
                signature_nodes.end()
            );

            signature_stack.clear();
            for (int u : signature_nodes) {
                while (!signature_stack.empty() && !tree.is_ancestor(signature_stack.back(), u)) {
                    signature_stack.pop_back();
                }
                if (!signature_stack.empty()) {
                    int p = signature_stack.back();
                    if (virtual_left_child[static_cast<std::size_t>(p)] == -1) {
                        virtual_left_child[static_cast<std::size_t>(p)] = u;
                    } else {
                        virtual_right_child[static_cast<std::size_t>(p)] = u;
                    }
                }
                signature_stack.push_back(u);
            }

            for (int i = static_cast<int>(signature_nodes.size()) - 1; i >= 0; --i) {
                int u = signature_nodes[static_cast<std::size_t>(i)];
                if (tree.is_leaf(u)) {
                    const SignatureValue leaf = leaf_signature(tree.label(u));
                    sig_hash1[static_cast<std::size_t>(u)] = leaf.first;
                    sig_hash2[static_cast<std::size_t>(u)] = leaf.second;
                } else {
                    int left = virtual_left_child[static_cast<std::size_t>(u)];
                    int right = virtual_right_child[static_cast<std::size_t>(u)];

                    SignatureValue left_sig;
                    if (left != -1) {
                        left_sig.first = sig_hash1[static_cast<std::size_t>(left)];
                        left_sig.second = sig_hash2[static_cast<std::size_t>(left)];
                    }

                    SignatureValue right_sig;
                    if (right != -1) {
                        right_sig.first = sig_hash1[static_cast<std::size_t>(right)];
                        right_sig.second = sig_hash2[static_cast<std::size_t>(right)];
                    }

                    SignatureValue current;
                    if (left_sig.empty() && right_sig.empty()) {
                        current = {};
                    } else if (left_sig.empty()) {
                        current = right_sig;
                    } else if (right_sig.empty()) {
                        current = left_sig;
                    } else {
                        current = internal_signature(left_sig, right_sig);
                    }

                    sig_hash1[static_cast<std::size_t>(u)] = current.first;
                    sig_hash2[static_cast<std::size_t>(u)] = current.second;
                }
            }

            const int root = signature_nodes.front();
            SignatureValue result{
                sig_hash1[static_cast<std::size_t>(root)],
                sig_hash2[static_cast<std::size_t>(root)]
            };

            for (int u : signature_nodes) {
                sig_hash1[static_cast<std::size_t>(u)] = 0;
                sig_hash2[static_cast<std::size_t>(u)] = 0;
                virtual_left_child[static_cast<std::size_t>(u)] = -1;
                virtual_right_child[static_cast<std::size_t>(u)] = -1;
            }

            return result;
        }

        bool compatible(const std::vector<std::uint32_t>& labels) const {
            if (labels.size() <= 2) {
                return true;
            }

            leaf_nodes1.clear();
            for (std::uint32_t label : labels) {
                leaf_nodes1.push_back(t1.node_of_label(label));
            }
            std::sort(leaf_nodes1.begin(), leaf_nodes1.end(), [&](int a, int b) {
                return t1.tin[static_cast<std::size_t>(a)] < t1.tin[static_cast<std::size_t>(b)];
            });

            leaf_nodes2.clear();
            for (std::uint32_t label : labels) {
                leaf_nodes2.push_back(t2.node_of_label(label));
            }
            std::sort(leaf_nodes2.begin(), leaf_nodes2.end(), [&](int a, int b) {
                return t2.tin[static_cast<std::size_t>(a)] < t2.tin[static_cast<std::size_t>(b)];
            });

            const SignatureValue sig1 = restricted_signature(t1, leaf_nodes1);
            const SignatureValue sig2 = restricted_signature(t2, leaf_nodes2);

            return !sig1.empty() && sig1 == sig2;
        }
    };

    static void require_same_leaf_set(const Tree& t1, const Tree& t2) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw LocalImproveError("input trees do not have the same leaf set");
        }
    }

    Options options_;
};

}  // namespace pace26::heuristics
