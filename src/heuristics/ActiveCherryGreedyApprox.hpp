#pragma once

#include "core/Forest.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class ActiveCherryGreedyApproxError : public std::runtime_error {
public:
    explicit ActiveCherryGreedyApproxError(const std::string& message)
        : std::runtime_error("ActiveCherryGreedyApprox error: " + message) {}
};

class ActiveCherryGreedyApprox {
private:
    using Tree = pace26::core::Tree;
    using LabelForest = pace26::core::LabelForest;
    using LabelComponent = pace26::core::LabelComponent;
    using Timer = pace26::core::Timer;

public:
    enum class Policy {
        Balanced,
        PreferSmallCuts,
        PreferBigProgress,
        PreferDifferentComponent,
        PreferLowConflictMass,
        PreferFewPendants,
        PreferImmediateGain,
        ConservativeSingleCut,
        DualityConservative,
        ResolveFinalCut,
        AggressiveMultiCut
    };

    struct Options {
        Policy policy = Policy::Balanced;

        /*
         * The loop is O(number of performed cuts/contracts).
         * Keep this finite so it never eats the whole 30s run.
         */
        std::size_t max_steps_multiplier = 6;

        /*
         * Only inspect the first K active cherries at each step.
         * This is why the heuristic stays fast on 10k+ leaves.
         */
        std::size_t candidate_sample_cap = 64;

        double guard_seconds = 0.25;

        double local_time_limit_seconds = 0.0;
        std::uint64_t sample_salt = 0;
        std::vector<int> rank_choice_script;

        std::size_t publish_interval_steps = 0;
        std::function<void(const LabelForest&)> publish_candidate;
    };

    ActiveCherryGreedyApprox() : ActiveCherryGreedyApprox(Options{}) {}

    explicit ActiveCherryGreedyApprox(Options options)
        : options_(options) {}

    LabelForest solve(const Tree& t1, const Tree& t2, const Timer* timer = nullptr) const {
        require_same_leaf_set(t1, t2);

        DynamicTree active_t1 = DynamicTree::from_tree(t1);
        DynamicTree forest_t2 = DynamicTree::from_tree(t2);

        std::uint32_t next_label = next_pseudo_label(t1, t2);
        const Timer::Clock::time_point local_start = Timer::Clock::now();

        auto should_stop_now = [&]() {
            if (timer != nullptr && timer->should_stop(options_.guard_seconds)) {
                return true;
            }

            if (options_.local_time_limit_seconds <= 0.0) {
                return false;
            }

            const auto now = Timer::Clock::now();
            const std::chrono::duration<double> elapsed = now - local_start;
            return elapsed.count() >= options_.local_time_limit_seconds;
        };

        contract_all_common_cherries(active_t1, forest_t2, next_label, timer);
        publish_progress(forest_t2, t1.leaf_labels);

        const std::size_t max_steps =
            std::max<std::size_t>(32, options_.max_steps_multiplier * t1.leaf_labels.size());

        std::size_t steps = 0;
        std::size_t discrepancy_step = 0;
        PathScratch scratch;
        MassScratch mass;

        while (active_t1.active_leaf_count > 2 && steps++ < max_steps) {
            if (should_stop_now()) {
                break;
            }

            remove_resolved_singleton_roots_from_t1(active_t1, forest_t2);
            contract_all_common_cherries(active_t1, forest_t2, next_label, timer);

            if (active_t1.active_leaf_count <= 2) {
                break;
            }

            std::size_t sample_start = 0;
            if (options_.sample_salt != 0 && !active_t1.nodes.empty()) {
                const std::uint64_t key =
                    mix64(options_.sample_salt ^
                          (steps * 0x9e3779b97f4a7c15ULL) ^
                          static_cast<std::uint64_t>(active_t1.active_leaf_count));
                sample_start = static_cast<std::size_t>(key % active_t1.nodes.size());
            }

            std::vector<std::pair<std::uint32_t, std::uint32_t>> cherries =
                active_t1.cherries_by_label(options_.candidate_sample_cap, sample_start);

            if (cherries.empty()) {
                break;
            }

            mass.start(forest_t2.nodes.size());

            Candidate best;
            bool have_best = false;
            const int scoring_total = options_.policy == Policy::DualityConservative
                ? static_cast<int>(t1.leaf_labels.size())
                : active_t1.active_leaf_count;

            if (discrepancy_step < options_.rank_choice_script.size()) {
                std::vector<Candidate> ranked;
                ranked.reserve(cherries.size());

                for (const auto& [a, b] : cherries) {
                    Candidate cand = build_candidate(forest_t2, mass, scratch, a, b);

                    if (!cand.usable) {
                        continue;
                    }

                    cand.score = score_candidate(cand, scoring_total);
                    ranked.push_back(std::move(cand));
                }

                if (ranked.empty()) {
                    break;
                }

                std::sort(ranked.begin(), ranked.end(), candidate_better);

                const int requested = std::max(0, options_.rank_choice_script[discrepancy_step]);
                const std::size_t choice = std::min<std::size_t>(
                    static_cast<std::size_t>(requested),
                    ranked.size() - 1
                );
                ++discrepancy_step;
                best = std::move(ranked[choice]);
                have_best = true;
            } else {
                for (const auto& [a, b] : cherries) {
                    Candidate cand = build_candidate(forest_t2, mass, scratch, a, b);

                    if (!cand.usable) {
                        continue;
                    }

                    cand.score = score_candidate(cand, scoring_total);

                    if (!have_best || candidate_better(cand, best)) {
                        best = std::move(cand);
                        have_best = true;
                    }
                }
            }

            if (!have_best) {
                break;
            }

            const bool changed = apply_candidate(active_t1, forest_t2, best, mass);

            if (!changed) {
                break;
            }

            contract_all_common_cherries(active_t1, forest_t2, next_label, timer);
            if (options_.publish_interval_steps != 0 &&
                steps % options_.publish_interval_steps == 0) {
                publish_progress(forest_t2, t1.leaf_labels);
            }
        }

        LabelForest out = forest_t2.to_label_forest(t1.leaf_labels);
        out.normalize();
        out.validate_partition_of(t1.leaf_labels);
        return out;
    }

private:
    struct PayloadNode {
        int left = -1;
        int right = -1;
        std::uint32_t leaf = 0;
    };

    struct DynNode {
        int parent = -1;
        int left = -1;
        int right = -1;

        /*
         * label > 0 means active pseudo-leaf.
         * label == 0 means internal node.
         */
        std::uint32_t label = 0;

        int payload_id = -1;
        int payload_size = 0;

        bool active = true;
    };

    struct PathScratch {
        std::vector<int> mark;
        std::vector<int> pos;
        std::vector<int> up_a;
        std::vector<int> up_b;
        std::vector<int> path;
        int stamp = 1;

        void ensure(std::size_t n) {
            if (mark.size() < n) {
                mark.assign(n, 0);
                pos.assign(n, -1);
            }
        }
    };

    struct MassScratch {
        std::vector<int> mark;
        std::vector<int> value;
        int stamp = 1;

        void start(std::size_t n) {
            if (mark.size() < n) {
                mark.assign(n, 0);
                value.assign(n, 0);
            }

            ++stamp;
            if (stamp == std::numeric_limits<int>::max()) {
                std::fill(mark.begin(), mark.end(), 0);
                stamp = 1;
            }
        }
    };

    class DynamicTree {
    public:
        int root = -1;
        int active_leaf_count = 0;
        int root_component_count = 1;

        std::vector<DynNode> nodes;
        std::vector<PayloadNode> payloads;
        std::unordered_map<std::uint32_t, int> label_to_node;
        std::vector<int> label_to_node_flat;

        static DynamicTree from_tree(const Tree& tree) {
            DynamicTree out;
            out.root = tree.root;
            out.nodes.resize(static_cast<std::size_t>(tree.node_count()));
            out.payloads.reserve(tree.leaf_labels.size() * 2 + 1);
            out.label_to_node.reserve(tree.leaf_labels.size() * 4 + 1);

            std::uint32_t max_label = 0;
            for (std::uint32_t label : tree.leaf_labels) {
                max_label = std::max(max_label, label);
            }
            const std::uint64_t flat_size =
                static_cast<std::uint64_t>(max_label) +
                static_cast<std::uint64_t>(tree.leaf_labels.size()) + 4ULL;
            const std::uint64_t flat_limit = std::max<std::uint64_t>(
                1000000ULL,
                static_cast<std::uint64_t>(tree.leaf_labels.size()) * 8ULL + 1024ULL
            );
            if (flat_size <= flat_limit) {
                out.label_to_node_flat.assign(
                    static_cast<std::size_t>(flat_size),
                    -1
                );
            }

            for (int u = 0; u < tree.node_count(); ++u) {
                DynNode node;
                node.parent = tree.parent[static_cast<std::size_t>(u)];
                node.active = true;

                if (tree.is_leaf(u)) {
                    node.label = tree.label(u);
                    node.left = -1;
                    node.right = -1;
                    node.payload_id = out.push_leaf_payload(tree.label(u));
                    node.payload_size = 1;

                    out.remember_label_node(node.label, u);
                    ++out.active_leaf_count;
                } else {
                    node.label = 0;
                    node.left = tree.left(u);
                    node.right = tree.right(u);
                    node.payload_id = -1;
                    node.payload_size = 0;
                }

                out.nodes[static_cast<std::size_t>(u)] = node;
            }

            out.root_component_count = 1;
            return out;
        }

        void remember_label_node(std::uint32_t label, int node) {
            label_to_node[label] = node;
            if (!label_to_node_flat.empty()) {
                if (label >= label_to_node_flat.size()) {
                    const std::size_t wanted = static_cast<std::size_t>(label) + 1;
                    if (wanted <= label_to_node_flat.size() * 2 &&
                        wanted <= 2000000) {
                        label_to_node_flat.resize(wanted, -1);
                    }
                }
                if (label < label_to_node_flat.size()) {
                    label_to_node_flat[static_cast<std::size_t>(label)] = node;
                }
            }
        }

        void forget_label_node(std::uint32_t label) {
            label_to_node.erase(label);
            if (label < label_to_node_flat.size()) {
                label_to_node_flat[static_cast<std::size_t>(label)] = -1;
            }
        }

        int push_leaf_payload(std::uint32_t label) {
            PayloadNode p;
            p.leaf = label;
            payloads.push_back(p);
            return static_cast<int>(payloads.size()) - 1;
        }

        int push_join_payload(int left_payload, int right_payload) {
            PayloadNode p;
            p.left = left_payload;
            p.right = right_payload;
            p.leaf = 0;
            payloads.push_back(p);
            return static_cast<int>(payloads.size()) - 1;
        }

        bool is_active_node(int u) const {
            return u >= 0 &&
                   u < static_cast<int>(nodes.size()) &&
                   nodes[static_cast<std::size_t>(u)].active;
        }

        bool is_leaflike(int u) const {
            if (!is_active_node(u)) {
                return false;
            }

            const DynNode& x = nodes[static_cast<std::size_t>(u)];
            return x.left == -1 && x.right == -1;
        }

        int node_of_label(std::uint32_t label) const {
            int u = -1;
            if (label < label_to_node_flat.size()) {
                u = label_to_node_flat[static_cast<std::size_t>(label)];
            } else {
                auto it = label_to_node.find(label);

                if (it == label_to_node.end()) {
                    return -1;
                }

                u = it->second;
            }

            if (!is_leaflike(u)) {
                return -1;
            }

            return u;
        }

        int root_of(int u) const {
            if (!is_active_node(u)) {
                return -1;
            }

            int x = u;

            while (x != -1 &&
                   is_active_node(x) &&
                   nodes[static_cast<std::size_t>(x)].parent != -1) {
                x = nodes[static_cast<std::size_t>(x)].parent;
            }

            return x;
        }

        bool are_siblings_by_label(std::uint32_t a_label, std::uint32_t b_label) const {
            int a = node_of_label(a_label);
            int b = node_of_label(b_label);

            if (a < 0 || b < 0) {
                return false;
            }

            int pa = nodes[static_cast<std::size_t>(a)].parent;
            int pb = nodes[static_cast<std::size_t>(b)].parent;

            if (pa < 0 || pa != pb || !is_active_node(pa)) {
                return false;
            }

            const DynNode& p = nodes[static_cast<std::size_t>(pa)];

            return (p.left == a && p.right == b) ||
                   (p.left == b && p.right == a);
        }

        std::vector<std::pair<std::uint32_t, std::uint32_t>>
        cherries_by_label(std::size_t sample_cap, std::size_t start_offset = 0) const {
            std::vector<std::pair<std::uint32_t, std::uint32_t>> out;

            out.reserve(
                std::min<std::size_t>(
                    nodes.size(),
                    sample_cap == 0 ? nodes.size() : sample_cap
                )
            );

            const std::size_t node_count = nodes.size();
            if (node_count == 0) {
                return out;
            }

            start_offset %= node_count;

            auto maybe_add_cherry = [&](int u) {
                if (!is_active_node(u)) {
                    return false;
                }

                const DynNode& n = nodes[static_cast<std::size_t>(u)];

                if (n.left < 0 || n.right < 0) {
                    return false;
                }

                if (!is_leaflike(n.left) || !is_leaflike(n.right)) {
                    return false;
                }

                std::uint32_t a = nodes[static_cast<std::size_t>(n.left)].label;
                std::uint32_t b = nodes[static_cast<std::size_t>(n.right)].label;

                if (a > b) {
                    std::swap(a, b);
                }

                out.push_back({a, b});
                return sample_cap != 0 && out.size() >= sample_cap;
            };

            if (start_offset == 0) {
                for (int u = 0; u < static_cast<int>(node_count); ++u) {
                    if (maybe_add_cherry(u)) {
                        break;
                    }
                }

                return out;
            }

            for (std::size_t step = 0; step < node_count; ++step) {
                const int u = static_cast<int>((start_offset + step) % node_count);

                if (maybe_add_cherry(u)) {
                    break;
                }
            }

            return out;
        }

        bool contract_cherry(
            std::uint32_t a_label,
            std::uint32_t b_label,
            std::uint32_t new_label
        ) {
            int a = node_of_label(a_label);
            int b = node_of_label(b_label);

            if (a < 0 || b < 0) {
                return false;
            }

            int p = nodes[static_cast<std::size_t>(a)].parent;

            if (p < 0 ||
                p != nodes[static_cast<std::size_t>(b)].parent ||
                !is_active_node(p)) {
                return false;
            }

            DynNode& parent = nodes[static_cast<std::size_t>(p)];

            if (!((parent.left == a && parent.right == b) ||
                  (parent.left == b && parent.right == a))) {
                return false;
            }

            const int payload_a = nodes[static_cast<std::size_t>(a)].payload_id;
            const int payload_b = nodes[static_cast<std::size_t>(b)].payload_id;

            const int size_a = nodes[static_cast<std::size_t>(a)].payload_size;
            const int size_b = nodes[static_cast<std::size_t>(b)].payload_size;

            nodes[static_cast<std::size_t>(a)].active = false;
            nodes[static_cast<std::size_t>(b)].active = false;

            nodes[static_cast<std::size_t>(a)].parent = -1;
            nodes[static_cast<std::size_t>(b)].parent = -1;

            forget_label_node(a_label);
            forget_label_node(b_label);

            parent.left = -1;
            parent.right = -1;
            parent.label = new_label;
            parent.payload_id = push_join_payload(payload_a, payload_b);
            parent.payload_size = size_a + size_b;
            parent.active = true;

            remember_label_node(new_label, p);

            --active_leaf_count;
            return true;
        }

        int active_child_count(int u, int* only_child = nullptr) const {
            int count = 0;
            int child = -1;

            const DynNode& n = nodes[static_cast<std::size_t>(u)];

            if (n.left >= 0 && is_active_node(n.left)) {
                ++count;
                child = n.left;
            }

            if (n.right >= 0 && is_active_node(n.right)) {
                ++count;
                child = n.right;
            }

            if (only_child != nullptr) {
                *only_child = child;
            }

            return count;
        }

        void replace_child(int parent, int old_child, int new_child) {
            if (parent < 0 || !is_active_node(parent)) {
                return;
            }

            DynNode& p = nodes[static_cast<std::size_t>(parent)];

            if (p.left == old_child) {
                p.left = new_child;
            } else if (p.right == old_child) {
                p.right = new_child;
            }

            if (new_child >= 0) {
                nodes[static_cast<std::size_t>(new_child)].parent = parent;
            }
        }

        void suppress_up(int u) {
            while (u >= 0 && is_active_node(u) && !is_leaflike(u)) {
                int child = -1;
                const int degree = active_child_count(u, &child);

                if (degree >= 2) {
                    break;
                }

                const int parent = nodes[static_cast<std::size_t>(u)].parent;

                if (degree == 1) {
                    if (parent >= 0) {
                        replace_child(parent, u, child);
                    } else {
                        nodes[static_cast<std::size_t>(child)].parent = -1;

                        if (root == u) {
                            root = child;
                        }
                    }
                } else {
                    if (parent >= 0) {
                        replace_child(parent, u, -1);
                    } else if (root == u) {
                        root = -1;
                    }
                }

                nodes[static_cast<std::size_t>(u)].active = false;
                nodes[static_cast<std::size_t>(u)].left = -1;
                nodes[static_cast<std::size_t>(u)].right = -1;
                nodes[static_cast<std::size_t>(u)].parent = -1;

                u = parent;
            }
        }

        bool cut_edge_above(int child) {
            if (!is_active_node(child)) {
                return false;
            }

            const int parent = nodes[static_cast<std::size_t>(child)].parent;

            if (parent < 0 || !is_active_node(parent)) {
                return false;
            }

            replace_child(parent, child, -1);

            nodes[static_cast<std::size_t>(child)].parent = -1;

            ++root_component_count;

            suppress_up(parent);

            return true;
        }

        const std::vector<int>& path_nodes(int a, int b, PathScratch& scratch) const {
            scratch.ensure(nodes.size());

            ++scratch.stamp;

            if (scratch.stamp == std::numeric_limits<int>::max()) {
                std::fill(scratch.mark.begin(), scratch.mark.end(), 0);
                scratch.stamp = 1;
            }

            scratch.up_a.clear();
            scratch.up_b.clear();
            scratch.path.clear();

            int x = a;

            while (x != -1) {
                scratch.mark[static_cast<std::size_t>(x)] = scratch.stamp;
                scratch.pos[static_cast<std::size_t>(x)] =
                    static_cast<int>(scratch.up_a.size());

                scratch.up_a.push_back(x);

                x = nodes[static_cast<std::size_t>(x)].parent;
            }

            x = b;
            int lca = -1;

            while (x != -1) {
                if (scratch.mark[static_cast<std::size_t>(x)] == scratch.stamp) {
                    lca = x;
                    break;
                }

                scratch.up_b.push_back(x);

                x = nodes[static_cast<std::size_t>(x)].parent;
            }

            if (lca < 0) {
                return scratch.path;
            }

            const int upto = scratch.pos[static_cast<std::size_t>(lca)];

            for (int i = 0; i <= upto; ++i) {
                scratch.path.push_back(scratch.up_a[static_cast<std::size_t>(i)]);
            }

            for (int i = static_cast<int>(scratch.up_b.size()) - 1; i >= 0; --i) {
                scratch.path.push_back(scratch.up_b[static_cast<std::size_t>(i)]);
            }

            return scratch.path;
        }

        void collect_pendant_children_on_path(
            const std::vector<int>& path,
            std::vector<int>& out
        ) const {
            out.clear();

            if (path.size() < 3) {
                return;
            }

            for (std::size_t i = 1; i + 1 < path.size(); ++i) {
                const int u = path[i];
                const int prev = path[i - 1];
                const int next = path[i + 1];

                const DynNode& n = nodes[static_cast<std::size_t>(u)];

                if (n.left >= 0 &&
                    is_active_node(n.left) &&
                    n.left != prev &&
                    n.left != next) {
                    out.push_back(n.left);
                }

                if (n.right >= 0 &&
                    is_active_node(n.right) &&
                    n.right != prev &&
                    n.right != next) {
                    out.push_back(n.right);
                }
            }
        }

        int active_mass(int u, MassScratch& scratch) const {
            if (!is_active_node(u)) {
                return 0;
            }

            const std::size_t idx = static_cast<std::size_t>(u);
            if (scratch.mark[idx] == scratch.stamp) {
                return scratch.value[idx];
            }

            const DynNode& n = nodes[idx];
            int total = 0;
            if (is_leaflike(u)) {
                total = n.payload_size;
            } else {
                if (n.left >= 0) {
                    total += active_mass(n.left, scratch);
                }
                if (n.right >= 0) {
                    total += active_mass(n.right, scratch);
                }
            }

            scratch.mark[idx] = scratch.stamp;
            scratch.value[idx] = total;
            return total;
        }

        int active_mass_or_one(int u, MassScratch& scratch) const {
            if (u < 0 || u >= static_cast<int>(nodes.size())) {
                return 1;
            }
            const int mass = active_mass(u, scratch);
            return mass > 0 ? mass : 1;
        }

        void collect_payload_leaves(
            int payload_id,
            std::vector<std::uint32_t>& out
        ) const {
            if (payload_id < 0 ||
                payload_id >= static_cast<int>(payloads.size())) {
                return;
            }

            const PayloadNode& p = payloads[static_cast<std::size_t>(payload_id)];

            if (p.leaf != 0) {
                out.push_back(p.leaf);
                return;
            }

            collect_payload_leaves(p.left, out);
            collect_payload_leaves(p.right, out);
        }

        void collect_component_leaves(
            int u,
            std::vector<std::uint32_t>& out
        ) const {
            if (!is_active_node(u)) {
                return;
            }

            const DynNode& n = nodes[static_cast<std::size_t>(u)];

            if (is_leaflike(u)) {
                collect_payload_leaves(n.payload_id, out);
                return;
            }

            if (n.left >= 0) {
                collect_component_leaves(n.left, out);
            }

            if (n.right >= 0) {
                collect_component_leaves(n.right, out);
            }
        }

        LabelForest to_label_forest(
            const std::vector<std::uint32_t>& expected_labels
        ) const {
            LabelForest forest;
            forest.components.reserve(static_cast<std::size_t>(root_component_count));
            std::unordered_set<std::uint32_t> seen_labels;
            seen_labels.reserve(expected_labels.size() * 2);

            for (int u = 0; u < static_cast<int>(nodes.size()); ++u) {
                if (!is_active_node(u) ||
                    nodes[static_cast<std::size_t>(u)].parent != -1) {
                    continue;
                }

                std::vector<std::uint32_t> labels;
                collect_component_leaves(u, labels);

                std::sort(labels.begin(), labels.end());
                labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

                if (!labels.empty()) {
                    for (std::uint32_t label : labels) {
                        seen_labels.insert(label);
                    }
                    forest.add_component(LabelComponent(std::move(labels)));
                }
            }

            /*
             * Defensive: should not be necessary, but guarantees partition output.
             */
            for (std::uint32_t label : expected_labels) {
                if (seen_labels.find(label) == seen_labels.end()) {
                    forest.components.emplace_back(label);
                }
            }

            forest.normalize();
            return forest;
        }
    };

    struct Candidate {
        bool usable = false;

        std::uint32_t a = 0;
        std::uint32_t b = 0;

        int na = -1;
        int nb = -1;

        int ra = -1;
        int rb = -1;

        bool same_component = false;

        int distance = 0;
        int conflict_mass = 0;
        int component_size = 0;
        int immediate_gain = 0;

        std::vector<int> pendants;

        double score = -std::numeric_limits<double>::infinity();
    };

    Options options_;

    static std::uint64_t mix64(std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    void publish_progress(
        const DynamicTree& forest,
        const std::vector<std::uint32_t>& expected_labels
    ) const {
        if (!options_.publish_candidate) {
            return;
        }

        try {
            LabelForest out = forest.to_label_forest(expected_labels);
            out.normalize();
            options_.publish_candidate(out);
        } catch (const std::exception&) {
            // Progress publishing must never affect the heuristic itself.
        }
    }

    static void require_same_leaf_set(const Tree& t1, const Tree& t2) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw ActiveCherryGreedyApproxError("input trees do not have the same leaf set");
        }
    }

    static std::uint32_t next_pseudo_label(const Tree& t1, const Tree& t2) {
        std::uint32_t mx = 0;

        for (std::uint32_t x : t1.leaf_labels) {
            mx = std::max(mx, x);
        }

        for (std::uint32_t x : t2.leaf_labels) {
            mx = std::max(mx, x);
        }

        if (mx == std::numeric_limits<std::uint32_t>::max()) {
            throw ActiveCherryGreedyApproxError("cannot allocate pseudo label");
        }

        return mx + 1;
    }

    static bool contract_one_common_cherry(
        DynamicTree& t1,
        DynamicTree& t2,
        std::uint32_t& next_label
    ) {
        const std::vector<std::pair<std::uint32_t, std::uint32_t>> cherries =
            t1.cherries_by_label(0);

        for (const auto& [a, b] : cherries) {
            if (!t2.are_siblings_by_label(a, b)) {
                continue;
            }

            const std::uint32_t label = next_label++;

            const bool ok1 = t1.contract_cherry(a, b, label);
            const bool ok2 = t2.contract_cherry(a, b, label);

            return ok1 && ok2;
        }

        return false;
    }

    static void contract_all_common_cherries(
        DynamicTree& t1,
        DynamicTree& t2,
        std::uint32_t& next_label,
        const Timer* timer
    ) {
        while (true) {
            if (timer != nullptr && timer->should_stop(0.05)) {
                return;
            }

            if (!contract_one_common_cherry(t1, t2, next_label)) {
                return;
            }
        }
    }

    static void remove_resolved_singleton_roots_from_t1(
        DynamicTree& t1,
        const DynamicTree& f
    ) {
        for (int u = 0; u < static_cast<int>(f.nodes.size()); ++u) {
            if (!f.is_leaflike(u)) {
                continue;
            }

            if (f.nodes[static_cast<std::size_t>(u)].parent != -1) {
                continue;
            }

            const std::uint32_t label = f.nodes[static_cast<std::size_t>(u)].label;

            const int v = t1.node_of_label(label);

            if (v >= 0) {
                t1.cut_edge_above(v);
            }
        }
    }

    Candidate build_candidate(
        const DynamicTree& f,
        MassScratch& mass,
        PathScratch& scratch,
        std::uint32_t a,
        std::uint32_t b
    ) const {
        Candidate cand;
        cand.a = a;
        cand.b = b;

        cand.na = f.node_of_label(a);
        cand.nb = f.node_of_label(b);

        if (cand.na < 0 || cand.nb < 0) {
            return cand;
        }

        cand.ra = f.root_of(cand.na);
        cand.rb = f.root_of(cand.nb);

        if (cand.ra < 0 || cand.rb < 0) {
            return cand;
        }

        cand.same_component = cand.ra == cand.rb;
        cand.usable = true;

        if (cand.same_component) {
            const std::vector<int>& path = f.path_nodes(cand.na, cand.nb, scratch);

            cand.distance = path.empty() ? 0 : static_cast<int>(path.size()) - 1;

            f.collect_pendant_children_on_path(path, cand.pendants);

            for (int p : cand.pendants) {
                cand.conflict_mass += f.active_mass_or_one(p, mass);
            }

            if (options_.policy == Policy::DualityConservative) {
                cand.component_size = f.active_mass_or_one(cand.ra, mass);
            }

            if (cand.pendants.size() == 1) {
                cand.immediate_gain = 1;
            }
        } else {
            const int ma = f.active_mass_or_one(cand.ra, mass);
            const int mb = f.active_mass_or_one(cand.rb, mass);

            cand.component_size = std::min(ma, mb);
            cand.conflict_mass = cand.component_size;
            cand.immediate_gain = 1;
        }

        return cand;
    }

    double score_candidate(const Candidate& cand, int total_active) const {
        const double denom = static_cast<double>(std::max(1, total_active));

        if (options_.policy == Policy::DualityConservative) {
            double score = cand.same_component ? -2.0 : 0.0;
            score -= 15.0 * static_cast<double>(cand.distance) / denom;
            score -= 24.0 * static_cast<double>(cand.pendants.size()) / denom;
            score -= 18.0 * static_cast<double>(cand.conflict_mass) / denom;
            score += 5.0 * static_cast<double>(cand.immediate_gain);
            score -= 2.0 * static_cast<double>(cand.component_size) / denom;
            if (cand.pendants.size() <= 1) {
                score += 2.0;
            }
            return score;
        }

        if (options_.policy == Policy::ResolveFinalCut) {
            double score = cand.same_component ? 40.0 : 45.0;
            if (cand.pendants.size() <= 1) {
                score += 80.0;
            }
            score += 3.0 * static_cast<double>(cand.immediate_gain);
            score -= 0.02 * static_cast<double>(cand.distance);
            return score;
        }

        double score = 0.0;

        if (!cand.same_component) {
            score += 3.0;
            score -= static_cast<double>(cand.component_size) / denom;
        } else {
            score -= 2.0;
            score -= 1.2 * static_cast<double>(cand.pendants.size());
            score -= 4.0 * static_cast<double>(cand.conflict_mass) / denom;
            score -= 0.25 * static_cast<double>(cand.distance);
        }

        switch (options_.policy) {
            case Policy::PreferSmallCuts:
                score -= 3.0 * static_cast<double>(cand.pendants.size());
                score -= 2.0 * static_cast<double>(cand.conflict_mass) / denom;
                break;

            case Policy::PreferBigProgress:
                score += 2.0 * static_cast<double>(cand.pendants.size());
                score += 1.5 * static_cast<double>(cand.conflict_mass) / denom;
                break;

            case Policy::PreferDifferentComponent:
                if (!cand.same_component) {
                    score += 4.0;
                }
                break;

            case Policy::PreferLowConflictMass:
                score -= 8.0 * static_cast<double>(cand.conflict_mass) / denom;
                score -= 0.75 * static_cast<double>(cand.pendants.size());
                break;

            case Policy::PreferFewPendants:
                score -= 4.0 * static_cast<double>(cand.pendants.size());
                score -= 1.0 * static_cast<double>(cand.conflict_mass) / denom;
                break;

            case Policy::PreferImmediateGain:
                score += 4.0 * static_cast<double>(cand.immediate_gain);
                if (!cand.same_component) {
                    score += 1.0;
                }
                break;

            case Policy::ConservativeSingleCut:
                score -= 5.0 * static_cast<double>(cand.pendants.size());
                score -= 6.0 * static_cast<double>(cand.conflict_mass) / denom;
                if (cand.pendants.size() <= 1) {
                    score += 2.0;
                }
                break;

            case Policy::DualityConservative:
            case Policy::ResolveFinalCut:
                break;

            case Policy::AggressiveMultiCut:
                score += 3.0 * static_cast<double>(cand.immediate_gain);
                if (cand.pendants.size() >= 2) {
                    score += 1.5;
                }
                score += 1.0 * static_cast<double>(cand.conflict_mass) / denom;
                break;

            case Policy::Balanced:
                break;
        }

        return score;
    }

    static bool candidate_better(const Candidate& a, const Candidate& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }

        if (a.same_component != b.same_component) {
            return !a.same_component;
        }

        if (a.conflict_mass != b.conflict_mass) {
            return a.conflict_mass < b.conflict_mass;
        }

        if (a.a != b.a) {
            return a.a < b.a;
        }

        return a.b < b.b;
    }

    bool apply_candidate(
        DynamicTree& t1,
        DynamicTree& f,
        const Candidate& cand,
        MassScratch& mass
    ) const {
        if (!cand.usable) {
            return false;
        }

        if (!cand.same_component) {
            int cut = -1;

            const int ma = f.active_mass_or_one(cand.ra, mass);
            const int mb = f.active_mass_or_one(cand.rb, mass);

            /*
             * If both endpoints are already singleton roots in F, cutting in F does
             * nothing. Cut in T1 to remove this active cherry from consideration.
             */
            if (f.nodes[static_cast<std::size_t>(cand.na)].parent == -1 &&
                f.nodes[static_cast<std::size_t>(cand.nb)].parent == -1) {
                const int ta = t1.node_of_label(cand.a);

                if (ta >= 0 && t1.cut_edge_above(ta)) {
                    return true;
                }

                const int tb = t1.node_of_label(cand.b);
                return tb >= 0 && t1.cut_edge_above(tb);
            }

            if (f.nodes[static_cast<std::size_t>(cand.na)].parent == -1) {
                cut = cand.nb;
            } else if (f.nodes[static_cast<std::size_t>(cand.nb)].parent == -1) {
                cut = cand.na;
            } else {
                cut = (ma <= mb) ? cand.na : cand.nb;
            }

            return f.cut_edge_above(cut);
        }

        if (!cand.pendants.empty()) {
            std::vector<int> sorted = cand.pendants;

            std::sort(sorted.begin(), sorted.end(), [&](int x, int y) {
                const int mx = f.active_mass_or_one(x, mass);
                const int my = f.active_mass_or_one(y, mass);

                if (mx != my) {
                    return mx < my;
                }

                return x < y;
            });

            bool changed = false;

            std::size_t limit = 1;

            if (options_.policy == Policy::PreferBigProgress) {
                limit = std::min<std::size_t>(3, sorted.size());
            } else if (options_.policy == Policy::AggressiveMultiCut) {
                limit = std::min<std::size_t>(5, sorted.size());
            } else if (options_.policy == Policy::ConservativeSingleCut) {
                limit = 1;
            } else if (options_.policy == Policy::DualityConservative ||
                       options_.policy == Policy::ResolveFinalCut) {
                limit = 1;
            }

            for (std::size_t i = 0; i < limit; ++i) {
                changed |= f.cut_edge_above(sorted[i]);
            }

            return changed;
        }

        const int ma = f.active_mass_or_one(cand.na, mass);
        const int mb = f.active_mass_or_one(cand.nb, mass);

        return f.cut_edge_above(ma <= mb ? cand.na : cand.nb);
    }
};

}  // namespace pace26::heuristics
