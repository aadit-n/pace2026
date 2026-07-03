#pragma once

#include "core/Forest.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class ForestCrossoverError : public std::runtime_error {
public:
    explicit ForestCrossoverError(const std::string& message)
        : std::runtime_error("Forest crossover error: " + message) {}
};

/*
 * Exact pairwise crossover of two complete valid agreement forests.
 *
 * Given parent forests F and G, every component conflict is cross-parent:
 * components inside F are mutually compatible, and components inside G are
 * mutually compatible. Therefore the maximum-weight compatible recombination
 * of non-singleton components is a weighted independent set in a bipartite
 * graph, solved exactly by min-cut.
 */
class ForestCrossover {
private:
    using Tree = pace26::core::Tree;
    using LabelComponent = pace26::core::LabelComponent;
    using LabelForest = pace26::core::LabelForest;
    using Timer = pace26::core::Timer;

public:
    struct Stats {
        std::size_t parent_a_components = 0;
        std::size_t parent_b_components = 0;
        std::size_t shared_components = 0;
        std::size_t left_candidates = 0;
        std::size_t right_candidates = 0;
        std::size_t conflict_edges = 0;
        std::size_t selected_shared = 0;
        std::size_t selected_left = 0;
        std::size_t selected_right = 0;
        std::size_t final_components = 0;
        std::size_t final_non_singletons = 0;
        std::size_t uncovered_leaves = 0;
        int final_gain = 0;
        std::int64_t total_weight = 0;
        std::int64_t selected_weight = 0;
        std::int64_t min_cut_weight = 0;
        bool stopped = false;
    };

    struct Options {
        double guard_seconds = 1.0;
        Stats* stats = nullptr;
    };

    ForestCrossover() : ForestCrossover(Options{}) {}

    explicit ForestCrossover(Options options)
        : options_(options) {}

    LabelForest solve(
        const Tree& t1,
        const Tree& t2,
        LabelForest parent_a,
        LabelForest parent_b,
        const Timer* timer = nullptr
    ) const {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw ForestCrossoverError("input trees do not have the same leaf set");
        }

        Stats* stats = options_.stats;
        if (stats != nullptr) {
            *stats = Stats{};
        }

        parent_a.normalize();
        parent_b.normalize();
        parent_a.validate_partition_of(t1.leaf_labels);
        parent_b.validate_partition_of(t1.leaf_labels);

        if (stats != nullptr) {
            stats->parent_a_components = parent_a.component_count();
            stats->parent_b_components = parent_b.component_count();
        }

        if (should_stop(timer)) {
            if (stats != nullptr) {
                stats->stopped = true;
            }
            return parent_a;
        }

        Solver solver(t1, t2, options_, stats, timer);
        return solver.run(std::move(parent_a), std::move(parent_b));
    }

private:
    struct VectorHash {
        std::size_t operator()(const std::vector<std::uint32_t>& values) const noexcept {
            std::uint64_t h = 0x9e3779b97f4a7c15ULL;
            for (std::uint32_t value : values) {
                h ^= static_cast<std::uint64_t>(value) +
                     0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdULL;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ULL;
            h ^= h >> 33;
            return static_cast<std::size_t>(h);
        }
    };

    struct Candidate {
        LabelComponent component;
        std::vector<int> leaf_ids;
        std::vector<int> edges1;
        std::vector<int> edges2;
        std::int64_t weight = 0;
    };

    struct FlowEdge {
        int to = 0;
        int rev = 0;
        std::int64_t cap = 0;
    };

    class Dinic {
    public:
        explicit Dinic(int node_count)
            : graph_(static_cast<std::size_t>(node_count)),
              level_(static_cast<std::size_t>(node_count), 0),
              it_(static_cast<std::size_t>(node_count), 0) {}

        void add_edge(int from, int to, std::int64_t cap) {
            FlowEdge forward{to, static_cast<int>(graph_[static_cast<std::size_t>(to)].size()), cap};
            FlowEdge backward{from, static_cast<int>(graph_[static_cast<std::size_t>(from)].size()), 0};
            graph_[static_cast<std::size_t>(from)].push_back(forward);
            graph_[static_cast<std::size_t>(to)].push_back(backward);
        }

        std::int64_t max_flow(int source, int sink) {
            std::int64_t flow = 0;
            while (bfs(source, sink)) {
                std::fill(it_.begin(), it_.end(), 0);
                while (true) {
                    const std::int64_t pushed =
                        dfs(source, sink, std::numeric_limits<std::int64_t>::max() / 4);
                    if (pushed == 0) {
                        break;
                    }
                    flow += pushed;
                }
            }
            return flow;
        }

        std::vector<char> reachable_from(int source) const {
            std::vector<char> seen(graph_.size(), 0);
            std::queue<int> q;
            seen[static_cast<std::size_t>(source)] = 1;
            q.push(source);

            while (!q.empty()) {
                const int u = q.front();
                q.pop();

                for (const FlowEdge& edge : graph_[static_cast<std::size_t>(u)]) {
                    if (edge.cap <= 0 || seen[static_cast<std::size_t>(edge.to)]) {
                        continue;
                    }
                    seen[static_cast<std::size_t>(edge.to)] = 1;
                    q.push(edge.to);
                }
            }

            return seen;
        }

    private:
        std::vector<std::vector<FlowEdge>> graph_;
        std::vector<int> level_;
        std::vector<int> it_;

        bool bfs(int source, int sink) {
            std::fill(level_.begin(), level_.end(), -1);
            std::queue<int> q;
            level_[static_cast<std::size_t>(source)] = 0;
            q.push(source);

            while (!q.empty()) {
                const int u = q.front();
                q.pop();

                for (const FlowEdge& edge : graph_[static_cast<std::size_t>(u)]) {
                    if (edge.cap <= 0 || level_[static_cast<std::size_t>(edge.to)] >= 0) {
                        continue;
                    }
                    level_[static_cast<std::size_t>(edge.to)] =
                        level_[static_cast<std::size_t>(u)] + 1;
                    q.push(edge.to);
                }
            }

            return level_[static_cast<std::size_t>(sink)] >= 0;
        }

        std::int64_t dfs(int u, int sink, std::int64_t flow) {
            if (u == sink) {
                return flow;
            }

            for (int& i = it_[static_cast<std::size_t>(u)];
                 i < static_cast<int>(graph_[static_cast<std::size_t>(u)].size());
                 ++i) {
                FlowEdge& edge = graph_[static_cast<std::size_t>(u)][static_cast<std::size_t>(i)];
                if (edge.cap <= 0 ||
                    level_[static_cast<std::size_t>(edge.to)] !=
                        level_[static_cast<std::size_t>(u)] + 1) {
                    continue;
                }

                const std::int64_t pushed = dfs(edge.to, sink, std::min(flow, edge.cap));
                if (pushed == 0) {
                    continue;
                }

                edge.cap -= pushed;
                graph_[static_cast<std::size_t>(edge.to)][static_cast<std::size_t>(edge.rev)].cap += pushed;
                return pushed;
            }

            return 0;
        }
    };

    class Solver {
    public:
        Solver(
            const Tree& input_t1,
            const Tree& input_t2,
            const Options& input_options,
            Stats* input_stats,
            const Timer* input_timer
        )
            : t1(input_t1),
              t2(input_t2),
              options(input_options),
              stats(input_stats),
              timer(input_timer) {
            labels = t1.leaf_labels;
            label_id.reserve(labels.size() * 2 + 1);
            for (std::size_t i = 0; i < labels.size(); ++i) {
                label_id.emplace(labels[i], static_cast<int>(i));
            }

            const std::int64_t n = static_cast<std::int64_t>(labels.size());
            gain_scale = n * n + 1;
        }

        LabelForest run(LabelForest parent_a, LabelForest parent_b) {
            std::vector<Candidate> all_left;
            std::unordered_map<std::vector<std::uint32_t>, int, VectorHash> left_of_labels;
            left_of_labels.reserve(parent_a.components.size() * 2 + 1);

            for (const LabelComponent& component : parent_a.components) {
                if (component.labels.size() <= 1) {
                    continue;
                }
                Candidate candidate = make_candidate(component);
                const int index = static_cast<int>(all_left.size());
                left_of_labels.emplace(candidate.component.labels, index);
                all_left.push_back(std::move(candidate));
            }

            std::vector<char> left_shared(all_left.size(), 0);

            for (const LabelComponent& component : parent_b.components) {
                if (component.labels.size() <= 1) {
                    continue;
                }

                LabelComponent normalized = component;
                normalized.normalize();

                auto existing = left_of_labels.find(normalized.labels);
                if (existing != left_of_labels.end()) {
                    left_shared[static_cast<std::size_t>(existing->second)] = 1;
                    continue;
                }

                right.push_back(make_candidate(normalized));
            }

            for (std::size_t i = 0; i < all_left.size(); ++i) {
                if (left_shared[i]) {
                    shared.push_back(std::move(all_left[i]));
                } else {
                    left.push_back(std::move(all_left[i]));
                }
            }

            if (stats != nullptr) {
                stats->shared_components = shared.size();
                stats->left_candidates = left.size();
                stats->right_candidates = right.size();
                stats->selected_shared = shared.size();
            }

            std::vector<std::pair<int, int>> conflicts = build_conflicts();
            LabelForest result = solve_mwis(conflicts);
            result.normalize();
            result.validate_partition_of(labels);
            return result;
        }

    private:
        const Tree& t1;
        const Tree& t2;
        const Options& options;
        Stats* stats = nullptr;
        const Timer* timer = nullptr;

        std::vector<std::uint32_t> labels;
        std::unordered_map<std::uint32_t, int> label_id;
        std::int64_t gain_scale = 1;

        std::vector<Candidate> shared;
        std::vector<Candidate> left;
        std::vector<Candidate> right;

        bool should_stop() const {
            return timer != nullptr && timer->should_stop(options.guard_seconds);
        }

        int label_index(std::uint32_t label) const {
            auto it = label_id.find(label);
            if (it == label_id.end()) {
                throw ForestCrossoverError("unknown leaf label");
            }
            return it->second;
        }

        std::int64_t component_weight(std::size_t size) const {
            const std::int64_t s = static_cast<std::int64_t>(size);
            const std::int64_t gain = s - 1;
            const std::int64_t tie_bonus = s * s - s;
            return gain * gain_scale + tie_bonus;
        }

        Candidate make_candidate(LabelComponent component) const {
            component.normalize();
            Candidate candidate;
            candidate.component = std::move(component);
            candidate.weight = component_weight(candidate.component.labels.size());

            candidate.leaf_ids.reserve(candidate.component.labels.size());
            for (std::uint32_t label : candidate.component.labels) {
                candidate.leaf_ids.push_back(label_index(label));
            }

            candidate.edges1 = induced_edges(t1, candidate.component.labels);
            candidate.edges2 = induced_edges(t2, candidate.component.labels);
            return candidate;
        }

        static std::vector<int> induced_edges(
            const Tree& tree,
            const std::vector<std::uint32_t>& component_labels
        ) {
            std::vector<int> edges;
            if (component_labels.size() <= 1) {
                return edges;
            }

            int root = tree.node_of_label(component_labels.front());
            for (std::size_t i = 1; i < component_labels.size(); ++i) {
                root = tree.lca(root, tree.node_of_label(component_labels[i]));
            }

            for (std::uint32_t label : component_labels) {
                int u = tree.node_of_label(label);
                while (u != root) {
                    edges.push_back(u);
                    u = tree.parent[static_cast<std::size_t>(u)];
                    if (u < 0) {
                        throw ForestCrossoverError("invalid parent path while computing induced edges");
                    }
                }
            }

            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            return edges;
        }

        std::vector<std::pair<int, int>> build_conflicts() const {
            std::vector<int> owner_leaf(labels.size(), -1);
            std::vector<int> owner_edge1(static_cast<std::size_t>(t1.node_count()), -1);
            std::vector<int> owner_edge2(static_cast<std::size_t>(t2.node_count()), -1);

            for (int i = 0; i < static_cast<int>(left.size()); ++i) {
                const Candidate& candidate = left[static_cast<std::size_t>(i)];
                for (int leaf : candidate.leaf_ids) {
                    owner_leaf[static_cast<std::size_t>(leaf)] = i;
                }
                for (int edge : candidate.edges1) {
                    owner_edge1[static_cast<std::size_t>(edge)] = i;
                }
                for (int edge : candidate.edges2) {
                    owner_edge2[static_cast<std::size_t>(edge)] = i;
                }
            }

            std::vector<std::pair<int, int>> conflicts;
            std::unordered_set<std::uint64_t> seen;
            seen.reserve(
                (left.size() + right.size()) * 4 + 1
            );

            auto add_conflict = [&](int left_index, int right_index) {
                if (left_index < 0 || right_index < 0) {
                    return;
                }
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(
                         static_cast<std::uint32_t>(left_index)) << 32) |
                    static_cast<std::uint32_t>(right_index);
                if (!seen.insert(key).second) {
                    return;
                }
                conflicts.push_back({left_index, right_index});
            };

            for (int j = 0; j < static_cast<int>(right.size()); ++j) {
                const Candidate& candidate = right[static_cast<std::size_t>(j)];
                for (int leaf : candidate.leaf_ids) {
                    add_conflict(owner_leaf[static_cast<std::size_t>(leaf)], j);
                }
                for (int edge : candidate.edges1) {
                    add_conflict(owner_edge1[static_cast<std::size_t>(edge)], j);
                }
                for (int edge : candidate.edges2) {
                    add_conflict(owner_edge2[static_cast<std::size_t>(edge)], j);
                }
            }

            if (stats != nullptr) {
                stats->conflict_edges = conflicts.size();
            }
            return conflicts;
        }

        LabelForest solve_mwis(const std::vector<std::pair<int, int>>& conflicts) const {
            const int source = 0;
            const int left_offset = 1;
            const int right_offset = left_offset + static_cast<int>(left.size());
            const int sink = right_offset + static_cast<int>(right.size());

            Dinic flow(sink + 1);

            std::int64_t total_weight = 0;
            for (int i = 0; i < static_cast<int>(left.size()); ++i) {
                total_weight += left[static_cast<std::size_t>(i)].weight;
                flow.add_edge(source, left_offset + i, left[static_cast<std::size_t>(i)].weight);
            }
            for (int j = 0; j < static_cast<int>(right.size()); ++j) {
                total_weight += right[static_cast<std::size_t>(j)].weight;
                flow.add_edge(right_offset + j, sink, right[static_cast<std::size_t>(j)].weight);
            }
            for (const Candidate& candidate : shared) {
                total_weight += candidate.weight;
            }

            const std::int64_t inf = std::max<std::int64_t>(1, total_weight + 1);
            for (const auto& [li, ri] : conflicts) {
                flow.add_edge(left_offset + li, right_offset + ri, inf);
            }

            const std::int64_t cut_weight = flow.max_flow(source, sink);
            const std::vector<char> reachable = flow.reachable_from(source);

            LabelForest forest;
            std::vector<char> covered(labels.size(), 0);
            std::int64_t selected_weight = 0;
            int gain = 0;

            auto add_selected = [&](const Candidate& candidate) {
                forest.add_component(candidate.component);
                selected_weight += candidate.weight;
                gain += static_cast<int>(candidate.component.labels.size()) - 1;
                for (int leaf : candidate.leaf_ids) {
                    covered[static_cast<std::size_t>(leaf)] = 1;
                }
            };

            for (const Candidate& candidate : shared) {
                add_selected(candidate);
            }

            std::size_t selected_left = 0;
            for (int i = 0; i < static_cast<int>(left.size()); ++i) {
                if (reachable[static_cast<std::size_t>(left_offset + i)] == 0) {
                    continue;
                }
                add_selected(left[static_cast<std::size_t>(i)]);
                ++selected_left;
            }

            std::size_t selected_right = 0;
            for (int j = 0; j < static_cast<int>(right.size()); ++j) {
                if (reachable[static_cast<std::size_t>(right_offset + j)] != 0) {
                    continue;
                }
                add_selected(right[static_cast<std::size_t>(j)]);
                ++selected_right;
            }

            std::size_t uncovered = 0;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (covered[i]) {
                    continue;
                }
                forest.components.emplace_back(labels[i]);
                ++uncovered;
            }

            forest.normalize();

            if (stats != nullptr) {
                stats->selected_left = selected_left;
                stats->selected_right = selected_right;
                stats->uncovered_leaves = uncovered;
                stats->final_components = forest.component_count();
                stats->final_non_singletons =
                    shared.size() + selected_left + selected_right;
                stats->final_gain = gain;
                stats->total_weight = total_weight;
                stats->selected_weight = selected_weight;
                stats->min_cut_weight = cut_weight;
                stats->stopped = should_stop();
            }

            return forest;
        }
    };

    bool should_stop(const Timer* timer) const {
        return timer != nullptr && timer->should_stop(options_.guard_seconds);
    }

    Options options_;
};

}  // namespace pace26::heuristics
