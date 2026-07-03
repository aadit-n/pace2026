#pragma once

#include "core/Forest.hpp"
#include "core/Tree.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class CherryPairApproxError : public std::runtime_error {
public:
    explicit CherryPairApproxError(const std::string& message)
        : std::runtime_error("CherryPairApprox error: " + message) {}
};

class CherryPairApprox {
private:
    using Tree = pace26::core::Tree;
    using LabelComponent = pace26::core::LabelComponent;
    using LabelForest = pace26::core::LabelForest;

    struct Candidate {
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        std::vector<int> edges1;
        std::vector<int> edges2;
    };

    static void require_same_leaf_set(const Tree& t1, const Tree& t2) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw CherryPairApproxError("input trees do not have the same leaf set");
        }
    }

    static std::vector<int> induced_edges(
        const Tree& tree,
        std::uint32_t a,
        std::uint32_t b
    ) {
        std::vector<int> edges;

        int root = tree.lca(tree.node_of_label(a), tree.node_of_label(b));

        for (std::uint32_t label : {a, b}) {
            int u = tree.node_of_label(label);

            while (u != root) {
                edges.push_back(u);
                u = tree.parent[static_cast<std::size_t>(u)];

                if (u < 0) {
                    throw CherryPairApproxError("invalid parent path");
                }
            }
        }

        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        return edges;
    }

    static void add_cherries_from_tree(
        const Tree& base,
        const Tree& other,
        std::vector<Candidate>& candidates
    ) {
        for (const auto& node : base.nodes) {
            if (node.is_leaf()) {
                continue;
            }

            const auto& left = base.nodes[static_cast<std::size_t>(node.left)];
            const auto& right = base.nodes[static_cast<std::size_t>(node.right)];

            if (!left.is_leaf() || !right.is_leaf()) {
                continue;
            }

            Candidate c;
            c.a = std::min(left.label, right.label);
            c.b = std::max(left.label, right.label);
            c.edges1 = induced_edges(base, c.a, c.b);
            c.edges2 = induced_edges(other, c.a, c.b);
            candidates.push_back(std::move(c));
        }
    }

    static bool edge_set_free(
        const std::vector<int>& edges,
        const std::vector<char>& used
    ) {
        for (int edge : edges) {
            if (used[static_cast<std::size_t>(edge)] != 0) {
                return false;
            }
        }

        return true;
    }

    static void mark_edges(
        const std::vector<int>& edges,
        std::vector<char>& used
    ) {
        for (int edge : edges) {
            used[static_cast<std::size_t>(edge)] = 1;
        }
    }

public:
    LabelForest solve(const Tree& t1, const Tree& t2) const {
        require_same_leaf_set(t1, t2);

        std::vector<Candidate> candidates;
        candidates.reserve(t1.leaf_labels.size());

        add_cherries_from_tree(t1, t2, candidates);

        const std::size_t first_direction_count = candidates.size();
        add_cherries_from_tree(t2, t1, candidates);

        for (std::size_t i = first_direction_count; i < candidates.size(); ++i) {
            std::swap(candidates[i].edges1, candidates[i].edges2);
        }

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& x, const Candidate& y) {
                const std::size_t x_cost = x.edges1.size() + x.edges2.size();
                const std::size_t y_cost = y.edges1.size() + y.edges2.size();

                if (x_cost != y_cost) {
                    return x_cost < y_cost;
                }

                if (x.a != y.a) {
                    return x.a < y.a;
                }

                return x.b < y.b;
            }
        );

        std::vector<char> used_edges1(static_cast<std::size_t>(t1.node_count()), 0);
        std::vector<char> used_edges2(static_cast<std::size_t>(t2.node_count()), 0);

        std::unordered_set<std::uint32_t> covered;
        covered.reserve(t1.leaf_labels.size() * 2);

        LabelForest forest;
        forest.components.reserve(t1.leaf_labels.size());

        for (const Candidate& c : candidates) {
            if (covered.find(c.a) != covered.end() ||
                covered.find(c.b) != covered.end()) {
                continue;
            }

            if (!edge_set_free(c.edges1, used_edges1) ||
                !edge_set_free(c.edges2, used_edges2)) {
                continue;
            }

            forest.add_component(LabelComponent(std::vector<std::uint32_t>{c.a, c.b}));
            covered.insert(c.a);
            covered.insert(c.b);
            mark_edges(c.edges1, used_edges1);
            mark_edges(c.edges2, used_edges2);
        }

        for (std::uint32_t label : t1.leaf_labels) {
            if (covered.find(label) == covered.end()) {
                forest.components.emplace_back(label);
            }
        }

        forest.normalize();
        forest.validate_partition_of(t1.leaf_labels);
        return forest;
    }
};

}  // namespace pace26::heuristics
