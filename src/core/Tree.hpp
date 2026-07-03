#pragma once

#include "io/NewickParser.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pace26::core {

class TreeError : public std::runtime_error {
public:
    explicit TreeError(const std::string& message)
        : std::runtime_error("Tree error: " + message) {}
};

struct TreeNode {
    std::uint32_t id = 0;
    std::uint32_t label = 0;

    int left = -1;
    int right = -1;

    bool is_leaf() const noexcept {
        return left < 0 && right < 0;
    }
};

/*
 * Solver-friendly rooted binary tree.
 *
 * This is deliberately array-based:
 * - node ids are integer indices;
 * - parent/depth/tin/tout/postorder are precomputed;
 * - leaf label -> node lookup is O(1) average;
 * - LCA is O(log n).
 */
class Tree {
public:
    std::vector<TreeNode> nodes;
    int root = -1;

    std::vector<int> parent;
    std::vector<int> depth;
    std::vector<int> tin;
    std::vector<int> tout;
    std::vector<int> postorder;

    std::vector<std::uint32_t> leaf_count_under;

    std::vector<std::uint32_t> leaf_labels;
    std::vector<int> leaf_to_node_flat;

    // Binary lifting table: up[j][u] = 2^j-th ancestor of u.
    std::vector<std::vector<int>> up;

    Tree() = default;

    static Tree from_newick(const pace26::io::NewickTree& source) {
        Tree tree;

        tree.nodes.reserve(source.nodes.size());

        for (const auto& n : source.nodes) {
            TreeNode out;
            out.id = n.id;
            out.label = n.label;
            out.left = n.left;
            out.right = n.right;
            tree.nodes.push_back(out);
        }

        tree.root = source.root;
        tree.rebuild_indices();

        return tree;
    }

    pace26::io::NewickTree to_newick_tree() const {
        pace26::io::NewickTree out;
        out.nodes.reserve(nodes.size());

        for (const TreeNode& n : nodes) {
            pace26::io::NewickNode node;
            node.id = n.id;
            node.label = n.label;
            node.left = n.left;
            node.right = n.right;
            out.nodes.push_back(node);
        }

        out.root = root;
        return out;
    }

    int node_count() const noexcept {
        return static_cast<int>(nodes.size());
    }

    int leaf_count() const noexcept {
        return static_cast<int>(leaf_labels.size());
    }

    bool empty() const noexcept {
        return nodes.empty();
    }

    const TreeNode& node(int u) const {
        check_node(u);
        return nodes[static_cast<std::size_t>(u)];
    }

    bool is_leaf(int u) const {
        return node(u).is_leaf();
    }

    std::uint32_t label(int u) const {
        const TreeNode& n = node(u);
        if (!n.is_leaf()) {
            throw TreeError("requested label of an internal node");
        }
        return n.label;
    }

    int left(int u) const {
        const TreeNode& n = node(u);
        if (n.is_leaf()) {
            throw TreeError("requested left child of a leaf");
        }
        return n.left;
    }

    int right(int u) const {
        const TreeNode& n = node(u);
        if (n.is_leaf()) {
            throw TreeError("requested right child of a leaf");
        }
        return n.right;
    }

    bool contains_label(std::uint32_t x) const {
        return x < leaf_to_node_flat.size() && leaf_to_node_flat[x] != -1;
    }

    int node_of_label(std::uint32_t x) const {
        if (x >= leaf_to_node_flat.size() || leaf_to_node_flat[x] == -1) {
            throw TreeError("leaf label " + std::to_string(x) + " not found");
        }
        return leaf_to_node_flat[x];
    }

    bool is_ancestor(int ancestor, int descendant) const {
        check_node(ancestor);
        check_node(descendant);
        return tin[static_cast<std::size_t>(ancestor)] <= tin[static_cast<std::size_t>(descendant)] &&
               tout[static_cast<std::size_t>(descendant)] <= tout[static_cast<std::size_t>(ancestor)];
    }

    int lca(int a, int b) const {
        check_node(a);
        check_node(b);

        if (is_ancestor(a, b)) return a;
        if (is_ancestor(b, a)) return b;

        int u = a;

        for (int j = static_cast<int>(up.size()) - 1; j >= 0; --j) {
            int candidate = up[static_cast<std::size_t>(j)][static_cast<std::size_t>(u)];
            if (!is_ancestor(candidate, b)) {
                u = candidate;
            }
        }

        return parent[static_cast<std::size_t>(u)];
    }

    std::vector<std::uint32_t> labels_under(int u) const {
        check_node(u);

        std::vector<std::uint32_t> labels;
        labels.reserve(leaf_count_under[static_cast<std::size_t>(u)]);

        std::vector<int> stack;
        stack.push_back(u);

        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();

            const TreeNode& n = nodes[static_cast<std::size_t>(v)];

            if (n.is_leaf()) {
                labels.push_back(n.label);
            } else {
                stack.push_back(n.left);
                stack.push_back(n.right);
            }
        }

        std::sort(labels.begin(), labels.end());
        return labels;
    }

    std::uint32_t max_node_or_label_id() const {
        std::uint32_t result = 0;

        for (const TreeNode& n : nodes) {
            result = std::max(result, n.id);
            result = std::max(result, n.label);
        }

        return result;
    }

    void rebuild_indices() {
        validate_basic_structure();

        const int m = node_count();

        parent.assign(static_cast<std::size_t>(m), -1);
        depth.assign(static_cast<std::size_t>(m), 0);
        tin.assign(static_cast<std::size_t>(m), 0);
        tout.assign(static_cast<std::size_t>(m), 0);
        postorder.clear();
        postorder.reserve(static_cast<std::size_t>(m));
        leaf_count_under.assign(static_cast<std::size_t>(m), 0);

        for (int u = 0; u < m; ++u) {
            const TreeNode& n = nodes[static_cast<std::size_t>(u)];
            if (!n.is_leaf()) {
                parent[static_cast<std::size_t>(n.left)] = u;
                parent[static_cast<std::size_t>(n.right)] = u;
            }
        }

        if (parent[static_cast<std::size_t>(root)] != -1) {
            throw TreeError("root has a parent");
        }

        build_dfs_indices();
        build_leaf_indices();
        build_lca_table();
    }

private:
    void check_node(int u) const {
        if (u < 0 || u >= node_count()) {
            throw TreeError("node index out of range");
        }
    }

    void validate_basic_structure() const {
        if (nodes.empty()) {
            throw TreeError("tree has no nodes");
        }

        if (root < 0 || root >= static_cast<int>(nodes.size())) {
            throw TreeError("root index out of range");
        }

        for (std::size_t i = 0; i < nodes.size(); ++i) {
            const TreeNode& n = nodes[i];

            if (n.is_leaf()) {
                if (n.label == 0) {
                    throw TreeError("leaf with label 0");
                }
            } else {
                if (n.left < 0 || n.right < 0) {
                    throw TreeError("internal node with missing child");
                }

                if (n.left >= static_cast<int>(nodes.size()) ||
                    n.right >= static_cast<int>(nodes.size())) {
                    throw TreeError("child index out of range");
                }

                if (n.left == n.right) {
                    throw TreeError("internal node has identical left/right child");
                }
            }
        }
    }

    void build_dfs_indices() {
        int timer = 0;

        std::vector<std::pair<int, bool>> stack;
        stack.reserve(nodes.size());
        stack.push_back({root, false});

        while (!stack.empty()) {
            auto [u, exiting] = stack.back();
            stack.pop_back();

            if (exiting) {
                tout[static_cast<std::size_t>(u)] = timer;
                postorder.push_back(u);
                continue;
            }

            tin[static_cast<std::size_t>(u)] = timer++;
            stack.push_back({u, true});

            const TreeNode& n = nodes[static_cast<std::size_t>(u)];

            if (!n.is_leaf()) {
                depth[static_cast<std::size_t>(n.left)] = depth[static_cast<std::size_t>(u)] + 1;
                depth[static_cast<std::size_t>(n.right)] = depth[static_cast<std::size_t>(u)] + 1;

                stack.push_back({n.right, false});
                stack.push_back({n.left, false});
            }
        }
    }

    void build_leaf_indices() {
        leaf_labels.clear();
        leaf_to_node_flat.clear();

        std::uint32_t max_label = 0;
        for (const auto& n : nodes) {
            if (n.is_leaf() && n.label > max_label) {
                max_label = n.label;
            }
        }
        
        // Failsafe for absurdly large labels that would cause huge memory allocations,
        // though PACE labels are guaranteed to be dense and small (<= N).
        if (max_label > 100000000) {
            throw TreeError("max label too large for flat array optimization");
        }
        
        leaf_to_node_flat.assign(max_label + 1, -1);
        leaf_labels.reserve((nodes.size() + 1) / 2);

        for (int u : postorder) {
            TreeNode& n = nodes[static_cast<std::size_t>(u)];

            if (n.is_leaf()) {
                leaf_count_under[static_cast<std::size_t>(u)] = 1;

                if (leaf_to_node_flat[n.label] != -1) {
                    throw TreeError("duplicate leaf label " + std::to_string(n.label));
                }

                leaf_to_node_flat[n.label] = u;
                leaf_labels.push_back(n.label);
            } else {
                const std::uint32_t left_count =
                    leaf_count_under[static_cast<std::size_t>(n.left)];
                const std::uint32_t right_count =
                    leaf_count_under[static_cast<std::size_t>(n.right)];

                leaf_count_under[static_cast<std::size_t>(u)] =
                    left_count + right_count;
            }
        }

        std::sort(leaf_labels.begin(), leaf_labels.end());
    }

    void build_lca_table() {
        const int m = node_count();

        int log = 1;
        while ((1 << log) <= std::max(1, m)) {
            ++log;
        }

        up.assign(
            static_cast<std::size_t>(log),
            std::vector<int>(static_cast<std::size_t>(m), root)
        );

        for (int u = 0; u < m; ++u) {
            int p = parent[static_cast<std::size_t>(u)];
            up[0][static_cast<std::size_t>(u)] = (p < 0 ? u : p);
        }

        for (int j = 1; j < log; ++j) {
            for (int u = 0; u < m; ++u) {
                int mid = up[static_cast<std::size_t>(j - 1)][static_cast<std::size_t>(u)];
                up[static_cast<std::size_t>(j)][static_cast<std::size_t>(u)] =
                    up[static_cast<std::size_t>(j - 1)][static_cast<std::size_t>(mid)];
            }
        }
    }
};

}  // namespace pace26::core