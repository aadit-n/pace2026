#pragma once

#include "NewickParser.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::reductions {

using pace26::io::NewickNode;
using pace26::io::NewickTree;

class ClusterDecompositionError : public std::runtime_error {
public:
    explicit ClusterDecompositionError(const std::string& message)
        : std::runtime_error("Cluster decomposition error: " + message) {}
};

struct ClusterDecompositionRecord {
    std::vector<std::uint32_t> cluster_labels;

    int root1 = -1;
    int root2 = -1;

    std::uint32_t x_up_label = 0;
    std::uint32_t x_down_label = 0;
};

struct ClusterSplit {
    bool found = false;

    ClusterDecompositionRecord record;

    std::uint32_t bottom_leaf_count = 0;
    std::uint32_t top_without_cluster_leaf_count = 0;
    bool open_trees_materialized = false;

    // Bottom pair: induced cluster trees.
    NewickTree bottom1;
    NewickTree bottom2;

    // Bottom pair with an extra marker taxon X_DOWN.
    NewickTree bottom1_with_marker;
    NewickTree bottom2_with_marker;

    // Top pair with the cluster replaced by X_UP.
    NewickTree top1_with_placeholder;
    NewickTree top2_with_placeholder;

    // Top pair with the cluster removed entirely.
    NewickTree top1_without_cluster;
    NewickTree top2_without_cluster;
};

struct ClusterSplitTemplate {
    bool found = false;

    int root1 = -1;
    int root2 = -1;
    std::uint32_t size = 0;
    std::vector<std::uint32_t> cluster_labels;
};

struct ClusterOpenTrees {
    NewickTree bottom1_with_marker;
    NewickTree bottom2_with_marker;
    NewickTree top1_with_placeholder;
    NewickTree top2_with_placeholder;
};

class ClusterDecomposition {
private:
    struct HashKey {
        std::uint32_t count = 0;
        std::uint64_t sum_hash = 0;
        std::uint64_t xor_hash = 0;

        bool operator==(const HashKey& other) const noexcept {
            return count == other.count &&
                   sum_hash == other.sum_hash &&
                   xor_hash == other.xor_hash;
        }
    };

    struct HashKeyHasher {
        std::size_t operator()(const HashKey& k) const noexcept {
            std::uint64_t x = 0x9e3779b97f4a7c15ULL;
            x ^= static_cast<std::uint64_t>(k.count) + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
            x ^= k.sum_hash + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);
            x ^= k.xor_hash + 0x9e3779b97f4a7c15ULL + (x << 6) + (x >> 2);

            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdULL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53ULL;
            x ^= x >> 33;

            return static_cast<std::size_t>(x);
        }
    };

    struct IndexInfo {
        std::vector<int> postorder;
        std::vector<int> parent;
        std::vector<std::uint32_t> leaf_count;
        std::vector<std::uint64_t> sum_hash;
        std::vector<std::uint64_t> xor_hash;
    };

    struct Candidate {
        int node1 = -1;
        int node2 = -1;
        std::uint32_t size = 0;
    };

    static std::uint64_t splitmix64(std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    static std::uint64_t label_sum_hash(std::uint32_t label) {
        return splitmix64(static_cast<std::uint64_t>(label) * 0x9e3779b97f4a7c15ULL);
    }

    static std::uint64_t label_xor_hash(std::uint32_t label) {
        return splitmix64(static_cast<std::uint64_t>(label) ^ 0xd1b54a32d192ed03ULL);
    }

    static void validate_tree(const NewickTree& tree, const char* name) {
        if (tree.root < 0) {
            throw ClusterDecompositionError(std::string(name) + " has no root");
        }

        if (tree.root >= static_cast<int>(tree.nodes.size())) {
            throw ClusterDecompositionError(std::string(name) + " root index out of range");
        }

        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            const NewickNode& node = tree.nodes[i];

            if (node.is_leaf()) {
                if (node.label == 0) {
                    throw ClusterDecompositionError(std::string(name) + " has leaf label 0");
                }
            } else {
                if (node.left < 0 || node.right < 0) {
                    throw ClusterDecompositionError(std::string(name) + " has non-binary internal node");
                }

                if (node.left >= static_cast<int>(tree.nodes.size()) ||
                    node.right >= static_cast<int>(tree.nodes.size())) {
                    throw ClusterDecompositionError(std::string(name) + " has child index out of range");
                }

                if (node.left == node.right) {
                    throw ClusterDecompositionError(std::string(name) + " has identical left/right child");
                }
            }
        }
    }

    static std::vector<int> iterative_postorder(const NewickTree& tree) {
        std::vector<int> order;
        order.reserve(tree.nodes.size());

        std::vector<std::pair<int, bool>> stack;
        stack.reserve(tree.nodes.size());
        stack.push_back({tree.root, false});

        while (!stack.empty()) {
            const auto [u, seen] = stack.back();
            stack.pop_back();

            if (seen) {
                order.push_back(u);
                continue;
            }

            stack.push_back({u, true});

            const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];
            if (!node.is_leaf()) {
                stack.push_back({node.right, false});
                stack.push_back({node.left, false});
            }
        }

        return order;
    }

    static IndexInfo build_index(const NewickTree& tree) {
        validate_tree(tree, "tree");

        IndexInfo idx;
        const std::size_t m = tree.nodes.size();

        idx.postorder = iterative_postorder(tree);
        idx.parent.assign(m, -1);
        idx.leaf_count.assign(m, 0);
        idx.sum_hash.assign(m, 0);
        idx.xor_hash.assign(m, 0);

        for (std::size_t i = 0; i < m; ++i) {
            const NewickNode& node = tree.nodes[i];

            if (!node.is_leaf()) {
                idx.parent[static_cast<std::size_t>(node.left)] = static_cast<int>(i);
                idx.parent[static_cast<std::size_t>(node.right)] = static_cast<int>(i);
            }
        }

        for (int u : idx.postorder) {
            const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];

            if (node.is_leaf()) {
                idx.leaf_count[static_cast<std::size_t>(u)] = 1;
                idx.sum_hash[static_cast<std::size_t>(u)] = label_sum_hash(node.label);
                idx.xor_hash[static_cast<std::size_t>(u)] = label_xor_hash(node.label);
            } else {
                const int l = node.left;
                const int r = node.right;

                idx.leaf_count[static_cast<std::size_t>(u)] =
                    idx.leaf_count[static_cast<std::size_t>(l)] +
                    idx.leaf_count[static_cast<std::size_t>(r)];

                idx.sum_hash[static_cast<std::size_t>(u)] =
                    idx.sum_hash[static_cast<std::size_t>(l)] +
                    idx.sum_hash[static_cast<std::size_t>(r)];

                idx.xor_hash[static_cast<std::size_t>(u)] =
                    idx.xor_hash[static_cast<std::size_t>(l)] ^
                    idx.xor_hash[static_cast<std::size_t>(r)];
            }
        }

        return idx;
    }

    static HashKey key_for(const IndexInfo& idx, int node) {
        HashKey key;
        key.count = idx.leaf_count[static_cast<std::size_t>(node)];
        key.sum_hash = idx.sum_hash[static_cast<std::size_t>(node)];
        key.xor_hash = idx.xor_hash[static_cast<std::size_t>(node)];
        return key;
    }

    static std::vector<std::uint32_t> collect_leaves(const NewickTree& tree, int root) {
        std::vector<std::uint32_t> labels;

        std::vector<int> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();

            const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];

            if (node.is_leaf()) {
                labels.push_back(node.label);
            } else {
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }

        std::sort(labels.begin(), labels.end());
        return labels;
    }

    static bool same_leaf_set(
        const NewickTree& t1,
        int node1,
        const NewickTree& t2,
        int node2
    ) {
        return collect_leaves(t1, node1) == collect_leaves(t2, node2);
    }

    static std::uint32_t max_node_or_label_id(const NewickTree& tree) {
        std::uint32_t result = 0;

        for (const NewickNode& node : tree.nodes) {
            result = std::max(result, node.id);
            result = std::max(result, node.label);
        }

        return result;
    }

    static std::uint32_t checked_increment(std::uint32_t value, const char* message) {
        if (value == std::numeric_limits<std::uint32_t>::max()) {
            throw ClusterDecompositionError(message);
        }
        return value + 1;
    }

    static int copy_subtree(
        const NewickTree& source,
        int u,
        NewickTree& target,
        std::uint32_t& next_internal_id
    ) {
        const NewickNode& node = source.nodes[static_cast<std::size_t>(u)];

        if (node.is_leaf()) {
            NewickNode copied;
            copied.id = node.id;
            copied.label = node.label;
            copied.left = -1;
            copied.right = -1;

            target.nodes.push_back(copied);
            return static_cast<int>(target.nodes.size() - 1);
        }

        const int left_new = copy_subtree(source, node.left, target, next_internal_id);
        const int right_new = copy_subtree(source, node.right, target, next_internal_id);

        NewickNode internal;
        internal.id = next_internal_id;
        internal.label = 0;
        internal.left = left_new;
        internal.right = right_new;

        next_internal_id = checked_increment(next_internal_id, "internal id overflow while copying subtree");

        target.nodes.push_back(internal);
        return static_cast<int>(target.nodes.size() - 1);
    }

    static NewickTree make_bottom_tree(
        const NewickTree& source,
        int cluster_root,
        std::uint32_t next_internal_id
    ) {
        NewickTree result;
        result.nodes.reserve(source.nodes.size());

        result.root = copy_subtree(source, cluster_root, result, next_internal_id);

        return result;
    }

    static int add_leaf(NewickTree& tree, std::uint32_t label) {
        NewickNode leaf;
        leaf.id = label;
        leaf.label = label;
        leaf.left = -1;
        leaf.right = -1;

        tree.nodes.push_back(leaf);
        return static_cast<int>(tree.nodes.size() - 1);
    }

    static NewickTree make_bottom_tree_with_marker(
        const NewickTree& source,
        int cluster_root,
        std::uint32_t x_down_label,
        std::uint32_t next_internal_id
    ) {
        NewickTree result;
        result.nodes.reserve(source.nodes.size() + 2);

        const int cluster_copy = copy_subtree(source, cluster_root, result, next_internal_id);
        const int marker = add_leaf(result, x_down_label);

        NewickNode root;
        root.id = next_internal_id;
        root.label = 0;
        root.left = cluster_copy;
        root.right = marker;

        next_internal_id = checked_increment(next_internal_id, "internal id overflow while making bottom marker tree");

        result.nodes.push_back(root);
        result.root = static_cast<int>(result.nodes.size() - 1);

        return result;
    }

    static int copy_with_placeholder(
        const NewickTree& source,
        int u,
        int cluster_root,
        std::uint32_t placeholder_label,
        NewickTree& target,
        std::uint32_t& next_internal_id
    ) {
        if (u == cluster_root) {
            return add_leaf(target, placeholder_label);
        }

        const NewickNode& node = source.nodes[static_cast<std::size_t>(u)];

        if (node.is_leaf()) {
            NewickNode copied;
            copied.id = node.id;
            copied.label = node.label;
            copied.left = -1;
            copied.right = -1;

            target.nodes.push_back(copied);
            return static_cast<int>(target.nodes.size() - 1);
        }

        const int left_new = copy_with_placeholder(
            source,
            node.left,
            cluster_root,
            placeholder_label,
            target,
            next_internal_id
        );

        const int right_new = copy_with_placeholder(
            source,
            node.right,
            cluster_root,
            placeholder_label,
            target,
            next_internal_id
        );

        NewickNode internal;
        internal.id = next_internal_id;
        internal.label = 0;
        internal.left = left_new;
        internal.right = right_new;

        next_internal_id = checked_increment(next_internal_id, "internal id overflow while making top placeholder tree");

        target.nodes.push_back(internal);
        return static_cast<int>(target.nodes.size() - 1);
    }

    static NewickTree make_top_tree_with_placeholder(
        const NewickTree& source,
        int cluster_root,
        std::uint32_t x_up_label,
        std::uint32_t next_internal_id
    ) {
        NewickTree result;
        result.nodes.reserve(source.nodes.size());

        result.root = copy_with_placeholder(
            source,
            source.root,
            cluster_root,
            x_up_label,
            result,
            next_internal_id
        );

        return result;
    }

    static int copy_without_labels(
        const NewickTree& source,
        int u,
        const std::unordered_set<std::uint32_t>& deleted,
        NewickTree& target,
        std::uint32_t& next_internal_id
    ) {
        const NewickNode& node = source.nodes[static_cast<std::size_t>(u)];

        if (node.is_leaf()) {
            if (deleted.find(node.label) != deleted.end()) {
                return -1;
            }

            NewickNode copied;
            copied.id = node.id;
            copied.label = node.label;
            copied.left = -1;
            copied.right = -1;

            target.nodes.push_back(copied);
            return static_cast<int>(target.nodes.size() - 1);
        }

        const int left_new = copy_without_labels(source, node.left, deleted, target, next_internal_id);
        const int right_new = copy_without_labels(source, node.right, deleted, target, next_internal_id);

        if (left_new < 0 && right_new < 0) {
            return -1;
        }

        // Suppress degree-2 nodes.
        if (left_new < 0) {
            return right_new;
        }

        if (right_new < 0) {
            return left_new;
        }

        NewickNode internal;
        internal.id = next_internal_id;
        internal.label = 0;
        internal.left = left_new;
        internal.right = right_new;

        next_internal_id = checked_increment(next_internal_id, "internal id overflow while making top without cluster");

        target.nodes.push_back(internal);
        return static_cast<int>(target.nodes.size() - 1);
    }

    static NewickTree make_top_tree_without_cluster(
        const NewickTree& source,
        const std::vector<std::uint32_t>& cluster_labels,
        std::uint32_t next_internal_id
    ) {
        std::unordered_set<std::uint32_t> deleted;
        deleted.reserve(cluster_labels.size() * 2);

        for (std::uint32_t x : cluster_labels) {
            deleted.insert(x);
        }

        NewickTree result;
        result.nodes.reserve(source.nodes.size());

        result.root = copy_without_labels(source, source.root, deleted, result, next_internal_id);

        if (result.root < 0) {
            throw ClusterDecompositionError("top tree without cluster became empty");
        }

        return result;
    }

    static std::uint32_t count_leaves(const NewickTree& tree) {
        std::uint32_t count = 0;

        for (const NewickNode& node : tree.nodes) {
            if (node.is_leaf()) {
                ++count;
            }
        }

        return count;
    }

    static bool valid_node_index(const NewickTree& tree, int node) {
        return node >= 0 && node < static_cast<int>(tree.nodes.size());
    }

    static Candidate find_best_common_cluster(
        const NewickTree& t1,
        const NewickTree& t2,
        const IndexInfo& idx1,
        const IndexInfo& idx2
    ) {
        const std::uint32_t n = count_leaves(t1);

        if (n != count_leaves(t2)) {
            throw ClusterDecompositionError("trees do not have the same number of leaves");
        }

        if (n < 5) {
            return Candidate{};
        }

        std::unordered_map<HashKey, int, HashKeyHasher> node_in_t2;
        node_in_t2.reserve(t2.nodes.size() * 2);

        for (int v : idx2.postorder) {
            const std::uint32_t size = idx2.leaf_count[static_cast<std::size_t>(v)];

            // Same filter as the .jj helper:
            // ignore size 1, 2, n-1, n.
            if (size <= 2 || size >= n - 1) {
                continue;
            }

            node_in_t2.emplace(key_for(idx2, v), v);
        }

        Candidate best;
        bool found = false;

        for (int u : idx1.postorder) {
            const std::uint32_t size = idx1.leaf_count[static_cast<std::size_t>(u)];

            if (size <= 2 || size >= n - 1) {
                continue;
            }

            auto it = node_in_t2.find(key_for(idx1, u));
            if (it == node_in_t2.end()) {
                continue;
            }

            const int v = it->second;

            // Hashes are used only to find candidates quickly.
            // Verify exactly before accepting.
            if (!same_leaf_set(t1, u, t2, v)) {
                continue;
            }

            const std::uint32_t current_distance =
                static_cast<std::uint32_t>(
                    std::abs(static_cast<int>(2 * size) - static_cast<int>(n))
                );

            std::uint32_t best_distance = std::numeric_limits<std::uint32_t>::max();

            if (found) {
                best_distance =
                    static_cast<std::uint32_t>(
                        std::abs(static_cast<int>(2 * best.size) - static_cast<int>(n))
                    );
            }

            if (!found || current_distance < best_distance) {
                found = true;
                best.node1 = u;
                best.node2 = v;
                best.size = size;
            }
        }

        return best;
    }

public:
    /*
     * Indexed common-cluster view for a tree pair.
     *
     * The template stores only node roots and the exact leaf set. It can be
     * cached across repeated solves of the same reduced pair, then materialized
     * with fresh marker labels when a recursive split is actually taken.
     */
    static ClusterSplitTemplate find_best_common_cluster_template(
        const NewickTree& t1,
        const NewickTree& t2
    ) {
        validate_tree(t1, "tree1");
        validate_tree(t2, "tree2");

        const IndexInfo idx1 = build_index(t1);
        const IndexInfo idx2 = build_index(t2);

        const Candidate best = find_best_common_cluster(t1, t2, idx1, idx2);

        ClusterSplitTemplate result;
        if (best.node1 < 0 || best.node2 < 0) {
            return result;
        }

        result.found = true;
        result.root1 = best.node1;
        result.root2 = best.node2;
        result.size = best.size;
        result.cluster_labels = collect_leaves(t1, best.node1);
        return result;
    }

    static bool template_matches(
        const NewickTree& t1,
        const NewickTree& t2,
        const ClusterSplitTemplate& templ
    ) {
        if (!templ.found) {
            return true;
        }

        if (!valid_node_index(t1, templ.root1) ||
            !valid_node_index(t2, templ.root2)) {
            return false;
        }

        std::vector<std::uint32_t> labels1 = collect_leaves(t1, templ.root1);
        if (labels1 != templ.cluster_labels) {
            return false;
        }

        std::vector<std::uint32_t> labels2 = collect_leaves(t2, templ.root2);
        return labels2 == templ.cluster_labels;
    }

    static ClusterOpenTrees materialize_open_trees(
        const NewickTree& t1,
        const NewickTree& t2,
        const ClusterDecompositionRecord& record
    ) {
        if (!valid_node_index(t1, record.root1) ||
            !valid_node_index(t2, record.root2)) {
            throw ClusterDecompositionError("invalid cluster roots while materializing open trees");
        }

        std::uint32_t next_internal_id =
            std::max(max_node_or_label_id(t1), max_node_or_label_id(t2));

        next_internal_id = std::max(next_internal_id, record.x_up_label);
        next_internal_id = std::max(next_internal_id, record.x_down_label);
        next_internal_id = checked_increment(next_internal_id, "internal id overflow before building open cluster trees");

        ClusterOpenTrees trees;
        trees.bottom1_with_marker =
            make_bottom_tree_with_marker(t1, record.root1, record.x_down_label, next_internal_id);
        trees.bottom2_with_marker =
            make_bottom_tree_with_marker(t2, record.root2, record.x_down_label, next_internal_id);
        trees.top1_with_placeholder =
            make_top_tree_with_placeholder(t1, record.root1, record.x_up_label, next_internal_id);
        trees.top2_with_placeholder =
            make_top_tree_with_placeholder(t2, record.root2, record.x_up_label, next_internal_id);
        return trees;
    }

    static ClusterSplit materialize_split(
        const NewickTree& t1,
        const NewickTree& t2,
        const ClusterSplitTemplate& templ,
        std::uint32_t& next_placeholder_label,
        bool include_open_trees = false
    ) {
        ClusterSplit split;

        if (!templ.found) {
            return split;
        }

        if (!template_matches(t1, t2, templ)) {
            throw ClusterDecompositionError("cached cluster template does not match tree pair");
        }

        const std::uint32_t x_up = next_placeholder_label;
        next_placeholder_label = checked_increment(next_placeholder_label, "placeholder id overflow");

        const std::uint32_t x_down = next_placeholder_label;
        next_placeholder_label = checked_increment(next_placeholder_label, "placeholder id overflow");

        split.found = true;
        split.record.root1 = templ.root1;
        split.record.root2 = templ.root2;
        split.record.cluster_labels = templ.cluster_labels;
        split.record.x_up_label = x_up;
        split.record.x_down_label = x_down;
        split.bottom_leaf_count = templ.size;

        const std::uint32_t n = count_leaves(t1);
        split.top_without_cluster_leaf_count =
            n >= templ.size ? n - templ.size : 0;

        std::uint32_t next_internal_id =
            std::max(max_node_or_label_id(t1), max_node_or_label_id(t2));

        next_internal_id = std::max(next_internal_id, x_up);
        next_internal_id = std::max(next_internal_id, x_down);
        next_internal_id = checked_increment(next_internal_id, "internal id overflow before building cluster split");

        split.bottom1 = make_bottom_tree(t1, templ.root1, next_internal_id);
        split.bottom2 = make_bottom_tree(t2, templ.root2, next_internal_id);

        split.top1_without_cluster =
            make_top_tree_without_cluster(t1, templ.cluster_labels, next_internal_id);
        split.top2_without_cluster =
            make_top_tree_without_cluster(t2, templ.cluster_labels, next_internal_id);

        if (include_open_trees) {
            ClusterOpenTrees open = materialize_open_trees(t1, t2, split.record);
            split.bottom1_with_marker = std::move(open.bottom1_with_marker);
            split.bottom2_with_marker = std::move(open.bottom2_with_marker);
            split.top1_with_placeholder = std::move(open.top1_with_placeholder);
            split.top2_with_placeholder = std::move(open.top2_with_placeholder);
            split.open_trees_materialized = true;
        }

        return split;
    }

    /*
     * Finds one balanced common cluster and builds the closed tree-pair variants.
     * Marker/placeholder OPEN variants are lazy by default and can be requested
     * with include_open_trees.
     *
     * This does NOT solve the subinstances.
     * It only prepares the decomposition.
     */
    static ClusterSplit split_once(
        const NewickTree& t1,
        const NewickTree& t2,
        std::uint32_t& next_placeholder_label,
        bool include_open_trees = false
    ) {
        ClusterSplitTemplate templ =
            find_best_common_cluster_template(t1, t2);
        return materialize_split(
            t1,
            t2,
            templ,
            next_placeholder_label,
            include_open_trees
        );
    }
};

}  // namespace pace26::reductions
