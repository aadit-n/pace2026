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

class ThreeTwoChainReductionError : public std::runtime_error {
public:
    explicit ThreeTwoChainReductionError(const std::string& message)
        : std::runtime_error("3-2 chain reduction error: " + message) {}
};

enum class ThreeTwoSourceTree : std::uint8_t {
    Tree1HasPendant3Chain,
    Tree2HasPendant3Chain
};

struct ThreeTwoChainReductionRecord {
    // The taxon deleted by this reduction.
    // During final reconstruction, this must be added back as a singleton component.
    std::uint32_t deleted_label = 0;

    // In the notation of the rule:
    // pendant_3_chain = (xi, xj, x3)
    // pendant_2_chain = (xi, x3)
    std::uint32_t xi = 0;
    std::uint32_t xj = 0;
    std::uint32_t x3 = 0;

    ThreeTwoSourceTree source = ThreeTwoSourceTree::Tree1HasPendant3Chain;
};

struct ThreeTwoChainReductionOutput {
    bool changed = false;

    NewickTree tree1;
    NewickTree tree2;

    std::vector<ThreeTwoChainReductionRecord> records;

    // All labels removed by this reduction.
    std::unordered_set<std::uint32_t> removed_labels;
};

class ThreeTwoChainReduction {
private:
    struct ParentInfo {
        std::vector<int> parent;
    };

    struct LeafLookup {
        std::unordered_map<std::uint32_t, int> node_of_label;
        std::vector<std::uint32_t> labels;
    };

    static void validate_tree(const NewickTree& tree, const char* name) {
        if (tree.root < 0) {
            throw ThreeTwoChainReductionError(std::string(name) + " has no root");
        }

        if (tree.root >= static_cast<int>(tree.nodes.size())) {
            throw ThreeTwoChainReductionError(std::string(name) + " root index out of range");
        }

        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            const NewickNode& node = tree.nodes[i];

            if (node.is_leaf()) {
                if (node.label == 0) {
                    throw ThreeTwoChainReductionError(std::string(name) + " has leaf label 0");
                }
            } else {
                if (node.left < 0 || node.right < 0) {
                    throw ThreeTwoChainReductionError(std::string(name) + " has non-binary internal node");
                }

                if (node.left >= static_cast<int>(tree.nodes.size()) ||
                    node.right >= static_cast<int>(tree.nodes.size())) {
                    throw ThreeTwoChainReductionError(std::string(name) + " has child index out of range");
                }

                if (node.left == node.right) {
                    throw ThreeTwoChainReductionError(std::string(name) + " has identical left/right child");
                }
            }
        }
    }

    static ParentInfo compute_parent_info(const NewickTree& tree) {
        ParentInfo info;
        info.parent.assign(tree.nodes.size(), -1);

        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            const NewickNode& node = tree.nodes[i];

            if (!node.is_leaf()) {
                info.parent[static_cast<std::size_t>(node.left)] = static_cast<int>(i);
                info.parent[static_cast<std::size_t>(node.right)] = static_cast<int>(i);
            }
        }

        return info;
    }

    static LeafLookup build_leaf_lookup(const NewickTree& tree) {
        LeafLookup lookup;
        lookup.node_of_label.reserve(tree.nodes.size() * 2);
        lookup.labels.reserve((tree.nodes.size() + 1) / 2);

        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            const NewickNode& node = tree.nodes[i];

            if (node.is_leaf()) {
                if (!lookup.node_of_label.emplace(node.label, static_cast<int>(i)).second) {
                    throw ThreeTwoChainReductionError("duplicate leaf label");
                }

                lookup.labels.push_back(node.label);
            }
        }

        return lookup;
    }

    static bool is_leaf_node(const NewickTree& tree, int u) {
        if (u < 0 || u >= static_cast<int>(tree.nodes.size())) {
            throw ThreeTwoChainReductionError("node index out of range");
        }

        return tree.nodes[static_cast<std::size_t>(u)].is_leaf();
    }

    static std::uint32_t cherry_sibling_label(
        const NewickTree& tree,
        const ParentInfo& parent_info,
        int leaf_node
    ) {
        if (!is_leaf_node(tree, leaf_node)) {
            throw ThreeTwoChainReductionError("expected leaf node");
        }

        const int parent = parent_info.parent[static_cast<std::size_t>(leaf_node)];
        if (parent < 0) {
            return 0;
        }

        const NewickNode& p = tree.nodes[static_cast<std::size_t>(parent)];

        int sibling = -1;
        if (p.left == leaf_node) {
            sibling = p.right;
        } else if (p.right == leaf_node) {
            sibling = p.left;
        } else {
            throw ThreeTwoChainReductionError("parent pointer inconsistency");
        }

        if (!is_leaf_node(tree, sibling)) {
            return 0;
        }

        return tree.nodes[static_cast<std::size_t>(sibling)].label;
    }

    static bool leaf_parent_is_grandparent_of_cherry(
        const NewickTree& tree,
        const ParentInfo& parent_info,
        int cherry_leaf_node,
        int upper_leaf_node
    ) {
        if (!is_leaf_node(tree, cherry_leaf_node) || !is_leaf_node(tree, upper_leaf_node)) {
            return false;
        }

        const int cherry_parent = parent_info.parent[static_cast<std::size_t>(cherry_leaf_node)];
        if (cherry_parent < 0) {
            return false;
        }

        const int grandparent = parent_info.parent[static_cast<std::size_t>(cherry_parent)];
        if (grandparent < 0) {
            return false;
        }

        const int upper_parent = parent_info.parent[static_cast<std::size_t>(upper_leaf_node)];
        return upper_parent == grandparent;
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
            throw ThreeTwoChainReductionError(message);
        }

        return value + 1;
    }

    static int copy_filtered_subtree(
        const NewickTree& source,
        int u,
        const std::unordered_set<std::uint32_t>& deleted,
        NewickTree& target,
        std::uint32_t& next_internal_id
    ) {
        if (u < 0 || u >= static_cast<int>(source.nodes.size())) {
            throw ThreeTwoChainReductionError("node index out of range while rebuilding filtered tree");
        }

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

        const int left_new = copy_filtered_subtree(
            source,
            node.left,
            deleted,
            target,
            next_internal_id
        );

        const int right_new = copy_filtered_subtree(
            source,
            node.right,
            deleted,
            target,
            next_internal_id
        );

        if (left_new < 0 && right_new < 0) {
            return -1;
        }

        // Suppress degree-2 internal nodes after deleting a leaf.
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

        next_internal_id = checked_increment(
            next_internal_id,
            "internal node id overflow while rebuilding 3-2 reduced tree"
        );

        target.nodes.push_back(internal);
        return static_cast<int>(target.nodes.size() - 1);
    }

    static NewickTree rebuild_filtered_tree(
        const NewickTree& source,
        const std::unordered_set<std::uint32_t>& deleted,
        std::uint32_t next_internal_id
    ) {
        NewickTree result;
        result.nodes.reserve(source.nodes.size());

        result.root = copy_filtered_subtree(
            source,
            source.root,
            deleted,
            result,
            next_internal_id
        );

        if (result.root < 0) {
            throw ThreeTwoChainReductionError("3-2 reduction deleted all leaves");
        }

        return result;
    }

    static ThreeTwoChainReductionOutput reduce_once(
        const NewickTree& t1,
        const NewickTree& t2
    ) {
        validate_tree(t1, "tree1");
        validate_tree(t2, "tree2");

        const ParentInfo p1 = compute_parent_info(t1);
        const ParentInfo p2 = compute_parent_info(t2);

        const LeafLookup l1 = build_leaf_lookup(t1);
        const LeafLookup l2 = build_leaf_lookup(t2);

        ThreeTwoChainReductionOutput output;
        output.changed = false;

        std::unordered_set<std::uint32_t> used_labels;

        for (std::uint32_t x : l1.labels) {
            if (used_labels.count(x)) {
                continue;
            }

            const auto x1_it = l1.node_of_label.find(x);
            const auto x2_it = l2.node_of_label.find(x);

            if (x1_it == l1.node_of_label.end() || x2_it == l2.node_of_label.end()) {
                throw ThreeTwoChainReductionError("trees do not have the same leaf set");
            }

            const int x1_node = x1_it->second;
            const int x2_node = x2_it->second;

            const std::uint32_t y = cherry_sibling_label(t1, p1, x1_node);
            const std::uint32_t z = cherry_sibling_label(t2, p2, x2_node);

            if (y == 0 || z == 0) {
                continue;
            }

            if (y == z) {
                continue;
            }

            if (used_labels.count(y) || used_labels.count(z)) {
                continue;
            }

            const auto y1_it = l1.node_of_label.find(y);
            const auto y2_it = l2.node_of_label.find(y);
            const auto z1_it = l1.node_of_label.find(z);
            const auto z2_it = l2.node_of_label.find(z);

            if (y1_it == l1.node_of_label.end() ||
                y2_it == l2.node_of_label.end() ||
                z1_it == l1.node_of_label.end() ||
                z2_it == l2.node_of_label.end()) {
                throw ThreeTwoChainReductionError("trees do not have identical leaf sets");
            }

            if (leaf_parent_is_grandparent_of_cherry(
                    t2,
                    p2,
                    x2_node,
                    y2_it->second
                )) {
                ThreeTwoChainReductionRecord record;
                record.deleted_label = z;
                record.xi = x;
                record.xj = z;
                record.x3 = y;
                record.source = ThreeTwoSourceTree::Tree2HasPendant3Chain;

                output.records.push_back(record);
                output.removed_labels.insert(z);
                output.changed = true;

                used_labels.insert(x);
                used_labels.insert(y);
                used_labels.insert(z);
            } else if (leaf_parent_is_grandparent_of_cherry(
                    t1,
                    p1,
                    x1_node,
                    z1_it->second
                )) {
                ThreeTwoChainReductionRecord record;
                record.deleted_label = y;
                record.xi = x;
                record.xj = y;
                record.x3 = z;
                record.source = ThreeTwoSourceTree::Tree1HasPendant3Chain;

                output.records.push_back(record);
                output.removed_labels.insert(y);
                output.changed = true;

                used_labels.insert(x);
                used_labels.insert(y);
                used_labels.insert(z);
            }
        }

        if (output.changed) {
            std::uint32_t next_internal_id = std::max(
                max_node_or_label_id(t1),
                max_node_or_label_id(t2)
            );
            next_internal_id = checked_increment(
                next_internal_id,
                "cannot allocate internal node id for 3-2 reduced trees"
            );

            output.tree1 = rebuild_filtered_tree(t1, output.removed_labels, next_internal_id);
            output.tree2 = rebuild_filtered_tree(t2, output.removed_labels, next_internal_id);
        } else {
            output.tree1 = t1;
            output.tree2 = t2;
        }

        return output;
    }

public:
    /*
     * Exhaustively applies the 3-2-chain reduction.
     *
     * This deliberately applies one deletion at a time and then recomputes parents/cherries.
     * That is safer than batching because a single deletion can suppress internal nodes and
     * create new local 3-2 patterns.
     */
    static ThreeTwoChainReductionOutput reduce(
        const NewickTree& original_t1,
        const NewickTree& original_t2
    ) {
        NewickTree current_t1 = original_t1;
        NewickTree current_t2 = original_t2;

        ThreeTwoChainReductionOutput total;
        total.changed = false;

        while (true) {
            ThreeTwoChainReductionOutput step = reduce_once(current_t1, current_t2);

            if (!step.changed) {
                total.tree1 = std::move(current_t1);
                total.tree2 = std::move(current_t2);
                return total;
            }

            total.changed = true;

            for (const auto& record : step.records) {
                total.records.push_back(record);
                total.removed_labels.insert(record.deleted_label);
            }

            current_t1 = std::move(step.tree1);
            current_t2 = std::move(step.tree2);
        }
    }
};

}  // namespace pace26::reductions