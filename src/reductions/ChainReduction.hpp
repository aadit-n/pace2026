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

class ChainReductionError : public std::runtime_error {
public:
    explicit ChainReductionError(const std::string& message)
        : std::runtime_error("Chain reduction error: " + message) {}
};

struct ChainReductionRecord {
    // Full common chain before reduction, ordered from root toward leaves.
    std::vector<std::uint32_t> full_chain;

    // The three labels retained in the reduced trees.
    std::vector<std::uint32_t> kept_prefix;

    // Deleted labels. These must be restored during final expansion.
    std::vector<std::uint32_t> removed_suffix;
};

struct ChainReductionOutput {
    bool changed = false;

    NewickTree tree1;
    NewickTree tree2;

    std::vector<ChainReductionRecord> records;

    // Labels removed from both trees.
    std::unordered_set<std::uint32_t> removed_labels;
};

class ChainReduction {
private:
    struct ParentInfo {
        std::vector<int> parent;
    };

    struct ChainOccurrence {
        std::vector<std::uint32_t> labels;  // root-to-leaf order
    };

    struct Position {
        int chain_id = -1;
        int pos = -1;
    };

    static void validate_tree(const NewickTree& tree, const char* name) {
        if (tree.root < 0) {
            throw ChainReductionError(std::string(name) + " has no root");
        }

        if (tree.root >= static_cast<int>(tree.nodes.size())) {
            throw ChainReductionError(std::string(name) + " root index out of range");
        }

        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            const NewickNode& node = tree.nodes[i];

            if (node.is_leaf()) {
                if (node.label == 0) {
                    throw ChainReductionError(std::string(name) + " has leaf label 0");
                }
            } else {
                if (node.left < 0 || node.right < 0) {
                    throw ChainReductionError(std::string(name) + " has non-binary internal node");
                }

                if (node.left >= static_cast<int>(tree.nodes.size()) ||
                    node.right >= static_cast<int>(tree.nodes.size())) {
                    throw ChainReductionError(std::string(name) + " has child index out of range");
                }

                if (node.left == node.right) {
                    throw ChainReductionError(std::string(name) + " has identical left/right child");
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

    static bool is_leaf_node(const NewickTree& tree, int u) {
        if (u < 0 || u >= static_cast<int>(tree.nodes.size())) {
            throw ChainReductionError("node index out of range");
        }
        return tree.nodes[static_cast<std::size_t>(u)].is_leaf();
    }

    static int sibling_of(const NewickTree& tree, const ParentInfo& parent_info, int u) {
        const int p = parent_info.parent[static_cast<std::size_t>(u)];
        if (p < 0) {
            return -1;
        }

        const NewickNode& parent = tree.nodes[static_cast<std::size_t>(p)];

        if (parent.left == u) {
            return parent.right;
        }

        if (parent.right == u) {
            return parent.left;
        }

        throw ChainReductionError("parent pointer inconsistency");
    }

    static bool one_leaf_one_internal(
        const NewickTree& tree,
        int u,
        int& leaf_child,
        int& internal_child
    ) {
        const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];

        if (node.is_leaf()) {
            return false;
        }

        const bool left_leaf = is_leaf_node(tree, node.left);
        const bool right_leaf = is_leaf_node(tree, node.right);

        if (left_leaf == right_leaf) {
            return false;
        }

        if (left_leaf) {
            leaf_child = node.left;
            internal_child = node.right;
        } else {
            leaf_child = node.right;
            internal_child = node.left;
        }

        return true;
    }

    static bool both_children_are_leaves(const NewickTree& tree, int u) {
        const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];

        if (node.is_leaf()) {
            return false;
        }

        return is_leaf_node(tree, node.left) && is_leaf_node(tree, node.right);
    }

    static std::vector<ChainOccurrence> extract_maximal_rooted_chains(const NewickTree& tree) {
        validate_tree(tree, "tree");

        const ParentInfo parent_info = compute_parent_info(tree);

        std::vector<ChainOccurrence> chains;

        for (std::size_t idx = 0; idx < tree.nodes.size(); ++idx) {
            const int u = static_cast<int>(idx);
            const NewickNode& node = tree.nodes[idx];

            if (node.is_leaf()) {
                continue;
            }

            int leaf_child = -1;
            int internal_child = -1;

            if (!one_leaf_one_internal(tree, u, leaf_child, internal_child)) {
                continue;
            }

            // Maximal-start test:
            // If u's sibling is a leaf, then the chain actually starts one level above u.
            const int parent = parent_info.parent[idx];
            if (parent >= 0) {
                const int sib = sibling_of(tree, parent_info, u);
                if (sib >= 0 && is_leaf_node(tree, sib)) {
                    continue;
                }
            }

            ChainOccurrence chain;
            chain.labels.push_back(tree.nodes[static_cast<std::size_t>(leaf_child)].label);

            int cur = internal_child;

            while (cur >= 0) {
                if (both_children_are_leaves(tree, cur)) {
                    const NewickNode& bottom = tree.nodes[static_cast<std::size_t>(cur)];

                    std::uint32_t a = tree.nodes[static_cast<std::size_t>(bottom.left)].label;
                    std::uint32_t b = tree.nodes[static_cast<std::size_t>(bottom.right)].label;

                    // Bottom cherry order is topologically irrelevant.
                    // Canonicalise it so both trees produce the same sequence.
                    if (a > b) {
                        std::swap(a, b);
                    }

                    chain.labels.push_back(a);
                    chain.labels.push_back(b);
                    break;
                }

                int next_leaf = -1;
                int next_internal = -1;

                if (!one_leaf_one_internal(tree, cur, next_leaf, next_internal)) {
                    break;
                }

                chain.labels.push_back(tree.nodes[static_cast<std::size_t>(next_leaf)].label);
                cur = next_internal;
            }

            if (chain.labels.size() >= 2) {
                chains.push_back(std::move(chain));
            }
        }

        return chains;
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
            throw ChainReductionError(message);
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
            throw ChainReductionError("node index out of range while rebuilding filtered tree");
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

        // Suppress degree-2 nodes after deletion.
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
            "internal node id overflow while rebuilding chain-reduced tree"
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
            throw ChainReductionError("chain reduction deleted all leaves");
        }

        return result;
    }

public:
    /*
     * Classic two-tree rooted common chain reduction.
     *
     * Rule:
     *   If C = (x1, x2, ..., xm) is a common rooted chain with m >= 4,
     *   keep x1, x2, x3 and delete x4..xm from both trees.
     *
     * This should be run after common subtree reduction and then repeated
     * together with subtree reduction until neither changes anything.
     */
    static ChainReductionOutput reduce(const NewickTree& t1, const NewickTree& t2) {
        validate_tree(t1, "tree1");
        validate_tree(t2, "tree2");

        const auto chains1 = extract_maximal_rooted_chains(t1);
        const auto chains2 = extract_maximal_rooted_chains(t2);

        std::unordered_map<std::uint32_t, Position> pos_in_t2;
        pos_in_t2.reserve(t2.nodes.size() * 2);

        for (int cid = 0; cid < static_cast<int>(chains2.size()); ++cid) {
            const auto& labels = chains2[static_cast<std::size_t>(cid)].labels;

            for (int p = 0; p < static_cast<int>(labels.size()); ++p) {
                pos_in_t2.emplace(labels[static_cast<std::size_t>(p)], Position{cid, p});
            }
        }

        ChainReductionOutput output;

        std::unordered_set<std::uint32_t> labels_already_used;
        labels_already_used.reserve(t1.nodes.size() * 2);

        for (const ChainOccurrence& chain1 : chains1) {
            const auto& a = chain1.labels;
            std::size_t i = 0;

            while (i < a.size()) {
                auto it = pos_in_t2.find(a[i]);
                if (it == pos_in_t2.end()) {
                    ++i;
                    continue;
                }

                const Position start = it->second;
                const auto& b = chains2[static_cast<std::size_t>(start.chain_id)].labels;

                std::size_t j = i;

                while (j < a.size()) {
                    auto jt = pos_in_t2.find(a[j]);
                    if (jt == pos_in_t2.end()) {
                        break;
                    }

                    const Position q = jt->second;

                    const int expected_pos =
                        start.pos + static_cast<int>(j - i);

                    if (q.chain_id != start.chain_id || q.pos != expected_pos) {
                        break;
                    }

                    if (q.pos < 0 || q.pos >= static_cast<int>(b.size())) {
                        break;
                    }

                    if (b[static_cast<std::size_t>(q.pos)] != a[j]) {
                        break;
                    }

                    ++j;
                }

                const std::size_t len = j - i;

                if (len >= 4) {
                    bool overlaps_previous_reduction = false;

                    for (std::size_t k = i; k < j; ++k) {
                        if (labels_already_used.find(a[k]) != labels_already_used.end()) {
                            overlaps_previous_reduction = true;
                            break;
                        }
                    }

                    if (!overlaps_previous_reduction) {
                        ChainReductionRecord record;

                        for (std::size_t k = i; k < j; ++k) {
                            record.full_chain.push_back(a[k]);
                            labels_already_used.insert(a[k]);
                        }

                        record.kept_prefix.push_back(a[i]);
                        record.kept_prefix.push_back(a[i + 1]);
                        record.kept_prefix.push_back(a[i + 2]);

                        for (std::size_t k = i + 3; k < j; ++k) {
                            record.removed_suffix.push_back(a[k]);
                            output.removed_labels.insert(a[k]);
                        }

                        output.records.push_back(std::move(record));
                    }
                }

                if (j == i) {
                    ++i;
                } else {
                    i = j;
                }
            }
        }

        output.changed = !output.removed_labels.empty();

        if (!output.changed) {
            output.tree1 = t1;
            output.tree2 = t2;
            return output;
        }

        std::uint32_t next_internal_id = std::max(
            max_node_or_label_id(t1),
            max_node_or_label_id(t2)
        );

        next_internal_id = checked_increment(
            next_internal_id,
            "cannot allocate internal node id for chain-reduced trees"
        );

        output.tree1 = rebuild_filtered_tree(t1, output.removed_labels, next_internal_id);
        output.tree2 = rebuild_filtered_tree(t2, output.removed_labels, next_internal_id);

        return output;
    }
};

}  // namespace pace26::reductions