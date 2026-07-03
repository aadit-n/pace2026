#pragma once

#include "io/NewickParser.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pace26::reductions {

using pace26::io::NewickNode;
using pace26::io::NewickTree;

class CommonSubtreeReductionError : public std::runtime_error {
public:
    explicit CommonSubtreeReductionError(const std::string& message)
        : std::runtime_error("Common subtree reduction error: " + message) {}
};

struct CommonSubtreeRecord {
    // Placeholder leaf label used in the reduced trees.
    std::uint32_t placeholder_label = 0;

    // Root node index of the contracted subtree in the original trees.
    int root1 = -1;
    int root2 = -1;

    // Exact canonical signature ID.
    int canonical_id = 0;

    // Number of original leaves inside this contracted subtree.
    std::uint32_t leaf_count = 0;

    // Original labels contained in this subtree.
    std::vector<std::uint32_t> leaves;

    // Newick representation of the original subtree without final semicolon.
    // Useful for expansion while writing final forest components.
    std::string expansion_newick;
};

struct CommonSubtreeReductionOutput {
    bool changed = false;

    NewickTree tree1;
    NewickTree tree2;

    std::vector<CommonSubtreeRecord> records;

    // Root node index in original tree -> placeholder label.
    std::unordered_map<int, std::uint32_t> replace1;
    std::unordered_map<int, std::uint32_t> replace2;
};

inline std::uint32_t checked_increment(std::uint32_t x, const char* message) {
    if (x == std::numeric_limits<std::uint32_t>::max()) {
        throw CommonSubtreeReductionError(message);
    }
    return x + 1;
}

inline std::uint32_t max_node_or_label_id(const NewickTree& tree) {
    std::uint32_t result = 0;

    for (const NewickNode& node : tree.nodes) {
        result = std::max(result, node.id);
        result = std::max(result, node.label);
    }

    return result;
}

inline std::uint32_t choose_next_placeholder_label(const NewickTree& t1, const NewickTree& t2) {
    const std::uint32_t max1 = max_node_or_label_id(t1);
    const std::uint32_t max2 = max_node_or_label_id(t2);
    const std::uint32_t mx = std::max(max1, max2);

    if (mx == std::numeric_limits<std::uint32_t>::max()) {
        throw CommonSubtreeReductionError("cannot allocate placeholder label");
    }

    return mx + 1;
}

class SignatureInterner {
private:
    struct Key {
        std::uint8_t type = 0; // 0 = leaf, 1 = internal
        std::uint64_t a = 0;
        std::uint64_t b = 0;

        bool operator==(const Key& other) const noexcept {
            return type == other.type && a == other.a && b == other.b;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            std::uint64_t x = 1469598103934665603ULL;
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

    std::unordered_map<Key, int, KeyHash> id_of_;
    int next_id_ = 1;

public:
    int leaf(std::uint32_t label) {
        Key key;
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
            throw CommonSubtreeReductionError("invalid child signature");
        }

        std::uint64_t a = static_cast<std::uint64_t>(left_sig);
        std::uint64_t b = static_cast<std::uint64_t>(right_sig);

        // The input trees are unordered bifurcating trees.
        // Therefore (A,B) and (B,A) are the same rooted topology.
        if (a > b) {
            std::swap(a, b);
        }

        Key key;
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

class CommonSubtreeReduction {
private:
    struct SignatureInfo {
        std::vector<int> postorder;
        std::vector<int> sig;
        std::vector<std::uint32_t> leaf_count;
    };

    struct Candidate {
        std::uint32_t leaf_count = 0;
        int sig = 0;
        int root1 = -1;
        int root2 = -1;
    };

    static void validate_tree(const NewickTree& tree, const char* name) {
        if (tree.root < 0) {
            throw CommonSubtreeReductionError(std::string(name) + " has no root");
        }

        if (tree.root >= static_cast<int>(tree.nodes.size())) {
            throw CommonSubtreeReductionError(std::string(name) + " root index is out of range");
        }

        for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
            const NewickNode& node = tree.nodes[i];

            if (node.is_leaf()) {
                if (node.label == 0) {
                    throw CommonSubtreeReductionError(std::string(name) + " has a leaf with label 0");
                }
            } else {
                if (node.left < 0 || node.right < 0) {
                    throw CommonSubtreeReductionError(std::string(name) + " has a non-binary internal node");
                }

                if (node.left >= static_cast<int>(tree.nodes.size()) ||
                    node.right >= static_cast<int>(tree.nodes.size())) {
                    throw CommonSubtreeReductionError(std::string(name) + " has child index out of range");
                }

                if (node.left == node.right) {
                    throw CommonSubtreeReductionError(std::string(name) + " has identical left/right child index");
                }
            }
        }
    }

    static std::vector<int> iterative_postorder(const NewickTree& tree) {
        validate_tree(tree, "tree");

        std::vector<int> order;
        order.reserve(tree.nodes.size());

        std::vector<std::pair<int, bool>> stack;
        stack.reserve(tree.nodes.size());
        stack.push_back({tree.root, false});

        while (!stack.empty()) {
            const auto [u, seen] = stack.back();
            stack.pop_back();

            if (u < 0 || u >= static_cast<int>(tree.nodes.size())) {
                throw CommonSubtreeReductionError("node index out of range during traversal");
            }

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

    static SignatureInfo compute_signatures(const NewickTree& tree, SignatureInterner& interner) {
        SignatureInfo info;

        const std::size_t m = tree.nodes.size();

        info.postorder = iterative_postorder(tree);
        info.sig.assign(m, 0);
        info.leaf_count.assign(m, 0);

        for (int u : info.postorder) {
            const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];

            if (node.is_leaf()) {
                info.sig[static_cast<std::size_t>(u)] = interner.leaf(node.label);
                info.leaf_count[static_cast<std::size_t>(u)] = 1;
            } else {
                const int l = node.left;
                const int r = node.right;

                const int left_sig = info.sig[static_cast<std::size_t>(l)];
                const int right_sig = info.sig[static_cast<std::size_t>(r)];

                info.sig[static_cast<std::size_t>(u)] = interner.internal(left_sig, right_sig);

                const std::uint32_t left_count = info.leaf_count[static_cast<std::size_t>(l)];
                const std::uint32_t right_count = info.leaf_count[static_cast<std::size_t>(r)];

                if (left_count > std::numeric_limits<std::uint32_t>::max() - right_count) {
                    throw CommonSubtreeReductionError("leaf count overflow");
                }

                info.leaf_count[static_cast<std::size_t>(u)] = left_count + right_count;
            }
        }

        return info;
    }

    static void mark_subtree_blocked(const NewickTree& tree, int root, std::vector<char>& blocked) {
        std::vector<int> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();

            if (u < 0 || u >= static_cast<int>(tree.nodes.size())) {
                throw CommonSubtreeReductionError("node index out of range while marking subtree");
            }

            if (blocked[static_cast<std::size_t>(u)]) {
                continue;
            }

            blocked[static_cast<std::size_t>(u)] = 1;

            const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];
            if (!node.is_leaf()) {
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }
    }

    static std::vector<std::uint32_t> collect_leaves(const NewickTree& tree, int root) {
        std::vector<std::uint32_t> leaves;

        std::vector<int> stack;
        stack.push_back(root);

        while (!stack.empty()) {
            const int u = stack.back();
            stack.pop_back();

            if (u < 0 || u >= static_cast<int>(tree.nodes.size())) {
                throw CommonSubtreeReductionError("node index out of range while collecting leaves");
            }

            const NewickNode& node = tree.nodes[static_cast<std::size_t>(u)];

            if (node.is_leaf()) {
                leaves.push_back(node.label);
            } else {
                stack.push_back(node.left);
                stack.push_back(node.right);
            }
        }

        std::sort(leaves.begin(), leaves.end());
        return leaves;
    }

    static std::string subtree_newick(const NewickTree& tree, int root) {
        if (root < 0 || root >= static_cast<int>(tree.nodes.size())) {
            throw CommonSubtreeReductionError("node index out of range while writing Newick");
        }

        const NewickNode& node = tree.nodes[static_cast<std::size_t>(root)];

        if (node.is_leaf()) {
            return std::to_string(node.label);
        }

        return "(" + subtree_newick(tree, node.left) + "," + subtree_newick(tree, node.right) + ")";
    }

    static int copy_reduced_subtree(
        const NewickTree& source,
        int u,
        const std::unordered_map<int, std::uint32_t>& replacement,
        NewickTree& target,
        std::uint32_t& next_internal_id
    ) {
        auto replace_it = replacement.find(u);
        if (replace_it != replacement.end()) {
            NewickNode leaf;
            leaf.id = replace_it->second;
            leaf.label = replace_it->second;
            leaf.left = -1;
            leaf.right = -1;

            target.nodes.push_back(leaf);
            return static_cast<int>(target.nodes.size() - 1);
        }

        const NewickNode& node = source.nodes[static_cast<std::size_t>(u)];

        if (node.is_leaf()) {
            NewickNode leaf;
            leaf.id = node.id;
            leaf.label = node.label;
            leaf.left = -1;
            leaf.right = -1;

            target.nodes.push_back(leaf);
            return static_cast<int>(target.nodes.size() - 1);
        }

        const int new_left = copy_reduced_subtree(source, node.left, replacement, target, next_internal_id);
        const int new_right = copy_reduced_subtree(source, node.right, replacement, target, next_internal_id);

        NewickNode internal;
        internal.id = next_internal_id;
        internal.label = 0;
        internal.left = new_left;
        internal.right = new_right;

        next_internal_id = checked_increment(next_internal_id, "internal node id overflow while rebuilding reduced tree");

        target.nodes.push_back(internal);
        return static_cast<int>(target.nodes.size() - 1);
    }

    static NewickTree rebuild_reduced_tree(
        const NewickTree& source,
        const std::unordered_map<int, std::uint32_t>& replacement,
        std::uint32_t next_internal_id
    ) {
        NewickTree reduced;
        reduced.nodes.reserve(source.nodes.size());

        reduced.root = copy_reduced_subtree(source, source.root, replacement, reduced, next_internal_id);

        return reduced;
    }

public:
    /*
     * Contracts every maximal common pendant subtree with at least min_leaf_count leaves.
     *
     * next_placeholder_label is passed by reference because this reduction may be called
     * repeatedly together with chain reductions, cluster reductions, etc.
     *
     * Recommended initial value:
     *     auto next = choose_next_placeholder_label(t1, t2);
     */
    static CommonSubtreeReductionOutput reduce(
        const NewickTree& t1,
        const NewickTree& t2,
        std::uint32_t& next_placeholder_label,
        std::uint32_t min_leaf_count = 2
    ) {
        validate_tree(t1, "tree1");
        validate_tree(t2, "tree2");

        if (min_leaf_count < 2) {
            throw CommonSubtreeReductionError("min_leaf_count must be at least 2");
        }

        SignatureInterner interner;

        SignatureInfo s1 = compute_signatures(t1, interner);
        SignatureInfo s2 = compute_signatures(t2, interner);

        std::unordered_map<int, int> node_of_signature_1;
        std::unordered_map<int, int> node_of_signature_2;

        node_of_signature_1.reserve(t1.nodes.size() * 2);
        node_of_signature_2.reserve(t2.nodes.size() * 2);

        for (int u : s1.postorder) {
            const std::uint32_t count = s1.leaf_count[static_cast<std::size_t>(u)];
            if (count >= min_leaf_count) {
                const int sig = s1.sig[static_cast<std::size_t>(u)];
                node_of_signature_1.emplace(sig, u);
            }
        }

        for (int v : s2.postorder) {
            const std::uint32_t count = s2.leaf_count[static_cast<std::size_t>(v)];
            if (count >= min_leaf_count) {
                const int sig = s2.sig[static_cast<std::size_t>(v)];
                node_of_signature_2.emplace(sig, v);
            }
        }

        std::vector<Candidate> candidates;

        for (const auto& [sig, u] : node_of_signature_1) {
            auto it = node_of_signature_2.find(sig);
            if (it == node_of_signature_2.end()) {
                continue;
            }

            const int v = it->second;

            const std::uint32_t count1 = s1.leaf_count[static_cast<std::size_t>(u)];
            const std::uint32_t count2 = s2.leaf_count[static_cast<std::size_t>(v)];

            if (count1 != count2) {
                throw CommonSubtreeReductionError("equal signatures produced different leaf counts");
            }

            Candidate c;
            c.leaf_count = count1;
            c.sig = sig;
            c.root1 = u;
            c.root2 = v;
            candidates.push_back(c);
        }

        // Maximal-first selection.
        // This prevents contracting both a big common subtree and its children.
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

        CommonSubtreeReductionOutput output;

        std::vector<char> blocked1(t1.nodes.size(), 0);
        std::vector<char> blocked2(t2.nodes.size(), 0);

        for (const Candidate& c : candidates) {
            if (blocked1[static_cast<std::size_t>(c.root1)] ||
                blocked2[static_cast<std::size_t>(c.root2)]) {
                continue;
            }

            const std::uint32_t placeholder = next_placeholder_label;
            next_placeholder_label = checked_increment(
                next_placeholder_label,
                "placeholder label overflow"
            );

            CommonSubtreeRecord record;
            record.placeholder_label = placeholder;
            record.root1 = c.root1;
            record.root2 = c.root2;
            record.canonical_id = c.sig;
            record.leaf_count = c.leaf_count;
            record.leaves = collect_leaves(t1, c.root1);
            record.expansion_newick = subtree_newick(t1, c.root1);

            output.records.push_back(std::move(record));
            output.replace1.emplace(c.root1, placeholder);
            output.replace2.emplace(c.root2, placeholder);

            mark_subtree_blocked(t1, c.root1, blocked1);
            mark_subtree_blocked(t2, c.root2, blocked2);
        }

        output.changed = !output.records.empty();

        if (!output.changed) {
            output.tree1 = t1;
            output.tree2 = t2;
            return output;
        }

        std::uint32_t next_internal_id = std::max(
            std::max(max_node_or_label_id(t1), max_node_or_label_id(t2)),
            next_placeholder_label
        );

        next_internal_id = checked_increment(
            next_internal_id,
            "cannot allocate internal node id for reduced trees"
        );

        output.tree1 = rebuild_reduced_tree(t1, output.replace1, next_internal_id);
        output.tree2 = rebuild_reduced_tree(t2, output.replace2, next_internal_id);

        return output;
    }
};

}  // namespace pace26::reductions