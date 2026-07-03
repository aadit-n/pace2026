#pragma once

#include "core/Forest.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <map>

namespace pace26::exact {

class InProcessExactMafError : public std::runtime_error {
public:
    explicit InProcessExactMafError(const std::string& message)
        : std::runtime_error("In-process exact MAF error: " + message) {}
};

class InProcessExactMaf {
private:
    using Tree = pace26::core::Tree;
    using LabelForest = pace26::core::LabelForest;
    using LabelComponent = pace26::core::LabelComponent;
    using Timer = pace26::core::Timer;

public:
    struct Options {
        bool enabled = true;
        std::size_t max_leaves = 64;
        double local_time_limit_seconds = 3.0;
        double global_guard_seconds = 1.0;
        double local_guard_seconds = 0.02;
        std::uint64_t max_candidate_masks_tested = 3000000ULL;
    };

    InProcessExactMaf()
        : options_(Options{}) {}

    explicit InProcessExactMaf(Options options)
        : options_(std::move(options)) {}

    std::optional<LabelForest> solve(
        const Tree& t1,
        const Tree& t2,
        const Timer* global_timer = nullptr,
        const LabelForest* incumbent_hint = nullptr
    ) const {
        if (!options_.enabled) {
            return std::nullopt;
        }

        require_same_leaf_set(t1, t2);

        const std::size_t n = t1.leaf_labels.size();
        if (n == 0 || n > options_.max_leaves) {
            return std::nullopt;
        }

        if (global_timer != nullptr &&
            global_timer->should_stop(options_.global_guard_seconds)) {
            return std::nullopt;
        }

        try {
            Solver solver(t1, t2, options_, global_timer, incumbent_hint);
            return solver.run();
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

private:
    struct SubCacheEntry {
        int cuts = 0;
        std::vector<std::vector<std::uint32_t>> components;
    };

    class Solver {
    public:
        const Tree& t1;
        const Tree& t2;
        const Options& options;
        const Timer* global_timer;
        Timer local_timer;

        struct Node {
            int left = -1;
            int right = -1;
            int parent = -1;
            bool active = true;
            bool is_leaf = false;
            std::uint32_t label = 0;
        };

        std::vector<Node> tree1;
        std::vector<Node> tree2;
        int root1 = -1;
        int root2 = -1;

        int active_leaves = 0;

        std::unordered_map<std::uint32_t, int> leaf_to_node1;
        std::unordered_map<std::uint32_t, int> leaf_to_node2;

        std::vector<std::vector<std::uint32_t>> contracted;

        enum class UndoType {
            SET_LEFT,
            SET_RIGHT,
            SET_PARENT,
            SET_ACTIVE,
            SET_ROOT,
            SET_ACTIVE_LEAVES,
            MERGE_CONTRACTED
        };

        struct UndoItem {
            UndoType type;
            int tree_idx; // 0 for tree1, 1 for tree2
            int node_idx;
            int old_val;
        };

        std::vector<UndoItem> undo_stack;
        std::vector<std::vector<std::uint32_t>> cut_components;

        int best_cuts = std::numeric_limits<int>::max();
        std::vector<std::vector<std::uint32_t>> best_components;

        mutable std::map<std::vector<std::uint32_t>, SubCacheEntry> sub_cache;
        std::uint64_t nodes_visited = 0;
        bool aborted = false;

        Solver(
            const Tree& input_t1,
            const Tree& input_t2,
            const Options& input_options,
            const Timer* input_global_timer,
            const LabelForest* incumbent_hint
        )
            : t1(input_t1),
              t2(input_t2),
              options(input_options),
              global_timer(input_global_timer),
              local_timer(std::max(0.001, input_options.local_time_limit_seconds)) {

            const int n1 = t1.node_count();
            tree1.resize(n1);
            for (int i = 0; i < n1; ++i) {
                tree1[i].left = t1.left(i);
                tree1[i].right = t1.right(i);
                tree1[i].parent = t1.parent[i];
                tree1[i].active = true;
                tree1[i].is_leaf = t1.is_leaf(i);
                tree1[i].label = t1.is_leaf(i) ? t1.label(i) : 0;
                if (tree1[i].is_leaf) {
                    leaf_to_node1[tree1[i].label] = i;
                }
            }
            root1 = t1.root;

            const int n2 = t2.node_count();
            tree2.resize(n2);
            for (int i = 0; i < n2; ++i) {
                tree2[i].left = t2.left(i);
                tree2[i].right = t2.right(i);
                tree2[i].parent = t2.parent[i];
                tree2[i].active = true;
                tree2[i].is_leaf = t2.is_leaf(i);
                tree2[i].label = t2.is_leaf(i) ? t2.label(i) : 0;
                if (tree2[i].is_leaf) {
                    leaf_to_node2[tree2[i].label] = i;
                }
            }
            root2 = t2.root;

            active_leaves = t1.leaf_count();

            contracted.resize(n1);
            for (int i = 0; i < n1; ++i) {
                if (tree1[i].is_leaf) {
                    contracted[i] = {tree1[i].label};
                }
            }

            if (incumbent_hint != nullptr) {
                best_cuts = incumbent_hint->component_count() - 1;
            }
        }

        std::optional<LabelForest> run() {
            search(0);

            if (best_cuts == std::numeric_limits<int>::max()) {
                return std::nullopt;
            }

            if (best_components.empty()) {
                return std::nullopt;
            }

            LabelForest forest;
            for (const auto& comp_labels : best_components) {
                forest.components.push_back(LabelComponent(comp_labels));
            }
            forest.normalize();
            forest.validate_partition_of(t1.leaf_labels);
            return forest;
        }

    private:
        void set_left(int tree_idx, int node, int val) {
            auto& t = (tree_idx == 0) ? tree1 : tree2;
            undo_stack.push_back({UndoType::SET_LEFT, tree_idx, node, t[node].left});
            t[node].left = val;
        }

        void set_right(int tree_idx, int node, int val) {
            auto& t = (tree_idx == 0) ? tree1 : tree2;
            undo_stack.push_back({UndoType::SET_RIGHT, tree_idx, node, t[node].right});
            t[node].right = val;
        }

        void set_parent(int tree_idx, int node, int val) {
            auto& t = (tree_idx == 0) ? tree1 : tree2;
            undo_stack.push_back({UndoType::SET_PARENT, tree_idx, node, t[node].parent});
            t[node].parent = val;
        }

        void set_active(int tree_idx, int node, bool val) {
            auto& t = (tree_idx == 0) ? tree1 : tree2;
            undo_stack.push_back({UndoType::SET_ACTIVE, tree_idx, node, static_cast<int>(t[node].active)});
            t[node].active = val;
        }

        void set_root(int tree_idx, int val) {
            auto& r = (tree_idx == 0) ? root1 : root2;
            undo_stack.push_back({UndoType::SET_ROOT, tree_idx, -1, r});
            r = val;
        }

        void set_active_leaves(int value) {
            undo_stack.push_back({UndoType::SET_ACTIVE_LEAVES, 0, -1, active_leaves});
            active_leaves = value;
        }

        void rollback(std::size_t watermark) {
            while (undo_stack.size() > watermark) {
                auto item = undo_stack.back();
                undo_stack.pop_back();
                auto& t = (item.tree_idx == 0) ? tree1 : tree2;
                switch (item.type) {
                    case UndoType::SET_LEFT:
                        t[item.node_idx].left = item.old_val;
                        break;
                    case UndoType::SET_RIGHT:
                        t[item.node_idx].right = item.old_val;
                        break;
                    case UndoType::SET_PARENT:
                        t[item.node_idx].parent = item.old_val;
                        break;
                    case UndoType::SET_ACTIVE:
                        t[item.node_idx].active = static_cast<bool>(item.old_val);
                        break;
                    case UndoType::SET_ROOT:
                        if (item.tree_idx == 0) root1 = item.old_val;
                        else root2 = item.old_val;
                        break;
                    case UndoType::SET_ACTIVE_LEAVES:
                        active_leaves = item.old_val;
                        break;
                    case UndoType::MERGE_CONTRACTED:
                        contracted[item.node_idx].resize(static_cast<std::size_t>(item.old_val));
                        break;
                }
            }
        }

        void deactivate_leaf(int tree_idx, int leaf) {
            auto& t = (tree_idx == 0) ? tree1 : tree2;
            set_active(tree_idx, leaf, false);
            int p = t[leaf].parent;
            if (p == -1) {
                set_root(tree_idx, -1);
                return;
            }
            int sib = (t[p].left == leaf) ? t[p].right : t[p].left;
            int g = t[p].parent;
            set_active(tree_idx, p, false);
            set_parent(tree_idx, sib, g);
            if (g == -1) {
                set_root(tree_idx, sib);
            } else {
                if (t[g].left == p) {
                    set_left(tree_idx, g, sib);
                } else {
                    set_right(tree_idx, g, sib);
                }
            }
        }

        std::vector<std::uint32_t> component_for_active_label(std::uint32_t label) const {
            auto it = leaf_to_node1.find(label);
            if (it == leaf_to_node1.end()) {
                return {label};
            }

            const auto& labels = contracted[static_cast<std::size_t>(it->second)];
            if (labels.empty()) {
                return {label};
            }

            return labels;
        }

        std::vector<std::uint32_t> expand_representative_component(
            const std::vector<std::uint32_t>& labels
        ) const {
            std::vector<std::uint32_t> expanded;

            for (std::uint32_t label : labels) {
                std::vector<std::uint32_t> component = component_for_active_label(label);
                expanded.insert(expanded.end(), component.begin(), component.end());
            }

            std::sort(expanded.begin(), expanded.end());
            expanded.erase(std::unique(expanded.begin(), expanded.end()), expanded.end());
            return expanded;
        }

        void cut_active_label(std::uint32_t label) {
            deactivate_leaf(0, leaf_to_node1.at(label));
            deactivate_leaf(1, leaf_to_node2.at(label));
            set_active_leaves(active_leaves - 1);
        }

        bool is_cherry(int tree_idx, int p) const {
            const auto& t = (tree_idx == 0) ? tree1 : tree2;
            if (!t[p].active || t[p].is_leaf) return false;
            int l = t[p].left;
            int r = t[p].right;
            return l != -1 && r != -1 && t[l].active && t[l].is_leaf && t[r].active && t[r].is_leaf;
        }

        bool contract_cherries() {
            bool changed = false;
            for (std::size_t i = 0; i < tree1.size(); ++i) {
                if (is_cherry(0, static_cast<int>(i))) {
                    int l1 = tree1[i].left;
                    int r1 = tree1[i].right;
                    std::uint32_t lbl_l = tree1[l1].label;
                    std::uint32_t lbl_r = tree1[r1].label;

                    int l2 = leaf_to_node2[lbl_l];
                    int r2 = leaf_to_node2[lbl_r];
                    if (tree2[l2].active && tree2[r2].active) {
                        int p2 = tree2[l2].parent;
                        if (p2 != -1 && p2 == tree2[r2].parent) {
                            deactivate_leaf(0, r1);
                            deactivate_leaf(1, r2);

                            undo_stack.push_back({UndoType::MERGE_CONTRACTED, 0, l1, static_cast<int>(contracted[l1].size())});
                            contracted[l1].insert(contracted[l1].end(), contracted[r1].begin(), contracted[r1].end());

                            set_active_leaves(active_leaves - 1);
                            changed = true;
                        }
                    }
                }
            }
            return changed;
        }

        void run_all_contractions() {
            while (contract_cherries()) {}
        }

        std::vector<int> get_active_cherries1() const {
            std::vector<int> cherries;
            for (std::size_t i = 0; i < tree1.size(); ++i) {
                if (is_cherry(0, static_cast<int>(i))) {
                    cherries.push_back(static_cast<int>(i));
                }
            }
            return cherries;
        }

        int choose_branch_cherry(const std::vector<int>& cherries) const {
            int best = cherries.front();
            std::size_t best_hanging_count = std::numeric_limits<std::size_t>::max();
            std::uint32_t best_min_label = std::numeric_limits<std::uint32_t>::max();
            std::uint32_t best_max_label = std::numeric_limits<std::uint32_t>::max();

            for (int cherry : cherries) {
                const int l1 = tree1[cherry].left;
                const int r1 = tree1[cherry].right;
                const std::uint32_t a = tree1[l1].label;
                const std::uint32_t b = tree1[r1].label;
                const int a2 = leaf_to_node2.at(a);
                const int b2 = leaf_to_node2.at(b);
                const std::size_t hanging_count =
                    get_hanging_subtree_roots(a2, b2).size();
                const std::uint32_t min_label = std::min(a, b);
                const std::uint32_t max_label = std::max(a, b);

                if (hanging_count < best_hanging_count ||
                    (hanging_count == best_hanging_count &&
                     (min_label < best_min_label ||
                      (min_label == best_min_label && max_label < best_max_label)))) {
                    best = cherry;
                    best_hanging_count = hanging_count;
                    best_min_label = min_label;
                    best_max_label = max_label;
                }
            }

            return best;
        }

        int compute_packing_lower_bound() const {
            std::vector<bool> used_leaves(tree1.size(), false);
            int pack_size = 0;
            for (std::size_t i = 0; i < tree1.size(); ++i) {
                if (is_cherry(0, static_cast<int>(i))) {
                    int l1 = tree1[i].left;
                    int r1 = tree1[i].right;
                    if (used_leaves[l1] || used_leaves[r1]) continue;

                    std::uint32_t lbl_l = tree1[l1].label;
                    std::uint32_t lbl_r = tree1[r1].label;
                    int l2 = leaf_to_node2.at(lbl_l);
                    int r2 = leaf_to_node2.at(lbl_r);

                    bool is_common = false;
                    if (tree2[l2].active && tree2[r2].active) {
                        int p2 = tree2[l2].parent;
                        if (p2 != -1 && p2 == tree2[r2].parent) {
                            is_common = true;
                        }
                    }

                    if (!is_common) {
                        pack_size++;
                        used_leaves[l1] = true;
                        used_leaves[r1] = true;
                    }
                }
            }
            return pack_size;
        }

        int find_lca2(int u, int v) const {
            std::vector<bool> visited(tree2.size(), false);
            int curr = u;
            while (curr != -1) {
                visited[curr] = true;
                curr = tree2[curr].parent;
            }
            curr = v;
            while (curr != -1) {
                if (visited[curr]) return curr;
                curr = tree2[curr].parent;
            }
            return -1;
        }

        std::vector<int> get_hanging_subtree_roots(int a, int b) const {
            int g = find_lca2(a, b);
            std::unordered_set<int> path_nodes;

            int curr = a;
            while (curr != g) {
                path_nodes.insert(curr);
                curr = tree2[curr].parent;
            }
            curr = b;
            while (curr != g) {
                path_nodes.insert(curr);
                curr = tree2[curr].parent;
            }
            path_nodes.insert(g);

            std::vector<int> roots;
            curr = a;
            while (curr != g) {
                int p = tree2[curr].parent;
                int sib = (tree2[p].left == curr) ? tree2[p].right : tree2[p].left;
                if (sib != -1 && !path_nodes.count(sib)) {
                    roots.push_back(sib);
                }
                curr = p;
            }
            curr = b;
            while (curr != g) {
                int p = tree2[curr].parent;
                int sib = (tree2[p].left == curr) ? tree2[p].right : tree2[p].left;
                if (sib != -1 && !path_nodes.count(sib)) {
                    roots.push_back(sib);
                }
                curr = p;
            }
            return roots;
        }

        void get_active_leaves(int r, std::vector<int>& leaves) const {
            if (r == -1 || !tree2[r].active) return;
            if (tree2[r].is_leaf) {
                leaves.push_back(r);
                return;
            }
            get_active_leaves(tree2[r].left, leaves);
            get_active_leaves(tree2[r].right, leaves);
        }

        static int build_filtered_newick(
            const Tree& tree,
            int u,
            const std::unordered_set<std::uint32_t>& subset,
            pace26::io::NewickTree& target
        ) {
            if (tree.is_leaf(u)) {
                std::uint32_t lbl = tree.label(u);
                if (subset.count(lbl)) {
                    pace26::io::NewickNode node;
                    node.id = lbl;
                    node.label = lbl;
                    node.left = -1;
                    node.right = -1;
                    target.nodes.push_back(node);
                    return static_cast<int>(target.nodes.size() - 1);
                }
                return -1;
            }
            int l_new = build_filtered_newick(tree, tree.left(u), subset, target);
            int r_new = build_filtered_newick(tree, tree.right(u), subset, target);
            if (l_new == -1) return r_new;
            if (r_new == -1) return l_new;

            pace26::io::NewickNode node;
            node.id = static_cast<std::uint32_t>(target.nodes.size() + 1000000);
            node.label = 0;
            node.left = l_new;
            node.right = r_new;
            target.nodes.push_back(node);
            return static_cast<int>(target.nodes.size() - 1);
        }

        static Tree filter_tree(const Tree& tree, const std::unordered_set<std::uint32_t>& subset) {
            pace26::io::NewickTree target;
            target.root = build_filtered_newick(tree, tree.root, subset, target);
            if (target.root < 0) {
                throw InProcessExactMafError("Filtered subtree has no leaves");
            }
            return Tree::from_newick(target);
        }

        bool should_stop() {
            if (local_timer.should_stop(options.local_guard_seconds)) {
                aborted = true;
                return true;
            }

            if (global_timer != nullptr &&
                global_timer->should_stop(options.global_guard_seconds)) {
                aborted = true;
                return true;
            }

            if (options.max_candidate_masks_tested > 0 &&
                nodes_visited >= options.max_candidate_masks_tested) {
                aborted = true;
                return true;
            }

            ++nodes_visited;
            return false;
        }

        void search(int cuts_so_far) {
            if (aborted || should_stop()) {
                return;
            }

            std::size_t watermark = undo_stack.size();
            run_all_contractions();

            if (active_leaves <= 1) {
                if (cuts_so_far < best_cuts) {
                    best_cuts = cuts_so_far;
                    best_components = cut_components;
                    if (active_leaves == 1) {
                        int active_root = -1;
                        if (root1 != -1 && tree1[root1].active) {
                            active_root = root1;
                        } else {
                            for (std::size_t i = 0; i < tree1.size(); ++i) {
                                if (tree1[i].active) {
                                    active_root = static_cast<int>(i);
                                    break;
                                }
                            }
                        }
                        if (active_root != -1) {
                            best_components.push_back(contracted[active_root]);
                        }
                    }
                }
                rollback(watermark);
                return;
            }

            int lb = compute_packing_lower_bound();
            if (cuts_so_far + lb >= best_cuts) {
                rollback(watermark);
                return;
            }

            std::vector<int> cherries = get_active_cherries1();
            if (cherries.empty()) {
                rollback(watermark);
                return;
            }

            int cherry_node = choose_branch_cherry(cherries);
            int a1 = tree1[cherry_node].left;
            int b1 = tree1[cherry_node].right;
            std::uint32_t lbl_a = tree1[a1].label;
            std::uint32_t lbl_b = tree1[b1].label;

            // Branch 1: Cut a
            {
                std::size_t branch_watermark = undo_stack.size();
                cut_components.push_back(component_for_active_label(lbl_a));
                cut_active_label(lbl_a);

                search(cuts_so_far + 1);

                cut_components.pop_back();
                rollback(branch_watermark);
            }

            // Branch 2: Cut b
            {
                std::size_t branch_watermark = undo_stack.size();
                cut_components.push_back(component_for_active_label(lbl_b));
                cut_active_label(lbl_b);

                search(cuts_so_far + 1);

                cut_components.pop_back();
                rollback(branch_watermark);
            }

            // Branch 3: Cut all hanging subtrees in tree2
            {
                std::size_t branch_watermark = undo_stack.size();
                int u2 = leaf_to_node2[lbl_a];
                int v2 = leaf_to_node2[lbl_b];
                std::vector<int> hanging_roots = get_hanging_subtree_roots(u2, v2);

                std::vector<std::vector<std::uint32_t>> all_sub_components;
                int total_sub_cuts = 0;
                bool subproblem_failed = false;

                for (int r : hanging_roots) {
                    std::vector<int> leaves2;
                    get_active_leaves(r, leaves2);
                    std::vector<std::uint32_t> S;
                    for (int lf : leaves2) {
                        S.push_back(tree2[lf].label);
                    }
                    if (S.empty()) continue;

                    std::sort(S.begin(), S.end());
                    int sub_cuts = 0;
                    std::vector<std::vector<std::uint32_t>> sub_components;

                    auto it = sub_cache.find(S);
                    if (it != sub_cache.end()) {
                        sub_cuts = it->second.cuts;
                        sub_components = it->second.components;
                    } else {
                        std::unordered_set<std::uint32_t> subset(S.begin(), S.end());
                        Tree sub_t1 = filter_tree(t1, subset);
                        Tree sub_t2 = filter_tree(t2, subset);

                        Options sub_opts = options;
                        sub_opts.max_leaves = S.size();
                        InProcessExactMaf sub_exact(sub_opts);
                        auto sub_res = sub_exact.solve(sub_t1, sub_t2, global_timer);
                        if (!sub_res.has_value()) {
                            subproblem_failed = true;
                            break;
                        }
                        sub_cuts = sub_res->component_count() - 1;
                        for (const auto& comp : sub_res->components) {
                            sub_components.push_back(expand_representative_component(comp.labels));
                        }
                        sub_cache[S] = {sub_cuts, sub_components};
                    }

                    total_sub_cuts += sub_cuts;
                    all_sub_components.insert(all_sub_components.end(), sub_components.begin(), sub_components.end());
                }

                if (!subproblem_failed && cuts_so_far + static_cast<int>(hanging_roots.size()) + total_sub_cuts < best_cuts) {
                    int branch3_cost = static_cast<int>(hanging_roots.size()) + total_sub_cuts;
                    for (int r : hanging_roots) {
                        std::vector<int> leaves2;
                        get_active_leaves(r, leaves2);
                        for (int lf : leaves2) {
                            std::uint32_t lbl = tree2[lf].label;
                            cut_active_label(lbl);
                        }
                    }

                    std::size_t num_added = all_sub_components.size();
                    cut_components.insert(cut_components.end(), all_sub_components.begin(), all_sub_components.end());

                    search(cuts_so_far + branch3_cost);

                    for (std::size_t i = 0; i < num_added; ++i) {
                        cut_components.pop_back();
                    }
                }
                rollback(branch_watermark);
            }

            rollback(watermark);
        }
    };

    static void require_same_leaf_set(const Tree& t1, const Tree& t2) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw InProcessExactMafError("input trees do not have the same leaf set");
        }
    }

    Options options_;
};

}  // namespace pace26::exact
