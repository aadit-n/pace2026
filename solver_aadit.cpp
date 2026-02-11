#include "genesis/genesis.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

using genesis::tree::Tree;

namespace {

struct PaceInstance {
    int tree_count = 0;
    int leaf_count = 0;
    std::vector<Tree> trees;
    std::vector<std::string> newick_lines;
};

struct SimpleTree {
    int root = -1;
    std::vector<std::vector<int>> children;
    std::vector<int> parent;
    std::vector<int> leaf_label;  // 0 for internal
};

struct TreeInfo {
    int root = -1;
    int node_count = 0;
    std::vector<std::vector<int>> children;
    std::vector<int> parent;
    std::vector<int> depth;
    std::vector<int> leaf_to_node;      // index by leaf label
    std::vector<int> node_to_leaf;      // 0 for internal
    std::vector<int> subtree_leaf_count;
    std::vector<std::vector<int>> up;   // binary lifting table
};

char* g_best_output = nullptr;
size_t g_best_output_len = 0;

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

Tree parse_newick_line(const std::string& newick_line) {
    genesis::tree::CommonTreeNewickReader reader;
    auto src = genesis::utils::from_string(newick_line);
    return reader.read(src);
}

PaceInstance read_instance_from_stdin() {
    PaceInstance instance;
    std::string line;

    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (line[0] == '#') {
            if (line.rfind("#p", 0) == 0) {
                std::istringstream ss(line);
                std::string tag;
                ss >> tag >> instance.tree_count >> instance.leaf_count;
                break;
            }
            continue;
        }
    }

    if (instance.tree_count <= 0 || instance.leaf_count <= 0) {
        throw std::runtime_error("Invalid or missing #p line.");
    }

    while (static_cast<int>(instance.newick_lines.size()) < instance.tree_count && std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        instance.newick_lines.push_back(line);
        instance.trees.push_back(parse_newick_line(line));
    }

    if (static_cast<int>(instance.newick_lines.size()) != instance.tree_count) {
        throw std::runtime_error("Input ended before all trees were read.");
    }

    return instance;
}

std::string normalize_forest_output(const std::vector<std::string>& forest_lines) {
    std::string out;
    for (const std::string& raw_line : forest_lines) {
        std::string line = trim(raw_line);
        if (line.empty()) continue;
        if (line.back() != ';') line.push_back(';');
        out += line;
        out.push_back('\n');
    }
    return out;
}

void set_best_output_buffer(const std::string& output) {
    sigset_t set;
    sigset_t old_set;
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    sigprocmask(SIG_BLOCK, &set, &old_set);

    char* new_buffer = nullptr;
    const size_t new_len = output.size();
    if (new_len > 0) {
        new_buffer = new char[new_len];
        std::memcpy(new_buffer, output.data(), new_len);
    }

    char* old_buffer = g_best_output;
    g_best_output = new_buffer;
    g_best_output_len = new_len;
    delete[] old_buffer;

    sigprocmask(SIG_SETMASK, &old_set, nullptr);
}

void publish_best_solution(const std::vector<std::string>& forest_lines) {
    set_best_output_buffer(normalize_forest_output(forest_lines));
}

void sigterm_handler(int) {
    if (g_best_output && g_best_output_len > 0) {
        const ssize_t unused = write(STDOUT_FILENO, g_best_output, g_best_output_len);
        (void)unused;
    }
    _Exit(0);
}

std::vector<std::string> build_singleton_forest(int leaf_count) {
    std::vector<std::string> forest;
    forest.reserve(leaf_count);
    for (int i = 1; i <= leaf_count; ++i) {
        forest.push_back(std::to_string(i) + ";");
    }
    return forest;
}

class NewickParser {
public:
    explicit NewickParser(std::string text) : s_(std::move(text)) {}

    SimpleTree parse() {
        SimpleTree t;
        skip_ws();
        const int root = parse_subtree(t);
        skip_ws();
        if (pos_ >= s_.size() || s_[pos_] != ';') {
            throw std::runtime_error("Expected ';' at end of Newick.");
        }
        ++pos_;
        skip_ws();
        if (pos_ != s_.size()) {
            throw std::runtime_error("Unexpected trailing characters after Newick.");
        }
        t.root = root;
        if (t.parent[root] != -1) {
            throw std::runtime_error("Internal parser error: root has parent.");
        }
        return t;
    }

private:
    std::string s_;
    size_t pos_ = 0;

    static bool is_delim(char c) {
        return c == ',' || c == '(' || c == ')' || c == ';';
    }

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) {
            ++pos_;
        }
    }

    int add_node(SimpleTree& t, int leaf_label) {
        const int id = static_cast<int>(t.children.size());
        t.children.push_back({});
        t.parent.push_back(-1);
        t.leaf_label.push_back(leaf_label);
        return id;
    }

    static int parse_leaf_label(const std::string& token) {
        size_t i = 0;
        while (i < token.size() && !std::isdigit(static_cast<unsigned char>(token[i]))) ++i;
        size_t j = i;
        while (j < token.size() && std::isdigit(static_cast<unsigned char>(token[j]))) ++j;
        if (i == j) {
            throw std::runtime_error("Leaf token does not contain numeric label: " + token);
        }
        return std::stoi(token.substr(i, j - i));
    }

    std::string read_token() {
        skip_ws();
        std::string tok;
        while (pos_ < s_.size() && !is_delim(s_[pos_])) {
            tok.push_back(s_[pos_]);
            ++pos_;
        }
        return trim(tok);
    }

    void skip_optional_token() {
        (void)read_token();
    }

    int parse_subtree(SimpleTree& t) {
        skip_ws();
        if (pos_ >= s_.size()) {
            throw std::runtime_error("Unexpected end of Newick.");
        }

        if (s_[pos_] == '(') {
            ++pos_;
            const int node = add_node(t, 0);

            while (true) {
                const int child = parse_subtree(t);
                t.children[node].push_back(child);
                t.parent[child] = node;

                skip_ws();
                if (pos_ >= s_.size()) {
                    throw std::runtime_error("Unexpected end while parsing children.");
                }
                if (s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (s_[pos_] == ')') {
                    ++pos_;
                    break;
                }
                throw std::runtime_error("Expected ',' or ')' in internal node.");
            }

            skip_optional_token();
            return node;
        }

        const std::string token = read_token();
        if (token.empty()) {
            throw std::runtime_error("Expected leaf token.");
        }

        const int label = parse_leaf_label(token);
        return add_node(t, label);
    }
};

SimpleTree parse_newick_simple(const std::string& line) {
    return NewickParser(line).parse();
}

TreeInfo build_tree_info(const SimpleTree& tree, int leaf_count) {
    TreeInfo info;
    info.root = tree.root;
    info.node_count = static_cast<int>(tree.children.size());
    info.children = tree.children;
    info.parent = tree.parent;
    info.node_to_leaf = tree.leaf_label;
    info.depth.assign(info.node_count, 0);
    info.subtree_leaf_count.assign(info.node_count, 0);
    info.leaf_to_node.assign(leaf_count + 1, -1);

    std::vector<int> stack{info.root};
    while (!stack.empty()) {
        const int v = stack.back();
        stack.pop_back();
        for (int c : info.children[v]) {
            info.depth[c] = info.depth[v] + 1;
            stack.push_back(c);
        }
    }

    for (int v = 0; v < info.node_count; ++v) {
        const int leaf = info.node_to_leaf[v];
        if (leaf > 0 && leaf <= leaf_count) {
            info.leaf_to_node[leaf] = v;
        }
    }

    std::vector<std::pair<int, bool>> post;
    post.reserve(info.node_count * 2);
    post.push_back({info.root, false});
    while (!post.empty()) {
        auto [v, done] = post.back();
        post.pop_back();
        if (!done) {
            post.push_back({v, true});
            for (int c : info.children[v]) post.push_back({c, false});
        } else {
            if (info.children[v].empty() && info.node_to_leaf[v] > 0) {
                info.subtree_leaf_count[v] = 1;
            } else {
                int sum = 0;
                for (int c : info.children[v]) sum += info.subtree_leaf_count[c];
                info.subtree_leaf_count[v] = sum;
            }
        }
    }

    int lg = 1;
    while ((1 << lg) <= std::max(1, info.node_count)) ++lg;
    info.up.assign(lg, std::vector<int>(info.node_count, info.root));
    for (int v = 0; v < info.node_count; ++v) {
        info.up[0][v] = (info.parent[v] == -1 ? v : info.parent[v]);
    }
    for (int k = 1; k < lg; ++k) {
        for (int v = 0; v < info.node_count; ++v) {
            info.up[k][v] = info.up[k - 1][info.up[k - 1][v]];
        }
    }

    return info;
}

int lca_nodes(int a, int b, const TreeInfo& info) {
    if (a == -1 || b == -1) return -1;
    if (info.depth[a] < info.depth[b]) std::swap(a, b);
    int diff = info.depth[a] - info.depth[b];
    for (size_t k = 0; k < info.up.size(); ++k) {
        if (diff & (1 << k)) a = info.up[k][a];
    }
    if (a == b) return a;
    for (int k = static_cast<int>(info.up.size()) - 1; k >= 0; --k) {
        if (info.up[k][a] != info.up[k][b]) {
            a = info.up[k][a];
            b = info.up[k][b];
        }
    }
    return info.parent[a];
}

int lca_of_component(const std::vector<int>& leaves, const TreeInfo& info) {
    if (leaves.empty()) return -1;
    int lca = info.leaf_to_node[leaves[0]];
    for (size_t i = 1; i < leaves.size(); ++i) {
        lca = lca_nodes(lca, info.leaf_to_node[leaves[i]], info);
    }
    return lca;
}

bool component_valid_in_tree(const std::vector<int>& leaves, const TreeInfo& info) {
    if (leaves.size() <= 1) return true;
    const int r = lca_of_component(leaves, info);
    if (r == -1) return false;
    return info.subtree_leaf_count[r] == static_cast<int>(leaves.size());
}

std::map<int, std::vector<int>> build_components(const std::vector<int>& component_of_leaf) {
    std::map<int, std::vector<int>> components;
    for (int leaf = 1; leaf < static_cast<int>(component_of_leaf.size()); ++leaf) {
        components[component_of_leaf[leaf]].push_back(leaf);
    }
    return components;
}

bool partition_valid_across_all_trees(
    const std::vector<int>& component_of_leaf,
    const std::vector<TreeInfo>& infos
) {
    const auto components = build_components(component_of_leaf);
    for (const auto& [id, leaves] : components) {
        (void)id;
        for (const auto& info : infos) {
            if (!component_valid_in_tree(leaves, info)) {
                return false;
            }
        }
    }
    return true;
}

bool are_sibling_leaves_in_tree(int a, int b, const TreeInfo& info) {
    const int na = (a >= 0 && a < static_cast<int>(info.leaf_to_node.size()) ? info.leaf_to_node[a] : -1);
    const int nb = (b >= 0 && b < static_cast<int>(info.leaf_to_node.size()) ? info.leaf_to_node[b] : -1);
    if (na == -1 || nb == -1) return false;
    const int pa = info.parent[na];
    const int pb = info.parent[nb];
    if (pa == -1 || pb == -1 || pa != pb) return false;
    if (info.children[pa].size() != 2) return false;
    for (int c : info.children[pa]) {
        if (info.node_to_leaf[c] == 0) return false;
    }
    return true;
}

std::vector<std::pair<int, int>> extract_cherries(const TreeInfo& info) {
    std::vector<std::pair<int, int>> cherries;
    for (int v = 0; v < info.node_count; ++v) {
        if (info.children[v].size() != 2) continue;
        const int c1 = info.children[v][0];
        const int c2 = info.children[v][1];
        const int l1 = info.node_to_leaf[c1];
        const int l2 = info.node_to_leaf[c2];
        if (l1 > 0 && l2 > 0) {
            cherries.push_back({l1, l2});
        }
    }
    return cherries;
}

void merge_component_ids(std::vector<int>& component_of_leaf, int keep_id, int remove_id) {
    if (keep_id == remove_id) return;
    for (int leaf = 1; leaf < static_cast<int>(component_of_leaf.size()); ++leaf) {
        if (component_of_leaf[leaf] == remove_id) component_of_leaf[leaf] = keep_id;
    }
}

int child_of_lca_on_path(int leaf_node, int lca, const TreeInfo& info) {
    int v = leaf_node;
    while (v != -1 && info.parent[v] != lca) {
        v = info.parent[v];
    }
    return v;
}

bool split_component_by_tree(
    std::vector<int>& component_of_leaf,
    const std::vector<int>& leaves,
    const TreeInfo& info,
    int& next_component_id
) {
    if (leaves.size() <= 1) return false;

    const int r = lca_of_component(leaves, info);
    if (r == -1) return false;

    std::unordered_map<int, std::vector<int>> groups;
    groups.reserve(leaves.size());

    for (int leaf : leaves) {
        const int node = info.leaf_to_node[leaf];
        const int child = child_of_lca_on_path(node, r, info);
        if (child == -1) {
            groups[-1].push_back(leaf);
        } else {
            groups[child].push_back(leaf);
        }
    }

    if (groups.size() <= 1) {
        if (leaves.size() <= 1) return false;
        component_of_leaf[leaves.back()] = next_component_id++;
        return true;
    }

    auto it = groups.begin();
    const std::vector<int> keep_group = it->second;
    const int keep_id = component_of_leaf[keep_group.front()];

    ++it;
    for (; it != groups.end(); ++it) {
        const int new_id = next_component_id++;
        for (int leaf : it->second) {
            component_of_leaf[leaf] = new_id;
        }
    }

    for (int leaf : keep_group) {
        component_of_leaf[leaf] = keep_id;
    }

    return true;
}

void enforce_validity_for_trees(
    std::vector<int>& component_of_leaf,
    const std::vector<TreeInfo>& infos,
    int max_tree_idx,
    int& next_component_id
) {
    while (true) {
        bool changed = false;
        const auto components = build_components(component_of_leaf);

        for (const auto& [comp_id, leaves] : components) {
            (void)comp_id;
            if (leaves.size() <= 1) continue;
            for (int ti = 0; ti <= max_tree_idx; ++ti) {
                if (!component_valid_in_tree(leaves, infos[ti])) {
                    if (split_component_by_tree(component_of_leaf, leaves, infos[ti], next_component_id)) {
                        changed = true;
                    }
                    break;
                }
            }
            if (changed) break;
        }

        if (!changed) break;
    }
}

std::string restricted_newick_dfs(
    int node,
    const TreeInfo& base,
    const std::vector<char>& in_component
) {
    const int leaf = base.node_to_leaf[node];
    if (leaf > 0) {
        return in_component[leaf] ? std::to_string(leaf) : std::string();
    }

    std::vector<std::string> parts;
    for (int child : base.children[node]) {
        std::string sub = restricted_newick_dfs(child, base, in_component);
        if (!sub.empty()) parts.push_back(std::move(sub));
    }

    if (parts.empty()) return {};
    if (parts.size() == 1) return parts[0];  // cleanup contraction

    std::string out = "(";
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out.push_back(',');
        out += parts[i];
    }
    out.push_back(')');
    return out;
}

std::vector<std::string> build_forest_from_partition(
    const std::vector<int>& component_of_leaf,
    const TreeInfo& base_tree,
    int leaf_count
) {
    auto components = build_components(component_of_leaf);

    std::vector<std::pair<int, std::string>> keyed;
    keyed.reserve(components.size());

    for (auto& [comp_id, leaves] : components) {
        (void)comp_id;
        std::sort(leaves.begin(), leaves.end());
        if (leaves.size() == 1) {
            keyed.push_back({leaves[0], std::to_string(leaves[0]) + ";"});
            continue;
        }

        std::vector<char> in_component(leaf_count + 1, 0);
        for (int leaf : leaves) in_component[leaf] = 1;
        std::string nw = restricted_newick_dfs(base_tree.root, base_tree, in_component);
        if (nw.empty()) {
            nw = std::to_string(leaves.front());
            for (size_t i = 1; i < leaves.size(); ++i) {
                nw = "(" + nw + "," + std::to_string(leaves[i]) + ")";
            }
        }
        nw.push_back(';');
        keyed.push_back({leaves.front(), std::move(nw)});
    }

    std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    std::vector<std::string> forest;
    forest.reserve(keyed.size());
    for (auto& kv : keyed) forest.push_back(std::move(kv.second));
    return forest;
}

bool can_merge_components(
    const std::vector<int>& leaves_a,
    const std::vector<int>& leaves_b,
    const std::vector<TreeInfo>& infos
) {
    std::vector<int> merged;
    merged.reserve(leaves_a.size() + leaves_b.size());
    merged.insert(merged.end(), leaves_a.begin(), leaves_a.end());
    merged.insert(merged.end(), leaves_b.begin(), leaves_b.end());

    for (const auto& info : infos) {
        if (!component_valid_in_tree(merged, info)) return false;
    }
    return true;
}

double read_stride_timeout_seconds(double default_value) {
    const char* env = std::getenv("STRIDE_TIMEOUT");
    if (!env) return default_value;
    char* end = nullptr;
    const double parsed = std::strtod(env, &end);
    if (end == env || parsed <= 0.0) return default_value;
    return parsed;
}

// Plug your algorithm here.
// Input: parsed PACE instance (tree_count, leaf_count, and parsed trees).
// Output: forest as one Newick tree per vector entry.
std::vector<std::string> solve(const PaceInstance& instance) {
    const int n = instance.leaf_count;
    const int m = instance.tree_count;

    std::vector<SimpleTree> parsed_trees;
    parsed_trees.reserve(m);
    for (const std::string& line : instance.newick_lines) {
        parsed_trees.push_back(parse_newick_simple(line));
    }

    std::vector<TreeInfo> infos;
    infos.reserve(m);
    for (const auto& t : parsed_trees) {
        infos.push_back(build_tree_info(t, n));
    }

    std::vector<int> component_of_leaf(n + 1, 0);
    for (int leaf = 1; leaf <= n; ++leaf) component_of_leaf[leaf] = leaf;
    int next_component_id = n + 1;

    // Phase 1: cherry-based heuristic on first two trees.
    if (m >= 2) {
        const auto cherries = extract_cherries(infos[0]);
        for (const auto& [a, b] : cherries) {
            if (are_sibling_leaves_in_tree(a, b, infos[1])) {
                merge_component_ids(component_of_leaf, component_of_leaf[a], component_of_leaf[b]);
            } else {
                const int na = infos[1].leaf_to_node[a];
                const int nb = infos[1].leaf_to_node[b];
                if (na != -1 && nb != -1) {
                    const int cut_leaf = (infos[1].depth[na] >= infos[1].depth[nb] ? a : b);
                    component_of_leaf[cut_leaf] = next_component_id++;
                }
            }
        }
    }

    // Make sure phase-1 result is valid for processed trees.
    enforce_validity_for_trees(component_of_leaf, infos, std::max(0, std::min(1, m - 1)), next_component_id);

    std::vector<std::string> best = build_forest_from_partition(component_of_leaf, infos[0], n);
    if (partition_valid_across_all_trees(component_of_leaf, infos)) {
        publish_best_solution(best);
    }

    // Phase 2: incrementally refine with remaining trees.
    for (int ti = 2; ti < m; ++ti) {
        enforce_validity_for_trees(component_of_leaf, infos, ti, next_component_id);
        if (partition_valid_across_all_trees(component_of_leaf, infos)) {
            best = build_forest_from_partition(component_of_leaf, infos[0], n);
            publish_best_solution(best);
        }
    }

    // Final full validity pass.
    enforce_validity_for_trees(component_of_leaf, infos, m - 1, next_component_id);
    if (partition_valid_across_all_trees(component_of_leaf, infos)) {
        best = build_forest_from_partition(component_of_leaf, infos[0], n);
        publish_best_solution(best);
    }

    // Phase 3: merge-improvement loop under time budget.
    const auto start = std::chrono::steady_clock::now();
    const double soft_timeout = read_stride_timeout_seconds(30.0);
    const auto deadline = start + std::chrono::milliseconds(static_cast<long long>(soft_timeout * 850.0));

    while (std::chrono::steady_clock::now() < deadline) {
        bool improved = false;
        auto components = build_components(component_of_leaf);
        std::vector<int> comp_ids;
        comp_ids.reserve(components.size());
        for (const auto& kv : components) comp_ids.push_back(kv.first);

        for (size_t i = 0; i < comp_ids.size() && std::chrono::steady_clock::now() < deadline; ++i) {
            for (size_t j = i + 1; j < comp_ids.size() && std::chrono::steady_clock::now() < deadline; ++j) {
                const int id_i = comp_ids[i];
                const int id_j = comp_ids[j];
                if (can_merge_components(components[id_i], components[id_j], infos)) {
                    merge_component_ids(component_of_leaf, id_i, id_j);
                    improved = true;
                    break;
                }
            }
            if (improved) break;
        }

        if (!improved) break;

        if (partition_valid_across_all_trees(component_of_leaf, infos)) {
            best = build_forest_from_partition(component_of_leaf, infos[0], n);
            publish_best_solution(best);
        }
    }

    return best;
}

}  // namespace

int main() {
    std::signal(SIGTERM, sigterm_handler);

    try {
        const PaceInstance instance = read_instance_from_stdin();

        // Keep a valid fallback in case SIGTERM arrives during solve().
        publish_best_solution(build_singleton_forest(instance.leaf_count));

        const std::vector<std::string> final_forest = solve(instance);
        publish_best_solution(final_forest);

        if (g_best_output && g_best_output_len > 0) {
            std::cout.write(g_best_output, static_cast<std::streamsize>(g_best_output_len));
        }
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
