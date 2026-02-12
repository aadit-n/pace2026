#include "genesis/genesis.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
    std::vector<uint64_t> sub_hash_a;   // topology hash A (unordered rooted)
    std::vector<uint64_t> sub_hash_b;   // topology hash B (independent mix)
};

using Bitmask = std::vector<uint64_t>;
using ClusterMap = std::unordered_map<uint64_t, std::vector<Bitmask>>;

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
    info.sub_hash_a.assign(info.node_count, 0);
    info.sub_hash_b.assign(info.node_count, 0);
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

    auto mix64a = [](uint64_t x) -> uint64_t {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    };
    auto mix64b = [](uint64_t x) -> uint64_t {
        x += 0x517cc1b727220a95ULL;
        x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
        x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
        return x ^ (x >> 33);
    };

    std::vector<std::pair<int, bool>> post_hash;
    post_hash.reserve(info.node_count * 2);
    post_hash.push_back({info.root, false});
    while (!post_hash.empty()) {
        auto [v, done] = post_hash.back();
        post_hash.pop_back();
        if (!done) {
            post_hash.push_back({v, true});
            for (int c : info.children[v]) post_hash.push_back({c, false});
            continue;
        }

        if (info.node_to_leaf[v] > 0) {
            info.sub_hash_a[v] = mix64a(static_cast<uint64_t>(info.node_to_leaf[v]) + 0x123456789abcdefULL);
            info.sub_hash_b[v] = mix64b(static_cast<uint64_t>(info.node_to_leaf[v]) + 0xfedcba987654321ULL);
            continue;
        }

        std::vector<uint64_t> child_hashes_a;
        std::vector<uint64_t> child_hashes_b;
        child_hashes_a.reserve(info.children[v].size());
        child_hashes_b.reserve(info.children[v].size());
        for (int c : info.children[v]) {
            child_hashes_a.push_back(info.sub_hash_a[c]);
            child_hashes_b.push_back(info.sub_hash_b[c]);
        }
        std::sort(child_hashes_a.begin(), child_hashes_a.end());
        std::sort(child_hashes_b.begin(), child_hashes_b.end());

        uint64_t ha = mix64a(0xCAFEBABE12345678ULL ^ static_cast<uint64_t>(child_hashes_a.size()));
        for (uint64_t ch : child_hashes_a) {
            ha = mix64a(ha ^ mix64a(ch + 0x9e3779b97f4a7c15ULL));
        }
        uint64_t hb = mix64b(0x3141592653589793ULL ^ static_cast<uint64_t>(child_hashes_b.size()));
        for (uint64_t ch : child_hashes_b) {
            hb = mix64b(hb ^ mix64b(ch + 0x517cc1b727220a95ULL));
        }
        info.sub_hash_a[v] = ha;
        info.sub_hash_b[v] = hb;
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

Bitmask make_empty_bitmask(int leaf_count) {
    return Bitmask((leaf_count + 63) / 64, 0ULL);
}

void bitmask_set(Bitmask& bm, int leaf) {
    const int idx = leaf - 1;
    bm[idx / 64] |= (1ULL << (idx % 64));
}

bool bitmask_intersects(const Bitmask& a, const Bitmask& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] & b[i]) return true;
    }
    return false;
}

void bitmask_or_assign(Bitmask& target, const Bitmask& src) {
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] |= src[i];
    }
}

bool bitmask_equal(const Bitmask& a, const Bitmask& b) {
    return a == b;
}

int bitmask_count(const Bitmask& bm) {
    int count = 0;
    for (uint64_t w : bm) count += __builtin_popcountll(w);
    return count;
}

std::vector<int> bitmask_to_leaves(const Bitmask& bm, int leaf_count) {
    std::vector<int> leaves;
    leaves.reserve(bitmask_count(bm));
    for (int leaf = 1; leaf <= leaf_count; ++leaf) {
        const int idx = leaf - 1;
        if (bm[idx / 64] & (1ULL << (idx % 64))) leaves.push_back(leaf);
    }
    return leaves;
}

uint64_t hashLeafSet(const Bitmask& bitmask) {
    uint64_t h = 1469598103934665603ULL;
    for (uint64_t w : bitmask) {
        h ^= w;
        h *= 1099511628211ULL;
    }
    return h;
}

std::vector<Bitmask> getClusters(const TreeInfo& info, int leaf_count) {
    std::vector<Bitmask> node_mask(info.node_count, make_empty_bitmask(leaf_count));
    std::vector<Bitmask> clusters;

    std::vector<std::pair<int, bool>> post;
    post.reserve(info.node_count * 2);
    post.push_back({info.root, false});
    while (!post.empty()) {
        auto [v, done] = post.back();
        post.pop_back();
        if (!done) {
            post.push_back({v, true});
            for (int c : info.children[v]) post.push_back({c, false});
            continue;
        }

        if (info.node_to_leaf[v] > 0) {
            bitmask_set(node_mask[v], info.node_to_leaf[v]);
        } else {
            for (int c : info.children[v]) bitmask_or_assign(node_mask[v], node_mask[c]);
            clusters.push_back(node_mask[v]);
        }
    }

    return clusters;
}

bool isClusterCompatible(const Bitmask& leaf_set, const ClusterMap& map) {
    const uint64_t h = hashLeafSet(leaf_set);
    auto it = map.find(h);
    if (it == map.end()) return false;
    for (const Bitmask& cand : it->second) {
        if (bitmask_equal(cand, leaf_set)) return true;
    }
    return false;
}

std::vector<std::vector<int>> compute_subtree_leaf_lists(const TreeInfo& info) {
    std::vector<std::vector<int>> leaves(info.node_count);
    std::vector<std::pair<int, bool>> post;
    post.reserve(info.node_count * 2);
    post.push_back({info.root, false});

    while (!post.empty()) {
        auto [v, done] = post.back();
        post.pop_back();
        if (!done) {
            post.push_back({v, true});
            for (int c : info.children[v]) post.push_back({c, false});
            continue;
        }

        if (info.node_to_leaf[v] > 0) {
            leaves[v].push_back(info.node_to_leaf[v]);
        } else {
            size_t total = 0;
            for (int c : info.children[v]) total += leaves[c].size();
            leaves[v].reserve(total);
            for (int c : info.children[v]) {
                leaves[v].insert(leaves[v].end(), leaves[c].begin(), leaves[c].end());
            }
        }
    }

    return leaves;
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

std::pair<bool, int> cluster_root(const TreeInfo& T, const std::vector<int>& leaves) {
    if (leaves.empty()) return {false, -1};
    if (leaves.size() == 1) {
        const int node = T.leaf_to_node[leaves[0]];
        return {node != -1, node};
    }
    const int r = lca_of_component(leaves, T);
    if (r == -1) return {false, -1};
    if (T.subtree_leaf_count[r] != static_cast<int>(leaves.size())) return {false, -1};
    return {true, r};
}

std::pair<bool, uint64_t> component_cluster_hash_all_trees(
    const std::vector<TreeInfo>& infos,
    const std::vector<int>& leaves,
    int upto_tree
) {
    if (leaves.size() <= 1) return {true, 0};
    if (infos.empty()) return {true, 0};
    const int last = std::min(upto_tree, static_cast<int>(infos.size()) - 1);
    if (last < 0) return {true, 0};

    uint64_t h = 0;
    uint64_t h2 = 0;
    bool h_set = false;
    for (int t = 0; t <= last; ++t) {
        auto [ok, r] = cluster_root(infos[t], leaves);
        if (!ok) return {false, 0};
        const uint64_t ht = infos[t].sub_hash_a[r];
        const uint64_t ht2 = infos[t].sub_hash_b[r];
        if (!h_set) {
            h = ht;
            h2 = ht2;
            h_set = true;
        } else if (ht != h || ht2 != h2) {
            return {false, 0};
        }
    }
    return {true, h};
}

bool component_topology_valid_across_trees(
    const std::vector<int>& leaves,
    const std::vector<TreeInfo>& infos,
    int upto_tree
) {
    if (leaves.size() <= 1) return true;
    if (infos.empty()) return true;
    auto [ok, _h] = component_cluster_hash_all_trees(infos, leaves, upto_tree);
    return ok;
}

bool isComponentValid(const TreeInfo& info, const std::vector<int>& comp_leaves) {
    if (comp_leaves.size() <= 1) return true;
    const int r = lca_of_component(comp_leaves, info);
    if (r == -1) return false;
    return info.subtree_leaf_count[r] == static_cast<int>(comp_leaves.size());
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
    const std::vector<TreeInfo>& infos,
    int upto_tree
) {
    const auto components = build_components(component_of_leaf);
    for (const auto& [id, leaves] : components) {
        (void)id;
        if (!component_topology_valid_across_trees(leaves, infos, upto_tree)) return false;
    }
    return true;
}

int count_components(const std::vector<int>& component_of_leaf) {
    std::unordered_set<int> ids;
    ids.reserve(component_of_leaf.size());
    for (int leaf = 1; leaf < static_cast<int>(component_of_leaf.size()); ++leaf) {
        ids.insert(component_of_leaf[leaf]);
    }
    return static_cast<int>(ids.size());
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

int child_side_under_lca(int leaf, int lca, const TreeInfo& info) {
    if (lca < 0) return -1;
    const int node = (leaf >= 0 && leaf < static_cast<int>(info.leaf_to_node.size())) ? info.leaf_to_node[leaf] : -1;
    if (node == -1) return -1;

    int child = child_of_lca_on_path(node, lca, info);
    if (child == -1 && node == lca) return -1;
    for (size_t i = 0; i < info.children[lca].size(); ++i) {
        if (info.children[lca][i] == child) return static_cast<int>(i);
    }
    return -1;
}

bool split_component_by_two_trees(
    std::vector<int>& component_of_leaf,
    const std::vector<int>& leaves,
    const TreeInfo& A,
    const TreeInfo& B,
    int& next_component_id
) {
    if (leaves.size() <= 1) return false;
    const int rA = lca_of_component(leaves, A);
    const int rB = lca_of_component(leaves, B);
    if (rA == -1 || rB == -1) return false;

    std::map<std::pair<int, int>, std::vector<int>> buckets;
    for (int leaf : leaves) {
        const int sideA = child_side_under_lca(leaf, rA, A);
        const int sideB = child_side_under_lca(leaf, rB, B);
        buckets[{sideA, sideB}].push_back(leaf);
    }
    if (buckets.size() <= 1) return false;

    int keep_id = component_of_leaf[leaves.front()];
    bool first = true;
    for (auto& kv : buckets) {
        const int bucket_id = first ? keep_id : next_component_id++;
        first = false;
        for (int leaf : kv.second) component_of_leaf[leaf] = bucket_id;
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

bool repair_all_components(
    std::vector<int>& component_of_leaf,
    const std::vector<TreeInfo>& infos,
    int upto_tree,
    int& next_component_id
) {
    bool any_change = false;
    while (true) {
        bool changed = false;
        const auto components = build_components(component_of_leaf);
        for (const auto& kv : components) {
            const std::vector<int>& leaves = kv.second;
            if (leaves.size() <= 1) continue;

            bool cluster_fail = false;
            const int last = std::min(upto_tree, static_cast<int>(infos.size()) - 1);
            for (int ti = 0; ti <= last; ++ti) {
                if (!isComponentValid(infos[ti], leaves)) {
                    cluster_fail = true;
                    break;
                }
            }

            if (cluster_fail) {
                for (int ti = 0; ti <= last; ++ti) {
                    if (!isComponentValid(infos[ti], leaves)) {
                        if (split_component_by_tree(component_of_leaf, leaves, infos[ti], next_component_id)) {
                            changed = true;
                            any_change = true;
                        }
                        break;
                    }
                }
                if (changed) break;
                continue;
            }

            if (!component_topology_valid_across_trees(leaves, infos, upto_tree)) {
                bool split = false;
                for (int i = 1; i <= last; ++i) {
                    if (split_component_by_two_trees(component_of_leaf, leaves, infos[0], infos[i], next_component_id)) {
                        split = true;
                        changed = true;
                        any_change = true;
                        break;
                    }
                }
                if (!split) {
                    if (split_component_by_tree(component_of_leaf, leaves, infos[0], next_component_id)) {
                        changed = true;
                        any_change = true;
                    }
                }
                if (changed) break;
            }
        }

        if (!changed) break;
    }
    return any_change;
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

std::string buildNewick(const TreeInfo& tree1, const std::vector<int>& comp_leaves, int leaf_count) {
    if (comp_leaves.empty()) return {};
    if (comp_leaves.size() == 1) return std::to_string(comp_leaves[0]) + ";";

    std::vector<char> in_component(leaf_count + 1, 0);
    for (int leaf : comp_leaves) in_component[leaf] = 1;

    std::string nw = restricted_newick_dfs(tree1.root, tree1, in_component);
    if (nw.empty()) {
        nw = std::to_string(comp_leaves.front());
        for (size_t i = 1; i < comp_leaves.size(); ++i) {
            nw = "(" + nw + "," + std::to_string(comp_leaves[i]) + ")";
        }
    }
    nw.push_back(';');
    return nw;
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

        std::string nw = buildNewick(base_tree, leaves, leaf_count);
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

bool validate_forest(
    const std::vector<int>& component_of_leaf,
    const std::vector<TreeInfo>& infos,
    int upto_tree
) {
    return partition_valid_across_all_trees(component_of_leaf, infos, upto_tree);
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

bool canMerge(const std::vector<int>& leaves, const std::vector<TreeInfo>& all_trees, int upto_tree) {
    return component_topology_valid_across_trees(leaves, all_trees, upto_tree);
}

bool canMerge_relaxed_cluster_only(
    const std::vector<int>& leaves,
    const std::vector<TreeInfo>& infos,
    int upto_tree
) {
    const int last = std::min(upto_tree, static_cast<int>(infos.size()) - 1);
    for (int ti = 0; ti <= last; ++ti) {
        if (!component_valid_in_tree(leaves, infos[ti])) return false;
    }
    return true;
}

bool try_speculative_merge(
    std::vector<int>& comp_of_leaf,
    const std::vector<int>& union_leaves,
    int upto_tree,
    const std::vector<TreeInfo>& infos,
    int& next_comp_id
) {
    std::vector<int> backup = comp_of_leaf;
    const int before = count_components(backup);

    const int new_id = next_comp_id++;
    for (int leaf : union_leaves) comp_of_leaf[leaf] = new_id;

    bool changed;
    do {
        changed = repair_all_components(comp_of_leaf, infos, upto_tree, next_comp_id);
    } while (changed);

    const int after = count_components(comp_of_leaf);
    if (after < before && validate_forest(comp_of_leaf, infos, upto_tree)) return true;

    comp_of_leaf.swap(backup);
    return false;
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
    std::vector<std::vector<Bitmask>> clusters_by_tree(m);
    std::vector<ClusterMap> cluster_maps(m);
    for (int t = 0; t < m; ++t) {
        clusters_by_tree[t] = getClusters(infos[t], n);
        for (const Bitmask& c : clusters_by_tree[t]) {
            cluster_maps[t][hashLeafSet(c)].push_back(c);
        }
    }

    const auto pivot_subtree_leaves = compute_subtree_leaf_lists(infos[0]);
    std::vector<int> pivot_internal_nodes;
    pivot_internal_nodes.reserve(infos[0].node_count);
    for (int v = 0; v < infos[0].node_count; ++v) {
        if (!infos[0].children[v].empty()) pivot_internal_nodes.push_back(v);
    }

    const auto start = std::chrono::steady_clock::now();
    const double soft_timeout = read_stride_timeout_seconds(30.0);
    const auto deadline = start + std::chrono::milliseconds(static_cast<long long>(soft_timeout * 850.0));

    auto run_one_attempt = [&](uint64_t seed) -> std::vector<int> {
        std::mt19937_64 rng(seed);
        std::vector<int> component_of_leaf(n + 1, -1);
        int next_component_id = n + 1;
        const int upto_tree = m - 1;

        // Phase 1: support-weighted compatible cluster packing.
        if (m >= 2) {
            std::vector<Bitmask> compatible_clusters;
            compatible_clusters.reserve(clusters_by_tree[0].size());
            for (const Bitmask& c : clusters_by_tree[0]) {
                if (isClusterCompatible(c, cluster_maps[1])) compatible_clusters.push_back(c);
            }

            std::shuffle(compatible_clusters.begin(), compatible_clusters.end(), rng);
            auto cluster_score = [&](const Bitmask& c) -> long long {
                int support = 0;
                for (int t = 0; t < m; ++t) {
                    if (isClusterCompatible(c, cluster_maps[t])) ++support;
                }
                return (static_cast<long long>(support) << 20) + bitmask_count(c);
            };
            std::sort(compatible_clusters.begin(), compatible_clusters.end(), [&](const Bitmask& a, const Bitmask& b) {
                const long long sa = cluster_score(a);
                const long long sb = cluster_score(b);
                if (sa != sb) return sa > sb;
                return bitmask_count(a) > bitmask_count(b);
            });

            Bitmask used_leaf = make_empty_bitmask(n);
            for (const Bitmask& c : compatible_clusters) {
                if (bitmask_intersects(c, used_leaf)) continue;
                std::vector<int> leaves = bitmask_to_leaves(c, n);
                if (leaves.size() <= 1) continue;
                const int cid = next_component_id++;
                for (int leaf : leaves) component_of_leaf[leaf] = cid;
                bitmask_or_assign(used_leaf, c);
            }
        }

        for (int leaf = 1; leaf <= n; ++leaf) {
            if (component_of_leaf[leaf] == -1) component_of_leaf[leaf] = next_component_id++;
        }

        // Phase 2: incremental refinement.
        if (m >= 2) {
            enforce_validity_for_trees(component_of_leaf, infos, 1, next_component_id);
            while (repair_all_components(component_of_leaf, infos, 1, next_component_id)) {}
        } else {
            enforce_validity_for_trees(component_of_leaf, infos, 0, next_component_id);
            while (repair_all_components(component_of_leaf, infos, 0, next_component_id)) {}
        }
        for (int ti = 2; ti < m; ++ti) {
            enforce_validity_for_trees(component_of_leaf, infos, ti, next_component_id);
            while (repair_all_components(component_of_leaf, infos, ti, next_component_id)) {}
        }
        while (repair_all_components(component_of_leaf, infos, upto_tree, next_component_id)) {}

        // Phase 3: targeted merge with speculative merge + repair.
        std::vector<int> node_order = pivot_internal_nodes;
        const int max_rounds = 10;
        for (int round = 0; round < max_rounds && std::chrono::steady_clock::now() < deadline; ++round) {
            std::shuffle(node_order.begin(), node_order.end(), rng);
            bool improved_round = false;

            for (int u : node_order) {
                if (std::chrono::steady_clock::now() >= deadline) break;
                const std::vector<int>& leaves_u = pivot_subtree_leaves[u];
                if (leaves_u.size() <= 1) continue;

                std::unordered_set<int> comp_ids_set;
                comp_ids_set.reserve(leaves_u.size());
                for (int leaf : leaves_u) comp_ids_set.insert(component_of_leaf[leaf]);
                if (comp_ids_set.size() <= 1) continue;

                const auto components = build_components(component_of_leaf);
                std::vector<int> union_leaves;
                for (int cid : comp_ids_set) {
                    auto it = components.find(cid);
                    if (it != components.end()) {
                        union_leaves.insert(union_leaves.end(), it->second.begin(), it->second.end());
                    }
                }
                std::sort(union_leaves.begin(), union_leaves.end());
                union_leaves.erase(std::unique(union_leaves.begin(), union_leaves.end()), union_leaves.end());
                if (union_leaves.size() <= 1) continue;

                if (!canMerge_relaxed_cluster_only(union_leaves, infos, upto_tree)) continue;
                if (try_speculative_merge(component_of_leaf, union_leaves, upto_tree, infos, next_component_id)) {
                    improved_round = true;
                }
            }

            if (!improved_round) break;
        }

        while (repair_all_components(component_of_leaf, infos, upto_tree, next_component_id)) {}
        if (!validate_forest(component_of_leaf, infos, upto_tree)) {
            std::vector<int> singleton(n + 1, 0);
            for (int leaf = 1; leaf <= n; ++leaf) singleton[leaf] = leaf;
            return singleton;
        }
        return component_of_leaf;
    };

    std::vector<int> best_comp(n + 1, 0);
    for (int leaf = 1; leaf <= n; ++leaf) best_comp[leaf] = leaf;
    int best_k = n;
    std::vector<std::string> best_forest = build_singleton_forest(n);

    const int attempts = (n <= 1500 ? 5 : (n <= 4000 ? 2 : 1));
    uint64_t base_seed = static_cast<uint64_t>(start.time_since_epoch().count());
    for (int a = 0; a < attempts && std::chrono::steady_clock::now() < deadline; ++a) {
        const uint64_t seed = base_seed + static_cast<uint64_t>(a) * 0x9e3779b97f4a7c15ULL;
        std::vector<int> comp = run_one_attempt(seed);
        const int k = count_components(comp);
        if (k < best_k) {
            best_k = k;
            best_comp = std::move(comp);
            best_forest = build_forest_from_partition(best_comp, infos[0], n);
            publish_best_solution(best_forest);  // publish only strict improvements
        }
    }

    if (best_k >= n) return build_singleton_forest(n);
    return best_forest;
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
