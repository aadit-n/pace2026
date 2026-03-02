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
#include <queue>
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
    std::vector<std::pair<uint64_t, uint64_t>> unrooted_hash;  // canonical unrooted hash per subtree root
    std::vector<uint8_t> unrooted_ready;  // cache flag for lazy unrooted hash computation
};

using Bitmask = std::vector<uint64_t>;
using ClusterMap = std::unordered_map<uint64_t, std::vector<Bitmask>>;
using HashPair = std::pair<uint64_t, uint64_t>;

struct HashPairHasher {
    size_t operator()(const HashPair& p) const noexcept {
        uint64_t x = p.first ^ (p.second + 0x9e3779b97f4a7c15ULL + (p.first << 6) + (p.first >> 2));
        return static_cast<size_t>(x);
    }
};

struct DSU {
    std::vector<int> p;
    std::vector<int> r;

    explicit DSU(int n) : p(n), r(n, 0) {
        std::iota(p.begin(), p.end(), 0);
    }

    int find(int x) {
        if (p[x] == x) return x;
        p[x] = find(p[x]);
        return p[x];
    }

    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) std::swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) ++r[a];
        return true;
    }
};

char* g_best_output = nullptr;
size_t g_best_output_len = 0;
uint64_t g_unrooted_hash_calls = 0;
uint64_t g_unrooted_hash_computed_new = 0;

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

std::vector<int> find_centers_of_tree(const std::vector<std::vector<int>>& adj);
std::pair<uint64_t, uint64_t> rooted_hash_from(
    int root,
    int parent,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>& node_label
);
std::pair<uint64_t, uint64_t> get_unrooted_hash_subtree(TreeInfo& T, int root);

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
    info.unrooted_hash.assign(info.node_count, {0, 0});
    info.unrooted_ready.assign(info.node_count, 0);
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

bool component_cluster_valid_all_trees(
    const std::vector<TreeInfo>& infos,
    const std::vector<int>& leaves,
    int upto_tree
) {
    if (leaves.size() <= 1) return true;
    if (infos.empty()) return true;
    const int last = std::min(upto_tree, static_cast<int>(infos.size()) - 1);
    if (last < 0) return true;

    for (int t = 0; t <= last; ++t) {
        auto [ok, r] = cluster_root(infos[t], leaves);
        if (!ok) return false;
        if (infos[t].subtree_leaf_count[r] != static_cast<int>(leaves.size())) return false;
    }
    return true;
}

std::vector<int> find_centers_of_tree(const std::vector<std::vector<int>>& adj) {
    const int n = static_cast<int>(adj.size());
    if (n == 0) return {};
    if (n == 1) return {0};

    std::vector<int> degree(n, 0);
    std::queue<int> q;
    for (int i = 0; i < n; ++i) {
        degree[i] = static_cast<int>(adj[i].size());
        if (degree[i] <= 1) q.push(i);
    }

    int remaining = n;
    while (remaining > 2 && !q.empty()) {
        const int layer_size = static_cast<int>(q.size());
        remaining -= layer_size;
        for (int i = 0; i < layer_size; ++i) {
            const int u = q.front();
            q.pop();
            for (int v : adj[u]) {
                if (--degree[v] == 1) q.push(v);
            }
        }
    }

    std::vector<int> centers;
    while (!q.empty()) {
        centers.push_back(q.front());
        q.pop();
    }
    if (centers.empty()) centers.push_back(0);
    return centers;
}

std::pair<uint64_t, uint64_t> rooted_hash_from(
    int root,
    int parent,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>& node_label
) {
    auto mix_a = [](uint64_t x) -> uint64_t {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    };
    auto mix_b = [](uint64_t x) -> uint64_t {
        x += 0x517cc1b727220a95ULL;
        x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
        x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
        return x ^ (x >> 33);
    };

    std::vector<std::pair<uint64_t, uint64_t>> child_hashes;
    for (int v : adj[root]) {
        if (v == parent) continue;
        child_hashes.push_back(rooted_hash_from(v, root, adj, node_label));
    }
    std::sort(child_hashes.begin(), child_hashes.end());

    uint64_t ha = mix_a(0xC001D00DULL);
    uint64_t hb = mix_b(0xBAD5EEDULL);
    if (node_label[root] >= 0) {
        ha = mix_a(ha ^ mix_a(static_cast<uint64_t>(node_label[root]) + 0x123456789abcdefULL));
        hb = mix_b(hb ^ mix_b(static_cast<uint64_t>(node_label[root]) + 0xfedcba987654321ULL));
    } else {
        ha = mix_a(ha ^ 0xABCDEF1234567890ULL);
        hb = mix_b(hb ^ 0x0123456789ABCDEFULL);
    }
    for (const auto& ch : child_hashes) {
        ha = mix_a(ha ^ mix_a(ch.first + 0x9e3779b97f4a7c15ULL));
        hb = mix_b(hb ^ mix_b(ch.second + 0x517cc1b727220a95ULL));
    }
    return {ha, hb};
}

std::pair<uint64_t, uint64_t> get_unrooted_hash_subtree(TreeInfo& T, int root) {
    ++g_unrooted_hash_calls;
    if (root < 0 || root >= T.node_count) return {0, 0};
    if (T.unrooted_ready[root]) return T.unrooted_hash[root];

    ++g_unrooted_hash_computed_new;
    std::vector<int> nodes;
    std::vector<int> parent_local;
    nodes.reserve(std::max(1, T.subtree_leaf_count[root] * 2));
    parent_local.reserve(std::max(1, T.subtree_leaf_count[root] * 2));

    std::vector<std::pair<int, int>> st;
    st.reserve(std::max(1, T.subtree_leaf_count[root] * 2));
    st.push_back({root, -1});

    while (!st.empty()) {
        auto [u, pl] = st.back();
        st.pop_back();
        const int uid = static_cast<int>(nodes.size());
        nodes.push_back(u);
        parent_local.push_back(pl);
        for (int c : T.children[u]) {
            st.push_back({c, uid});
        }
    }

    std::vector<std::vector<int>> adj(nodes.size());
    std::vector<int> node_label(nodes.size(), -1);
    for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
        const int u = nodes[i];
        if (T.node_to_leaf[u] > 0) node_label[i] = T.node_to_leaf[u];
        const int p = parent_local[i];
        if (p != -1) {
            adj[i].push_back(p);
            adj[p].push_back(i);
        }
    }

    const std::vector<int> centers = find_centers_of_tree(adj);
    auto h0 = rooted_hash_from(centers[0], -1, adj, node_label);
    std::pair<uint64_t, uint64_t> h = h0;
    if (centers.size() > 1) {
        auto h1 = rooted_hash_from(centers[1], -1, adj, node_label);
        h = std::min(h0, h1);
    }
    T.unrooted_hash[root] = h;
    T.unrooted_ready[root] = 1;
    return h;
}

std::pair<uint64_t, uint64_t> component_unrooted_hash(TreeInfo& T, const std::vector<int>& leaves) {
    auto [ok, r] = cluster_root(T, leaves);
    if (!ok) return {0, 0};
    if (T.subtree_leaf_count[r] != static_cast<int>(leaves.size())) return {0, 0};
    return get_unrooted_hash_subtree(T, r);
}

bool component_agrees_all_trees(
    const std::vector<int>& leaves,
    int upto_tree,
    std::vector<TreeInfo>& trees
) {
    if (leaves.size() <= 1) return true;
    if (trees.empty()) return true;
    const int last = std::min(upto_tree, static_cast<int>(trees.size()) - 1);
    if (last < 0) return true;

    const int sz = static_cast<int>(leaves.size());
    for (int t = 0; t <= last; ++t) {
        auto [ok, r] = cluster_root(trees[t], leaves);
        if (!ok || r == -1) return false;
        if (trees[t].subtree_leaf_count[r] != sz) return false;
    }

    const auto h0 = component_unrooted_hash(trees[0], leaves);
    for (int t = 1; t <= last; ++t) {
        const auto ht = component_unrooted_hash(trees[t], leaves);
        if (ht != h0) return false;
    }
    return true;
}

bool component_topology_valid_across_trees(
    const std::vector<int>& leaves,
    std::vector<TreeInfo>& infos,
    int upto_tree
) {
    return component_agrees_all_trees(leaves, upto_tree, infos);
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
    std::vector<TreeInfo>& infos,
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
    std::vector<TreeInfo>& infos,
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

            if (!component_agrees_all_trees(leaves, upto_tree, infos)) {
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
    std::vector<TreeInfo>& infos,
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

bool canMerge(const std::vector<int>& leaves, std::vector<TreeInfo>& all_trees, int upto_tree) {
    return component_agrees_all_trees(leaves, upto_tree, all_trees);
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
    std::vector<TreeInfo>& infos,
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
    if (
        after < before &&
        component_agrees_all_trees(union_leaves, upto_tree, infos) &&
        validate_forest(comp_of_leaf, infos, upto_tree)
    ) {
        return true;
    }

    comp_of_leaf.swap(backup);
    return false;
}

int count_conflicting_components(
    const std::vector<int>& component_of_leaf,
    std::vector<TreeInfo>& infos,
    int upto_tree
) {
    int conflicts = 0;
    const auto components = build_components(component_of_leaf);
    for (const auto& kv : components) {
        const auto& leaves = kv.second;
        if (leaves.size() <= 1) continue;
        if (!component_agrees_all_trees(leaves, upto_tree, infos)) ++conflicts;
    }
    return conflicts;
}

void cut_leaf_to_new_component(std::vector<int>& component_of_leaf, int leaf, int& next_component_id) {
    component_of_leaf[leaf] = next_component_id++;
}

std::vector<int> choose_conflict_witness(
    const std::vector<int>& leaves,
    const TreeInfo& A,
    const TreeInfo& B
) {
    std::vector<int> witness;
    if (leaves.size() <= 3) return leaves;

    const int rA = lca_of_component(leaves, A);
    const int rB = lca_of_component(leaves, B);
    if (rA == -1 || rB == -1) {
        return {leaves[0], leaves[1], leaves[2]};
    }

    std::unordered_map<long long, std::vector<int>> buckets;
    buckets.reserve(leaves.size());
    for (int leaf : leaves) {
        const int sa = child_side_under_lca(leaf, rA, A);
        const int sb = child_side_under_lca(leaf, rB, B);
        const long long key = (static_cast<long long>(sa) << 32) ^ static_cast<unsigned>(sb);
        buckets[key].push_back(leaf);
    }
    std::vector<std::vector<int>> groups;
    groups.reserve(buckets.size());
    for (auto& it : buckets) groups.push_back(std::move(it.second));
    std::sort(groups.begin(), groups.end(), [](const auto& x, const auto& y) {
        return x.size() > y.size();
    });

    for (const auto& g : groups) {
        if (!g.empty()) witness.push_back(g[0]);
        if (witness.size() == 3) break;
    }
    for (int leaf : leaves) {
        if (witness.size() == 3) break;
        if (std::find(witness.begin(), witness.end(), leaf) == witness.end()) witness.push_back(leaf);
    }
    return witness;
}

std::vector<int> solve_wz_cut_engine(
    int n,
    int m,
    std::vector<TreeInfo>& infos,
    const std::vector<std::pair<int, int>>& shared_cherries,
    std::chrono::steady_clock::time_point mode_deadline
) {
    std::vector<int> comp(n + 1, 1);
    int next_component_id = 2;
    const int upto2 = std::min(1, m - 1);
    const int upto_all = m - 1;

    auto evaluate = [&](const std::vector<int>& part) -> int {
        return count_conflicting_components(part, infos, upto2) * 100000 + count_components(part);
    };

    int best_score = evaluate(comp);
    int stall = 0;
    std::mt19937_64 rng(static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));

    while (std::chrono::steady_clock::now() < mode_deadline) {
        bool progressed = false;

        for (const auto& ch : shared_cherries) {
            const int a = ch.first;
            const int b = ch.second;
            if (comp[a] == comp[b]) continue;
            const auto components = build_components(comp);
            auto ita = components.find(comp[a]);
            auto itb = components.find(comp[b]);
            if (ita == components.end() || itb == components.end()) continue;
            std::vector<int> union_leaves = ita->second;
            union_leaves.insert(union_leaves.end(), itb->second.begin(), itb->second.end());
            std::sort(union_leaves.begin(), union_leaves.end());
            union_leaves.erase(std::unique(union_leaves.begin(), union_leaves.end()), union_leaves.end());
            if (!canMerge_relaxed_cluster_only(union_leaves, infos, upto2)) continue;
            if (try_speculative_merge(comp, union_leaves, upto2, infos, next_component_id)) {
                progressed = true;
            }
            if (std::chrono::steady_clock::now() >= mode_deadline) break;
        }

        std::vector<int> conflict_component;
        {
            const auto components = build_components(comp);
            for (const auto& kv : components) {
                if (kv.second.size() <= 1) continue;
                if (!component_agrees_all_trees(kv.second, upto2, infos)) {
                    conflict_component = kv.second;
                    break;
                }
            }
        }

        if (conflict_component.empty()) break;
        const std::vector<int> witness = choose_conflict_witness(conflict_component, infos[0], infos[1]);
        if (witness.empty()) break;

        std::vector<int> best_state = comp;
        int best_local = evaluate(comp);

        auto try_cut_plan = [&](const std::vector<int>& cut_leaves) {
            std::vector<int> cand = comp;
            int cand_next = next_component_id;
            for (int leaf : cut_leaves) cut_leaf_to_new_component(cand, leaf, cand_next);
            while (repair_all_components(cand, infos, upto2, cand_next)) {}
            const int score = evaluate(cand);
            if (score < best_local) {
                best_local = score;
                best_state = std::move(cand);
            }
        };

        try_cut_plan(witness);  // cut-all approx step
        for (int leaf : witness) {
            try_cut_plan({leaf});  // beam-like shallow branch
        }

        if (best_local < best_score) {
            best_score = best_local;
            comp = std::move(best_state);
            progressed = true;
            stall = 0;
        } else {
            ++stall;
        }

        if (!progressed && stall >= 8) stall = 0;
    }

    int next_all = next_component_id;
    enforce_validity_for_trees(comp, infos, upto_all, next_all);
    while (repair_all_components(comp, infos, upto_all, next_all)) {}
    return comp;
}

bool exact_wz_dfs(
    std::vector<int>& part,
    int cuts_used,
    int cut_budget,
    int& next_component_id,
    std::vector<TreeInfo>& infos,
    std::chrono::steady_clock::time_point deadline
) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
    if (validate_forest(part, infos, 1)) return true;
    if (cuts_used >= cut_budget) return false;

    std::vector<int> conflict_component;
    const auto components = build_components(part);
    for (const auto& kv : components) {
        if (kv.second.size() <= 1) continue;
        if (!component_agrees_all_trees(kv.second, 1, infos)) {
            conflict_component = kv.second;
            break;
        }
    }
    if (conflict_component.empty()) return false;

    const auto witness = choose_conflict_witness(conflict_component, infos[0], infos[1]);
    for (int leaf : witness) {
        std::vector<int> child = part;
        int child_next = next_component_id;
        cut_leaf_to_new_component(child, leaf, child_next);
        while (repair_all_components(child, infos, 1, child_next)) {}
        if (exact_wz_dfs(child, cuts_used + 1, cut_budget, child_next, infos, deadline)) {
            part = std::move(child);
            next_component_id = child_next;
            return true;
        }
    }
    return false;
}

std::vector<int> try_exact_wz(
    int n,
    std::vector<TreeInfo>& infos,
    int k_max,
    std::chrono::steady_clock::time_point deadline
) {
    std::vector<int> best(n + 1, 1);
    for (int k = 0; k <= k_max; ++k) {
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::vector<int> cur(n + 1, 1);
        int next_id = 2;
        if (exact_wz_dfs(cur, 0, k, next_id, infos, deadline)) return cur;
    }
    return best;
}

std::vector<int> lns_improve_partition(
    std::vector<int> base,
    int n,
    int upto_tree,
    std::vector<TreeInfo>& infos,
    const std::vector<int>& leaf_neighbor,
    std::chrono::steady_clock::time_point deadline
) {
    std::mt19937_64 rng(static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^ 0xA55A5AA5ULL);
    std::vector<int> best = std::move(base);
    int best_k = count_components(best);
    int next_component_id = n + 100;

    while (std::chrono::steady_clock::now() < deadline) {
        std::vector<int> cur = best;
        auto components = build_components(cur);
        std::vector<int> ids;
        ids.reserve(components.size());
        for (const auto& kv : components) ids.push_back(kv.first);
        if (ids.empty()) break;
        std::shuffle(ids.begin(), ids.end(), rng);
        int destroy_count = std::max(1, static_cast<int>(ids.size() * (0.10 + (rng() % 21) / 100.0)));
        destroy_count = std::min(destroy_count, static_cast<int>(ids.size()));

        for (int i = 0; i < destroy_count; ++i) {
            auto it = components.find(ids[i]);
            if (it == components.end()) continue;
            for (int leaf : it->second) cur[leaf] = next_component_id++;
        }
        while (repair_all_components(cur, infos, upto_tree, next_component_id)) {}

        for (int trial = 0; trial < 128 && std::chrono::steady_clock::now() < deadline; ++trial) {
            int a = 1 + static_cast<int>(rng() % static_cast<uint64_t>(n));
            int b = leaf_neighbor[a];
            if (b == -1) b = 1 + static_cast<int>(rng() % static_cast<uint64_t>(n));
            if (a == b || cur[a] == cur[b]) continue;
            auto cur_comps = build_components(cur);
            auto ita = cur_comps.find(cur[a]);
            auto itb = cur_comps.find(cur[b]);
            if (ita == cur_comps.end() || itb == cur_comps.end()) continue;
            std::vector<int> union_leaves = ita->second;
            union_leaves.insert(union_leaves.end(), itb->second.begin(), itb->second.end());
            std::sort(union_leaves.begin(), union_leaves.end());
            union_leaves.erase(std::unique(union_leaves.begin(), union_leaves.end()), union_leaves.end());
            if (!canMerge_relaxed_cluster_only(union_leaves, infos, upto_tree)) continue;
            (void)try_speculative_merge(cur, union_leaves, upto_tree, infos, next_component_id);
        }

        if (validate_forest(cur, infos, upto_tree)) {
            int k = count_components(cur);
            if (k < best_k) {
                best_k = k;
                best = std::move(cur);
            }
        }
    }
    return best;
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
    std::vector<std::vector<std::vector<int>>> subtree_leaves_by_tree(m);
    for (int t = 0; t < m; ++t) {
        subtree_leaves_by_tree[t] = compute_subtree_leaf_lists(infos[t]);
    }
    const auto& pivot_subtree_leaves = subtree_leaves_by_tree[0];

    std::vector<std::unordered_map<HashPair, std::vector<int>, HashPairHasher>> nodes_by_hash(m);
    for (int t = 0; t < m; ++t) {
        for (int v = 0; v < infos[t].node_count; ++v) {
            if (infos[t].subtree_leaf_count[v] < 2) continue;
            nodes_by_hash[t][infos[t].unrooted_hash[v]].push_back(v);
        }
    }

    std::vector<int> common_cluster_nodes0;
    common_cluster_nodes0.reserve(infos[0].node_count);
    for (int v = 0; v < infos[0].node_count; ++v) {
        if (infos[0].subtree_leaf_count[v] < 2) continue;
        const HashPair h = infos[0].unrooted_hash[v];
        bool in_all = true;
        for (int t = 1; t < m; ++t) {
            if (nodes_by_hash[t].find(h) == nodes_by_hash[t].end()) {
                in_all = false;
                break;
            }
        }
        if (in_all) common_cluster_nodes0.push_back(v);
    }
    std::sort(common_cluster_nodes0.begin(), common_cluster_nodes0.end(), [&](int a, int b) {
        return infos[0].subtree_leaf_count[a] > infos[0].subtree_leaf_count[b];
    });
    std::vector<int> pivot_internal_nodes;
    pivot_internal_nodes.reserve(infos[0].node_count);
    for (int v = 0; v < infos[0].node_count; ++v) {
        if (!infos[0].children[v].empty()) pivot_internal_nodes.push_back(v);
    }

    std::vector<int> leaf_neighbor(n + 1, -1);
    for (int leaf = 1; leaf <= n; ++leaf) {
        const int ln = infos[0].leaf_to_node[leaf];
        if (ln == -1) continue;
        const int p = infos[0].parent[ln];
        if (p == -1) continue;
        for (int c : infos[0].children[p]) {
            if (c == ln) continue;
            if (!pivot_subtree_leaves[c].empty()) {
                leaf_neighbor[leaf] = pivot_subtree_leaves[c][0];
                break;
            }
        }
    }

    std::vector<std::pair<int, int>> shared_cherries;
    if (m >= 2) {
        const auto ch0 = extract_cherries(infos[0]);
        const auto ch1 = extract_cherries(infos[1]);
        std::unordered_set<uint64_t> s1;
        s1.reserve(ch1.size() * 2 + 1);
        for (auto [a, b] : ch1) {
            if (a > b) std::swap(a, b);
            s1.insert((static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b));
        }
        shared_cherries.reserve(ch0.size());
        for (auto [a, b] : ch0) {
            if (a > b) std::swap(a, b);
            const uint64_t key = (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b);
            if (s1.find(key) != s1.end()) shared_cherries.push_back({a, b});
        }
    }

    const auto start = std::chrono::steady_clock::now();
    const double soft_timeout = read_stride_timeout_seconds(30.0);
    const auto deadline = start + std::chrono::milliseconds(static_cast<long long>(soft_timeout * 850.0));

    auto run_one_attempt = [&](uint64_t seed, std::chrono::steady_clock::time_point attempt_deadline) -> std::vector<int> {
        std::mt19937_64 rng(seed);
        std::vector<int> component_of_leaf(n + 1, -1);
        int next_component_id = n + 1;
        const int upto_tree = m - 1;
        const auto attempt_start = std::chrono::steady_clock::now();
        if (attempt_start >= attempt_deadline) {
            std::vector<int> singleton(n + 1, 0);
            for (int leaf = 1; leaf <= n; ++leaf) singleton[leaf] = leaf;
            return singleton;
        }
        const auto budget = std::chrono::duration_cast<std::chrono::milliseconds>(attempt_deadline - attempt_start).count();
        const auto phase1_end = attempt_start + std::chrono::milliseconds(static_cast<long long>(budget * 0.20));
        const auto phase2_end = attempt_start + std::chrono::milliseconds(static_cast<long long>(budget * 0.80));

        std::vector<int> best_valid_partition(n + 1, 0);
        for (int leaf = 1; leaf <= n; ++leaf) best_valid_partition[leaf] = leaf;
        int best_valid_k = n;
        auto update_best_valid = [&]() {
            if (validate_forest(component_of_leaf, infos, upto_tree)) {
                const int k = count_components(component_of_leaf);
                if (k < best_valid_k) {
                    best_valid_k = k;
                    best_valid_partition = component_of_leaf;
                }
            }
        };

        // Phase A: common-subtree contraction (large-first) using canonical unrooted hashes.
        {
            DSU dsu(n + 1);
            for (int v0 : common_cluster_nodes0) {
                const std::vector<int>& leaves = pivot_subtree_leaves[v0];
                if (leaves.size() < 2) continue;

                const int r0 = dsu.find(leaves[0]);
                bool all_same = true;
                for (size_t i = 1; i < leaves.size(); ++i) {
                    if (dsu.find(leaves[i]) != r0) {
                        all_same = false;
                        break;
                    }
                }
                if (all_same) continue;

                const int first = leaves[0];
                for (size_t i = 1; i < leaves.size(); ++i) {
                    dsu.unite(first, leaves[i]);
                }
            }

            for (int leaf = 1; leaf <= n; ++leaf) {
                component_of_leaf[leaf] = dsu.find(leaf);
            }
        }

        for (int leaf = 1; leaf <= n; ++leaf) {
            if (component_of_leaf[leaf] == -1) component_of_leaf[leaf] = next_component_id++;
        }
        update_best_valid();

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
        update_best_valid();

        // Phase 3a: small moves (pairwise + common-cluster-guided), 60% time slice.
        while (std::chrono::steady_clock::now() < phase2_end) {
            bool improved_batch = false;

            for (int trial = 0; trial < 64 && std::chrono::steady_clock::now() < phase2_end; ++trial) {
                int a = 1 + static_cast<int>(rng() % static_cast<uint64_t>(n));
                int b = leaf_neighbor[a];
                if (b == -1) b = 1 + static_cast<int>(rng() % static_cast<uint64_t>(n));
                if (a == b) continue;
                if (component_of_leaf[a] == component_of_leaf[b]) continue;

                const auto components = build_components(component_of_leaf);
                auto ita = components.find(component_of_leaf[a]);
                auto itb = components.find(component_of_leaf[b]);
                if (ita == components.end() || itb == components.end()) continue;

                std::vector<int> union_leaves = ita->second;
                union_leaves.insert(union_leaves.end(), itb->second.begin(), itb->second.end());
                std::sort(union_leaves.begin(), union_leaves.end());
                union_leaves.erase(std::unique(union_leaves.begin(), union_leaves.end()), union_leaves.end());
                if (!canMerge_relaxed_cluster_only(union_leaves, infos, upto_tree)) continue;
                if (try_speculative_merge(component_of_leaf, union_leaves, upto_tree, infos, next_component_id)) {
                    improved_batch = true;
                    update_best_valid();
                }
            }

            std::vector<int> guided = common_cluster_nodes0;
            std::shuffle(guided.begin(), guided.end(), rng);
            int guided_limit = 0;
            for (int v0 : guided) {
                if (std::chrono::steady_clock::now() >= phase2_end) break;
                if (++guided_limit > 128) break;
                const std::vector<int>& leaves = pivot_subtree_leaves[v0];
                if (leaves.size() <= 1) continue;

                std::unordered_set<int> comp_ids;
                comp_ids.reserve(leaves.size());
                for (int leaf : leaves) comp_ids.insert(component_of_leaf[leaf]);
                if (comp_ids.size() <= 1) continue;

                const auto components = build_components(component_of_leaf);
                std::vector<int> union_leaves;
                for (int cid : comp_ids) {
                    auto it = components.find(cid);
                    if (it != components.end()) {
                        union_leaves.insert(union_leaves.end(), it->second.begin(), it->second.end());
                    }
                }
                std::sort(union_leaves.begin(), union_leaves.end());
                union_leaves.erase(std::unique(union_leaves.begin(), union_leaves.end()), union_leaves.end());
                if (!canMerge_relaxed_cluster_only(union_leaves, infos, upto_tree)) continue;
                if (try_speculative_merge(component_of_leaf, union_leaves, upto_tree, infos, next_component_id)) {
                    improved_batch = true;
                    update_best_valid();
                }
            }

            if (!improved_batch && std::chrono::steady_clock::now() > phase1_end) break;
        }

        // Phase 3b: larger targeted merges, final 20% time slice.
        std::vector<int> node_order = pivot_internal_nodes;
        const int max_rounds = 10;
        int no_improve_rounds = 0;
        bool switch_to_small_only = false;
        for (int round = 0; round < max_rounds && std::chrono::steady_clock::now() < attempt_deadline; ++round) {
            std::shuffle(node_order.begin(), node_order.end(), rng);
            bool improved_round = false;

            for (int u : node_order) {
                if (std::chrono::steady_clock::now() >= attempt_deadline) break;
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
                    update_best_valid();
                }
            }

            if (!improved_round) {
                ++no_improve_rounds;
            } else {
                no_improve_rounds = 0;
            }
            if (!improved_round && no_improve_rounds >= 2 && std::chrono::steady_clock::now() > (attempt_start + std::chrono::milliseconds(static_cast<long long>(budget * 0.70)))) {
                switch_to_small_only = true;
                break;
            }
            if (!improved_round) break;
        }

        if (switch_to_small_only) {
            while (std::chrono::steady_clock::now() < attempt_deadline) {
                int a = 1 + static_cast<int>(rng() % static_cast<uint64_t>(n));
                int b = leaf_neighbor[a];
                if (b == -1) b = 1 + static_cast<int>(rng() % static_cast<uint64_t>(n));
                if (a == b) continue;
                if (component_of_leaf[a] == component_of_leaf[b]) continue;
                const auto components = build_components(component_of_leaf);
                auto ita = components.find(component_of_leaf[a]);
                auto itb = components.find(component_of_leaf[b]);
                if (ita == components.end() || itb == components.end()) continue;
                std::vector<int> union_leaves = ita->second;
                union_leaves.insert(union_leaves.end(), itb->second.begin(), itb->second.end());
                std::sort(union_leaves.begin(), union_leaves.end());
                union_leaves.erase(std::unique(union_leaves.begin(), union_leaves.end()), union_leaves.end());
                if (!canMerge_relaxed_cluster_only(union_leaves, infos, upto_tree)) continue;
                if (try_speculative_merge(component_of_leaf, union_leaves, upto_tree, infos, next_component_id)) {
                    update_best_valid();
                }
            }
        }

        while (repair_all_components(component_of_leaf, infos, upto_tree, next_component_id)) {}
        update_best_valid();
        if (!validate_forest(component_of_leaf, infos, upto_tree)) return best_valid_partition;
        return (count_components(component_of_leaf) <= best_valid_k ? component_of_leaf : best_valid_partition);
    };

    std::vector<int> best_comp(n + 1, 0);
    for (int leaf = 1; leaf <= n; ++leaf) best_comp[leaf] = leaf;
    int best_k = n;
    std::vector<std::string> best_forest = build_singleton_forest(n);

    auto maybe_update_best = [&](std::vector<int> cand) {
        if (!validate_forest(cand, infos, m - 1)) return;
        const int k = count_components(cand);
        if (k < best_k) {
            best_k = k;
            best_comp = std::move(cand);
            best_forest = build_forest_from_partition(best_comp, infos[0], n);
            publish_best_solution(best_forest);
        }
    };

    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - start).count();
    const auto mode2_end = start + std::chrono::milliseconds(static_cast<long long>(total_ms * 0.15));
    const auto mode1_end = start + std::chrono::milliseconds(static_cast<long long>(total_ms * 0.70));
    const auto mode3_end = start + std::chrono::milliseconds(static_cast<long long>(total_ms * 0.90));

    // Mode 2: exact-on-easy (time-boxed).
    if (m >= 2 && n <= 120 && std::chrono::steady_clock::now() < mode2_end) {
        std::vector<int> exact = try_exact_wz(n, infos, 6, mode2_end);
        int next_id = n + 1;
        while (repair_all_components(exact, infos, m - 1, next_id)) {}
        maybe_update_best(std::move(exact));
    }

    // Mode 1: WZ-style conflict-driven cut engine.
    if (m >= 2 && std::chrono::steady_clock::now() < mode1_end) {
        std::vector<int> wz = solve_wz_cut_engine(n, m, infos, shared_cherries, mode1_end);
        maybe_update_best(std::move(wz));
    }

    // Mode 3: previous high-quality constructor as strong baseline.
    uint64_t base_seed = static_cast<uint64_t>(start.time_since_epoch().count());
    int a = 0;
    while (std::chrono::steady_clock::now() < mode3_end) {
        const uint64_t seed = base_seed + static_cast<uint64_t>(a) * 0x9e3779b97f4a7c15ULL;
        std::vector<int> comp = run_one_attempt(seed, mode3_end);
        maybe_update_best(std::move(comp));
        ++a;
    }

    // Mode 4: LNS destroy-repair on incumbent.
    if (std::chrono::steady_clock::now() < deadline) {
        std::vector<int> improved = lns_improve_partition(best_comp, n, m - 1, infos, leaf_neighbor, deadline);
        maybe_update_best(std::move(improved));
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
        std::cerr << "#stats unrooted_hash_calls=" << g_unrooted_hash_calls
                  << " unrooted_hash_computed_new=" << g_unrooted_hash_computed_new << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
