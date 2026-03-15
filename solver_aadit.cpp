#include "genesis/genesis.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

using genesis::tree::Tree;
using Clock = std::chrono::steady_clock;

uint64_t mix64(uint64_t x);

struct PaceInstance {
    int tree_count = 0;
    int leaf_count = 0;
    std::vector<std::string> newick_lines;
};

struct Signature {
    uint64_t h = 0;
    int sz = 0;
    bool operator==(const Signature& o) const { return h == o.h && sz == o.sz; }
};

struct SignatureHasher {
    size_t operator()(const Signature& s) const {
        uint64_t x = s.h ^ (static_cast<uint64_t>(s.sz) * 0x9e3779b97f4a7c15ULL);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        return static_cast<size_t>(x);
    }
};

struct SimpleTree {
    int root = -1;
    std::vector<std::vector<int>> children;
    std::vector<int> parent;
    std::vector<int> leaf_label; // 0 internal, 1..n leaf
};

struct TreeData {
    int n_nodes = 0;
    int root = -1;
    std::vector<int> parent;
    std::vector<int> child0;
    std::vector<int> child1;
    std::vector<char> is_leaf;
    std::vector<int> leaf_label;
    std::vector<int> node_of_leaf;

    std::vector<int> depth;
    std::vector<std::vector<int>> up;

    std::vector<int> sub_count;
    std::vector<uint64_t> sub_hash;
    std::vector<uint64_t> topo_a;
    std::vector<uint64_t> topo_b;

    std::unordered_set<Signature, SignatureHasher> cluster_sigs;
};

struct DynNode {
    int parent = -1;
    int left = -1;
    int right = -1;
    int label = 0; // current active pseudo-leaf label, 0 for internal
    bool active = true;

    // Contraction history tree for expansion to original labels.
    int comp_left = -1;
    int comp_right = -1;
    std::vector<int> merged_original; // non-empty iff this node is a pseudo leaf

    // Rooted subtree signature for active pseudo leaves.
    uint64_t leafset_hash = 0;
    int payload_size = 0;
    uint64_t topo_a = 0;
    uint64_t topo_b = 0;
};

struct DynamicTree {
    int root = -1;
    std::vector<DynNode> nodes;
    std::unordered_map<int, int> label_to_node; // active pseudo-label -> node
    std::vector<char> is_cherry;
    std::vector<int> cherry_pos;
    std::vector<int> cherry_nodes;
};

struct ReducedSubtreeSig {
    uint64_t leafset_hash = 0;
    int leaf_count = 0;
    uint64_t topo_a = 0;
    uint64_t topo_b = 0;
};

struct ReducedSubtreeKey {
    uint64_t leafset_hash = 0;
    int leaf_count = 0;
    uint64_t topo_a = 0;
    uint64_t topo_b = 0;

    bool operator==(const ReducedSubtreeKey& o) const {
        return leafset_hash == o.leafset_hash &&
               leaf_count == o.leaf_count &&
               topo_a == o.topo_a &&
               topo_b == o.topo_b;
    }
};

struct ReducedSubtreeKeyHasher {
    size_t operator()(const ReducedSubtreeKey& s) const {
        uint64_t x = s.leafset_hash;
        x ^= mix64(static_cast<uint64_t>(s.leaf_count) + 0x9e3779b97f4a7c15ULL);
        x ^= mix64(s.topo_a + 0x517cc1b727220a95ULL);
        x ^= mix64(s.topo_b + 0x94d049bb133111ebULL);
        return static_cast<size_t>(mix64(x));
    }
};

volatile sig_atomic_t g_terminate = 0;
volatile sig_atomic_t g_output_written = 0;
char* g_best_output = nullptr;
size_t g_best_output_len = 0;
int g_last_leaf_count = 1;

std::string trim(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    size_t j = s.size();
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) --j;
    return s.substr(i, j - i);
}

void write_best_once() {
    if (g_output_written) return;
    g_output_written = 1;
    if (g_best_output && g_best_output_len > 0) {
        const ssize_t unused = write(STDOUT_FILENO, g_best_output, g_best_output_len);
        (void)unused;
    }
}

void sig_handler(int) {
    g_terminate = 1;
    write_best_once();
    _Exit(0);
}

void set_best_output_buffer(const std::string& output) {
    sigset_t set;
    sigset_t old;
    sigemptyset(&set);
    sigaddset(&set, SIGTERM);
    sigaddset(&set, SIGINT);
    sigprocmask(SIG_BLOCK, &set, &old);

    char* nb = nullptr;
    if (!output.empty()) {
        nb = new char[output.size()];
        std::memcpy(nb, output.data(), output.size());
    }
    char* oldp = g_best_output;
    g_best_output = nb;
    g_best_output_len = output.size();
    delete[] oldp;

    sigprocmask(SIG_SETMASK, &old, nullptr);
}

std::vector<std::string> singleton_forest(int n) {
    std::vector<std::string> out;
    out.reserve(n);
    for (int i = 1; i <= n; ++i) out.push_back(std::to_string(i) + ";");
    return out;
}

std::string normalize_output(const std::vector<std::string>& lines) {
    std::string out;
    for (std::string s : lines) {
        s = trim(s);
        if (s.empty()) continue;
        if (s.back() != ';') s.push_back(';');
        out += s;
        out.push_back('\n');
    }
    return out;
}

void publish_best_solution(const std::vector<std::string>& forest) {
    set_best_output_buffer(normalize_output(forest));
}

bool parse_with_genesis(const std::string& line) {
    try {
        genesis::tree::CommonTreeNewickReader reader;
        auto src = genesis::utils::from_string(line);
        Tree t = reader.read(src);
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

class NewickParser {
public:
    explicit NewickParser(std::string s) : s_(std::move(s)) {}

    bool parse(SimpleTree& t) {
        try {
            skip_ws();
            int r = parse_subtree(t);
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ';') return false;
            ++pos_;
            skip_ws();
            if (pos_ != s_.size()) return false;
            t.root = r;
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    std::string s_;
    size_t pos_ = 0;

    static bool delim(char c) { return c == '(' || c == ')' || c == ',' || c == ';'; }

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    int add_node(SimpleTree& t, int lbl) {
        int id = static_cast<int>(t.children.size());
        t.children.push_back({});
        t.parent.push_back(-1);
        t.leaf_label.push_back(lbl);
        return id;
    }

    std::string read_token() {
        skip_ws();
        std::string tok;
        while (pos_ < s_.size() && !delim(s_[pos_])) {
            tok.push_back(s_[pos_]);
            ++pos_;
        }
        return trim(tok);
    }

    static int parse_label(const std::string& tok) {
        size_t i = 0;
        while (i < tok.size() && !std::isdigit(static_cast<unsigned char>(tok[i]))) ++i;
        size_t j = i;
        while (j < tok.size() && std::isdigit(static_cast<unsigned char>(tok[j]))) ++j;
        if (i == j) throw std::runtime_error("missing leaf label");
        return std::stoi(tok.substr(i, j - i));
    }

    int parse_subtree(SimpleTree& t) {
        skip_ws();
        if (pos_ >= s_.size()) throw std::runtime_error("eof");
        if (s_[pos_] == '(') {
            ++pos_;
            int u = add_node(t, 0);
            int a = parse_subtree(t);
            t.children[u].push_back(a);
            t.parent[a] = u;
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ',') throw std::runtime_error("comma");
            ++pos_;
            int b = parse_subtree(t);
            t.children[u].push_back(b);
            t.parent[b] = u;
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ')') throw std::runtime_error("close");
            ++pos_;
            (void)read_token();
            return u;
        }

        std::string tok = read_token();
        if (tok.empty()) throw std::runtime_error("empty leaf");
        return add_node(t, parse_label(tok));
    }
};

bool read_instance(PaceInstance& inst) {
    std::string line;
    bool have_p = false;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("#p", 0) == 0) {
                std::istringstream ss(line);
                std::string tag;
                ss >> tag >> inst.tree_count >> inst.leaf_count;
                if (inst.tree_count > 0 && inst.leaf_count > 0) {
                    have_p = true;
                    g_last_leaf_count = inst.leaf_count;
                }
            }
            continue;
        }
        if (!have_p) continue;
        if (static_cast<int>(inst.newick_lines.size()) < inst.tree_count) {
            inst.newick_lines.push_back(line);
        }
    }
    return have_p && static_cast<int>(inst.newick_lines.size()) == inst.tree_count;
}

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

uint64_t instance_seed(const PaceInstance& inst) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h ^= static_cast<uint64_t>(inst.leaf_count) + 0x9e3779b97f4a7c15ULL;
    for (const auto& s : inst.newick_lines) {
        for (unsigned char c : s) {
            h ^= static_cast<uint64_t>(c);
            h *= 0x100000001b3ULL;
        }
        h = mix64(h);
    }
    return h;
}

bool build_tree_data(const SimpleTree& st, int n, const std::vector<uint64_t>& zob, TreeData& td) {
    td.n_nodes = static_cast<int>(st.children.size());
    td.root = st.root;
    td.parent = st.parent;
    td.child0.assign(td.n_nodes, -1);
    td.child1.assign(td.n_nodes, -1);
    td.is_leaf.assign(td.n_nodes, 0);
    td.leaf_label = st.leaf_label;
    td.node_of_leaf.assign(n + 1, -1);

    int leaves = 0;
    std::vector<int> seen(n + 1, 0);
    for (int u = 0; u < td.n_nodes; ++u) {
        if (st.children[u].empty()) {
            td.is_leaf[u] = 1;
            int x = st.leaf_label[u];
            if (x < 1 || x > n || seen[x]) return false;
            seen[x] = 1;
            td.node_of_leaf[x] = u;
            ++leaves;
        } else {
            if (st.children[u].size() != 2) return false;
            td.child0[u] = st.children[u][0];
            td.child1[u] = st.children[u][1];
        }
    }
    if (leaves != n) return false;

    td.depth.assign(td.n_nodes, 0);
    std::vector<int> bfs{td.root};
    for (size_t i = 0; i < bfs.size(); ++i) {
        int u = bfs[i];
        if (!td.is_leaf[u]) {
            int a = td.child0[u], b = td.child1[u];
            td.depth[a] = td.depth[u] + 1;
            td.depth[b] = td.depth[u] + 1;
            bfs.push_back(a);
            bfs.push_back(b);
        }
    }

    int lg = 1;
    while ((1 << lg) <= std::max(1, td.n_nodes)) ++lg;
    td.up.assign(lg, std::vector<int>(td.n_nodes, td.root));
    for (int v = 0; v < td.n_nodes; ++v) td.up[0][v] = (td.parent[v] == -1 ? v : td.parent[v]);
    for (int k = 1; k < lg; ++k) {
        for (int v = 0; v < td.n_nodes; ++v) td.up[k][v] = td.up[k - 1][td.up[k - 1][v]];
    }

    td.sub_count.assign(td.n_nodes, 0);
    td.sub_hash.assign(td.n_nodes, 0);
    td.topo_a.assign(td.n_nodes, 0);
    td.topo_b.assign(td.n_nodes, 0);
    td.cluster_sigs.clear();

    std::vector<std::pair<int, int>> stck;
    stck.reserve(td.n_nodes * 2);
    stck.push_back({td.root, 0});
    while (!stck.empty()) {
        auto [u, phase] = stck.back();
        stck.pop_back();
        if (!phase) {
            stck.push_back({u, 1});
            if (!td.is_leaf[u]) {
                stck.push_back({td.child0[u], 0});
                stck.push_back({td.child1[u], 0});
            }
            continue;
        }
        if (td.is_leaf[u]) {
            int x = td.leaf_label[u];
            td.sub_count[u] = 1;
            td.sub_hash[u] = zob[x];
            td.topo_a[u] = mix64(static_cast<uint64_t>(x) + 0x1111111111111111ULL);
            td.topo_b[u] = mix64(static_cast<uint64_t>(x) + 0x2222222222222222ULL);
        } else {
            int a = td.child0[u], b = td.child1[u];
            td.sub_count[u] = td.sub_count[a] + td.sub_count[b];
            td.sub_hash[u] = td.sub_hash[a] ^ td.sub_hash[b];
            uint64_t aa = td.topo_a[a], ab = td.topo_a[b];
            uint64_t ba = td.topo_b[a], bb = td.topo_b[b];
            if (aa > ab) std::swap(aa, ab);
            if (ba > bb) std::swap(ba, bb);
            td.topo_a[u] = mix64(aa ^ (ab + 0x9e3779b97f4a7c15ULL));
            td.topo_b[u] = mix64(ba ^ (bb + 0x517cc1b727220a95ULL));
        }
        td.cluster_sigs.insert(Signature{td.sub_hash[u], td.sub_count[u]});
    }

    return true;
}

int lca_node(const TreeData& t, int a, int b) {
    if (t.depth[a] < t.depth[b]) std::swap(a, b);
    int d = t.depth[a] - t.depth[b];
    for (size_t i = 0; i < t.up.size(); ++i) if (d & (1 << i)) a = t.up[i][a];
    if (a == b) return a;
    for (int i = static_cast<int>(t.up.size()) - 1; i >= 0; --i) {
        if (t.up[i][a] != t.up[i][b]) {
            a = t.up[i][a];
            b = t.up[i][b];
        }
    }
    return t.parent[a];
}

int lca_of_leaves(const TreeData& t, const std::vector<int>& leaves) {
    int l = t.node_of_leaf[leaves[0]];
    for (size_t i = 1; i < leaves.size(); ++i) l = lca_node(t, l, t.node_of_leaf[leaves[i]]);
    return l;
}

bool component_valid_both_trees(
    const std::vector<int>& leaves,
    const std::vector<uint64_t>& zob,
    const std::vector<TreeData>& trees
) {
    if (leaves.empty()) return false;
    uint64_t h = 0;
    for (int x : leaves) h ^= zob[x];
    int sz = static_cast<int>(leaves.size());
    Signature s{h, sz};

    std::pair<uint64_t, uint64_t> ref{0, 0};
    bool have_ref = false;
    for (const auto& t : trees) {
        if (!t.cluster_sigs.count(s)) return false;
        if (sz == 1) continue;
        int r = lca_of_leaves(t, leaves);
        if (t.sub_count[r] != sz || t.sub_hash[r] != h) return false;
        std::pair<uint64_t, uint64_t> tp{t.topo_a[r], t.topo_b[r]};
        if (!have_ref) {
            ref = tp;
            have_ref = true;
        } else if (tp != ref) {
            return false;
        }
    }
    return true;
}

std::string restricted_newick_dfs(int u, const TreeData& t, const std::vector<char>& in_comp) {
    if (t.is_leaf[u]) {
        int x = t.leaf_label[u];
        return in_comp[x] ? std::to_string(x) : std::string();
    }
    std::string a = restricted_newick_dfs(t.child0[u], t, in_comp);
    std::string b = restricted_newick_dfs(t.child1[u], t, in_comp);
    if (a.empty() && b.empty()) return {};
    if (a.empty()) return b;
    if (b.empty()) return a;
    return "(" + a + "," + b + ")";
}

std::vector<std::string> forest_from_partition(
    const std::vector<std::vector<int>>& comps,
    int n,
    const TreeData& base_tree
) {
    std::vector<std::pair<int, std::string>> lines;
    lines.reserve(comps.size());
    for (auto leaves : comps) {
        if (leaves.empty()) continue;
        std::sort(leaves.begin(), leaves.end());
        std::string line;
        if (leaves.size() == 1) {
            line = std::to_string(leaves[0]) + ";";
        } else {
            std::vector<char> in_comp(n + 1, 0);
            for (int x : leaves) in_comp[x] = 1;
            line = restricted_newick_dfs(base_tree.root, base_tree, in_comp);
            if (line.empty()) {
                line = std::to_string(leaves[0]);
                for (size_t i = 1; i < leaves.size(); ++i) {
                    line = "(" + line + "," + std::to_string(leaves[i]) + ")";
                }
            }
            line.push_back(';');
        }
        lines.push_back({leaves[0], line});
    }
    std::sort(lines.begin(), lines.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<std::string> out;
    out.reserve(lines.size());
    for (auto& p : lines) out.push_back(std::move(p.second));
    return out;
}

DynamicTree build_dynamic_tree(const SimpleTree& st) {
    DynamicTree dt;
    dt.nodes.resize(st.children.size());
    dt.is_cherry.assign(st.children.size(), 0);
    dt.cherry_pos.assign(st.children.size(), -1);
    dt.root = st.root;
    for (size_t u = 0; u < st.children.size(); ++u) {
        DynNode& dn = dt.nodes[u];
        dn.parent = st.parent[u];
        dn.left = -1;
        dn.right = -1;
        dn.active = true;
        dn.comp_left = -1;
        dn.comp_right = -1;

        if (st.children[u].empty()) {
            dn.label = st.leaf_label[u];
            dn.merged_original = {st.leaf_label[u]};
            dn.leafset_hash = mix64(static_cast<uint64_t>(st.leaf_label[u]) * 0x9e3779b97f4a7c15ULL + 0xabcULL);
            dn.payload_size = 1;
            dn.topo_a = mix64(static_cast<uint64_t>(st.leaf_label[u]) + 0x1111111111111111ULL);
            dn.topo_b = mix64(static_cast<uint64_t>(st.leaf_label[u]) + 0x2222222222222222ULL);
            dt.label_to_node[dn.label] = static_cast<int>(u);
        } else {
            dn.label = 0;
            dn.left = st.children[u][0];
            dn.right = st.children[u][1];
        }
    }
    for (int u = 0; u < static_cast<int>(dt.nodes.size()); ++u) {
        int l = dt.nodes[u].left;
        int r = dt.nodes[u].right;
        if (l != -1 && r != -1 &&
            dt.nodes[l].active && dt.nodes[r].active &&
            dt.nodes[l].left == -1 && dt.nodes[l].right == -1 &&
            dt.nodes[r].left == -1 && dt.nodes[r].right == -1) {
            dt.is_cherry[u] = 1;
            dt.cherry_pos[u] = static_cast<int>(dt.cherry_nodes.size());
            dt.cherry_nodes.push_back(u);
        }
    }
    return dt;
}

bool is_leaflike(const DynamicTree& t, int u) {
    return u >= 0 && u < static_cast<int>(t.nodes.size()) &&
           t.nodes[u].active && t.nodes[u].left == -1 && t.nodes[u].right == -1;
}

void ensure_cherry_storage(DynamicTree& t) {
    if (t.is_cherry.size() < t.nodes.size()) t.is_cherry.resize(t.nodes.size(), 0);
    if (t.cherry_pos.size() < t.nodes.size()) t.cherry_pos.resize(t.nodes.size(), -1);
}

void remove_cherry_node(DynamicTree& t, int u) {
    if (u < 0 || u >= static_cast<int>(t.cherry_pos.size())) return;
    int pos = t.cherry_pos[u];
    if (pos == -1) return;
    int last = t.cherry_nodes.back();
    t.cherry_nodes[pos] = last;
    t.cherry_pos[last] = pos;
    t.cherry_nodes.pop_back();
    t.cherry_pos[u] = -1;
    t.is_cherry[u] = 0;
}

void refresh_local_cherry_status(DynamicTree& t, int u) {
    if (u < 0 || u >= static_cast<int>(t.nodes.size())) return;
    ensure_cherry_storage(t);

    bool now = false;
    if (t.nodes[u].active) {
        int l = t.nodes[u].left;
        int r = t.nodes[u].right;
        now = (l != -1 && r != -1 && is_leaflike(t, l) && is_leaflike(t, r));
    }
    if (now) {
        if (t.cherry_pos[u] == -1) {
            t.cherry_pos[u] = static_cast<int>(t.cherry_nodes.size());
            t.cherry_nodes.push_back(u);
        }
        t.is_cherry[u] = 1;
    } else {
        remove_cherry_node(t, u);
    }
}

void refresh_cherry_neighborhood(DynamicTree& t, int u) {
    if (u < 0 || u >= static_cast<int>(t.nodes.size())) return;
    refresh_local_cherry_status(t, u);
    int p = t.nodes[u].parent;
    if (p != -1) {
        refresh_local_cherry_status(t, p);
        int gp = t.nodes[p].parent;
        if (gp != -1) refresh_local_cherry_status(t, gp);
    }
    int l = t.nodes[u].left;
    int r = t.nodes[u].right;
    if (l != -1) refresh_local_cherry_status(t, l);
    if (r != -1) refresh_local_cherry_status(t, r);
}

bool is_active_leaf(const DynamicTree& t, int u) {
    return is_leaflike(t, u);
}

int degree_children(const DynamicTree& t, int u) {
    int d = 0;
    if (t.nodes[u].left != -1 && t.nodes[t.nodes[u].left].active) ++d;
    if (t.nodes[u].right != -1 && t.nodes[t.nodes[u].right].active) ++d;
    return d;
}

void replace_child(DynamicTree& t, int p, int oldc, int newc) {
    if (p == -1) return;
    if (t.nodes[p].left == oldc) t.nodes[p].left = newc;
    else if (t.nodes[p].right == oldc) t.nodes[p].right = newc;
}

void suppress_up(DynamicTree& t, int u) {
    while (u != -1 && t.nodes[u].active) {
        int deg = degree_children(t, u);
        if (deg != 1) break;
        remove_cherry_node(t, u);
        int child = (t.nodes[u].left != -1 && t.nodes[t.nodes[u].left].active) ? t.nodes[u].left : t.nodes[u].right;
        int p = t.nodes[u].parent;
        if (p == -1) {
            t.nodes[child].parent = -1;
            t.root = child;
            t.nodes[u].active = false;
            t.nodes[u].left = t.nodes[u].right = -1;
            refresh_cherry_neighborhood(t, child);
            break;
        }
        replace_child(t, p, u, child);
        t.nodes[child].parent = p;
        t.nodes[u].active = false;
        t.nodes[u].left = t.nodes[u].right = -1;
        refresh_cherry_neighborhood(t, child);
        refresh_local_cherry_status(t, p);
        u = p;
    }
    if (u != -1) refresh_cherry_neighborhood(t, u);
}

void cut_edge_above(DynamicTree& t, int child) {
    if (child < 0 || child >= static_cast<int>(t.nodes.size())) return;
    if (!t.nodes[child].active) return;
    int p = t.nodes[child].parent;
    if (p == -1) return;
    remove_cherry_node(t, p);
    replace_child(t, p, child, -1);
    t.nodes[child].parent = -1;
    refresh_cherry_neighborhood(t, child);
    suppress_up(t, p);
}

bool are_siblings_by_label(const DynamicTree& t, int la, int lb) {
    auto ia = t.label_to_node.find(la);
    auto ib = t.label_to_node.find(lb);
    if (ia == t.label_to_node.end() || ib == t.label_to_node.end()) return false;
    int a = ia->second;
    int b = ib->second;
    if (!is_active_leaf(t, a) || !is_active_leaf(t, b)) return false;
    int pa = t.nodes[a].parent;
    int pb = t.nodes[b].parent;
    if (pa == -1 || pa != pb) return false;
    int l = t.nodes[pa].left, r = t.nodes[pa].right;
    return (l == a && r == b) || (l == b && r == a);
}

bool contract_cherry(DynamicTree& t, int la, int lb, int new_label) {
    auto ia = t.label_to_node.find(la);
    auto ib = t.label_to_node.find(lb);
    if (ia == t.label_to_node.end() || ib == t.label_to_node.end()) return false;
    int a = ia->second;
    int b = ib->second;
    if (!is_active_leaf(t, a) || !is_active_leaf(t, b)) return false;
    int p = t.nodes[a].parent;
    if (p == -1 || t.nodes[b].parent != p) return false;

    int nid = static_cast<int>(t.nodes.size());
    t.nodes.push_back(DynNode{});
    ensure_cherry_storage(t);
    DynNode& nn = t.nodes[nid];
    nn.parent = p;
    nn.left = -1;
    nn.right = -1;
    nn.label = new_label;
    nn.active = true;
    nn.comp_left = a;
    nn.comp_right = b;
    nn.merged_original = t.nodes[a].merged_original;
    nn.merged_original.insert(nn.merged_original.end(), t.nodes[b].merged_original.begin(), t.nodes[b].merged_original.end());
    nn.leafset_hash = t.nodes[a].leafset_hash ^ t.nodes[b].leafset_hash;
    nn.payload_size = t.nodes[a].payload_size + t.nodes[b].payload_size;
    uint64_t aa = t.nodes[a].topo_a, ab = t.nodes[b].topo_a;
    uint64_t ba = t.nodes[a].topo_b, bb = t.nodes[b].topo_b;
    if (aa > ab) std::swap(aa, ab);
    if (ba > bb) std::swap(ba, bb);
    nn.topo_a = mix64(aa ^ (ab + 0x9e3779b97f4a7c15ULL));
    nn.topo_b = mix64(ba ^ (bb + 0x517cc1b727220a95ULL));

    t.label_to_node.erase(la);
    t.label_to_node.erase(lb);
    t.label_to_node[new_label] = nid;
    t.cherry_pos[nid] = -1;
    t.is_cherry[nid] = 0;

    int left = t.nodes[p].left;
    int right = t.nodes[p].right;
    if ((left == a && right == b) || (left == b && right == a)) {
        t.nodes[p].left = nid;
        t.nodes[p].right = -1;
    } else {
        return false;
    }

    remove_cherry_node(t, p);
    remove_cherry_node(t, a);
    remove_cherry_node(t, b);
    t.nodes[a].active = false;
    t.nodes[b].active = false;
    t.nodes[a].parent = -1;
    t.nodes[b].parent = -1;

    refresh_cherry_neighborhood(t, nid);
    suppress_up(t, p);
    return true;
}

std::vector<std::pair<int, int>> cherries_by_label(const DynamicTree& t) {
    std::vector<std::pair<int, int>> out;
    out.reserve(t.cherry_nodes.size());
    for (int u : t.cherry_nodes) {
        if (u < 0 || u >= static_cast<int>(t.nodes.size())) continue;
        if (!t.nodes[u].active) continue;
        int l = t.nodes[u].left;
        int r = t.nodes[u].right;
        if (!is_active_leaf(t, l) || !is_active_leaf(t, r)) continue;
        int a = t.nodes[l].label;
        int b = t.nodes[r].label;
        if (a > b) std::swap(a, b);
        out.push_back({a, b});
    }
    return out;
}

bool contract_all_common_cherries(DynamicTree& t1, DynamicTree& t2, int& next_label) {
    bool any = false;
    while (true) {
        auto cherries = cherries_by_label(t1);
        bool contracted = false;
        for (auto [a, b] : cherries) {
            if (!are_siblings_by_label(t1, a, b) || !are_siblings_by_label(t2, a, b)) continue;
            int nl = next_label++;
            if (contract_cherry(t1, a, b, nl) && contract_cherry(t2, a, b, nl)) {
                contracted = true;
                any = true;
                break; // restart scan; cherry set changed
            }
        }
        if (!contracted) break;
    }
    return any;
}

int root_of(const DynamicTree& t, int u) {
    int x = u;
    while (x != -1 && t.nodes[x].active && t.nodes[x].parent != -1) x = t.nodes[x].parent;
    return x;
}

std::vector<int> path_nodes(const DynamicTree& t, int a, int b) {
    std::unordered_map<int, int> pos;
    std::vector<int> upa;
    int x = a;
    while (x != -1) {
        pos[x] = static_cast<int>(upa.size());
        upa.push_back(x);
        x = t.nodes[x].parent;
    }
    std::vector<int> upb;
    x = b;
    int lca = -1;
    while (x != -1) {
        if (pos.find(x) != pos.end()) {
            lca = x;
            break;
        }
        upb.push_back(x);
        x = t.nodes[x].parent;
    }
    if (lca == -1) return {};

    std::vector<int> res;
    int ia = pos[lca];
    for (int i = 0; i <= ia; ++i) res.push_back(upa[i]);
    for (int i = static_cast<int>(upb.size()) - 1; i >= 0; --i) res.push_back(upb[i]);
    return res;
}

std::vector<int> pendant_children_on_path(const DynamicTree& t, const std::vector<int>& path) {
    std::vector<int> out;
    if (path.size() < 3) return out;
    for (size_t i = 1; i + 1 < path.size(); ++i) {
        int u = path[i];
        int prev = path[i - 1];
        int next = path[i + 1];
        int l = t.nodes[u].left;
        int r = t.nodes[u].right;
        if (l != -1 && t.nodes[l].active && l != prev && l != next) out.push_back(l);
        if (r != -1 && t.nodes[r].active && r != prev && r != next) out.push_back(r);
    }
    return out;
}

int count_common_cherries(const DynamicTree& t1, const DynamicTree& f) {
    auto cherries = cherries_by_label(t1);
    int cnt = 0;
    for (auto [a, b] : cherries) {
        if (are_siblings_by_label(f, a, b)) ++cnt;
    }
    return cnt;
}

int active_leaf_count(const DynamicTree& t) {
    int c = 0;
    for (const auto& n : t.nodes) if (n.active && n.left == -1 && n.right == -1) ++c;
    return c;
}

int count_root_components(const DynamicTree& t) {
    int c = 0;
    for (const auto& n : t.nodes) if (n.active && n.parent == -1) ++c;
    return c;
}

std::vector<int> merge_sorted_components(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> out;
    out.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) out.push_back(a[i++]);
        else if (b[j] < a[i]) out.push_back(b[j++]);
        else {
            out.push_back(a[i]);
            ++i;
            ++j;
        }
    }
    while (i < a.size()) out.push_back(a[i++]);
    while (j < b.size()) out.push_back(b[j++]);
    return out;
}

std::vector<std::vector<int>> greedy_merge_partition(
    std::vector<std::vector<int>> comps,
    const std::vector<uint64_t>& zob,
    const std::vector<TreeData>& trees
) {
    for (auto& comp : comps) {
        std::sort(comp.begin(), comp.end());
        comp.erase(std::unique(comp.begin(), comp.end()), comp.end());
    }

    if (comps.size() > 256) return comps;

    while (true) {
        int best_i = -1;
        int best_j = -1;
        size_t best_union_size = 0;

        for (int i = 0; i < static_cast<int>(comps.size()); ++i) {
            if (comps[i].empty()) continue;
            for (int j = i + 1; j < static_cast<int>(comps.size()); ++j) {
                if (comps[j].empty()) continue;
                size_t max_possible = comps[i].size() + comps[j].size();
                if (max_possible <= best_union_size) continue;
                auto merged = merge_sorted_components(comps[i], comps[j]);
                if (merged.size() <= best_union_size) continue;
                if (!component_valid_both_trees(merged, zob, trees)) continue;
                best_union_size = merged.size();
                best_i = i;
                best_j = j;
            }
        }

        if (best_i == -1) break;
        comps[best_i] = merge_sorted_components(comps[best_i], comps[best_j]);
        comps.erase(comps.begin() + best_j);
    }

    return comps;
}

void collect_component_leaves(const DynamicTree& t, int u, std::vector<int>& out) {
    if (u == -1 || !t.nodes[u].active) return;
    if (t.nodes[u].left == -1 && t.nodes[u].right == -1) {
        const auto& m = t.nodes[u].merged_original;
        out.insert(out.end(), m.begin(), m.end());
        return;
    }
    if (t.nodes[u].left != -1) collect_component_leaves(t, t.nodes[u].left, out);
    if (t.nodes[u].right != -1) collect_component_leaves(t, t.nodes[u].right, out);
}

void collect_active_postorder(const DynamicTree& t, std::vector<int>& order) {
    order.clear();
    if (t.root == -1 || t.root >= static_cast<int>(t.nodes.size()) || !t.nodes[t.root].active) return;

    std::vector<std::pair<int, int>> st;
    st.reserve(t.nodes.size());
    st.push_back({t.root, 0});
    while (!st.empty()) {
        auto [u, phase] = st.back();
        st.pop_back();
        if (u == -1 || !t.nodes[u].active) continue;
        if (phase == 1) {
            order.push_back(u);
            continue;
        }
        st.push_back({u, 1});
        if (t.nodes[u].right != -1) st.push_back({t.nodes[u].right, 0});
        if (t.nodes[u].left != -1) st.push_back({t.nodes[u].left, 0});
    }
}

std::vector<int> compute_active_leaf_masses(const DynamicTree& t) {
    std::vector<int> mass(t.nodes.size(), 0);
    std::vector<int> order;
    collect_active_postorder(t, order);
    for (int u : order) {
        if (is_leaflike(t, u)) {
            mass[u] = t.nodes[u].payload_size;
        } else {
            int total = 0;
            if (t.nodes[u].left != -1 && t.nodes[t.nodes[u].left].active) total += mass[t.nodes[u].left];
            if (t.nodes[u].right != -1 && t.nodes[t.nodes[u].right].active) total += mass[t.nodes[u].right];
            mass[u] = total;
        }
    }
    return mass;
}

void compute_active_subtree_signatures(
    const DynamicTree& t,
    std::vector<ReducedSubtreeSig>& sigs
) {
    sigs.assign(t.nodes.size(), ReducedSubtreeSig{});
    std::vector<int> order;
    collect_active_postorder(t, order);
    for (int u : order) {
        if (is_leaflike(t, u)) {
            sigs[u].leafset_hash = t.nodes[u].leafset_hash;
            sigs[u].leaf_count = t.nodes[u].payload_size;
            sigs[u].topo_a = t.nodes[u].topo_a;
            sigs[u].topo_b = t.nodes[u].topo_b;
        } else {
            int l = t.nodes[u].left;
            int r = t.nodes[u].right;
            if (l == -1 || r == -1 || !t.nodes[l].active || !t.nodes[r].active) continue;
            sigs[u].leaf_count = sigs[l].leaf_count + sigs[r].leaf_count;
            sigs[u].leafset_hash = sigs[l].leafset_hash ^ sigs[r].leafset_hash;
            uint64_t aa = sigs[l].topo_a, ab = sigs[r].topo_a;
            uint64_t ba = sigs[l].topo_b, bb = sigs[r].topo_b;
            if (aa > ab) std::swap(aa, ab);
            if (ba > bb) std::swap(ba, bb);
            sigs[u].topo_a = mix64(aa ^ (ab + 0x9e3779b97f4a7c15ULL));
            sigs[u].topo_b = mix64(ba ^ (bb + 0x517cc1b727220a95ULL));
        }
    }
}

void mark_active_subtree(const DynamicTree& t, int u, std::vector<char>& blocked) {
    if (u == -1 || !t.nodes[u].active || blocked[u]) return;
    std::vector<int> st{u};
    while (!st.empty()) {
        int x = st.back();
        st.pop_back();
        if (x == -1 || !t.nodes[x].active || blocked[x]) continue;
        blocked[x] = 1;
        if (t.nodes[x].left != -1) st.push_back(t.nodes[x].left);
        if (t.nodes[x].right != -1) st.push_back(t.nodes[x].right);
    }
}

void deactivate_active_subtree(DynamicTree& t, int u) {
    if (u == -1 || !t.nodes[u].active) return;
    std::vector<int> st{u};
    while (!st.empty()) {
        int x = st.back();
        st.pop_back();
        if (x == -1 || !t.nodes[x].active) continue;
        remove_cherry_node(t, x);
        if (is_leaflike(t, x)) t.label_to_node.erase(t.nodes[x].label);
        int l = t.nodes[x].left;
        int r = t.nodes[x].right;
        t.nodes[x].active = false;
        t.nodes[x].parent = -1;
        if (l != -1) st.push_back(l);
        if (r != -1) st.push_back(r);
    }
}

bool contract_active_subtree_root(
    DynamicTree& t,
    int u,
    int new_label,
    const ReducedSubtreeSig& sig
) {
    if (u == -1 || !t.nodes[u].active) return false;

    std::vector<int> leaves;
    leaves.reserve(sig.leaf_count);
    collect_component_leaves(t, u, leaves);
    std::sort(leaves.begin(), leaves.end());
    leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
    if (leaves.empty()) return false;

    int p = t.nodes[u].parent;
    int nid = static_cast<int>(t.nodes.size());
    t.nodes.push_back(DynNode{});
    ensure_cherry_storage(t);

    DynNode& nn = t.nodes[nid];
    nn.parent = p;
    nn.left = -1;
    nn.right = -1;
    nn.label = new_label;
    nn.active = true;
    nn.comp_left = u;
    nn.comp_right = -1;
    nn.merged_original = std::move(leaves);
    nn.leafset_hash = sig.leafset_hash;
    nn.payload_size = sig.leaf_count;
    nn.topo_a = sig.topo_a;
    nn.topo_b = sig.topo_b;

    if (p == -1) {
        t.root = nid;
    } else {
        replace_child(t, p, u, nid);
    }

    deactivate_active_subtree(t, u);
    t.label_to_node[new_label] = nid;
    t.cherry_pos[nid] = -1;
    t.is_cherry[nid] = 0;
    refresh_cherry_neighborhood(t, nid);
    if (p != -1) refresh_cherry_neighborhood(t, p);
    return true;
}

bool contract_all_common_pendant_subtrees(
    DynamicTree& t1,
    DynamicTree& t2,
    int& next_label,
    const std::vector<uint64_t>& zob,
    const std::vector<TreeData>& trees
) {
    std::vector<ReducedSubtreeSig> sig1, sig2;
    compute_active_subtree_signatures(t1, sig1);
    compute_active_subtree_signatures(t2, sig2);

    std::unordered_map<ReducedSubtreeKey, int, ReducedSubtreeKeyHasher> map1;
    std::unordered_map<ReducedSubtreeKey, int, ReducedSubtreeKeyHasher> map2;
    for (int u = 0; u < static_cast<int>(t1.nodes.size()); ++u) {
        if (!t1.nodes[u].active || is_leaflike(t1, u) || sig1[u].leaf_count < 2) continue;
        map1[ReducedSubtreeKey{sig1[u].leafset_hash, sig1[u].leaf_count, sig1[u].topo_a, sig1[u].topo_b}] = u;
    }
    for (int u = 0; u < static_cast<int>(t2.nodes.size()); ++u) {
        if (!t2.nodes[u].active || is_leaflike(t2, u) || sig2[u].leaf_count < 2) continue;
        map2[ReducedSubtreeKey{sig2[u].leafset_hash, sig2[u].leaf_count, sig2[u].topo_a, sig2[u].topo_b}] = u;
    }

    struct Match { int u1; int u2; ReducedSubtreeSig sig; };
    std::vector<Match> matches;
    matches.reserve(std::min(map1.size(), map2.size()));
    for (const auto& [key, u1] : map1) {
        auto it = map2.find(key);
        if (it == map2.end()) continue;
        matches.push_back(Match{u1, it->second, sig1[u1]});
    }
    std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b) {
        return a.sig.leaf_count > b.sig.leaf_count;
    });

    std::vector<char> blocked1(t1.nodes.size(), 0), blocked2(t2.nodes.size(), 0);
    bool any = false;
    for (const auto& m : matches) {
        if (blocked1[m.u1] || blocked2[m.u2]) continue;
        std::vector<int> leaves;
        leaves.reserve(m.sig.leaf_count);
        collect_component_leaves(t1, m.u1, leaves);
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        if (static_cast<int>(leaves.size()) != m.sig.leaf_count) continue;
        if (!component_valid_both_trees(leaves, zob, trees)) continue;

        int nl = next_label++;
        mark_active_subtree(t1, m.u1, blocked1);
        mark_active_subtree(t2, m.u2, blocked2);
        if (contract_active_subtree_root(t1, m.u1, nl, m.sig) &&
            contract_active_subtree_root(t2, m.u2, nl, m.sig)) {
            any = true;
        }
    }
    return any;
}

bool exhaust_common_reductions(
    DynamicTree& t1,
    DynamicTree& t2,
    int& next_label,
    const std::vector<uint64_t>& zob,
    const std::vector<TreeData>& trees
) {
    bool any = false;
    while (true) {
        bool changed = false;
        changed |= contract_all_common_cherries(t1, t2, next_label);
        changed |= contract_all_common_pendant_subtrees(t1, t2, next_label, zob, trees);
        any |= changed;
        if (!changed) break;
    }
    return any;
}

struct CherryCandidate {
    int a = -1;
    int b = -1;
    double score = -std::numeric_limits<double>::infinity();
    bool common = false;
    bool same_component = false;
    int na = -1;
    int nb = -1;
    int ra = -1;
    int rb = -1;
    int distance = 0;
    int pendant_count = 0;
    int conflict_mass = 0;
    int component_size = 0;
    int immediate_gain = 0;
    std::vector<int> path;
    std::vector<int> pendants;
};

CherryCandidate build_cherry_candidate(
    const DynamicTree& t1,
    const DynamicTree& f,
    const std::vector<int>& f_mass,
    int a,
    int b
) {
    CherryCandidate cand;
    cand.a = a;
    cand.b = b;
    cand.common = are_siblings_by_label(f, a, b);
    cand.same_component = false;
    cand.immediate_gain = cand.common ? 2 : 0;

    auto ia = f.label_to_node.find(a);
    auto ib = f.label_to_node.find(b);
    if (ia == f.label_to_node.end() || ib == f.label_to_node.end()) return cand;

    cand.na = ia->second;
    cand.nb = ib->second;
    if (!is_active_leaf(f, cand.na) || !is_active_leaf(f, cand.nb)) return cand;

    cand.ra = root_of(f, cand.na);
    cand.rb = root_of(f, cand.nb);
    if (cand.ra == -1 || cand.rb == -1) return cand;

    cand.same_component = (cand.ra == cand.rb);
    if (cand.same_component) {
        cand.path = path_nodes(f, cand.na, cand.nb);
        cand.distance = cand.path.empty() ? 0 : static_cast<int>(cand.path.size()) - 1;
        cand.pendants = pendant_children_on_path(f, cand.path);
        cand.pendant_count = static_cast<int>(cand.pendants.size());
        for (int u : cand.pendants) {
            if (u >= 0 && u < static_cast<int>(f_mass.size())) cand.conflict_mass += f_mass[u];
        }
        if (cand.ra >= 0 && cand.ra < static_cast<int>(f_mass.size())) cand.component_size = f_mass[cand.ra];
        if (cand.pendant_count == 1) cand.immediate_gain += 1;
    } else {
        int ma = (cand.ra >= 0 && cand.ra < static_cast<int>(f_mass.size())) ? f_mass[cand.ra] : 0;
        int mb = (cand.rb >= 0 && cand.rb < static_cast<int>(f_mass.size())) ? f_mass[cand.rb] : 0;
        cand.component_size = std::min(ma, mb);
        cand.conflict_mass = cand.component_size;
        cand.immediate_gain += 1;
    }

    cand.score =
        (cand.common ? 12.0 : 0.0) +
        (cand.same_component ? 0.0 : 4.0) -
        1.5 * static_cast<double>(cand.distance) -
        2.0 * static_cast<double>(cand.pendant_count) -
        0.02 * static_cast<double>(cand.conflict_mass) +
        3.0 * static_cast<double>(cand.immediate_gain) -
        0.01 * static_cast<double>(cand.component_size);

    (void)t1;
    return cand;
}

CherryCandidate select_scored_cherry(
    const DynamicTree& t1,
    const DynamicTree& f,
    uint64_t& rng,
    double alpha
) {
    auto cherries = cherries_by_label(t1);
    if (cherries.empty()) return {};

    auto f_mass = compute_active_leaf_masses(f);
    std::vector<CherryCandidate> candidates;
    candidates.reserve(cherries.size());
    double best = -std::numeric_limits<double>::infinity();
    double worst = std::numeric_limits<double>::infinity();
    for (auto [a, b] : cherries) {
        auto cand = build_cherry_candidate(t1, f, f_mass, a, b);
        if (cand.na == -1 || cand.nb == -1) continue;
        best = std::max(best, cand.score);
        worst = std::min(worst, cand.score);
        candidates.push_back(std::move(cand));
    }
    if (candidates.empty()) return {};

    double cutoff = best - alpha * (best - worst);
    std::vector<int> rcl;
    rcl.reserve(candidates.size());
    for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
        if (candidates[i].score >= cutoff) rcl.push_back(i);
    }
    if (rcl.empty()) rcl.push_back(0);

    rng = mix64(rng + 0x6a09e667f3bcc909ULL);
    return candidates[rcl[static_cast<size_t>(rng % rcl.size())]];
}

void apply_cut_plan(DynamicTree& f, const std::vector<int>& cuts) {
    for (int u : cuts) cut_edge_above(f, u);
}

double evaluate_reduced_state(
    const DynamicTree& t1,
    const DynamicTree& f
) {
    auto cherries = cherries_by_label(t1);
    auto f_mass = compute_active_leaf_masses(f);
    int comp_count = count_root_components(f);
    int sampled_pendants = 0;
    int sampled_conflict_mass = 0;

    int sample_limit = std::min<int>(6, cherries.size());
    for (int i = 0; i < sample_limit; ++i) {
        auto [a, b] = cherries[i];
        auto ia = f.label_to_node.find(a);
        auto ib = f.label_to_node.find(b);
        if (ia == f.label_to_node.end() || ib == f.label_to_node.end()) continue;
        int na = ia->second;
        int nb = ib->second;
        if (!is_active_leaf(f, na) || !is_active_leaf(f, nb)) continue;

        int ra = root_of(f, na);
        int rb = root_of(f, nb);
        if (ra == -1 || rb == -1) continue;

        if (ra != rb) {
            sampled_conflict_mass += std::min(f_mass[ra], f_mass[rb]);
            continue;
        }

        auto path = path_nodes(f, na, nb);
        auto pendants = pendant_children_on_path(f, path);
        sampled_pendants += static_cast<int>(pendants.size());
        for (int u : pendants) sampled_conflict_mass += f_mass[u];
    }

    return
        -20.0 * static_cast<double>(comp_count) +
         0.75 * static_cast<double>(cherries.size()) -
         1.75 * static_cast<double>(sampled_pendants) -
         0.02 * static_cast<double>(sampled_conflict_mass);
}

std::vector<int> choose_cut_plan(
    const DynamicTree& t1,
    const DynamicTree& f,
    const CherryCandidate& cand,
    int next_label,
    uint64_t& rng
) {
    std::vector<std::vector<int>> plans;
    if (!cand.same_component) {
        plans.push_back({cand.na});
        plans.push_back({cand.nb});
    } else {
        auto f_mass = compute_active_leaf_masses(f);
        std::vector<int> sorted = cand.pendants;
        std::sort(sorted.begin(), sorted.end(), [&](int x, int y) {
            return f_mass[x] < f_mass[y];
        });

        // On harder same-component conflicts, scoring every pendant cut is
        // expensive and tends to hurt completed-run count more than it helps.
        size_t individual_limit = std::min<size_t>(4, sorted.size());
        for (size_t i = 0; i < individual_limit; ++i) {
            plans.push_back({sorted[i]});
        }
        plans.push_back({cand.na});
        plans.push_back({cand.nb});
        if (cand.pendants.size() > 1) {
            std::vector<int> bundle;
            for (size_t i = 0; i < (sorted.size() + 1) / 2; ++i) bundle.push_back(sorted[i]);
            plans.push_back(std::move(bundle));
            if (cand.pendants.size() <= 6) {
                std::vector<int> classic{cand.na, cand.nb};
                classic.insert(classic.end(), cand.pendants.begin(), cand.pendants.end());
                plans.push_back(std::move(classic));
            }
        }
    }
    if (plans.empty()) return {};

    double best = -std::numeric_limits<double>::infinity();
    std::vector<int> best_indices;
    for (int i = 0; i < static_cast<int>(plans.size()); ++i) {
        DynamicTree tt = t1;
        DynamicTree ff = f;
        int nl = next_label;
        apply_cut_plan(ff, plans[i]);
        contract_all_common_cherries(tt, ff, nl);
        double val = evaluate_reduced_state(tt, ff);
        if (val > best + 1e-9) {
            best = val;
            best_indices = {i};
        } else if (std::abs(val - best) <= 1e-9) {
            best_indices.push_back(i);
        }
    }

    rng = mix64(rng + 0x94d049bb133111ebULL);
    return plans[best_indices[static_cast<size_t>(rng % best_indices.size())]];
}

bool use_expensive_same_component_plan(
    int total_leaves,
    const std::vector<int>& path,
    const std::vector<int>& pendants
) {
    // The rollout-style chooser pays off on small/medium instances,
    // but on the largest instances it can prevent any completed run.
    if (total_leaves > 4000) return false;
    if (path.size() > 64) return false;
    if (pendants.size() > 10) return false;
    return true;
}

struct ThreeApproxResult {
    std::vector<std::vector<int>> comps;
    bool complete = false;
};

ThreeApproxResult run_three_approx(
    DynamicTree t1,
    DynamicTree f,
    Clock::time_point deadline,
    int n,
    int incumbent_components,
    uint64_t seed,
    double alpha
) {
    (void)alpha;
    int next_label = n + 1;
    uint64_t rng = seed ^ 0x9e3779b97f4a7c15ULL;
    bool timed_out_or_terminated = false;

    // Common-subtree precontraction (light): exhaust common cherries before branching.
    contract_all_common_cherries(t1, f, next_label);

    while (!g_terminate && Clock::now() < deadline && active_leaf_count(t1) > 2) {
        // --- Step 1: remove singleton leaves in F that are roots ---
        // For each leaf in F that is a root, cut its corresponding edge in T1.
        for (int u = 0; u < static_cast<int>(f.nodes.size()); ++u) {
            if (!f.nodes[u].active) continue;
            if (f.nodes[u].left == -1 && f.nodes[u].right == -1 && f.nodes[u].parent == -1) {
                int lbl = f.nodes[u].label;
                auto it = t1.label_to_node.find(lbl);
                if (it != t1.label_to_node.end()) {
                    int v = it->second;
                    cut_edge_above(t1, v);
                }
            }
        }

        // Cuts never decrease the number of forest components, so once the
        // current forest cannot beat the incumbent we can abandon this restart.
        if (incumbent_components < std::numeric_limits<int>::max() &&
            count_root_components(f) >= incumbent_components) {
            break;
        }

        auto cherries = cherries_by_label(t1);
        if (cherries.empty()) break;

        bool did_common_contract = false;
        rng = mix64(rng + 0x9e3779b97f4a7c15ULL);
        size_t start_idx = cherries.empty() ? 0 : static_cast<size_t>(rng % cherries.size());
        for (size_t step = 0; step < cherries.size(); ++step) {
            auto [a, b] = cherries[(start_idx + step) % cherries.size()];
            if (are_siblings_by_label(f, a, b)) {
                int nl = next_label++;
                if (contract_cherry(t1, a, b, nl) && contract_cherry(f, a, b, nl)) {
                    did_common_contract = true;
                    break;
                }
            }
        }
        if (did_common_contract) continue;

        rng = mix64(rng + 0x517cc1b727220a95ULL);
        auto [a, b] = cherries[static_cast<size_t>(rng % cherries.size())];
        auto ia = f.label_to_node.find(a);
        auto ib = f.label_to_node.find(b);
        if (ia == f.label_to_node.end() || ib == f.label_to_node.end()) break;
        int na = ia->second;
        int nb = ib->second;
        if (!is_active_leaf(f, na) || !is_active_leaf(f, nb)) continue;

        int ra = root_of(f, na);
        int rb = root_of(f, nb);
        if (ra == -1 || rb == -1) break;

        if (ra != rb) {
            DynamicTree fa = f;
            DynamicTree fb = f;
            cut_edge_above(fa, na);
            cut_edge_above(fb, nb);
            int ma = count_common_cherries(t1, fa);
            int mb = count_common_cherries(t1, fb);
            if (ma > mb) {
                f = std::move(fa);
            } else if (mb > ma) {
                f = std::move(fb);
            } else {
                rng = mix64(rng + 0x94d049bb133111ebULL);
                if ((rng & 1ULL) == 0ULL) f = std::move(fa);
                else f = std::move(fb);
            }
            continue;
        }

        auto path = path_nodes(f, na, nb);
        auto pendants = pendant_children_on_path(f, path);
        if (pendants.size() == 1) {
            cut_edge_above(f, pendants[0]);
        } else {
            if (use_expensive_same_component_plan(n, path, pendants)) {
                auto f_mass = compute_active_leaf_masses(f);
                auto cand = build_cherry_candidate(t1, f, f_mass, a, b);
                auto plan = choose_cut_plan(t1, f, cand, next_label, rng);
                if (!plan.empty()) {
                    apply_cut_plan(f, plan);
                    continue;
                }
            }
            {
                cut_edge_above(f, na);
                cut_edge_above(f, nb);
                for (int c : pendants) cut_edge_above(f, c);
            }
        }
    }
    if (g_terminate || Clock::now() >= deadline) {
        timed_out_or_terminated = true;
    }

    ThreeApproxResult res;
    for (int u = 0; u < static_cast<int>(f.nodes.size()); ++u) {
        if (!f.nodes[u].active || f.nodes[u].parent != -1) continue;
        std::vector<int> leaves;
        leaves.reserve(16);
        collect_component_leaves(f, u, leaves);
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        if (!leaves.empty()) res.comps.push_back(std::move(leaves));
    }
    // Only suppress publication for runs interrupted by timeout/signal.
    // Natural algorithmic termination paths remain publishable.
    res.complete = !timed_out_or_terminated;
    return res;
}

std::vector<std::string> solve(const PaceInstance& inst) {
    int n = inst.leaf_count;
    if (n <= 0) return singleton_forest(1);
    if (inst.tree_count != 2) return singleton_forest(n);

    std::vector<uint64_t> zob(n + 1, 0);
    for (int x = 1; x <= n; ++x) zob[x] = mix64(static_cast<uint64_t>(x) * 0x9e3779b97f4a7c15ULL + 0xabcULL);

    std::vector<SimpleTree> parsed;
    std::vector<TreeData> trees;
    parsed.reserve(2);
    trees.reserve(2);

    for (const auto& line : inst.newick_lines) {
        if (!parse_with_genesis(line)) return singleton_forest(n);
        SimpleTree st;
        if (!NewickParser(line).parse(st)) return singleton_forest(n);
        TreeData td;
        if (!build_tree_data(st, n, zob, td)) return singleton_forest(n);
        parsed.push_back(std::move(st));
        trees.push_back(std::move(td));
    }

    const auto start = Clock::now();
    double timeout_sec = 295.0;
    if (const char* env = std::getenv("STRIDE_TIMEOUT")) {
        char* end = nullptr;
        double v = std::strtod(env, &end);
        if (end != env && v > 0.0) {
            timeout_sec = (v > 1000.0 ? (v / 1000.0) : v);
        }
    }
    auto soft_deadline = start + std::chrono::milliseconds(
        std::max<long long>(1, static_cast<long long>(timeout_sec * 1000.0) - 200)
    );

    auto best_out = singleton_forest(n);
    publish_best_solution(best_out);

    DynamicTree dt1_base = build_dynamic_tree(parsed[0]);
    DynamicTree dt2_base = build_dynamic_tree(parsed[1]);

    uint64_t base_seed = instance_seed(inst);
    uint64_t iter = 0;
    while (!g_terminate && Clock::now() < soft_deadline) {
        uint64_t seed = mix64(base_seed ^ iter);
        auto res = run_three_approx(
            dt1_base,
            dt2_base,
            soft_deadline,
            n,
            static_cast<int>(best_out.size()),
            seed,
            0.15
        );
        if (!res.complete || res.comps.empty()) {
            ++iter;
            continue;
        }
        auto out = forest_from_partition(res.comps, n, trees[0]);
        if (!out.empty() && static_cast<int>(out.size()) < static_cast<int>(best_out.size())) {
            best_out = out;
            publish_best_solution(best_out);
        }
        ++iter;
    }
    return best_out;
}

} // namespace

int main() {
    std::signal(SIGTERM, sig_handler);
    std::signal(SIGINT, sig_handler);

    PaceInstance inst;
    if (!read_instance(inst)) {
        set_best_output_buffer(normalize_output(singleton_forest(std::max(1, g_last_leaf_count))));
        write_best_once();
        return 0;
    }

    publish_best_solution(singleton_forest(std::max(1, inst.leaf_count)));

    std::vector<std::string> out;
    try {
        out = solve(inst);
    } catch (...) {
        out = singleton_forest(std::max(1, inst.leaf_count));
    }

    if (out.empty()) out = singleton_forest(std::max(1, inst.leaf_count));
    publish_best_solution(out);
    write_best_once();
    return 0;
}
