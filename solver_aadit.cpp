#include "genesis/genesis.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
};

struct DynamicTree {
    int root = -1;
    std::vector<DynNode> nodes;
    std::unordered_map<int, int> label_to_node; // active pseudo-label -> node
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
            dt.label_to_node[dn.label] = static_cast<int>(u);
        } else {
            dn.label = 0;
            dn.left = st.children[u][0];
            dn.right = st.children[u][1];
        }
    }
    return dt;
}

bool is_active_leaf(const DynamicTree& t, int u) {
    return u >= 0 && u < static_cast<int>(t.nodes.size()) && t.nodes[u].active && t.nodes[u].left == -1 && t.nodes[u].right == -1;
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
        int child = (t.nodes[u].left != -1 && t.nodes[t.nodes[u].left].active) ? t.nodes[u].left : t.nodes[u].right;
        int p = t.nodes[u].parent;
        if (p == -1) {
            t.nodes[child].parent = -1;
            t.root = child;
            t.nodes[u].active = false;
            t.nodes[u].left = t.nodes[u].right = -1;
            break;
        }
        replace_child(t, p, u, child);
        t.nodes[child].parent = p;
        t.nodes[u].active = false;
        t.nodes[u].left = t.nodes[u].right = -1;
        u = p;
    }
}

void cut_edge_above(DynamicTree& t, int child) {
    if (child < 0 || child >= static_cast<int>(t.nodes.size())) return;
    if (!t.nodes[child].active) return;
    int p = t.nodes[child].parent;
    if (p == -1) return;
    replace_child(t, p, child, -1);
    t.nodes[child].parent = -1;
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

    t.label_to_node.erase(la);
    t.label_to_node.erase(lb);
    t.label_to_node[new_label] = nid;

    int left = t.nodes[p].left;
    int right = t.nodes[p].right;
    if ((left == a && right == b) || (left == b && right == a)) {
        t.nodes[p].left = nid;
        t.nodes[p].right = -1;
    } else {
        return false;
    }

    t.nodes[a].active = false;
    t.nodes[b].active = false;
    t.nodes[a].parent = -1;
    t.nodes[b].parent = -1;

    suppress_up(t, p);
    return true;
}

std::vector<std::pair<int, int>> cherries_by_label(const DynamicTree& t) {
    std::vector<std::pair<int, int>> out;
    for (int u = 0; u < static_cast<int>(t.nodes.size()); ++u) {
        if (!t.nodes[u].active) continue;
        int l = t.nodes[u].left;
        int r = t.nodes[u].right;
        if (l == -1 || r == -1) continue;
        if (is_active_leaf(t, l) && is_active_leaf(t, r)) {
            int a = t.nodes[l].label;
            int b = t.nodes[r].label;
            if (a > b) std::swap(a, b);
            out.push_back({a, b});
        }
    }
    return out;
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

int active_leaf_count(const DynamicTree& t) {
    int c = 0;
    for (const auto& n : t.nodes) if (n.active && n.left == -1 && n.right == -1) ++c;
    return c;
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

struct ThreeApproxResult {
    std::vector<std::vector<int>> comps;
    bool complete = false;
};

ThreeApproxResult run_three_approx(
    DynamicTree t1,
    DynamicTree f,
    Clock::time_point deadline,
    int n,
    uint64_t seed
) {
    int next_label = n + 1;
    uint64_t rng = seed ^ 0x9e3779b97f4a7c15ULL;
    bool timed_out_or_terminated = false;

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
            cut_edge_above(f, na);
            cut_edge_above(f, nb);
            continue;
        }

        auto path = path_nodes(f, na, nb);
        auto pendants = pendant_children_on_path(f, path);
        if (pendants.size() == 1) {
            cut_edge_above(f, pendants[0]);
        } else {
            cut_edge_above(f, na);
            cut_edge_above(f, nb);
            for (int c : pendants) cut_edge_above(f, c);
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
        auto res = run_three_approx(dt1_base, dt2_base, soft_deadline, n, seed);
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
