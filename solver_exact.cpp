#include "genesis/genesis.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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
    bool has_aux_root = false;
    std::vector<std::string> newick_lines;
    struct TreeDecomposition {
        int width = -1;
        std::vector<std::vector<int>> bags;
        std::vector<std::pair<int, int>> edges;
        bool valid = false;
    } treedecomp;
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

struct PayloadNode {
    int left = -1;
    int right = -1;
    int leaf_label = -1;
    int size = 0;
};

struct DynNode {
    int parent = -1;
    int left = -1;
    int right = -1;
    int label = 0; // current active pseudo-leaf label, 0 for internal
    bool active = true;

    // Expansion payload for active pseudo leaves.
    int payload_id = -1;

    // Rooted subtree signature for active pseudo leaves.
    uint64_t leafset_hash = 0;
    int payload_size = 0;
    uint64_t topo_a = 0;
    uint64_t topo_b = 0;
};

struct DynamicTree {
    int root = -1;
    std::vector<DynNode> nodes;
    std::vector<PayloadNode> payloads;
    std::vector<int> label_to_node; // active pseudo-label -> node, -1 if inactive
    std::vector<char> is_cherry;
    std::vector<int> cherry_pos;
    std::vector<int> cherry_nodes;
};

struct UndoLog {
    enum class Kind {
        Root,
        NodeParent,
        NodeLeft,
        NodeRight,
        NodeActive,
        LabelSlot,
        IsCherrySlot,
        CherryPosSlot,
        CherryNodeSlot,
        NodesSize,
        PayloadsSize,
        LabelSize,
        IsCherrySize,
        CherryPosSize,
        CherryNodesSize
    };

    struct Entry {
        Kind kind;
        DynamicTree* tree;
        int index;
        int old_value;
        int aux;
    };

    std::vector<Entry> entries;

    size_t mark() const { return entries.size(); }

    void undo_to(size_t mark) {
        while (entries.size() > mark) {
            Entry e = entries.back();
            entries.pop_back();
            switch (e.kind) {
                case Kind::Root:
                    e.tree->root = e.old_value;
                    break;
                case Kind::NodeParent:
                    e.tree->nodes[e.index].parent = e.old_value;
                    break;
                case Kind::NodeLeft:
                    e.tree->nodes[e.index].left = e.old_value;
                    break;
                case Kind::NodeRight:
                    e.tree->nodes[e.index].right = e.old_value;
                    break;
                case Kind::NodeActive:
                    e.tree->nodes[e.index].active = static_cast<bool>(e.old_value);
                    break;
                case Kind::LabelSlot:
                    e.tree->label_to_node[e.index] = e.old_value;
                    break;
                case Kind::IsCherrySlot:
                    e.tree->is_cherry[e.index] = static_cast<char>(e.old_value);
                    break;
                case Kind::CherryPosSlot:
                    e.tree->cherry_pos[e.index] = e.old_value;
                    break;
                case Kind::CherryNodeSlot:
                    e.tree->cherry_nodes[e.index] = e.old_value;
                    break;
                case Kind::NodesSize:
                    e.tree->nodes.resize(static_cast<size_t>(e.old_value));
                    break;
                case Kind::PayloadsSize:
                    e.tree->payloads.resize(static_cast<size_t>(e.old_value));
                    break;
                case Kind::LabelSize:
                    e.tree->label_to_node.resize(static_cast<size_t>(e.old_value));
                    break;
                case Kind::IsCherrySize:
                    e.tree->is_cherry.resize(static_cast<size_t>(e.old_value));
                    break;
                case Kind::CherryPosSize:
                    e.tree->cherry_pos.resize(static_cast<size_t>(e.old_value));
                    break;
                case Kind::CherryNodesSize:
                    e.tree->cherry_nodes.resize(static_cast<size_t>(e.old_value));
                    if (e.old_value > 0) {
                        e.tree->cherry_nodes[static_cast<size_t>(e.old_value - 1)] = e.aux;
                    }
                    break;
            }
        }
    }
};

int find_node_of_label(const DynamicTree& t, int label) {
    if (label < 0 || label >= static_cast<int>(t.label_to_node.size())) return -1;
    return t.label_to_node[label];
}

void collect_active_postorder(const DynamicTree& t, std::vector<int>& order);
void push_undo(UndoLog* log, UndoLog::Kind kind, DynamicTree* tree, int index, int old_value, int aux = 0);

int push_payload(DynamicTree& t, PayloadNode payload, UndoLog* log = nullptr) {
    push_undo(log, UndoLog::Kind::PayloadsSize, &t, -1, static_cast<int>(t.payloads.size()));
    t.payloads.push_back(std::move(payload));
    return static_cast<int>(t.payloads.size()) - 1;
}

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

class IntArrayParser {
public:
    explicit IntArrayParser(std::string s) : s_(std::move(s)) {}

    bool parse_treedecomp(PaceInstance::TreeDecomposition& td) {
        try {
            skip_ws();
            expect('[');
            td.width = parse_int();
            expect(',');
            td.bags = parse_nested_int_lists();
            expect(',');
            td.edges.clear();
            auto raw_edges = parse_nested_int_lists();
            for (const auto& edge : raw_edges) {
                if (edge.size() != 2) return false;
                td.edges.push_back({edge[0], edge[1]});
            }
            expect(']');
            skip_ws();
            td.valid = (pos_ == s_.size());
            return td.valid;
        } catch (...) {
            return false;
        }
    }

private:
    std::string s_;
    size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
    }

    void expect(char c) {
        skip_ws();
        if (pos_ >= s_.size() || s_[pos_] != c) throw std::runtime_error("unexpected token");
        ++pos_;
    }

    int parse_int() {
        skip_ws();
        size_t start = pos_;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
        while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) ++pos_;
        if (start == pos_) throw std::runtime_error("expected int");
        return std::stoi(s_.substr(start, pos_ - start));
    }

    std::vector<int> parse_int_list() {
        std::vector<int> out;
        expect('[');
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            return out;
        }
        while (true) {
            out.push_back(parse_int());
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            break;
        }
        expect(']');
        return out;
    }

    std::vector<std::vector<int>> parse_nested_int_lists() {
        std::vector<std::vector<int>> out;
        expect('[');
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            return out;
        }
        while (true) {
            out.push_back(parse_int_list());
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            break;
        }
        expect(']');
        return out;
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
            } else if (line.rfind("#x treedecomp", 0) == 0) {
                size_t pos = line.find('[');
                if (pos != std::string::npos) {
                    PaceInstance::TreeDecomposition td;
                    if (IntArrayParser(line.substr(pos)).parse_treedecomp(td)) {
                        inst.treedecomp = std::move(td);
                    }
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

struct ParsedInstance {
    int tree_count = 0;
    int leaf_count = 0;
    bool has_aux_root = false;
    std::vector<uint64_t> zob;
    std::vector<SimpleTree> parsed;
    std::vector<TreeData> trees;
    std::vector<std::vector<int>> td_candidates;
};

struct SolveComponentsResult {
    bool complete = false;
    std::vector<std::vector<int>> components;
};

void collect_subtree_leaves(const TreeData& t, int u, std::vector<int>& out) {
    if (u == -1) return;
    if (t.is_leaf[u]) {
        out.push_back(t.leaf_label[u]);
        return;
    }
    collect_subtree_leaves(t, t.child0[u], out);
    collect_subtree_leaves(t, t.child1[u], out);
}

uint64_t hash_of_leaves(const std::vector<int>& leaves, const std::vector<uint64_t>& zob) {
    uint64_t h = 0;
    for (int x : leaves) h ^= zob[x];
    return h;
}

bool is_common_cluster(
    const std::vector<int>& leaves,
    const std::vector<uint64_t>& zob,
    const std::vector<TreeData>& trees
) {
    if (leaves.empty()) return false;
    uint64_t h = hash_of_leaves(leaves, zob);
    int sz = static_cast<int>(leaves.size());
    for (const auto& t : trees) {
        int r = lca_of_leaves(t, leaves);
        if (r == -1) return false;
        if (t.sub_count[r] != sz || t.sub_hash[r] != h) return false;
    }
    return true;
}

std::string restricted_newick_dfs_remap(
    int u,
    const TreeData& t,
    const std::vector<char>& in_subset,
    const std::vector<int>& remap
) {
    if (t.is_leaf[u]) {
        int x = t.leaf_label[u];
        return in_subset[x] ? std::to_string(remap[x]) : std::string();
    }
    std::string a = restricted_newick_dfs_remap(t.child0[u], t, in_subset, remap);
    std::string b = restricted_newick_dfs_remap(t.child1[u], t, in_subset, remap);
    if (a.empty() && b.empty()) return {};
    if (a.empty()) return b;
    if (b.empty()) return a;
    return "(" + a + "," + b + ")";
}

std::string compress_cluster_newick_dfs(
    int u,
    const TreeData& t,
    uint64_t cluster_hash,
    int cluster_size,
    const std::vector<char>& in_cluster,
    const std::vector<int>& outside_remap,
    int meta_label
) {
    if (t.sub_count[u] == cluster_size && t.sub_hash[u] == cluster_hash) {
        return std::to_string(meta_label);
    }
    if (t.is_leaf[u]) {
        int x = t.leaf_label[u];
        if (in_cluster[x]) return {};
        return std::to_string(outside_remap[x]);
    }
    std::string a = compress_cluster_newick_dfs(
        t.child0[u], t, cluster_hash, cluster_size, in_cluster, outside_remap, meta_label
    );
    std::string b = compress_cluster_newick_dfs(
        t.child1[u], t, cluster_hash, cluster_size, in_cluster, outside_remap, meta_label
    );
    if (a.empty() && b.empty()) return {};
    if (a.empty()) return b;
    if (b.empty()) return a;
    return "(" + a + "," + b + ")";
}

std::vector<std::vector<int>> extract_treedecomp_leaf_candidates(
    const PaceInstance::TreeDecomposition& td,
    int n
) {
    std::vector<std::vector<int>> out;
    if (!td.valid || td.bags.empty()) return out;

    std::vector<std::vector<int>> graph(td.bags.size());
    for (const auto& [a1, b1] : td.edges) {
        int a = a1 - 1;
        int b = b1 - 1;
        if (a < 0 || b < 0 || a >= static_cast<int>(td.bags.size()) || b >= static_cast<int>(td.bags.size())) continue;
        graph[static_cast<size_t>(a)].push_back(b);
        graph[static_cast<size_t>(b)].push_back(a);
    }

    std::vector<std::vector<int>> bag_leafsets(td.bags.size());
    for (size_t i = 0; i < td.bags.size(); ++i) {
        for (int x : td.bags[i]) {
            if (1 <= x && x <= n) bag_leafsets[i].push_back(x);
        }
        std::sort(bag_leafsets[i].begin(), bag_leafsets[i].end());
        bag_leafsets[i].erase(std::unique(bag_leafsets[i].begin(), bag_leafsets[i].end()), bag_leafsets[i].end());
    }

    for (const auto& [a1, b1] : td.edges) {
        int a = a1 - 1;
        int b = b1 - 1;
        if (a < 0 || b < 0 || a >= static_cast<int>(graph.size()) || b >= static_cast<int>(graph.size())) continue;

        std::vector<char> seen(graph.size(), 0);
        std::vector<int> stack{a};
        seen[static_cast<size_t>(a)] = 1;
        std::vector<char> in_side(n + 1, 0);
        while (!stack.empty()) {
            int u = stack.back();
            stack.pop_back();
            for (int x : bag_leafsets[static_cast<size_t>(u)]) in_side[x] = 1;
            for (int v : graph[static_cast<size_t>(u)]) {
                if ((u == a && v == b) || (u == b && v == a)) continue;
                if (seen[static_cast<size_t>(v)]) continue;
                seen[static_cast<size_t>(v)] = 1;
                stack.push_back(v);
            }
        }

        std::vector<int> side;
        std::vector<int> other;
        side.reserve(n);
        other.reserve(n);
        for (int x = 1; x <= n; ++x) {
            if (in_side[x]) side.push_back(x);
            else other.push_back(x);
        }
        if (side.empty() || other.empty()) continue;
        if (side.size() > other.size()) side.swap(other);
        if (side.size() <= 1 || side.size() >= static_cast<size_t>(n)) continue;
        out.push_back(std::move(side));
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        if (a.size() != b.size()) return a.size() > b.size();
        return a < b;
    });
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

bool parse_instance_data(const PaceInstance& inst, ParsedInstance& data) {
    data.tree_count = inst.tree_count;
    data.leaf_count = inst.leaf_count;
    data.has_aux_root = inst.has_aux_root;

    int n = inst.leaf_count;
    if (n <= 0) return false;

    data.zob.assign(n + 1, 0);
    for (int x = 1; x <= n; ++x) {
        data.zob[x] = mix64(static_cast<uint64_t>(x) * 0x9e3779b97f4a7c15ULL + 0xabcULL);
    }

    data.parsed.clear();
    data.trees.clear();
    data.parsed.reserve(static_cast<size_t>(inst.tree_count));
    data.trees.reserve(static_cast<size_t>(inst.tree_count));
    for (const auto& line : inst.newick_lines) {
        if (!parse_with_genesis(line)) return false;
        SimpleTree st;
        if (!NewickParser(line).parse(st)) return false;
        TreeData td;
        if (!build_tree_data(st, n, data.zob, td)) return false;
        data.parsed.push_back(std::move(st));
        data.trees.push_back(std::move(td));
    }

    data.td_candidates = extract_treedecomp_leaf_candidates(inst.treedecomp, n);
    return true;
}

DynamicTree build_dynamic_tree(const SimpleTree& st) {
    DynamicTree dt;
    dt.nodes.resize(st.children.size());
    dt.payloads.reserve(st.children.size());
    dt.is_cherry.assign(st.children.size(), 0);
    dt.cherry_pos.assign(st.children.size(), -1);
    dt.root = st.root;
    int max_label = 0;
    for (int x : st.leaf_label) max_label = std::max(max_label, x);
    dt.label_to_node.assign(static_cast<size_t>(max_label + 1), -1);
    for (size_t u = 0; u < st.children.size(); ++u) {
        DynNode& dn = dt.nodes[u];
        dn.parent = st.parent[u];
        dn.left = -1;
        dn.right = -1;
        dn.active = true;
        dn.payload_id = -1;

        if (st.children[u].empty()) {
            dn.label = st.leaf_label[u];
            dn.payload_id = push_payload(dt, PayloadNode{-1, -1, st.leaf_label[u], 1});
            dn.leafset_hash = mix64(static_cast<uint64_t>(st.leaf_label[u]) * 0x9e3779b97f4a7c15ULL + 0xabcULL);
            dn.payload_size = 1;
            dn.topo_a = mix64(static_cast<uint64_t>(st.leaf_label[u]) + 0x1111111111111111ULL);
            dn.topo_b = mix64(static_cast<uint64_t>(st.leaf_label[u]) + 0x2222222222222222ULL);
            dt.label_to_node[static_cast<size_t>(dn.label)] = static_cast<int>(u);
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

void push_undo(UndoLog* log, UndoLog::Kind kind, DynamicTree* tree, int index, int old_value, int aux) {
    if (log) log->entries.push_back({kind, tree, index, old_value, aux});
}

void ensure_label_storage(DynamicTree& t, int label, UndoLog* log = nullptr) {
    if (label < static_cast<int>(t.label_to_node.size())) return;
    push_undo(log, UndoLog::Kind::LabelSize, &t, -1, static_cast<int>(t.label_to_node.size()));
    t.label_to_node.resize(static_cast<size_t>(label + 1), -1);
}

void ensure_cherry_storage(DynamicTree& t, UndoLog* log = nullptr) {
    if (t.is_cherry.size() < t.nodes.size()) {
        push_undo(log, UndoLog::Kind::IsCherrySize, &t, -1, static_cast<int>(t.is_cherry.size()));
        t.is_cherry.resize(t.nodes.size(), 0);
    }
    if (t.cherry_pos.size() < t.nodes.size()) {
        push_undo(log, UndoLog::Kind::CherryPosSize, &t, -1, static_cast<int>(t.cherry_pos.size()));
        t.cherry_pos.resize(t.nodes.size(), -1);
    }
}

void set_root(DynamicTree& t, int value, UndoLog* log = nullptr) {
    if (t.root == value) return;
    push_undo(log, UndoLog::Kind::Root, &t, -1, t.root);
    t.root = value;
}

void set_node_parent(DynamicTree& t, int u, int value, UndoLog* log = nullptr) {
    if (t.nodes[u].parent == value) return;
    push_undo(log, UndoLog::Kind::NodeParent, &t, u, t.nodes[u].parent);
    t.nodes[u].parent = value;
}

void set_node_left(DynamicTree& t, int u, int value, UndoLog* log = nullptr) {
    if (t.nodes[u].left == value) return;
    push_undo(log, UndoLog::Kind::NodeLeft, &t, u, t.nodes[u].left);
    t.nodes[u].left = value;
}

void set_node_right(DynamicTree& t, int u, int value, UndoLog* log = nullptr) {
    if (t.nodes[u].right == value) return;
    push_undo(log, UndoLog::Kind::NodeRight, &t, u, t.nodes[u].right);
    t.nodes[u].right = value;
}

void set_node_active(DynamicTree& t, int u, bool value, UndoLog* log = nullptr) {
    if (t.nodes[u].active == value) return;
    push_undo(log, UndoLog::Kind::NodeActive, &t, u, static_cast<int>(t.nodes[u].active));
    t.nodes[u].active = value;
}

void set_label_slot(DynamicTree& t, int label, int value, UndoLog* log = nullptr) {
    ensure_label_storage(t, label, log);
    if (t.label_to_node[static_cast<size_t>(label)] == value) return;
    push_undo(log, UndoLog::Kind::LabelSlot, &t, label, t.label_to_node[static_cast<size_t>(label)]);
    t.label_to_node[static_cast<size_t>(label)] = value;
}

void set_is_cherry_slot(DynamicTree& t, int u, char value, UndoLog* log = nullptr) {
    ensure_cherry_storage(t, log);
    if (t.is_cherry[static_cast<size_t>(u)] == value) return;
    push_undo(log, UndoLog::Kind::IsCherrySlot, &t, u, t.is_cherry[static_cast<size_t>(u)]);
    t.is_cherry[static_cast<size_t>(u)] = value;
}

void set_cherry_pos_slot(DynamicTree& t, int u, int value, UndoLog* log = nullptr) {
    ensure_cherry_storage(t, log);
    if (t.cherry_pos[static_cast<size_t>(u)] == value) return;
    push_undo(log, UndoLog::Kind::CherryPosSlot, &t, u, t.cherry_pos[static_cast<size_t>(u)]);
    t.cherry_pos[static_cast<size_t>(u)] = value;
}

void set_cherry_node_slot(DynamicTree& t, int idx, int value, UndoLog* log = nullptr) {
    if (t.cherry_nodes[static_cast<size_t>(idx)] == value) return;
    push_undo(log, UndoLog::Kind::CherryNodeSlot, &t, idx, t.cherry_nodes[static_cast<size_t>(idx)]);
    t.cherry_nodes[static_cast<size_t>(idx)] = value;
}

void push_node(DynamicTree& t, DynNode node, UndoLog* log = nullptr) {
    push_undo(log, UndoLog::Kind::NodesSize, &t, -1, static_cast<int>(t.nodes.size()));
    t.nodes.push_back(std::move(node));
}

void resize_nodes(DynamicTree& t, size_t new_size, UndoLog* log = nullptr) {
    if (t.nodes.size() == new_size) return;
    push_undo(log, UndoLog::Kind::NodesSize, &t, -1, static_cast<int>(t.nodes.size()));
    t.nodes.resize(new_size);
}

void push_cherry_node(DynamicTree& t, int value, UndoLog* log = nullptr) {
    int old_size = static_cast<int>(t.cherry_nodes.size());
    int old_tail = old_size > 0 ? t.cherry_nodes.back() : 0;
    push_undo(log, UndoLog::Kind::CherryNodesSize, &t, -1, old_size, old_tail);
    t.cherry_nodes.push_back(value);
}

void pop_cherry_node(DynamicTree& t, UndoLog* log = nullptr) {
    int old_size = static_cast<int>(t.cherry_nodes.size());
    int old_tail = old_size > 0 ? t.cherry_nodes.back() : 0;
    push_undo(log, UndoLog::Kind::CherryNodesSize, &t, -1, old_size, old_tail);
    t.cherry_nodes.pop_back();
}

void remove_cherry_node(DynamicTree& t, int u, UndoLog* log = nullptr) {
    if (u < 0 || u >= static_cast<int>(t.cherry_pos.size())) return;
    int pos = t.cherry_pos[u];
    if (pos == -1) return;
    int last = t.cherry_nodes.back();
    set_cherry_node_slot(t, pos, last, log);
    set_cherry_pos_slot(t, last, pos, log);
    pop_cherry_node(t, log);
    set_cherry_pos_slot(t, u, -1, log);
    set_is_cherry_slot(t, u, 0, log);
}

void refresh_local_cherry_status(DynamicTree& t, int u, UndoLog* log = nullptr) {
    if (u < 0 || u >= static_cast<int>(t.nodes.size())) return;
    ensure_cherry_storage(t, log);

    bool now = false;
    if (t.nodes[u].active) {
        int l = t.nodes[u].left;
        int r = t.nodes[u].right;
        now = (l != -1 && r != -1 && is_leaflike(t, l) && is_leaflike(t, r));
    }
    if (now) {
        if (t.cherry_pos[u] == -1) {
            set_cherry_pos_slot(t, u, static_cast<int>(t.cherry_nodes.size()), log);
            push_cherry_node(t, u, log);
        }
        set_is_cherry_slot(t, u, 1, log);
    } else {
        remove_cherry_node(t, u, log);
    }
}

void refresh_cherry_neighborhood(DynamicTree& t, int u, UndoLog* log = nullptr) {
    if (u < 0 || u >= static_cast<int>(t.nodes.size())) return;
    refresh_local_cherry_status(t, u, log);
    int p = t.nodes[u].parent;
    if (p != -1) {
        refresh_local_cherry_status(t, p, log);
        int gp = t.nodes[p].parent;
        if (gp != -1) refresh_local_cherry_status(t, gp, log);
    }
    int l = t.nodes[u].left;
    int r = t.nodes[u].right;
    if (l != -1) refresh_local_cherry_status(t, l, log);
    if (r != -1) refresh_local_cherry_status(t, r, log);
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

void replace_child(DynamicTree& t, int p, int oldc, int newc, UndoLog* log = nullptr) {
    if (p == -1) return;
    if (t.nodes[p].left == oldc) set_node_left(t, p, newc, log);
    else if (t.nodes[p].right == oldc) set_node_right(t, p, newc, log);
}

void suppress_up(DynamicTree& t, int u, UndoLog* log = nullptr) {
    while (u != -1 && t.nodes[u].active) {
        int deg = degree_children(t, u);
        if (deg != 1) break;
        remove_cherry_node(t, u, log);
        int child = (t.nodes[u].left != -1 && t.nodes[t.nodes[u].left].active) ? t.nodes[u].left : t.nodes[u].right;
        int p = t.nodes[u].parent;
        if (p == -1) {
            set_node_parent(t, child, -1, log);
            set_root(t, child, log);
            set_node_active(t, u, false, log);
            set_node_left(t, u, -1, log);
            set_node_right(t, u, -1, log);
            refresh_cherry_neighborhood(t, child, log);
            break;
        }
        replace_child(t, p, u, child, log);
        set_node_parent(t, child, p, log);
        set_node_active(t, u, false, log);
        set_node_left(t, u, -1, log);
        set_node_right(t, u, -1, log);
        refresh_cherry_neighborhood(t, child, log);
        refresh_local_cherry_status(t, p, log);
        u = p;
    }
    if (u != -1) refresh_cherry_neighborhood(t, u, log);
}

void cut_edge_above(DynamicTree& t, int child, UndoLog* log = nullptr) {
    if (child < 0 || child >= static_cast<int>(t.nodes.size())) return;
    if (!t.nodes[child].active) return;
    int p = t.nodes[child].parent;
    if (p == -1) return;
    remove_cherry_node(t, p, log);
    replace_child(t, p, child, -1, log);
    set_node_parent(t, child, -1, log);
    refresh_cherry_neighborhood(t, child, log);
    suppress_up(t, p, log);
}

bool are_siblings_by_label(const DynamicTree& t, int la, int lb) {
    int a = find_node_of_label(t, la);
    int b = find_node_of_label(t, lb);
    if (a == -1 || b == -1) return false;
    if (!is_active_leaf(t, a) || !is_active_leaf(t, b)) return false;
    int pa = t.nodes[a].parent;
    int pb = t.nodes[b].parent;
    if (pa == -1 || pa != pb) return false;
    int l = t.nodes[pa].left, r = t.nodes[pa].right;
    return (l == a && r == b) || (l == b && r == a);
}

bool contract_cherry(DynamicTree& t, int la, int lb, int new_label, UndoLog* log = nullptr) {
    int a = find_node_of_label(t, la);
    int b = find_node_of_label(t, lb);
    if (a == -1 || b == -1) return false;
    if (!is_active_leaf(t, a) || !is_active_leaf(t, b)) return false;
    int p = t.nodes[a].parent;
    if (p == -1 || t.nodes[b].parent != p) return false;
    int left = t.nodes[p].left;
    int right = t.nodes[p].right;
    if (!((left == a && right == b) || (left == b && right == a))) return false;

    DynNode nn;
    nn.parent = p;
    nn.left = -1;
    nn.right = -1;
    nn.label = new_label;
    nn.active = true;
    nn.payload_id = push_payload(
        t,
        PayloadNode{
            t.nodes[a].payload_id,
            t.nodes[b].payload_id,
            -1,
            t.nodes[a].payload_size + t.nodes[b].payload_size
        },
        log
    );
    nn.leafset_hash = t.nodes[a].leafset_hash ^ t.nodes[b].leafset_hash;
    nn.payload_size = t.nodes[a].payload_size + t.nodes[b].payload_size;
    uint64_t aa = t.nodes[a].topo_a, ab = t.nodes[b].topo_a;
    uint64_t ba = t.nodes[a].topo_b, bb = t.nodes[b].topo_b;
    if (aa > ab) std::swap(aa, ab);
    if (ba > bb) std::swap(ba, bb);
    nn.topo_a = mix64(aa ^ (ab + 0x9e3779b97f4a7c15ULL));
    nn.topo_b = mix64(ba ^ (bb + 0x517cc1b727220a95ULL));

    int nid = static_cast<int>(t.nodes.size());
    push_node(t, std::move(nn), log);
    ensure_cherry_storage(t, log);
    ensure_label_storage(t, new_label, log);
    set_label_slot(t, la, -1, log);
    set_label_slot(t, lb, -1, log);
    set_label_slot(t, new_label, nid, log);

    set_node_left(t, p, nid, log);
    set_node_right(t, p, -1, log);

    remove_cherry_node(t, p, log);
    remove_cherry_node(t, a, log);
    remove_cherry_node(t, b, log);
    set_node_active(t, a, false, log);
    set_node_active(t, b, false, log);
    set_node_parent(t, a, -1, log);
    set_node_parent(t, b, -1, log);

    refresh_cherry_neighborhood(t, nid, log);
    suppress_up(t, p, log);
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

template <typename Fn>
void for_each_cherry_label_pair(const DynamicTree& t, Fn&& fn) {
    for (int u : t.cherry_nodes) {
        if (u < 0 || u >= static_cast<int>(t.nodes.size())) continue;
        if (!t.nodes[u].active) continue;
        int l = t.nodes[u].left;
        int r = t.nodes[u].right;
        if (!is_active_leaf(t, l) || !is_active_leaf(t, r)) continue;
        int a = t.nodes[l].label;
        int b = t.nodes[r].label;
        if (a > b) std::swap(a, b);
        if (!fn(a, b)) break;
    }
}

bool contract_all_common_cherries(DynamicTree& t1, DynamicTree& t2, int& next_label, UndoLog* log = nullptr) {
    bool any = false;
    while (true) {
        auto cherries = cherries_by_label(t1);
        bool contracted = false;
        for (auto [a, b] : cherries) {
            if (!are_siblings_by_label(t1, a, b) || !are_siblings_by_label(t2, a, b)) continue;
            int nl = next_label++;
            if (contract_cherry(t1, a, b, nl, log) && contract_cherry(t2, a, b, nl, log)) {
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

void collect_payload_leaves(const DynamicTree& t, int payload_id, std::vector<int>& out) {
    if (payload_id < 0 || payload_id >= static_cast<int>(t.payloads.size())) return;
    const auto& payload = t.payloads[static_cast<size_t>(payload_id)];
    if (payload.leaf_label != -1) {
        out.push_back(payload.leaf_label);
        return;
    }
    collect_payload_leaves(t, payload.left, out);
    collect_payload_leaves(t, payload.right, out);
}

void collect_component_leaves(const DynamicTree& t, int u, std::vector<int>& out) {
    if (u == -1 || !t.nodes[u].active) return;
    if (t.nodes[u].left == -1 && t.nodes[u].right == -1) {
        collect_payload_leaves(t, t.nodes[u].payload_id, out);
        return;
    }
    if (t.nodes[u].left != -1) collect_component_leaves(t, t.nodes[u].left, out);
    if (t.nodes[u].right != -1) collect_component_leaves(t, t.nodes[u].right, out);
}

std::vector<std::vector<int>> components_from_dynamic(const DynamicTree& t) {
    std::vector<std::vector<int>> comps;
    for (int u = 0; u < static_cast<int>(t.nodes.size()); ++u) {
        if (!t.nodes[u].active || t.nodes[u].parent != -1) continue;
        std::vector<int> leaves;
        leaves.reserve(16);
        collect_component_leaves(t, u, leaves);
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        if (!leaves.empty()) comps.push_back(std::move(leaves));
    }
    std::sort(comps.begin(), comps.end(), [](const auto& a, const auto& b) {
        return a.front() < b.front();
    });
    return comps;
}

std::string encode_components_state(const std::vector<std::vector<int>>& comps, int next_idx) {
    size_t reserve = 16;
    for (const auto& comp : comps) reserve += comp.size() * 4 + 1;
    std::string key;
    key.reserve(reserve);
    key += std::to_string(next_idx);
    key.push_back('|');
    for (const auto& comp : comps) {
        for (int x : comp) {
            key += std::to_string(x);
            key.push_back(',');
        }
        key.push_back(';');
    }
    return key;
}

void bits_set(std::vector<uint64_t>& bits, int label) {
    int idx = label - 1;
    bits[static_cast<size_t>(idx >> 6)] |= (1ULL << (idx & 63));
}

bool bits_intersect(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & b[i]) != 0ULL) return true;
    }
    return false;
}

bool bits_subset_of(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if ((a[i] & ~b[i]) != 0ULL) return false;
    }
    return true;
}

bool bits_equal(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    return a == b;
}

void bits_or_assign(std::vector<uint64_t>& dst, const std::vector<uint64_t>& src) {
    for (size_t i = 0; i < dst.size(); ++i) dst[i] |= src[i];
}

void collect_payload_bits(const DynamicTree& t, int payload_id, std::vector<uint64_t>& bits) {
    if (payload_id < 0 || payload_id >= static_cast<int>(t.payloads.size())) return;
    const auto& payload = t.payloads[static_cast<size_t>(payload_id)];
    if (payload.leaf_label != -1) {
        bits_set(bits, payload.leaf_label);
        return;
    }
    collect_payload_bits(t, payload.left, bits);
    collect_payload_bits(t, payload.right, bits);
}

std::vector<std::vector<uint64_t>> compute_active_leaf_bits(const DynamicTree& t, int n) {
    const size_t words = static_cast<size_t>((n + 63) >> 6);
    std::vector<std::vector<uint64_t>> bits(t.nodes.size(), std::vector<uint64_t>(words, 0ULL));
    std::vector<int> order;
    collect_active_postorder(t, order);
    for (int u : order) {
        if (is_leaflike(t, u)) {
            collect_payload_bits(t, t.nodes[u].payload_id, bits[static_cast<size_t>(u)]);
            continue;
        }
        int l = t.nodes[u].left;
        int r = t.nodes[u].right;
        if (l != -1 && t.nodes[l].active) bits_or_assign(bits[static_cast<size_t>(u)], bits[static_cast<size_t>(l)]);
        if (r != -1 && t.nodes[r].active) bits_or_assign(bits[static_cast<size_t>(u)], bits[static_cast<size_t>(r)]);
    }
    return bits;
}

bool is_union_of_root_components(
    const std::vector<uint64_t>& target,
    const std::vector<std::vector<uint64_t>>& root_bits
) {
    std::vector<uint64_t> acc(target.size(), 0ULL);
    bool used = false;
    for (const auto& comp : root_bits) {
        if (!bits_intersect(comp, target)) continue;
        if (!bits_subset_of(comp, target)) return false;
        bits_or_assign(acc, comp);
        used = true;
    }
    return used && bits_equal(acc, target);
}

bool apply_reduction_rule_against(DynamicTree& dst, const DynamicTree& src, int n) {
    auto src_bits = compute_active_leaf_bits(src, n);
    std::vector<std::vector<uint64_t>> root_bits;
    for (int u = 0; u < static_cast<int>(src.nodes.size()); ++u) {
        if (!src.nodes[u].active || src.nodes[u].parent != -1) continue;
        root_bits.push_back(src_bits[static_cast<size_t>(u)]);
    }

    auto dst_bits = compute_active_leaf_bits(dst, n);
    for (int u = 0; u < static_cast<int>(dst.nodes.size()); ++u) {
        if (!dst.nodes[u].active || dst.nodes[u].parent == -1) continue;
        if (!is_union_of_root_components(dst_bits[static_cast<size_t>(u)], root_bits)) continue;
        cut_edge_above(dst, u);
        return true;
    }
    return false;
}

bool apply_pair_reduction_rule(DynamicTree& f1, DynamicTree& f2, int n) {
    bool any = false;
    while (true) {
        bool changed = false;
        changed |= apply_reduction_rule_against(f1, f2, n);
        changed |= apply_reduction_rule_against(f2, f1, n);
        if (!changed) break;
        any = true;
    }
    return any;
}

DynamicTree build_dynamic_forest(const std::vector<SimpleTree>& forest) {
    DynamicTree dt;
    int total_nodes = 0;
    int max_label = 0;
    for (const auto& st : forest) {
        total_nodes += static_cast<int>(st.children.size());
        for (int x : st.leaf_label) max_label = std::max(max_label, x);
    }

    dt.nodes.resize(static_cast<size_t>(total_nodes));
    dt.payloads.reserve(static_cast<size_t>(total_nodes));
    dt.is_cherry.assign(static_cast<size_t>(total_nodes), 0);
    dt.cherry_pos.assign(static_cast<size_t>(total_nodes), -1);
    dt.label_to_node.assign(static_cast<size_t>(max_label + 1), -1);
    dt.root = -1;

    int offset = 0;
    for (const auto& st : forest) {
        if (dt.root == -1 && st.root != -1) dt.root = st.root + offset;
        for (int u = 0; u < static_cast<int>(st.children.size()); ++u) {
            DynNode node;
            node.parent = (st.parent[u] == -1 ? -1 : st.parent[u] + offset);
            node.left = -1;
            node.right = -1;
            node.active = true;
            node.payload_id = -1;

            if (st.children[u].empty()) {
                node.label = st.leaf_label[u];
                node.payload_id = push_payload(dt, PayloadNode{-1, -1, st.leaf_label[u], 1});
                node.leafset_hash = mix64(static_cast<uint64_t>(st.leaf_label[u]) * 0x9e3779b97f4a7c15ULL + 0xabcULL);
                node.payload_size = 1;
                node.topo_a = mix64(static_cast<uint64_t>(st.leaf_label[u]) + 0x1111111111111111ULL);
                node.topo_b = mix64(static_cast<uint64_t>(st.leaf_label[u]) + 0x2222222222222222ULL);
                dt.label_to_node[static_cast<size_t>(node.label)] = offset + u;
            } else {
                node.label = 0;
                node.left = st.children[u][0] + offset;
                node.right = st.children[u][1] + offset;
            }
            dt.nodes[static_cast<size_t>(offset + u)] = std::move(node);
        }
        offset += static_cast<int>(st.children.size());
    }

    for (int u = 0; u < static_cast<int>(dt.nodes.size()); ++u) {
        int l = dt.nodes[u].left;
        int r = dt.nodes[u].right;
        if (l != -1 && r != -1 &&
            dt.nodes[l].active && dt.nodes[r].active &&
            dt.nodes[l].left == -1 && dt.nodes[l].right == -1 &&
            dt.nodes[r].left == -1 && dt.nodes[r].right == -1) {
            dt.is_cherry[static_cast<size_t>(u)] = 1;
            dt.cherry_pos[static_cast<size_t>(u)] = static_cast<int>(dt.cherry_nodes.size());
            dt.cherry_nodes.push_back(u);
        }
    }

    return dt;
}

bool build_dynamic_forest_from_components(
    const std::vector<std::vector<int>>& comps,
    int n,
    const TreeData& base_tree,
    DynamicTree& out
) {
    std::vector<std::string> lines = forest_from_partition(comps, n, base_tree);
    std::vector<SimpleTree> parsed;
    parsed.reserve(lines.size());
    for (const auto& line : lines) {
        SimpleTree st;
        if (!NewickParser(line).parse(st)) return false;
        parsed.push_back(std::move(st));
    }
    out = build_dynamic_forest(parsed);
    return true;
}

void collect_active_postorder(const DynamicTree& t, std::vector<int>& order) {
    order.clear();
    std::vector<std::pair<int, int>> st;
    st.reserve(t.nodes.size());
    for (int u = static_cast<int>(t.nodes.size()) - 1; u >= 0; --u) {
        if (!t.nodes[u].active || t.nodes[u].parent != -1) continue;
        st.push_back({u, 0});
    }
    while (!st.empty()) {
        const auto node_phase = st.back();
        st.pop_back();
        int u = node_phase.first;
        int phase = node_phase.second;
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

struct ExactSearchContext {
    int n = 0;
    Clock::time_point deadline;
    const TreeData* rebuild_tree = nullptr;
    const std::vector<DynamicTree>* original_forests = nullptr;
    int best_order = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> best_components;
    bool complete = true;
    std::unordered_set<std::string> seen_states;
};

struct PairSolveResult {
    bool complete = false;
    int order = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> components;
};

struct PairSearchContext {
    int n = 0;
    Clock::time_point deadline;
    int best_order = std::numeric_limits<int>::max();
    std::vector<std::vector<int>> best_components;
    bool complete = true;
};

bool exact_time_expired(ExactSearchContext& ctx) {
    if (!g_terminate && Clock::now() < ctx.deadline) return false;
    ctx.complete = false;
    return true;
}

void update_exact_best(const std::vector<std::vector<int>>& comps, ExactSearchContext& ctx) {
    if (comps.empty()) return;
    int order = static_cast<int>(comps.size());
    if (order >= ctx.best_order) return;
    ctx.best_order = order;
    ctx.best_components = comps;
}

bool pair_time_expired(PairSearchContext& ctx) {
    if (!g_terminate && Clock::now() < ctx.deadline) return false;
    ctx.complete = false;
    return true;
}

void update_pair_best(const std::vector<std::vector<int>>& comps, PairSearchContext& ctx) {
    if (comps.empty()) return;
    int order = static_cast<int>(comps.size());
    if (order >= ctx.best_order) return;
    ctx.best_order = order;
    ctx.best_components = comps;
}

bool make_label_singleton(DynamicTree& t, int label) {
    int u = find_node_of_label(t, label);
    if (u == -1 || !t.nodes[u].active || t.nodes[u].parent == -1) return false;
    cut_edge_above(t, u);
    return true;
}

void normalize_exact_pair(DynamicTree& f1, DynamicTree& f2, int n, int& next_label) {
    while (true) {
        bool changed = false;
        changed |= apply_pair_reduction_rule(f1, f2, n);
        changed |= contract_all_common_cherries(f1, f2, next_label);
        if (!changed) break;
    }
}

void exact_multi_search(DynamicTree base, int next_idx, ExactSearchContext& ctx);

void exact_pair_only_search(DynamicTree f1, DynamicTree f2, PairSearchContext& ctx) {
    if (pair_time_expired(ctx)) return;

    int next_label = std::max(
        ctx.n + 1,
        std::max(
            static_cast<int>(f1.label_to_node.size()),
            static_cast<int>(f2.label_to_node.size())
        )
    );
    normalize_exact_pair(f1, f2, ctx.n, next_label);
    int roots1 = count_root_components(f1);
    int roots2 = count_root_components(f2);
    if (std::max(roots1, roots2) >= ctx.best_order) return;
    if (pair_time_expired(ctx)) return;

    auto cherries = cherries_by_label(f2);
    if (cherries.empty()) {
        update_pair_best(components_from_dynamic(f2), ctx);
        return;
    }

    int a = cherries[0].first;
    int b = cherries[0].second;
    int na = find_node_of_label(f1, a);
    int nb = find_node_of_label(f1, b);
    if (na == -1 || nb == -1 || !is_active_leaf(f1, na) || !is_active_leaf(f1, nb)) return;

    auto branch_make_singleton = [&](int label) {
        DynamicTree nf1 = f1;
        DynamicTree nf2 = f2;
        bool changed1 = make_label_singleton(nf1, label);
        bool changed2 = make_label_singleton(nf2, label);
        if (!changed1 && !changed2) return;
        exact_pair_only_search(std::move(nf1), std::move(nf2), ctx);
    };

    int ra = root_of(f1, na);
    int rb = root_of(f1, nb);
    if (ra == -1 || rb == -1) return;
    if (ra != rb) {
        branch_make_singleton(a);
        if (pair_time_expired(ctx)) return;
        branch_make_singleton(b);
        return;
    }

    auto pendants = pendant_children_on_path(f1, path_nodes(f1, na, nb));
    branch_make_singleton(a);
    if (pair_time_expired(ctx)) return;
    branch_make_singleton(b);
    if (pair_time_expired(ctx)) return;
    if (pendants.empty()) return;

    DynamicTree nf1 = f1;
    for (int u : pendants) cut_edge_above(nf1, u);
    exact_pair_only_search(std::move(nf1), std::move(f2), ctx);
}

PairSolveResult solve_pair_exact(
    DynamicTree f1,
    DynamicTree f2,
    int n,
    Clock::time_point deadline,
    int initial_best_order = std::numeric_limits<int>::max(),
    const std::vector<std::vector<int>>* initial_best_components = nullptr
) {
    PairSearchContext ctx;
    ctx.n = n;
    ctx.deadline = deadline;
    ctx.best_order = initial_best_order;
    if (initial_best_components && static_cast<int>(initial_best_components->size()) == initial_best_order) {
        ctx.best_components = *initial_best_components;
    }

    exact_pair_only_search(std::move(f1), std::move(f2), ctx);

    PairSolveResult out;
    out.complete = ctx.complete && !ctx.best_components.empty();
    out.order = ctx.best_order;
    if (out.complete) out.components = std::move(ctx.best_components);
    return out;
}

void exact_pair_search(DynamicTree f1, DynamicTree f2, int next_idx, ExactSearchContext& ctx) {
    if (exact_time_expired(ctx)) return;

    int next_label = std::max(
        ctx.n + 1,
        std::max(
            static_cast<int>(f1.label_to_node.size()),
            static_cast<int>(f2.label_to_node.size())
        )
    );
    normalize_exact_pair(f1, f2, ctx.n, next_label);
    int roots1 = count_root_components(f1);
    int roots2 = count_root_components(f2);
    if (std::max(roots1, roots2) >= ctx.best_order) return;
    if (exact_time_expired(ctx)) return;

    auto cherries = cherries_by_label(f2);
    if (cherries.empty()) {
        auto comps = components_from_dynamic(f2);
        if (static_cast<int>(comps.size()) >= ctx.best_order) return;
        DynamicTree next_base;
        if (!build_dynamic_forest_from_components(comps, ctx.n, *ctx.rebuild_tree, next_base)) {
            ctx.complete = false;
            return;
        }
        exact_multi_search(std::move(next_base), next_idx, ctx);
        return;
    }

    int a = cherries[0].first;
    int b = cherries[0].second;
    int na = find_node_of_label(f1, a);
    int nb = find_node_of_label(f1, b);
    if (na == -1 || nb == -1 || !is_active_leaf(f1, na) || !is_active_leaf(f1, nb)) return;

    auto branch_make_singleton = [&](int label) {
        DynamicTree nf1 = f1;
        DynamicTree nf2 = f2;
        bool changed1 = make_label_singleton(nf1, label);
        bool changed2 = make_label_singleton(nf2, label);
        if (!changed1 && !changed2) return;
        exact_pair_search(std::move(nf1), std::move(nf2), next_idx, ctx);
    };

    int ra = root_of(f1, na);
    int rb = root_of(f1, nb);
    if (ra == -1 || rb == -1) return;
    if (ra != rb) {
        branch_make_singleton(a);
        if (exact_time_expired(ctx)) return;
        branch_make_singleton(b);
        return;
    }

    auto pendants = pendant_children_on_path(f1, path_nodes(f1, na, nb));
    branch_make_singleton(a);
    if (exact_time_expired(ctx)) return;
    branch_make_singleton(b);
    if (exact_time_expired(ctx)) return;
    if (pendants.empty()) return;

    DynamicTree nf1 = f1;
    for (int u : pendants) cut_edge_above(nf1, u);
    exact_pair_search(std::move(nf1), std::move(f2), next_idx, ctx);
}

void exact_multi_search(DynamicTree base, int next_idx, ExactSearchContext& ctx) {
    if (exact_time_expired(ctx)) return;
    auto comps = components_from_dynamic(base);
    if (static_cast<int>(comps.size()) >= ctx.best_order) return;
    if (!ctx.seen_states.insert(encode_components_state(comps, next_idx)).second) return;

    if (next_idx >= static_cast<int>(ctx.original_forests->size())) {
        update_exact_best(comps, ctx);
        return;
    }

    exact_pair_search(
        std::move(base),
        (*ctx.original_forests)[static_cast<size_t>(next_idx)],
        next_idx + 1,
        ctx
    );
}

SolveComponentsResult solve_components_basic(const ParsedInstance& data, Clock::time_point deadline) {
    SolveComponentsResult result;
    if (g_terminate || Clock::now() >= deadline) return result;

    int n = data.leaf_count;
    std::vector<std::vector<int>> initial_partition;
    initial_partition.reserve(static_cast<size_t>(n));
    for (int x = 1; x <= n; ++x) initial_partition.push_back({x});
    initial_partition = greedy_merge_partition(std::move(initial_partition), data.zob, data.trees);

    std::vector<DynamicTree> originals;
    originals.reserve(data.parsed.size());
    for (const auto& st : data.parsed) originals.push_back(build_dynamic_tree(st));

    if (data.tree_count == 1) {
        result.complete = true;
        result.components = std::move(initial_partition);
        return result;
    }

    const auto now = Clock::now();
    const auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    const long long bootstrap_budget_ms = std::min<long long>(
        20000,
        std::max<long long>(1000, remaining_ms / 3)
    );
    Clock::time_point bootstrap_deadline = std::min(
        deadline,
        now + std::chrono::milliseconds(bootstrap_budget_ms)
    );

    const bool enable_full_pair_bootstrap =
        (data.tree_count <= 24 && n <= 24) ||
        (data.tree_count <= 8 && n <= 64);

    int lower_bound = 0;
    int seed_second = 1;
    std::vector<int> row0_score(static_cast<size_t>(data.tree_count), 0);
    bool pair_lb_complete = enable_full_pair_bootstrap;

    for (int j = 1; j < data.tree_count; ++j) {
        PairSolveResult pair_res = solve_pair_exact(
            originals[0],
            originals[static_cast<size_t>(j)],
            n,
            bootstrap_deadline,
            static_cast<int>(initial_partition.size()),
            &initial_partition
        );
        if (!pair_res.complete) {
            pair_lb_complete = false;
            break;
        }
        row0_score[static_cast<size_t>(j)] = pair_res.order;
        lower_bound = std::max(lower_bound, pair_res.order);
        if (pair_res.order > row0_score[static_cast<size_t>(seed_second)]) seed_second = j;
    }

    if (pair_lb_complete) {
        for (int i = 1; i < data.tree_count; ++i) {
            for (int j = i + 1; j < data.tree_count; ++j) {
                PairSolveResult pair_res = solve_pair_exact(
                    originals[static_cast<size_t>(i)],
                    originals[static_cast<size_t>(j)],
                    n,
                    bootstrap_deadline,
                    static_cast<int>(initial_partition.size()),
                    &initial_partition
                );
                if (!pair_res.complete) {
                    pair_lb_complete = false;
                    break;
                }
                lower_bound = std::max(lower_bound, pair_res.order);
            }
            if (!pair_lb_complete) break;
        }
    }

    if (pair_lb_complete && lower_bound == static_cast<int>(initial_partition.size())) {
        result.complete = true;
        result.components = std::move(initial_partition);
        return result;
    }

    std::vector<std::vector<int>> best_upper = initial_partition;
    int best_upper_order = static_cast<int>(best_upper.size());

    if (seed_second >= 1 && enable_full_pair_bootstrap && Clock::now() < bootstrap_deadline) {
        std::vector<int> merge_order;
        merge_order.reserve(data.tree_count);
        merge_order.push_back(0);
        merge_order.push_back(seed_second);
        for (int j = 1; j < data.tree_count; ++j) {
            if (j == seed_second) continue;
            merge_order.push_back(j);
        }
        std::stable_sort(merge_order.begin() + 2, merge_order.end(), [&](int a, int b) {
            if (row0_score[static_cast<size_t>(a)] != row0_score[static_cast<size_t>(b)]) {
                return row0_score[static_cast<size_t>(a)] > row0_score[static_cast<size_t>(b)];
            }
            return a < b;
        });

        PairSolveResult seed_pair = solve_pair_exact(
            originals[0],
            originals[static_cast<size_t>(seed_second)],
            n,
            bootstrap_deadline
        );
        if (seed_pair.complete) {
            std::vector<std::vector<int>> current_components = std::move(seed_pair.components);
            DynamicTree current_base;
            if (build_dynamic_forest_from_components(current_components, n, data.trees[0], current_base)) {
                bool greedy_complete = true;
                for (size_t pos = 2; pos < merge_order.size(); ++pos) {
                    PairSolveResult step_res = solve_pair_exact(
                        current_base,
                        originals[static_cast<size_t>(merge_order[pos])],
                        n,
                        bootstrap_deadline
                    );
                    if (!step_res.complete) {
                        greedy_complete = false;
                        break;
                    }
                    current_components = std::move(step_res.components);
                    if (!build_dynamic_forest_from_components(current_components, n, data.trees[0], current_base)) {
                        greedy_complete = false;
                        break;
                    }
                }
                if (greedy_complete && static_cast<int>(current_components.size()) < best_upper_order) {
                    best_upper_order = static_cast<int>(current_components.size());
                    best_upper = std::move(current_components);
                }
            }
        }
    }

    if (pair_lb_complete && lower_bound == best_upper_order) {
        result.complete = true;
        result.components = std::move(best_upper);
        return result;
    }

    std::vector<DynamicTree> ordered_originals;
    ordered_originals.reserve(originals.size());
    ordered_originals.push_back(originals[0]);
    ordered_originals.push_back(originals[static_cast<size_t>(seed_second)]);
    std::vector<int> remaining;
    remaining.reserve(originals.size());
    for (int j = 1; j < data.tree_count; ++j) {
        if (j == seed_second) continue;
        remaining.push_back(j);
    }
    std::stable_sort(remaining.begin(), remaining.end(), [&](int a, int b) {
        if (row0_score[static_cast<size_t>(a)] != row0_score[static_cast<size_t>(b)]) {
            return row0_score[static_cast<size_t>(a)] > row0_score[static_cast<size_t>(b)];
        }
        return a < b;
    });
    for (int idx : remaining) ordered_originals.push_back(originals[static_cast<size_t>(idx)]);

    ExactSearchContext ctx;
    ctx.n = n;
    ctx.deadline = deadline;
    ctx.rebuild_tree = &data.trees[0];
    ctx.original_forests = &ordered_originals;
    ctx.best_order = best_upper_order;
    ctx.best_components = best_upper;

    exact_multi_search(ordered_originals[0], 1, ctx);

    result.complete = ctx.complete && !ctx.best_components.empty();
    if (result.complete) result.components = std::move(ctx.best_components);
    return result;
}

std::vector<int> choose_cluster_candidate(const ParsedInstance& data) {
    const int n = data.leaf_count;
    auto better = [n](const std::vector<int>& a, const std::vector<int>& b) {
        int ba = std::min<int>(a.size(), n - static_cast<int>(a.size()));
        int bb = std::min<int>(b.size(), n - static_cast<int>(b.size()));
        if (ba != bb) return ba > bb;
        if (a.size() != b.size()) return a.size() > b.size();
        return a < b;
    };

    std::vector<int> best;
    for (const auto& cand : data.td_candidates) {
        if (cand.size() <= 1 || cand.size() >= static_cast<size_t>(n)) continue;
        if (!is_common_cluster(cand, data.zob, data.trees)) continue;
        if (best.empty() || better(cand, best)) best = cand;
    }
    if (!best.empty() || data.has_aux_root) return best;

    std::unordered_set<Signature, SignatureHasher> seen;
    const auto& t0 = data.trees[0];
    for (int u = 0; u < t0.n_nodes; ++u) {
        if (t0.is_leaf[u]) continue;
        int sz = t0.sub_count[u];
        if (sz <= 1 || sz >= n) continue;
        Signature sig{t0.sub_hash[u], sz};
        if (!seen.insert(sig).second) continue;
        bool maybe = true;
        for (size_t i = 1; i < data.trees.size(); ++i) {
            if (!data.trees[i].cluster_sigs.count(sig)) {
                maybe = false;
                break;
            }
        }
        if (!maybe) continue;
        std::vector<int> cand;
        cand.reserve(static_cast<size_t>(sz));
        collect_subtree_leaves(t0, u, cand);
        std::sort(cand.begin(), cand.end());
        if (!is_common_cluster(cand, data.zob, data.trees)) continue;
        if (best.empty() || better(cand, best)) best = std::move(cand);
    }
    return best;
}

PaceInstance build_cluster_subinstance(const ParsedInstance& data, const std::vector<int>& cluster) {
    PaceInstance sub;
    sub.tree_count = data.tree_count;
    sub.leaf_count = static_cast<int>(cluster.size()) + 1;
    sub.has_aux_root = true;
    sub.newick_lines.reserve(data.trees.size());

    std::vector<char> in_cluster(data.leaf_count + 1, 0);
    std::vector<int> remap(data.leaf_count + 1, 0);
    for (size_t i = 0; i < cluster.size(); ++i) {
        in_cluster[cluster[i]] = 1;
        remap[cluster[i]] = static_cast<int>(i + 1);
    }
    int rho_label = sub.leaf_count;

    for (const auto& t : data.trees) {
        std::string inner = restricted_newick_dfs_remap(t.root, t, in_cluster, remap);
        if (inner.empty()) {
            sub.newick_lines.clear();
            return sub;
        }
        sub.newick_lines.push_back("(" + std::to_string(rho_label) + "," + inner + ");");
    }
    return sub;
}

PaceInstance build_quotient_subinstance(const ParsedInstance& data, const std::vector<int>& cluster) {
    PaceInstance out;
    out.tree_count = data.tree_count;
    out.leaf_count = data.leaf_count - static_cast<int>(cluster.size()) + 1;
    out.newick_lines.reserve(data.trees.size());

    std::vector<char> in_cluster(data.leaf_count + 1, 0);
    for (int x : cluster) in_cluster[x] = 1;

    std::vector<int> outside_remap(data.leaf_count + 1, 0);
    int next = 1;
    for (int x = 1; x <= data.leaf_count; ++x) {
        if (!in_cluster[x]) outside_remap[x] = next++;
    }
    int meta_label = out.leaf_count;
    uint64_t cluster_hash = hash_of_leaves(cluster, data.zob);
    int cluster_size = static_cast<int>(cluster.size());

    for (const auto& t : data.trees) {
        std::string line = compress_cluster_newick_dfs(
            t.root, t, cluster_hash, cluster_size, in_cluster, outside_remap, meta_label
        );
        if (line.empty()) {
            out.newick_lines.clear();
            return out;
        }
        if (line.back() != ';') line.push_back(';');
        out.newick_lines.push_back(std::move(line));
    }
    return out;
}

SolveComponentsResult combine_cluster_results(
    int original_n,
    const std::vector<int>& cluster,
    const SolveComponentsResult& sub_res,
    const SolveComponentsResult& quotient_res
) {
    SolveComponentsResult out;
    if (!sub_res.complete || !quotient_res.complete) return out;

    std::vector<char> in_cluster(original_n + 1, 0);
    for (int x : cluster) in_cluster[x] = 1;

    std::vector<int> cluster_new_to_old(cluster.size() + 2, 0);
    for (size_t i = 0; i < cluster.size(); ++i) cluster_new_to_old[i + 1] = cluster[i];
    int rho_label = static_cast<int>(cluster.size()) + 1;

    std::vector<int> outside_old;
    outside_old.reserve(original_n - static_cast<int>(cluster.size()));
    for (int x = 1; x <= original_n; ++x) if (!in_cluster[x]) outside_old.push_back(x);
    std::vector<int> quotient_new_to_old(outside_old.size() + 2, 0);
    for (size_t i = 0; i < outside_old.size(); ++i) quotient_new_to_old[i + 1] = outside_old[i];
    int meta_label = static_cast<int>(outside_old.size()) + 1;

    std::vector<int> root_attach;
    std::vector<std::vector<int>> merged;
    for (const auto& comp : sub_res.components) {
        bool has_rho = false;
        std::vector<int> mapped;
        for (int label : comp) {
            if (label == rho_label) {
                has_rho = true;
                continue;
            }
            if (1 <= label && label < static_cast<int>(cluster_new_to_old.size())) {
                mapped.push_back(cluster_new_to_old[label]);
            }
        }
        std::sort(mapped.begin(), mapped.end());
        mapped.erase(std::unique(mapped.begin(), mapped.end()), mapped.end());
        if (has_rho) root_attach = std::move(mapped);
        else if (!mapped.empty()) merged.push_back(std::move(mapped));
    }
    if (root_attach.empty()) return out;

    bool found_meta = false;
    for (const auto& comp : quotient_res.components) {
        bool has_meta = false;
        std::vector<int> mapped;
        for (int label : comp) {
            if (label == meta_label) {
                has_meta = true;
                continue;
            }
            if (1 <= label && label < static_cast<int>(quotient_new_to_old.size())) {
                mapped.push_back(quotient_new_to_old[label]);
            }
        }
        if (has_meta) {
            mapped.insert(mapped.end(), root_attach.begin(), root_attach.end());
            found_meta = true;
        }
        std::sort(mapped.begin(), mapped.end());
        mapped.erase(std::unique(mapped.begin(), mapped.end()), mapped.end());
        if (!mapped.empty()) merged.push_back(std::move(mapped));
    }
    if (!found_meta) return out;

    std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
        if (a.empty() || b.empty()) return a.size() < b.size();
        return a.front() < b.front();
    });
    out.complete = true;
    out.components = std::move(merged);
    return out;
}

SolveComponentsResult solve_components_recursive(const PaceInstance& inst, Clock::time_point deadline);

SolveComponentsResult solve_components_recursive(const ParsedInstance& data, Clock::time_point deadline) {
    if (g_terminate || Clock::now() >= deadline) return {};

    std::vector<int> cluster = choose_cluster_candidate(data);
    if (!cluster.empty()) {
        PaceInstance sub_inst = build_cluster_subinstance(data, cluster);
        PaceInstance quotient_inst = build_quotient_subinstance(data, cluster);
        if (!sub_inst.newick_lines.empty() && !quotient_inst.newick_lines.empty()) {
            SolveComponentsResult sub_res = solve_components_recursive(sub_inst, deadline);
            if (!sub_res.complete) return {};
            SolveComponentsResult quotient_res = solve_components_recursive(quotient_inst, deadline);
            if (!quotient_res.complete) return {};
            SolveComponentsResult combined = combine_cluster_results(
                data.leaf_count, cluster, sub_res, quotient_res
            );
            if (combined.complete) return combined;
        }
    }

    return solve_components_basic(data, deadline);
}

SolveComponentsResult solve_components_recursive(const PaceInstance& inst, Clock::time_point deadline) {
    ParsedInstance data;
    if (!parse_instance_data(inst, data)) return {};
    return solve_components_recursive(data, deadline);
}

std::vector<std::string> solve(const PaceInstance& inst) {
    if (inst.leaf_count <= 0) return singleton_forest(1);

    ParsedInstance top;
    if (!parse_instance_data(inst, top)) return {};

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

    SolveComponentsResult res = solve_components_recursive(top, soft_deadline);
    if (!res.complete || res.components.empty()) return {};
    return forest_from_partition(res.components, top.leaf_count, top.trees[0]);
}

} // namespace

int main() {
    std::signal(SIGTERM, sig_handler);
    std::signal(SIGINT, sig_handler);

    PaceInstance inst;
    if (!read_instance(inst)) {
        return 0;
    }

    std::vector<std::string> out;
    try {
        out = solve(inst);
    } catch (...) {
        out.clear();
    }

    if (!out.empty()) {
        publish_best_solution(out);
        write_best_once();
    }
    return 0;
}
