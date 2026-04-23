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

using Clock = std::chrono::steady_clock;

uint64_t mix64(uint64_t x);

struct PaceInstance {
    int tree_count = 0;
    int leaf_count = 0;
    std::vector<std::string> newick_lines;
};

struct SimpleTree {
    int root = -1;
    std::vector<std::vector<int>> children;
    std::vector<int> parent;
    std::vector<int> leaf_label; // 0 internal, 1..n leaf
};

struct TreeData {
    int root = -1;
    std::vector<int> parent;
    std::vector<int> child0;
    std::vector<int> child1;
    std::vector<char> is_leaf;
    std::vector<int> leaf_label;
};

struct ReducedInstance {
    SimpleTree t1;
    SimpleTree t2;
    std::vector<std::vector<int>> expansion; // reduced label -> original leaves
    int reduced_leaf_count = 0;
};

struct ReductionRuleStats {
    const char* name = "";
    uint64_t calls = 0;
    uint64_t applied = 0;
    uint64_t leaves_removed = 0;
    long long total_us = 0;
};

struct ReductionBuildStats {
    int input_leaves = 0;
    int output_leaves = 0;
    int outer_cycles = 0;
    ReductionRuleStats common_subtrees{"common_subtrees"};
    ReductionRuleStats rooted_chains{"rooted_chains"};
};

struct PayloadNode {
    int left = -1;
    int right = -1;
    int leaf_label = -1;
};

struct DynNode {
    int parent = -1;
    int left = -1;
    int right = -1;
    int label = 0; // current active pseudo-leaf label, 0 for internal
    bool active = true;

    // Expansion payload for active pseudo leaves.
    int payload_id = -1;
    int payload_size = 0;
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

bool build_tree_data(const SimpleTree& st, int n, TreeData& td) {
    int n_nodes = static_cast<int>(st.children.size());
    td.root = st.root;
    td.parent = st.parent;
    td.child0.assign(n_nodes, -1);
    td.child1.assign(n_nodes, -1);
    td.is_leaf.assign(n_nodes, 0);
    td.leaf_label = st.leaf_label;

    int leaves = 0;
    std::vector<int> seen(n + 1, 0);
    for (int u = 0; u < n_nodes; ++u) {
        if (st.children[u].empty()) {
            td.is_leaf[u] = 1;
            int x = st.leaf_label[u];
            if (x < 1 || x > n || seen[x]) return false;
            seen[x] = 1;
            ++leaves;
        } else {
            if (st.children[u].size() != 2) return false;
            td.child0[u] = st.children[u][0];
            td.child1[u] = st.children[u][1];
        }
    }
    if (leaves != n) return false;

    return true;
}

std::vector<int> simple_tree_leaf_labels(const SimpleTree& st) {
    std::vector<int> out;
    out.reserve(st.leaf_label.size());
    for (int x : st.leaf_label) {
        if (x > 0) out.push_back(x);
    }
    std::sort(out.begin(), out.end());
    return out;
}

int simple_tree_leaf_count(const SimpleTree& st) {
    int cnt = 0;
    for (int x : st.leaf_label) {
        if (x > 0) ++cnt;
    }
    return cnt;
}

bool reduction_profile_enabled() {
    static int enabled = []() {
        const char* env = std::getenv("STRIDE_REDUCTION_PROFILE");
        return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

void emit_reduction_profile(const ReductionBuildStats& stats) {
    if (!reduction_profile_enabled()) return;
    auto emit_rule = [&](const ReductionRuleStats& rule) {
        std::cerr
            << "REDUCE_RULE"
            << " name=" << rule.name
            << " calls=" << rule.calls
            << " applied=" << rule.applied
            << " leaves_removed=" << rule.leaves_removed
            << " us=" << rule.total_us
            << '\n';
    };
    std::cerr
        << "REDUCE"
        << " input_leaves=" << stats.input_leaves
        << " output_leaves=" << stats.output_leaves
        << " outer_cycles=" << stats.outer_cycles
        << '\n';
    emit_rule(stats.common_subtrees);
    emit_rule(stats.rooted_chains);
}

std::vector<int> gather_expanded_labels(
    const std::vector<int>& reduced_labels,
    const std::vector<std::vector<int>>& expansion
) {
    std::vector<int> out;
    for (int x : reduced_labels) {
        if (x >= 0 && x < static_cast<int>(expansion.size())) {
            out.insert(out.end(), expansion[static_cast<size_t>(x)].begin(), expansion[static_cast<size_t>(x)].end());
        } else {
            out.push_back(x);
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct SimpleNodeInfo {
    std::vector<int> leaves;
    std::string topo_key;
    std::string exact_key;
};

struct SimpleChainInfo {
    bool is_chain = false;
    std::vector<int> ordered_leaves;
};

int add_simple_node(SimpleTree& st, int lbl) {
    int id = static_cast<int>(st.children.size());
    st.children.push_back({});
    st.parent.push_back(-1);
    st.leaf_label.push_back(lbl);
    return id;
}

std::string join_ints_key(const std::vector<int>& vals) {
    std::string s;
    for (int x : vals) {
        s += std::to_string(x);
        s.push_back(',');
    }
    return s;
}

SimpleNodeInfo compute_simple_info_dfs(
    const SimpleTree& st,
    int u,
    std::vector<SimpleNodeInfo>& info
) {
    if (st.children[u].empty()) {
        SimpleNodeInfo cur;
        cur.leaves = {st.leaf_label[u]};
        cur.topo_key = "L" + std::to_string(st.leaf_label[u]);
        cur.exact_key = join_ints_key(cur.leaves) + "#" + cur.topo_key;
        info[static_cast<size_t>(u)] = cur;
        return cur;
    }

    auto a = compute_simple_info_dfs(st, st.children[u][0], info);
    auto b = compute_simple_info_dfs(st, st.children[u][1], info);
    SimpleNodeInfo cur;
    cur.leaves = a.leaves;
    cur.leaves.insert(cur.leaves.end(), b.leaves.begin(), b.leaves.end());
    std::sort(cur.leaves.begin(), cur.leaves.end());
    if (a.topo_key > b.topo_key) std::swap(a.topo_key, b.topo_key);
    cur.topo_key = "(" + a.topo_key + "|" + b.topo_key + ")";
    cur.exact_key = join_ints_key(cur.leaves) + "#" + cur.topo_key;
    info[static_cast<size_t>(u)] = cur;
    return cur;
}

SimpleChainInfo compute_simple_chain_dfs(
    const SimpleTree& st,
    int u,
    std::vector<SimpleChainInfo>& info
) {
    if (st.children[u].empty()) {
        SimpleChainInfo cur;
        info[static_cast<size_t>(u)] = cur;
        return cur;
    }

    int a = st.children[u][0];
    int b = st.children[u][1];
    SimpleChainInfo left_info = compute_simple_chain_dfs(st, a, info);
    SimpleChainInfo right_info = compute_simple_chain_dfs(st, b, info);
    bool a_leaf = st.children[a].empty();
    bool b_leaf = st.children[b].empty();

    if (a_leaf && b_leaf) {
        SimpleChainInfo cur;
        cur.is_chain = true;
        int la = st.leaf_label[a];
        int lb = st.leaf_label[b];
        if (la <= lb) cur.ordered_leaves = {la, lb};
        else cur.ordered_leaves = {lb, la};
        info[static_cast<size_t>(u)] = cur;
        return cur;
    }

    if (a_leaf == b_leaf) {
        SimpleChainInfo cur;
        info[static_cast<size_t>(u)] = cur;
        return cur;
    }

    int leaf_child = a_leaf ? a : b;
    SimpleChainInfo spine = a_leaf ? right_info : left_info;
    if (!spine.is_chain) {
        SimpleChainInfo cur;
        info[static_cast<size_t>(u)] = cur;
        return cur;
    }

    SimpleChainInfo cur;
    cur.is_chain = true;
    cur.ordered_leaves.reserve(spine.ordered_leaves.size() + 1);
    cur.ordered_leaves.push_back(st.leaf_label[leaf_child]);
    cur.ordered_leaves.insert(cur.ordered_leaves.end(), spine.ordered_leaves.begin(), spine.ordered_leaves.end());
    info[static_cast<size_t>(u)] = cur;
    return cur;
}

bool reduce_common_rooted_chains_once(
    SimpleTree& t1,
    SimpleTree& t2,
    std::vector<std::vector<int>>& expansion,
    int& next_label
) {
    constexpr int kMinReducibleChainLength = 3;
    std::vector<SimpleChainInfo> chain1(t1.children.size());
    std::vector<SimpleChainInfo> chain2(t2.children.size());
    compute_simple_chain_dfs(t1, t1.root, chain1);
    compute_simple_chain_dfs(t2, t2.root, chain2);

    struct Candidate {
        std::string key;
        int u1 = -1;
        int u2 = -1;
        int size = 0;
        std::vector<int> ordered_leaves;
    };

    std::unordered_map<std::string, int> map1;
    std::unordered_map<std::string, int> map2;
    for (int u = 0; u < static_cast<int>(t1.children.size()); ++u) {
        const auto& ci = chain1[static_cast<size_t>(u)];
        if (!ci.is_chain || static_cast<int>(ci.ordered_leaves.size()) < kMinReducibleChainLength) continue;
        map1.emplace(join_ints_key(ci.ordered_leaves), u);
    }
    for (int u = 0; u < static_cast<int>(t2.children.size()); ++u) {
        const auto& ci = chain2[static_cast<size_t>(u)];
        if (!ci.is_chain || static_cast<int>(ci.ordered_leaves.size()) < kMinReducibleChainLength) continue;
        map2.emplace(join_ints_key(ci.ordered_leaves), u);
    }

    std::vector<Candidate> candidates;
    candidates.reserve(std::min(map1.size(), map2.size()));
    for (const auto& kv : map1) {
        auto it = map2.find(kv.first);
        if (it == map2.end()) continue;
        const auto& leaves = chain1[static_cast<size_t>(kv.second)].ordered_leaves;
        candidates.push_back({kv.first, kv.second, it->second, static_cast<int>(leaves.size()), leaves});
    }
    if (candidates.empty()) return false;

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.size != b.size) return a.size > b.size;
        return a.key < b.key;
    });

    std::unordered_set<int> used_labels;
    std::unordered_map<std::string, int> selected;
    for (const auto& cand : candidates) {
        bool disjoint = true;
        for (int x : cand.ordered_leaves) {
            if (used_labels.count(x)) {
                disjoint = false;
                break;
            }
        }
        if (!disjoint) continue;
        int new_label = next_label++;
        if (new_label >= static_cast<int>(expansion.size())) {
            expansion.resize(static_cast<size_t>(new_label + 1));
        }
        expansion[static_cast<size_t>(new_label)] = gather_expanded_labels(cand.ordered_leaves, expansion);
        selected[cand.key] = new_label;
        for (int x : cand.ordered_leaves) used_labels.insert(x);
    }
    if (selected.empty()) return false;

    auto rebuild = [&](const SimpleTree& src, const std::vector<SimpleChainInfo>& info, auto&& rebuild_ref, int u, SimpleTree& out) -> int {
        const auto& cur = info[static_cast<size_t>(u)];
        if (cur.is_chain && static_cast<int>(cur.ordered_leaves.size()) >= kMinReducibleChainLength) {
            auto it = selected.find(join_ints_key(cur.ordered_leaves));
            if (it != selected.end()) {
                return add_simple_node(out, it->second);
            }
        }
        if (src.children[u].empty()) {
            return add_simple_node(out, src.leaf_label[u]);
        }
        int nu = add_simple_node(out, 0);
        int a = rebuild_ref(src, info, rebuild_ref, src.children[u][0], out);
        int b = rebuild_ref(src, info, rebuild_ref, src.children[u][1], out);
        out.children[nu].push_back(a);
        out.children[nu].push_back(b);
        out.parent[a] = nu;
        out.parent[b] = nu;
        return nu;
    };

    SimpleTree nt1;
    nt1.root = rebuild(t1, chain1, rebuild, t1.root, nt1);
    SimpleTree nt2;
    nt2.root = rebuild(t2, chain2, rebuild, t2.root, nt2);
    t1 = std::move(nt1);
    t2 = std::move(nt2);
    return true;
}

bool reduce_common_subtrees_once(
    SimpleTree& t1,
    SimpleTree& t2,
    std::vector<std::vector<int>>& expansion,
    int& next_label
) {
    std::vector<SimpleNodeInfo> info1(t1.children.size());
    std::vector<SimpleNodeInfo> info2(t2.children.size());
    compute_simple_info_dfs(t1, t1.root, info1);
    compute_simple_info_dfs(t2, t2.root, info2);

    std::unordered_map<std::string, int> map1;
    std::unordered_map<std::string, int> map2;
    for (int u = 0; u < static_cast<int>(t1.children.size()); ++u) {
        if (info1[static_cast<size_t>(u)].leaves.size() >= 2) {
            map1.emplace(info1[static_cast<size_t>(u)].exact_key, u);
        }
    }
    for (int u = 0; u < static_cast<int>(t2.children.size()); ++u) {
        if (info2[static_cast<size_t>(u)].leaves.size() >= 2) {
            map2.emplace(info2[static_cast<size_t>(u)].exact_key, u);
        }
    }

    struct Candidate {
        std::string key;
        int u1 = -1;
        int u2 = -1;
        int size = 0;
        std::vector<int> leaves;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(std::min(map1.size(), map2.size()));
    for (const auto& kv : map1) {
        auto it = map2.find(kv.first);
        if (it == map2.end()) continue;
        const auto& inf = info1[static_cast<size_t>(kv.second)];
        candidates.push_back({kv.first, kv.second, it->second, static_cast<int>(inf.leaves.size()), inf.leaves});
    }
    if (candidates.empty()) return false;

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.size != b.size) return a.size > b.size;
        return a.key < b.key;
    });

    std::unordered_set<int> used_labels;
    std::unordered_map<std::string, int> selected;
    for (const auto& cand : candidates) {
        bool disjoint = true;
        for (int x : cand.leaves) {
            if (used_labels.count(x)) {
                disjoint = false;
                break;
            }
        }
        if (!disjoint) continue;
        int new_label = next_label++;
        if (new_label >= static_cast<int>(expansion.size())) {
            expansion.resize(static_cast<size_t>(new_label + 1));
        }
        expansion[static_cast<size_t>(new_label)] = gather_expanded_labels(cand.leaves, expansion);
        selected[cand.key] = new_label;
        for (int x : cand.leaves) used_labels.insert(x);
    }

    if (selected.empty()) return false;

    auto rebuild = [&](const SimpleTree& src, const std::vector<SimpleNodeInfo>& info, auto&& rebuild_ref, int u, SimpleTree& out) -> int {
        const auto& cur = info[static_cast<size_t>(u)];
        auto it = selected.find(cur.exact_key);
        if (it != selected.end()) {
            return add_simple_node(out, it->second);
        }
        if (src.children[u].empty()) {
            return add_simple_node(out, src.leaf_label[u]);
        }
        int nu = add_simple_node(out, 0);
        int a = rebuild_ref(src, info, rebuild_ref, src.children[u][0], out);
        int b = rebuild_ref(src, info, rebuild_ref, src.children[u][1], out);
        out.children[nu].push_back(a);
        out.children[nu].push_back(b);
        out.parent[a] = nu;
        out.parent[b] = nu;
        return nu;
    };

    SimpleTree nt1;
    nt1.root = rebuild(t1, info1, rebuild, t1.root, nt1);
    SimpleTree nt2;
    nt2.root = rebuild(t2, info2, rebuild, t2.root, nt2);
    t1 = std::move(nt1);
    t2 = std::move(nt2);
    return true;
}

ReducedInstance build_reduced_instance(
    const SimpleTree& in1,
    const SimpleTree& in2,
    const std::vector<std::vector<int>>& base_expansion
);

ReducedInstance build_reduced_instance(const SimpleTree& in1, const SimpleTree& in2, int original_n) {
    std::vector<std::vector<int>> base_expansion(static_cast<size_t>(original_n + 1));
    for (int i = 1; i <= original_n; ++i) base_expansion[static_cast<size_t>(i)] = {i};
    return build_reduced_instance(in1, in2, base_expansion);
}

ReducedInstance build_reduced_instance(
    const SimpleTree& in1,
    const SimpleTree& in2,
    const std::vector<std::vector<int>>& base_expansion
) {
    ReductionBuildStats reduction_stats;
    ReducedInstance red;
    red.t1 = in1;
    red.t2 = in2;
    reduction_stats.input_leaves = simple_tree_leaf_count(red.t1);
    int max_label = 0;
    for (int x : red.t1.leaf_label) max_label = std::max(max_label, x);
    for (int x : red.t2.leaf_label) max_label = std::max(max_label, x);
    red.expansion = base_expansion;
    if (static_cast<int>(red.expansion.size()) <= max_label) {
        red.expansion.resize(static_cast<size_t>(max_label + 1));
    }
    int next_label = max_label + 1;

    auto apply_rule = [&](ReductionRuleStats& stats, auto&& fn) -> bool {
        ++stats.calls;
        int before = simple_tree_leaf_count(red.t1);
        auto started = Clock::now();
        bool changed = fn(red.t1, red.t2, red.expansion, next_label);
        stats.total_us += std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now() - started
        ).count();
        if (changed) {
            ++stats.applied;
            int after = simple_tree_leaf_count(red.t1);
            if (after < before) stats.leaves_removed += static_cast<uint64_t>(before - after);
        }
        return changed;
    };

    while (true) {
        bool changed = false;
        ++reduction_stats.outer_cycles;

        while (apply_rule(
            reduction_stats.common_subtrees,
            [&](SimpleTree& t1, SimpleTree& t2, std::vector<std::vector<int>>& expansion, int& nl) {
                return reduce_common_subtrees_once(t1, t2, expansion, nl);
            })) {
            changed = true;
        }

        if (apply_rule(
            reduction_stats.rooted_chains,
            [&](SimpleTree& t1, SimpleTree& t2, std::vector<std::vector<int>>& expansion, int& nl) {
                return reduce_common_rooted_chains_once(t1, t2, expansion, nl);
            })) {
            changed = true;
            continue;
        }

        if (!changed) break;
    }

    std::vector<int> labels = simple_tree_leaf_labels(red.t1);
    std::unordered_map<int, int> remap;
    std::vector<std::vector<int>> new_expansion(1);
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        remap[labels[static_cast<size_t>(i)]] = i + 1;
        new_expansion.push_back(red.expansion[static_cast<size_t>(labels[static_cast<size_t>(i)])]);
    }

    auto renumber_tree = [&](SimpleTree& st) {
        for (int& x : st.leaf_label) {
            if (x > 0) x = remap[x];
        }
    };
    renumber_tree(red.t1);
    renumber_tree(red.t2);
    red.expansion = std::move(new_expansion);
    red.reduced_leaf_count = static_cast<int>(labels.size());
    reduction_stats.output_leaves = red.reduced_leaf_count;
    emit_reduction_profile(reduction_stats);
    return red;
}

std::vector<std::vector<int>> singleton_partition_from_expansion(
    const std::vector<std::vector<int>>& expansion
) {
    std::vector<std::vector<int>> comps;
    for (size_t i = 1; i < expansion.size(); ++i) {
        if (expansion[i].empty()) continue;
        for (int x : expansion[i]) comps.push_back({x});
    }
    return comps;
}

std::vector<int> all_expanded_leaves(const std::vector<std::vector<int>>& expansion) {
    std::vector<int> leaves;
    for (size_t i = 1; i < expansion.size(); ++i) {
        leaves.insert(leaves.end(), expansion[i].begin(), expansion[i].end());
    }
    std::sort(leaves.begin(), leaves.end());
    leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
    return leaves;
}

std::vector<std::vector<int>> filter_expansion(
    const std::vector<std::vector<int>>& expansion,
    const std::vector<char>& keep
) {
    std::vector<std::vector<int>> out(expansion.size());
    size_t lim = std::min(expansion.size(), keep.size());
    for (size_t i = 1; i < lim; ++i) {
        if (keep[i]) out[i] = expansion[i];
    }
    return out;
}

bool find_common_cluster(
    const SimpleTree& t1,
    const SimpleTree& t2,
    std::vector<int>& cluster_labels
) {
    cluster_labels.clear();
    if (t1.children.empty() || t2.children.empty()) return false;
    int n = static_cast<int>(simple_tree_leaf_labels(t1).size());
    if (n < 6) return false;

    std::vector<SimpleNodeInfo> info1(t1.children.size());
    std::vector<SimpleNodeInfo> info2(t2.children.size());
    compute_simple_info_dfs(t1, t1.root, info1);
    compute_simple_info_dfs(t2, t2.root, info2);

    std::unordered_set<std::string> clusters2;
    for (int u = 0; u < static_cast<int>(t2.children.size()); ++u) {
        const auto& leaves = info2[static_cast<size_t>(u)].leaves;
        if (static_cast<int>(leaves.size()) >= 3 && static_cast<int>(leaves.size()) <= n - 3) {
            clusters2.insert(join_ints_key(leaves));
        }
    }

    int best_balance = std::numeric_limits<int>::max();
    int best_size = -1;
    for (int u = 0; u < static_cast<int>(t1.children.size()); ++u) {
        const auto& leaves = info1[static_cast<size_t>(u)].leaves;
        int sz = static_cast<int>(leaves.size());
        if (sz < 3 || sz > n - 3) continue;
        std::string key = join_ints_key(leaves);
        if (!clusters2.count(key)) continue;
        int balance = std::abs(n - 2 * sz);
        if (balance < best_balance || (balance == best_balance && sz > best_size)) {
            best_balance = balance;
            best_size = sz;
            cluster_labels = leaves;
        }
    }
    return !cluster_labels.empty();
}

int restrict_simple_tree_dfs(
    const SimpleTree& src,
    int u,
    const std::vector<char>& keep,
    SimpleTree& out
) {
    if (src.children[u].empty()) {
        int lbl = src.leaf_label[u];
        if (lbl >= 0 && lbl < static_cast<int>(keep.size()) && keep[static_cast<size_t>(lbl)]) {
            return add_simple_node(out, lbl);
        }
        return -1;
    }

    int a = restrict_simple_tree_dfs(src, src.children[u][0], keep, out);
    int b = restrict_simple_tree_dfs(src, src.children[u][1], keep, out);
    if (a == -1 && b == -1) return -1;
    if (a == -1) return b;
    if (b == -1) return a;
    int nu = add_simple_node(out, 0);
    out.children[nu].push_back(a);
    out.children[nu].push_back(b);
    out.parent[a] = nu;
    out.parent[b] = nu;
    return nu;
}

bool restrict_simple_tree(
    const SimpleTree& src,
    const std::vector<char>& keep,
    SimpleTree& out
) {
    out = SimpleTree{};
    int r = restrict_simple_tree_dfs(src, src.root, keep, out);
    if (r == -1) return false;
    out.root = r;
    return true;
}

std::vector<std::vector<int>> expand_reduced_components(
    const std::vector<std::vector<int>>& comps,
    const std::vector<std::vector<int>>& expansion
) {
    std::vector<std::vector<int>> out;
    out.reserve(comps.size());
    for (const auto& comp : comps) {
        std::vector<int> leaves;
        for (int x : comp) {
            if (x >= 0 && x < static_cast<int>(expansion.size())) {
                leaves.insert(leaves.end(), expansion[static_cast<size_t>(x)].begin(), expansion[static_cast<size_t>(x)].end());
            }
        }
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        if (!leaves.empty()) out.push_back(std::move(leaves));
    }
    return out;
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
            dn.payload_id = push_payload(dt, PayloadNode{-1, -1, st.leaf_label[u]});
            dn.payload_size = 1;
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
            -1
        },
        log
    );
    nn.payload_size = t.nodes[a].payload_size + t.nodes[b].payload_size;

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

int count_common_cherries(const DynamicTree& t1, const DynamicTree& f) {
    int cnt = 0;
    for_each_cherry_label_pair(t1, [&](int a, int b) {
        if (are_siblings_by_label(f, a, b)) ++cnt;
        return true;
    });
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

struct EliteSolution {
    int components = 0;
    uint64_t hash = 0;
    std::vector<int> comp_of_leaf;
};

bool is_medium_instance_size(int n) {
    return n >= 80 && n <= 350;
}

bool is_large_instance_size(int n) {
    return n > 350;
}

bool should_use_swapped_orientation(uint64_t iter, int n) {
    if (n < 24 || n >= 1000) return false;
    uint64_t mod = is_large_instance_size(n) ? 6ULL : 8ULL;
    return (mix64(iter ^ 0xa0761d6478bd642fULL) % mod) == 0ULL;
}

int elite_pool_limit_for_size(int n) {
    return is_medium_instance_size(n) ? 8 : 4;
}

double search_phase_fraction(
    Clock::time_point now,
    Clock::time_point start,
    Clock::time_point deadline
) {
    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - start).count();
    if (total_ns <= 0) return 1.0;
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
    if (elapsed_ns <= 0) return 0.0;
    if (elapsed_ns >= total_ns) return 1.0;
    return static_cast<double>(elapsed_ns) / static_cast<double>(total_ns);
}

bool configure_elite_guidance(
    const std::vector<EliteSolution>& elite_pool,
    uint64_t iter,
    Clock::time_point start,
    Clock::time_point deadline,
    const std::vector<int>*& elite_comp_map,
    double& elite_bonus
) {
    elite_comp_map = nullptr;
    elite_bonus = 0.0;
    if (elite_pool.empty()) return false;

    double phase = search_phase_fraction(Clock::now(), start, deadline);
    if (phase < 0.80) return false;

    bool use = false;
    if (phase >= 0.95) {
        use = true;
        elite_bonus = 6.0;
    } else if (phase >= 0.90) {
        use = ((iter & 3ULL) != 0ULL);
        elite_bonus = 4.5;
    } else {
        use = ((iter & 1ULL) != 0ULL);
        elite_bonus = 3.0;
    }
    if (!use) return false;

    const auto& elite = elite_pool[static_cast<size_t>(iter % elite_pool.size())];
    elite_comp_map = &elite.comp_of_leaf;
    return true;
}

uint64_t elite_partition_hash(const std::vector<int>& comp_of_leaf) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 1; i < comp_of_leaf.size(); ++i) {
        h ^= mix64((static_cast<uint64_t>(i) << 32) ^ static_cast<uint64_t>(comp_of_leaf[i] + 1));
        h *= 0x100000001b3ULL;
    }
    return h;
}

EliteSolution build_elite_solution(const std::vector<std::vector<int>>& comps, int n) {
    EliteSolution elite;
    elite.components = static_cast<int>(comps.size());
    elite.comp_of_leaf.assign(static_cast<size_t>(n + 1), 0);
    for (int i = 0; i < static_cast<int>(comps.size()); ++i) {
        for (int x : comps[static_cast<size_t>(i)]) {
            if (x >= 1 && x <= n) elite.comp_of_leaf[static_cast<size_t>(x)] = i + 1;
        }
    }
    elite.hash = elite_partition_hash(elite.comp_of_leaf);
    return elite;
}

void maybe_add_elite_solution(
    std::vector<EliteSolution>& pool,
    const std::vector<std::vector<int>>& comps,
    int n,
    int pool_limit
) {
    if (comps.empty()) return;
    EliteSolution elite = build_elite_solution(comps, n);
    for (const auto& cur : pool) {
        if (cur.hash == elite.hash && cur.comp_of_leaf == elite.comp_of_leaf) return;
    }
    pool.push_back(std::move(elite));
    std::sort(pool.begin(), pool.end(), [](const EliteSolution& a, const EliteSolution& b) {
        if (a.components != b.components) return a.components < b.components;
        return a.hash < b.hash;
    });
    if (pool_limit > 0 && static_cast<int>(pool.size()) > pool_limit) {
        pool.resize(static_cast<size_t>(pool_limit));
    }
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

void collect_active_postorder(const DynamicTree& t, std::vector<int>& order) {
    order.clear();
    if (t.root == -1 || t.root >= static_cast<int>(t.nodes.size()) || !t.nodes[t.root].active) return;

    std::vector<std::pair<int, int>> st;
    st.reserve(t.nodes.size());
    st.push_back({t.root, 0});
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

double score_cherry_candidate(const CherryCandidate& cand, int total_leaves) {
    if (is_medium_instance_size(total_leaves)) {
        return
            (cand.common ? 10.0 : 0.0) +
            (cand.same_component ? 0.0 : 5.0) -
            1.0 * static_cast<double>(cand.distance) -
            2.6 * static_cast<double>(cand.pendant_count) -
            0.012 * static_cast<double>(cand.conflict_mass) +
            3.5 * static_cast<double>(cand.immediate_gain) -
            0.004 * static_cast<double>(cand.component_size);
    }

    if (is_large_instance_size(total_leaves)) {
        return
            (cand.common ? 11.0 : 0.0) +
            (cand.same_component ? 0.0 : 4.75) -
            1.15 * static_cast<double>(cand.distance) -
            2.25 * static_cast<double>(cand.pendant_count) -
            0.014 * static_cast<double>(cand.conflict_mass) +
            3.6 * static_cast<double>(cand.immediate_gain) -
            0.005 * static_cast<double>(cand.component_size);
    }

    return
        (cand.common ? 12.0 : 0.0) +
        (cand.same_component ? 0.0 : 4.0) -
        1.5 * static_cast<double>(cand.distance) -
        2.0 * static_cast<double>(cand.pendant_count) -
        0.02 * static_cast<double>(cand.conflict_mass) +
        3.0 * static_cast<double>(cand.immediate_gain) -
        0.01 * static_cast<double>(cand.component_size);
}

CherryCandidate build_cherry_candidate(
    const DynamicTree& t1,
    const DynamicTree& f,
    const std::vector<int>& f_mass,
    int a,
    int b,
    int total_leaves
) {
    CherryCandidate cand;
    cand.a = a;
    cand.b = b;
    cand.common = are_siblings_by_label(f, a, b);
    cand.same_component = false;
    cand.immediate_gain = cand.common ? 2 : 0;

    cand.na = find_node_of_label(f, a);
    cand.nb = find_node_of_label(f, b);
    if (cand.na == -1 || cand.nb == -1) return cand;
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

    cand.score = score_cherry_candidate(cand, total_leaves);

    (void)t1;
    return cand;
}

void apply_cut_plan(DynamicTree& f, const std::vector<int>& cuts, UndoLog* log = nullptr) {
    for (int u : cuts) cut_edge_above(f, u, log);
}

double evaluate_reduced_state(
    const DynamicTree& t1,
    const DynamicTree& f,
    int total_leaves
) {
    auto f_mass = compute_active_leaf_masses(f);
    int comp_count = count_root_components(f);
    int sampled_pendants = 0;
    int sampled_conflict_mass = 0;
    int active_leaves = active_leaf_count(t1);

    int cherry_count = 0;
    int sampled = 0;
    int sample_cap = (is_medium_instance_size(total_leaves) || total_leaves <= 64)
        ? std::numeric_limits<int>::max()
        : (is_large_instance_size(total_leaves) ? 10 : 6);
    for_each_cherry_label_pair(t1, [&](int a, int b) {
        ++cherry_count;
        if (sampled >= sample_cap) return true;
        ++sampled;
        int na = find_node_of_label(f, a);
        int nb = find_node_of_label(f, b);
        if (na == -1 || nb == -1) return true;
        if (!is_active_leaf(f, na) || !is_active_leaf(f, nb)) return true;

        int ra = root_of(f, na);
        int rb = root_of(f, nb);
        if (ra == -1 || rb == -1) return true;

        if (ra != rb) {
            sampled_conflict_mass += std::min(f_mass[ra], f_mass[rb]);
            return true;
        }

        auto path = path_nodes(f, na, nb);
        auto pendants = pendant_children_on_path(f, path);
        sampled_pendants += static_cast<int>(pendants.size());
        for (int u : pendants) sampled_conflict_mass += f_mass[u];
        return true;
    });

    if (is_medium_instance_size(total_leaves)) {
        return
            -26.0 * static_cast<double>(comp_count) +
             1.10 * static_cast<double>(cherry_count) -
             1.30 * static_cast<double>(sampled_pendants) -
             0.015 * static_cast<double>(sampled_conflict_mass) -
             0.04 * static_cast<double>(active_leaves);
    }

    if (is_large_instance_size(total_leaves)) {
        return
            -21.0 * static_cast<double>(comp_count) +
             0.90 * static_cast<double>(cherry_count) -
             1.55 * static_cast<double>(sampled_pendants) -
             0.016 * static_cast<double>(sampled_conflict_mass);
    }

    return
        -20.0 * static_cast<double>(comp_count) +
         0.75 * static_cast<double>(cherry_count) -
         1.75 * static_cast<double>(sampled_pendants) -
         0.02 * static_cast<double>(sampled_conflict_mass);
}

std::vector<int> choose_cut_plan(
    DynamicTree& t1,
    DynamicTree& f,
    const CherryCandidate& cand,
    int next_label,
    uint64_t& rng,
    int total_leaves
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

        size_t individual_limit = is_medium_instance_size(total_leaves)
            ? std::min<size_t>(8, sorted.size())
            : std::min<size_t>(4, sorted.size());
        for (size_t i = 0; i < individual_limit; ++i) {
            plans.push_back({sorted[i]});
        }
        plans.push_back({cand.na});
        plans.push_back({cand.nb});
        if (cand.pendants.size() > 1) {
            std::vector<int> bundle;
            for (size_t i = 0; i < (sorted.size() + 1) / 2; ++i) bundle.push_back(sorted[i]);
            plans.push_back(std::move(bundle));
            if (is_medium_instance_size(total_leaves)) {
                plans.push_back(sorted);
            }
            if (cand.pendants.size() <= 6 || (is_medium_instance_size(total_leaves) && cand.pendants.size() <= 8)) {
                std::vector<int> classic{cand.na, cand.nb};
                classic.insert(classic.end(), cand.pendants.begin(), cand.pendants.end());
                plans.push_back(std::move(classic));
            }
        }
    }
    if (plans.empty()) return {};

    double best = -std::numeric_limits<double>::infinity();
    std::vector<int> best_indices;
    UndoLog undo;
    for (int i = 0; i < static_cast<int>(plans.size()); ++i) {
        size_t mark = undo.mark();
        int nl = next_label;
        apply_cut_plan(f, plans[i], &undo);
        contract_all_common_cherries(t1, f, nl, &undo);
        double val = evaluate_reduced_state(t1, f, total_leaves) - 0.05 * static_cast<double>(plans[i].size());
        undo.undo_to(mark);
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

std::string serialize_payload_canon(
    const DynamicTree& t,
    int payload_id,
    std::vector<std::string>& cache
) {
    if (payload_id < 0 || payload_id >= static_cast<int>(t.payloads.size())) return {};
    if (!cache[static_cast<size_t>(payload_id)].empty()) return cache[static_cast<size_t>(payload_id)];
    const auto& p = t.payloads[static_cast<size_t>(payload_id)];
    if (p.leaf_label != -1) {
        cache[static_cast<size_t>(payload_id)] = "L" + std::to_string(p.leaf_label);
        return cache[static_cast<size_t>(payload_id)];
    }
    std::string a = serialize_payload_canon(t, p.left, cache);
    std::string b = serialize_payload_canon(t, p.right, cache);
    if (a > b) std::swap(a, b);
    cache[static_cast<size_t>(payload_id)] = "(" + a + "," + b + ")";
    return cache[static_cast<size_t>(payload_id)];
}

std::string serialize_active_node_canon(
    const DynamicTree& t,
    int u,
    std::vector<std::string>& payload_cache
) {
    if (u == -1 || !t.nodes[u].active) return {};
    if (t.nodes[u].left == -1 && t.nodes[u].right == -1) {
        return serialize_payload_canon(t, t.nodes[u].payload_id, payload_cache);
    }
    std::string a = serialize_active_node_canon(t, t.nodes[u].left, payload_cache);
    std::string b = serialize_active_node_canon(t, t.nodes[u].right, payload_cache);
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a > b) std::swap(a, b);
    return "(" + a + "," + b + ")";
}

std::string serialize_active_forest_canon(const DynamicTree& t) {
    std::vector<std::string> payload_cache(t.payloads.size());
    std::vector<std::string> roots;
    roots.reserve(count_root_components(t));
    for (int u = 0; u < static_cast<int>(t.nodes.size()); ++u) {
        if (!t.nodes[u].active || t.nodes[u].parent != -1) continue;
        roots.push_back(serialize_active_node_canon(t, u, payload_cache));
    }
    std::sort(roots.begin(), roots.end());
    std::string out;
    for (const auto& s : roots) {
        out += s;
        out.push_back('|');
    }
    return out;
}

struct CanonHash {
    uint64_t a = 0;
    uint64_t b = 0;
};

bool operator==(const CanonHash& x, const CanonHash& y) {
    return x.a == y.a && x.b == y.b;
}

bool operator<(const CanonHash& x, const CanonHash& y) {
    if (x.a != y.a) return x.a < y.a;
    return x.b < y.b;
}

uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

CanonHash hash_leaf_label(int label) {
    uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(label + 1));
    return {
        mix64(0x9e3779b97f4a7c15ULL ^ x),
        mix64(0xc2b2ae3d27d4eb4fULL ^ rotl64(x, 17))
    };
}

CanonHash hash_unordered_pair(CanonHash x, CanonHash y, uint64_t tag) {
    if (y < x) std::swap(x, y);
    return {
        mix64(tag ^ x.a ^ rotl64(y.a, 11) ^ 0x243f6a8885a308d3ULL),
        mix64((tag << 1) ^ x.b ^ rotl64(y.b, 23) ^ 0x13198a2e03707344ULL)
    };
}

CanonHash hash_payload_canon(
    const DynamicTree& t,
    int payload_id,
    std::vector<CanonHash>& cache,
    std::vector<char>& seen
) {
    if (payload_id < 0 || payload_id >= static_cast<int>(t.payloads.size())) return {};
    if (seen[static_cast<size_t>(payload_id)]) return cache[static_cast<size_t>(payload_id)];
    seen[static_cast<size_t>(payload_id)] = 1;
    const auto& p = t.payloads[static_cast<size_t>(payload_id)];
    if (p.leaf_label != -1) {
        cache[static_cast<size_t>(payload_id)] = hash_leaf_label(p.leaf_label);
        return cache[static_cast<size_t>(payload_id)];
    }
    CanonHash a = hash_payload_canon(t, p.left, cache, seen);
    CanonHash b = hash_payload_canon(t, p.right, cache, seen);
    cache[static_cast<size_t>(payload_id)] = hash_unordered_pair(a, b, 0x94d049bb133111ebULL);
    return cache[static_cast<size_t>(payload_id)];
}

CanonHash hash_active_node_canon(
    const DynamicTree& t,
    int u,
    std::vector<CanonHash>& payload_cache,
    std::vector<char>& payload_seen
) {
    if (u == -1 || !t.nodes[u].active) return {};
    if (t.nodes[u].left == -1 && t.nodes[u].right == -1) {
        return hash_payload_canon(t, t.nodes[u].payload_id, payload_cache, payload_seen);
    }
    CanonHash a = hash_active_node_canon(t, t.nodes[u].left, payload_cache, payload_seen);
    CanonHash b = hash_active_node_canon(t, t.nodes[u].right, payload_cache, payload_seen);
    if (a == CanonHash{}) return b;
    if (b == CanonHash{}) return a;
    return hash_unordered_pair(a, b, 0xda942042e4dd58b5ULL);
}

CanonHash hash_active_forest_canon(const DynamicTree& t) {
    std::vector<CanonHash> payload_cache(t.payloads.size());
    std::vector<char> payload_seen(t.payloads.size(), 0);
    std::vector<CanonHash> roots;
    roots.reserve(count_root_components(t));
    for (int u = 0; u < static_cast<int>(t.nodes.size()); ++u) {
        if (!t.nodes[u].active || t.nodes[u].parent != -1) continue;
        roots.push_back(hash_active_node_canon(t, u, payload_cache, payload_seen));
    }
    std::sort(roots.begin(), roots.end());
    CanonHash h{0x6a09e667f3bcc909ULL, 0xbb67ae8584caa73bULL};
    for (const auto& root : roots) {
        h = {
            mix64(h.a ^ root.a ^ rotl64(root.b, 7)),
            mix64(h.b ^ root.b ^ rotl64(root.a, 19))
        };
    }
    return h;
}

struct ExactStateKey {
    CanonHash t1_hash;
    CanonHash f_hash;
    int active_leaves = 0;
    int forest_components = 0;
};

bool operator==(const ExactStateKey& x, const ExactStateKey& y) {
    return x.t1_hash == y.t1_hash &&
           x.f_hash == y.f_hash &&
           x.active_leaves == y.active_leaves &&
           x.forest_components == y.forest_components;
}

struct ExactStateKeyHash {
    size_t operator()(const ExactStateKey& key) const {
        uint64_t h = mix64(key.t1_hash.a ^ rotl64(key.t1_hash.b, 9));
        h ^= mix64(key.f_hash.a ^ rotl64(key.f_hash.b, 17));
        h ^= mix64((static_cast<uint64_t>(static_cast<uint32_t>(key.active_leaves)) << 32) ^
                   static_cast<uint32_t>(key.forest_components));
        return static_cast<size_t>(h);
    }
};

ExactStateKey exact_state_key(const DynamicTree& t1, const DynamicTree& f) {
    ExactStateKey key;
    key.t1_hash = hash_active_forest_canon(t1);
    key.f_hash = hash_active_forest_canon(f);
    key.active_leaves = active_leaf_count(t1);
    key.forest_components = count_root_components(f);
    return key;
}

struct ExactKernelResult {
    bool solved = false;
    std::vector<std::vector<int>> comps;
};

struct ExactSearchStats {
    int reduced_n = 0;
    int threshold = 24;
    int incumbent_before = 0;
    int incumbent_after = 0;
    long long elapsed_ms = 0;
    uint64_t nodes = 0;
    uint64_t memo_hits = 0;
    uint64_t bound_prunes = 0;
    uint64_t deadline_prunes = 0;
    bool triggered = false;
    bool improved = false;
};

bool exact_profile_enabled() {
    static int enabled = []() {
        const char* env = std::getenv("STRIDE_EXACT_PROFILE");
        return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

void emit_exact_profile(const char* phase, const ExactSearchStats& stats) {
    if (!exact_profile_enabled()) return;
    std::cerr
        << "EXACT"
        << " phase=" << phase
        << " reduced_n=" << stats.reduced_n
        << " threshold=" << stats.threshold
        << " triggered=" << (stats.triggered ? 1 : 0)
        << " incumbent_before=" << stats.incumbent_before
        << " incumbent_after=" << stats.incumbent_after
        << " improved=" << (stats.improved ? 1 : 0)
        << " ms=" << stats.elapsed_ms
        << " nodes=" << stats.nodes
        << " memo_hits=" << stats.memo_hits
        << " bound_prunes=" << stats.bound_prunes
        << " deadline_prunes=" << stats.deadline_prunes
        << '\n';
}

void collect_current_components(const DynamicTree& f, std::vector<std::vector<int>>& comps) {
    comps.clear();
    for (int u = 0; u < static_cast<int>(f.nodes.size()); ++u) {
        if (!f.nodes[u].active || f.nodes[u].parent != -1) continue;
        std::vector<int> leaves;
        collect_component_leaves(f, u, leaves);
        std::sort(leaves.begin(), leaves.end());
        leaves.erase(std::unique(leaves.begin(), leaves.end()), leaves.end());
        if (!leaves.empty()) comps.push_back(std::move(leaves));
    }
}

void exact_kernel_dfs(
    DynamicTree t1,
    DynamicTree f,
    int next_label,
    Clock::time_point deadline,
    int& best_components,
    std::vector<std::vector<int>>& best_comps,
    std::unordered_map<ExactStateKey, int, ExactStateKeyHash>& memo,
    ExactSearchStats* stats = nullptr
) {
    if (stats) ++stats->nodes;
    if (g_terminate || Clock::now() >= deadline) {
        if (stats) ++stats->deadline_prunes;
        return;
    }

    contract_all_common_cherries(t1, f, next_label);
    int current_components = count_root_components(f);
    if (current_components >= best_components) {
        if (stats) ++stats->bound_prunes;
        return;
    }

    ExactStateKey key = exact_state_key(t1, f);
    auto it = memo.find(key);
    if (it != memo.end() && it->second <= current_components) {
        if (stats) {
            ++stats->memo_hits;
            ++stats->bound_prunes;
        }
        return;
    }
    memo[key] = current_components;

    auto cherries = cherries_by_label(t1);
    if (cherries.empty() || active_leaf_count(t1) <= 2) {
        best_components = current_components;
        collect_current_components(f, best_comps);
        return;
    }

    auto f_mass = compute_active_leaf_masses(f);
    CherryCandidate best_cand;
    bool have = false;
    for (auto [a, b] : cherries) {
        auto cand = build_cherry_candidate(t1, f, f_mass, a, b, active_leaf_count(t1));
        if (!have || cand.score > best_cand.score) {
            best_cand = std::move(cand);
            have = true;
        }
    }
    if (!have || best_cand.na == -1 || best_cand.nb == -1) {
        return;
    }

    std::vector<std::vector<int>> plans;
    if (!best_cand.same_component) {
        plans.push_back({best_cand.na});
        plans.push_back({best_cand.nb});
    } else if (best_cand.pendant_count == 1) {
        plans.push_back({best_cand.pendants[0]});
    } else {
        plans.push_back({best_cand.na});
        plans.push_back({best_cand.nb});
        if (!best_cand.pendants.empty()) {
            plans.push_back(best_cand.pendants);
        }
    }

    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(plans.size());
    for (int i = 0; i < static_cast<int>(plans.size()); ++i) {
        UndoLog undo;
        size_t mark = undo.mark();
        int nl = next_label;
        apply_cut_plan(f, plans[static_cast<size_t>(i)], &undo);
        contract_all_common_cherries(t1, f, nl, &undo);
        double val = evaluate_reduced_state(t1, f, active_leaf_count(t1)) - 0.05 * static_cast<double>(plans[static_cast<size_t>(i)].size());
        undo.undo_to(mark);
        ranked.push_back({-val, i});
    }
    std::sort(ranked.begin(), ranked.end());

    for (const auto& rank : ranked) {
        const auto& plan = plans[static_cast<size_t>(rank.second)];
        DynamicTree nf = f;
        apply_cut_plan(nf, plan);
        exact_kernel_dfs(t1, nf, next_label, deadline, best_components, best_comps, memo, stats);
        if (g_terminate || Clock::now() >= deadline) return;
    }
}

ExactKernelResult maybe_solve_exact_kernel(
    const DynamicTree& dt1_base,
    const DynamicTree& dt2_base,
    int reduced_n,
    int heuristic_components,
    Clock::time_point soft_deadline,
    const char* phase = "initial"
) {
    ExactKernelResult res;
    ExactSearchStats stats;
    stats.reduced_n = reduced_n;
    stats.incumbent_before = heuristic_components;
    stats.incumbent_after = heuristic_components;
    if (reduced_n > stats.threshold) {
        emit_exact_profile(phase, stats);
        return res;
    }
    if (heuristic_components <= 1) {
        emit_exact_profile(phase, stats);
        return res;
    }
    if (heuristic_components - 1 > 18) {
        emit_exact_profile(phase, stats);
        return res;
    }
    if (Clock::now() + std::chrono::milliseconds(50) >= soft_deadline) {
        emit_exact_profile(phase, stats);
        return res;
    }

    int best_components = heuristic_components;
    std::vector<std::vector<int>> best_comps;
    std::unordered_map<ExactStateKey, int, ExactStateKeyHash> memo;
    auto exact_deadline = std::min(soft_deadline, Clock::now() + std::chrono::milliseconds(1200));
    stats.triggered = true;
    auto exact_start = Clock::now();
    ExactSearchStats* stats_ptr = exact_profile_enabled() ? &stats : nullptr;
    exact_kernel_dfs(
        dt1_base,
        dt2_base,
        reduced_n + 1,
        exact_deadline,
        best_components,
        best_comps,
        memo,
        stats_ptr
    );
    stats.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - exact_start
    ).count();
    stats.incumbent_after = best_components;
    if (!best_comps.empty() && best_components < heuristic_components) {
        res.solved = true;
        res.comps = std::move(best_comps);
        stats.improved = true;
    }
    emit_exact_profile(phase, stats);
    return res;
}

ThreeApproxResult run_three_approx(
    DynamicTree t1,
    DynamicTree f,
    Clock::time_point deadline,
    int n,
    int incumbent_components,
    uint64_t seed,
    const std::vector<int>* elite_comp_map = nullptr,
    double elite_bonus = 0.0
);

std::vector<std::vector<int>> reduced_singleton_components(int reduced_n) {
    std::vector<std::vector<int>> comps;
    comps.reserve(reduced_n);
    for (int i = 1; i <= reduced_n; ++i) comps.push_back({i});
    return comps;
}

std::vector<std::vector<int>> solve_direct_reduced(
    const ReducedInstance& reduced,
    Clock::time_point deadline,
    uint64_t seed_base
) {
    int reduced_n = reduced.reduced_leaf_count;
    if (reduced_n <= 0) return {};
    if (reduced_n == 1) return {{1}};
    if (reduced_n == 2) return {{1, 2}};

    DynamicTree dt1_base = build_dynamic_tree(reduced.t1);
    DynamicTree dt2_base = build_dynamic_tree(reduced.t2);

    std::vector<std::vector<int>> best_reduced = reduced_singleton_components(reduced_n);
    int best_components = static_cast<int>(best_reduced.size());

    auto exact = maybe_solve_exact_kernel(
        dt1_base,
        dt2_base,
        reduced_n,
        best_components,
        deadline,
        "initial"
    );
    if (exact.solved && !exact.comps.empty()) {
        best_reduced = exact.comps;
        best_components = static_cast<int>(best_reduced.size());
    }

    std::vector<EliteSolution> elite_pool;
    int elite_limit = elite_pool_limit_for_size(reduced_n);
    uint64_t iter = 0;
    auto search_start = Clock::now();

    while (!g_terminate && Clock::now() < deadline) {
        uint64_t seed = mix64(seed_base ^ iter);
        const std::vector<int>* elite_comp_map = nullptr;
        double elite_bonus = 0.0;
        configure_elite_guidance(elite_pool, iter, search_start, deadline, elite_comp_map, elite_bonus);
        bool swapped = should_use_swapped_orientation(iter, reduced_n);
        auto res = run_three_approx(
            swapped ? dt2_base : dt1_base,
            swapped ? dt1_base : dt2_base,
            deadline,
            reduced_n,
            best_components,
            seed,
            elite_comp_map,
            elite_bonus
        );
        if (!res.complete || res.comps.empty()) {
            ++iter;
            continue;
        }
        if (static_cast<int>(res.comps.size()) <= best_components + 2) {
            maybe_add_elite_solution(elite_pool, res.comps, reduced_n, elite_limit);
        }
        if (static_cast<int>(res.comps.size()) < best_components) {
            best_reduced = res.comps;
            best_components = static_cast<int>(best_reduced.size());
        }
        ++iter;
    }

    return best_reduced;
}

std::vector<std::vector<int>> solve_decomposed_candidate(
    const SimpleTree& in1,
    const SimpleTree& in2,
    const std::vector<std::vector<int>>& base_expansion,
    Clock::time_point deadline,
    uint64_t seed_base,
    int incumbent_components,
    int depth = 0
) {
    if (g_terminate || Clock::now() >= deadline) {
        return singleton_partition_from_expansion(base_expansion);
    }
    if (incumbent_components <= 1) {
        return singleton_partition_from_expansion(base_expansion);
    }

    ReducedInstance reduced = build_reduced_instance(in1, in2, base_expansion);
    if (reduced.reduced_leaf_count <= 0) return {};
    if (reduced.reduced_leaf_count <= 2) return {all_expanded_leaves(reduced.expansion)};

    if (depth < 8) {
        std::vector<int> cluster_labels;
        if (find_common_cluster(reduced.t1, reduced.t2, cluster_labels)) {
            std::vector<char> in_cluster(reduced.expansion.size(), 0);
            for (int x : cluster_labels) {
                if (x >= 0 && x < static_cast<int>(in_cluster.size())) in_cluster[static_cast<size_t>(x)] = 1;
            }
            std::vector<char> out_cluster = in_cluster;
            for (size_t i = 1; i < out_cluster.size(); ++i) out_cluster[i] = static_cast<char>(!out_cluster[i]);

            SimpleTree in_t1, in_t2, out_t1, out_t2;
            if (restrict_simple_tree(reduced.t1, in_cluster, in_t1) &&
                restrict_simple_tree(reduced.t2, in_cluster, in_t2) &&
                restrict_simple_tree(reduced.t1, out_cluster, out_t1) &&
                restrict_simple_tree(reduced.t2, out_cluster, out_t2)) {
                auto now = Clock::now();
                auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
                if (ms_left < 10) {
                    return singleton_partition_from_expansion(reduced.expansion);
                }
                long long inside_ms = std::max<long long>(
                    1,
                    ms_left * static_cast<long long>(cluster_labels.size()) /
                    std::max(1, reduced.reduced_leaf_count)
                );
                auto inside_deadline = now + std::chrono::milliseconds(inside_ms);
                auto inside = solve_decomposed_candidate(
                    in_t1,
                    in_t2,
                    filter_expansion(reduced.expansion, in_cluster),
                    inside_deadline,
                    mix64(seed_base ^ 0x91e10da5c79e7b1dULL),
                    std::max(1, incumbent_components - 1),
                    depth + 1
                );
                if (static_cast<int>(inside.size()) >= incumbent_components) {
                    return singleton_partition_from_expansion(reduced.expansion);
                }
                auto outside = solve_decomposed_candidate(
                    out_t1,
                    out_t2,
                    filter_expansion(reduced.expansion, out_cluster),
                    deadline,
                    mix64(seed_base ^ 0xbf58476d1ce4e5b9ULL),
                    std::max(1, incumbent_components - static_cast<int>(inside.size())),
                    depth + 1
                );
                inside.insert(inside.end(), outside.begin(), outside.end());
                if (static_cast<int>(inside.size()) >= incumbent_components) {
                    return singleton_partition_from_expansion(reduced.expansion);
                }
                return inside;
            }
        }
    }

    auto reduced_comps = solve_direct_reduced(reduced, deadline, seed_base);
    return expand_reduced_components(reduced_comps, reduced.expansion);
}

ThreeApproxResult run_three_approx(
    DynamicTree t1,
    DynamicTree f,
    Clock::time_point deadline,
    int n,
    int incumbent_components,
    uint64_t seed,
    const std::vector<int>* elite_comp_map,
    double elite_bonus
) {
    int next_label = n + 1;
    uint64_t rng = seed ^ 0x9e3779b97f4a7c15ULL;
    bool timed_out_or_terminated = false;
    bool medium_mode = is_medium_instance_size(n);

    // Exhaust common cherries before branching.
    contract_all_common_cherries(t1, f, next_label);

    while (!g_terminate && Clock::now() < deadline && active_leaf_count(t1) > 2) {
        // --- Step 1: remove singleton leaves in F that are roots ---
        // For each leaf in F that is a root, cut its corresponding edge in T1.
        for (int u = 0; u < static_cast<int>(f.nodes.size()); ++u) {
            if (!f.nodes[u].active) continue;
            if (f.nodes[u].left == -1 && f.nodes[u].right == -1 && f.nodes[u].parent == -1) {
                int lbl = f.nodes[u].label;
                int v = find_node_of_label(t1, lbl);
                if (v != -1) {
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

        size_t chosen_idx = 0;
        if (medium_mode || elite_comp_map) {
            auto f_mass = compute_active_leaf_masses(f);
            struct RankedCherry {
                double score = -std::numeric_limits<double>::infinity();
                size_t idx = 0;
            };
            std::vector<RankedCherry> ranked;
            ranked.reserve(cherries.size());
            for (size_t i = 0; i < cherries.size(); ++i) {
                auto [ca, cb] = cherries[i];
                auto cand = build_cherry_candidate(t1, f, f_mass, ca, cb, n);
                double score = cand.score;
                if (elite_comp_map && elite_comp_map->size() > static_cast<size_t>(n)) {
                    int ida = (*elite_comp_map)[static_cast<size_t>(ca)];
                    int idb = (*elite_comp_map)[static_cast<size_t>(cb)];
                    if (ida != 0 && ida == idb) score += elite_bonus;
                }
                ranked.push_back({score, i});
            }
            std::sort(ranked.begin(), ranked.end(), [](const RankedCherry& a, const RankedCherry& b) {
                if (a.score != b.score) return a.score > b.score;
                return a.idx < b.idx;
            });
            size_t top_k = medium_mode
                ? std::min<size_t>(4, ranked.size())
                : std::min<size_t>(elite_bonus >= 5.0 ? 2 : 3, ranked.size());
            rng = mix64(rng + 0x517cc1b727220a95ULL);
            chosen_idx = ranked[static_cast<size_t>(rng % top_k)].idx;
        } else {
            rng = mix64(rng + 0x517cc1b727220a95ULL);
            chosen_idx = static_cast<size_t>(rng % cherries.size());
        }
        auto [a, b] = cherries[chosen_idx];
        int na = find_node_of_label(f, a);
        int nb = find_node_of_label(f, b);
        if (na == -1 || nb == -1) break;
        if (!is_active_leaf(f, na) || !is_active_leaf(f, nb)) continue;

        int ra = root_of(f, na);
        int rb = root_of(f, nb);
        if (ra == -1 || rb == -1) break;

        if (ra != rb) {
            UndoLog undo;
            size_t mark = undo.mark();
            cut_edge_above(f, na, &undo);
            double ma = medium_mode ? evaluate_reduced_state(t1, f, n) : static_cast<double>(count_common_cherries(t1, f));
            undo.undo_to(mark);

            mark = undo.mark();
            cut_edge_above(f, nb, &undo);
            double mb = medium_mode ? evaluate_reduced_state(t1, f, n) : static_cast<double>(count_common_cherries(t1, f));
            undo.undo_to(mark);

            if (ma > mb) {
                cut_edge_above(f, na);
            } else if (mb > ma) {
                cut_edge_above(f, nb);
            } else {
                rng = mix64(rng + 0x94d049bb133111ebULL);
                if ((rng & 1ULL) == 0ULL) cut_edge_above(f, na);
                else cut_edge_above(f, nb);
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
                auto cand = build_cherry_candidate(t1, f, f_mass, a, b, n);
                auto plan = choose_cut_plan(t1, f, cand, next_label, rng, n);
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

    std::vector<SimpleTree> parsed;
    std::vector<TreeData> trees;
    parsed.reserve(2);
    trees.reserve(2);

    for (const auto& line : inst.newick_lines) {
        SimpleTree st;
        if (!NewickParser(line).parse(st)) return singleton_forest(n);
        TreeData td;
        if (!build_tree_data(st, n, td)) return singleton_forest(n);
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
    int best_components = static_cast<int>(best_out.size());

    const TreeData& original_t1 = trees[0];
    std::vector<std::vector<int>> identity_expansion(static_cast<size_t>(n + 1));
    for (int i = 1; i <= n; ++i) identity_expansion[static_cast<size_t>(i)] = {i};

    uint64_t base_seed = instance_seed(inst);
    auto decomp_deadline = std::min(
        soft_deadline,
        start + std::chrono::milliseconds(
            std::max<long long>(1, static_cast<long long>(timeout_sec * 1000.0 * 0.35))
        )
    );
    if (!g_terminate && Clock::now() + std::chrono::milliseconds(20) < decomp_deadline) {
        auto decomp_comps = solve_decomposed_candidate(
            parsed[0],
            parsed[1],
            identity_expansion,
            decomp_deadline,
            mix64(base_seed ^ 0x243f6a8885a308d3ULL),
            best_components
        );
        auto decomp_out = forest_from_partition(decomp_comps, n, original_t1);
        if (!decomp_out.empty() && static_cast<int>(decomp_out.size()) < best_components) {
            best_out = decomp_out;
            best_components = static_cast<int>(best_out.size());
            publish_best_solution(best_out);
        }
    }

    ReducedInstance reduced = build_reduced_instance(parsed[0], parsed[1], n);
    if (reduced.reduced_leaf_count <= 0) return best_out;

    DynamicTree dt1_base = build_dynamic_tree(reduced.t1);
    DynamicTree dt2_base = build_dynamic_tree(reduced.t2);
    int reduced_n = reduced.reduced_leaf_count;

    std::vector<EliteSolution> elite_pool;
    int elite_limit = elite_pool_limit_for_size(reduced_n);
    uint64_t iter = 0;
    while (!g_terminate && Clock::now() < soft_deadline) {
        uint64_t seed = mix64(base_seed ^ iter);
        const std::vector<int>* elite_comp_map = nullptr;
        double elite_bonus = 0.0;
        configure_elite_guidance(elite_pool, iter, start, soft_deadline, elite_comp_map, elite_bonus);
        bool swapped = should_use_swapped_orientation(iter, reduced_n);
        auto res = run_three_approx(
            swapped ? dt2_base : dt1_base,
            swapped ? dt1_base : dt2_base,
            soft_deadline,
            reduced_n,
            best_components,
            seed,
            elite_comp_map,
            elite_bonus
        );
        if (!res.complete || res.comps.empty()) {
            ++iter;
            continue;
        }
        if (static_cast<int>(res.comps.size()) <= best_components + 2) {
            maybe_add_elite_solution(elite_pool, res.comps, reduced_n, elite_limit);
        }
        auto expanded = expand_reduced_components(res.comps, reduced.expansion);
        auto out = forest_from_partition(expanded, n, original_t1);
        if (!out.empty() && static_cast<int>(out.size()) < best_components) {
            best_out = out;
            best_components = static_cast<int>(out.size());
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
