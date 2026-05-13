#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>
#include <functional>

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
    int active_leaf_count = 0;
    int root_component_count = 1;
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
        ActiveLeafCount,
        RootComponentCount,
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
                case Kind::ActiveLeafCount:
                    e.tree->active_leaf_count = e.old_value;
                    break;
                case Kind::RootComponentCount:
                    e.tree->root_component_count = e.old_value;
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

bool pace_profile_enabled() {
    static int enabled = []() {
        const char* env = std::getenv("PACE_PROFILE");
        return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

long long elapsed_ms_since(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
}

void emit_pace_profile(
    const char* pass,
    int original_n,
    int reduced_n,
    int score_before,
    int score_after,
    long long elapsed_ms,
    const char* reason = ""
) {
    if (!pace_profile_enabled()) return;
    std::cerr
        << "PACE_PROFILE"
        << " pass=" << pass
        << " original_n=" << original_n
        << " reduced_n=" << reduced_n
        << " before=" << score_before
        << " after=" << score_after
        << " improved=" << (score_after >= 0 && score_before >= 0 && score_after < score_before ? 1 : 0)
        << " ms=" << elapsed_ms;
    if (reason && reason[0] != '\0') {
        std::cerr << " reason=" << reason;
    }
    std::cerr << '\n';
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
    s.reserve(vals.size() * 8);
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
        candidates.push_back({kv.first, static_cast<int>(leaves.size()), leaves});
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
        int size = 0;
        std::vector<int> leaves;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(std::min(map1.size(), map2.size()));

    for (const auto& kv : map1) {
        auto it = map2.find(kv.first);
        if (it == map2.end()) continue;

        const auto& inf = info1[static_cast<size_t>(kv.second)];
        candidates.push_back({kv.first, static_cast<int>(inf.leaves.size()), inf.leaves});
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
    if (n < 4) return false;
    int min_cluster = (n > 1000 ? 2 : 3);

    std::vector<SimpleNodeInfo> info1(t1.children.size());
    std::vector<SimpleNodeInfo> info2(t2.children.size());
    compute_simple_info_dfs(t1, t1.root, info1);
    compute_simple_info_dfs(t2, t2.root, info2);

    std::unordered_set<std::string> clusters2;
    for (int u = 0; u < static_cast<int>(t2.children.size()); ++u) {
        const auto& leaves = info2[static_cast<size_t>(u)].leaves;
        if (static_cast<int>(leaves.size()) >= min_cluster &&
            static_cast<int>(leaves.size()) <= n - min_cluster) {
            clusters2.insert(join_ints_key(leaves));
        }
    }

    double best_score = -1e100;
    int best_size = -1;
    for (int u = 0; u < static_cast<int>(t1.children.size()); ++u) {
        const auto& leaves = info1[static_cast<size_t>(u)].leaves;
        int sz = static_cast<int>(leaves.size());
        if (sz < min_cluster || sz > n - min_cluster) continue;
        std::string key = join_ints_key(leaves);
        if (!clusters2.count(key)) continue;
        double score = static_cast<double>(std::min(sz, n - sz))
                    - 0.15 * static_cast<double>(std::abs(n - 2 * sz));

        if (score > best_score || (score == best_score && sz > best_size)) {
            best_score = score;
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
            ++dt.active_leaf_count;
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
    dt.root_component_count = 1;
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

    // Cutting one edge creates one additional forest component.
    push_undo(log, UndoLog::Kind::RootComponentCount, &t, -1, t.root_component_count);
    ++t.root_component_count;

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

    // Two active leaves become one active leaf.
    push_undo(log, UndoLog::Kind::ActiveLeafCount, &t, -1, t.active_leaf_count);
    --t.active_leaf_count;

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

void fill_cherries_by_label(const DynamicTree& t, std::vector<std::pair<int, int>>& out) {
    out.clear();
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

bool contract_one_common_cherry(
    DynamicTree& t1,
    DynamicTree& t2,
    int& next_label,
    UndoLog* log = nullptr
) {
    for (int u : t1.cherry_nodes) {
        if (u < 0 || u >= static_cast<int>(t1.nodes.size())) continue;
        if (!t1.nodes[static_cast<size_t>(u)].active) continue;

        int l = t1.nodes[static_cast<size_t>(u)].left;
        int r = t1.nodes[static_cast<size_t>(u)].right;

        if (!is_active_leaf(t1, l) || !is_active_leaf(t1, r)) continue;

        int a = t1.nodes[static_cast<size_t>(l)].label;
        int b = t1.nodes[static_cast<size_t>(r)].label;

        if (a > b) std::swap(a, b);

        if (!are_siblings_by_label(t2, a, b)) continue;

        int nl = next_label++;
        bool ok1 = contract_cherry(t1, a, b, nl, log);
        bool ok2 = contract_cherry(t2, a, b, nl, log);

        return ok1 && ok2;
    }

    return false;
}

bool contract_all_common_cherries(
    DynamicTree& t1,
    DynamicTree& t2,
    int& next_label,
    UndoLog* log = nullptr
) {
    bool any = false;

    while (contract_one_common_cherry(t1, t2, next_label, log)) {
        any = true;
    }

    return any;
}

int root_of(const DynamicTree& t, int u) {
    int x = u;
    while (x != -1 && t.nodes[x].active && t.nodes[x].parent != -1) x = t.nodes[x].parent;
    return x;
}

struct PathScratch {
    std::vector<int> mark;
    std::vector<int> pos;
    std::vector<int> upa;
    std::vector<int> upb;
    std::vector<int> result;
    int stamp = 1;

    void ensure_size(size_t n) {
        if (mark.size() < n) {
            mark.assign(n, 0);
            pos.assign(n, -1);
        }
    }
};

const std::vector<int>& path_nodes_fast(
    const DynamicTree& t,
    int a,
    int b,
    PathScratch& scratch
) {
    scratch.ensure_size(t.nodes.size());

    ++scratch.stamp;
    if (scratch.stamp == std::numeric_limits<int>::max()) {
        std::fill(scratch.mark.begin(), scratch.mark.end(), 0);
        scratch.stamp = 1;
    }

    scratch.upa.clear();
    scratch.upb.clear();
    scratch.result.clear();

    int x = a;
    while (x != -1) {
        scratch.mark[static_cast<size_t>(x)] = scratch.stamp;
        scratch.pos[static_cast<size_t>(x)] = static_cast<int>(scratch.upa.size());
        scratch.upa.push_back(x);
        x = t.nodes[static_cast<size_t>(x)].parent;
    }

    x = b;
    int lca = -1;
    while (x != -1) {
        if (scratch.mark[static_cast<size_t>(x)] == scratch.stamp) {
            lca = x;
            break;
        }
        scratch.upb.push_back(x);
        x = t.nodes[static_cast<size_t>(x)].parent;
    }

    if (lca == -1) {
        return scratch.result;
    }

    int ia = scratch.pos[static_cast<size_t>(lca)];
    for (int i = 0; i <= ia; ++i) {
        scratch.result.push_back(scratch.upa[static_cast<size_t>(i)]);
    }

    for (int i = static_cast<int>(scratch.upb.size()) - 1; i >= 0; --i) {
        scratch.result.push_back(scratch.upb[static_cast<size_t>(i)]);
    }

    return scratch.result;
}

void fill_pendant_children_on_path(
    const DynamicTree& t,
    const std::vector<int>& path,
    std::vector<int>& out
) {
    out.clear();

    if (path.size() < 3) return;

    for (size_t i = 1; i + 1 < path.size(); ++i) {
        int u = path[i];
        int prev = path[i - 1];
        int next = path[i + 1];

        int l = t.nodes[u].left;
        int r = t.nodes[u].right;

        if (l != -1 && t.nodes[l].active && l != prev && l != next) out.push_back(l);
        if (r != -1 && t.nodes[r].active && r != prev && r != next) out.push_back(r);
    }
}

int count_common_cherries(const DynamicTree& t1, const DynamicTree& f) {
    int cnt = 0;
    for_each_cherry_label_pair(t1, [&](int a, int b) {
        if (are_siblings_by_label(f, a, b)) ++cnt;
        return true;
    });
    return cnt;
}

int exact_conflict_lower_bound(const DynamicTree& t1, const DynamicTree& f) {
    std::vector<char> used(f.nodes.size(), 0);
    PathScratch path_scratch;
    int lb = 0;
    for_each_cherry_label_pair(t1, [&](int a, int b) {
        if (are_siblings_by_label(f, a, b)) return true;
        int na = find_node_of_label(f, a);
        int nb = find_node_of_label(f, b);
        if (na == -1 || nb == -1) return true;
        if (!is_active_leaf(f, na) || !is_active_leaf(f, nb)) return true;

        int ra = root_of(f, na);
        int rb = root_of(f, nb);
        if (ra == -1 || rb == -1) return true;

        if (ra != rb) {
            if (used[static_cast<size_t>(ra)] || used[static_cast<size_t>(rb)]) return true;
            used[static_cast<size_t>(ra)] = 1;
            used[static_cast<size_t>(rb)] = 1;
            ++lb;
            return true;
        }

        const auto& path = path_nodes_fast(f, na, nb, path_scratch);
        std::vector<int> zone;
        for (size_t i = 1; i + 1 < path.size(); ++i) zone.push_back(path[i]);
        if (zone.empty()) {
            int p = f.nodes[na].parent;
            if (p == -1) return true;
            zone.push_back(p);
        }

        for (int u : zone) {
            if (u >= 0 && u < static_cast<int>(used.size()) && used[static_cast<size_t>(u)]) {
                return true;
            }
        }
        for (int u : zone) {
            if (u >= 0 && u < static_cast<int>(used.size())) used[static_cast<size_t>(u)] = 1;
        }
        ++lb;
        return true;
    });
    return lb;
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

int elite_pool_limit_for_size(int n) {
    return is_medium_instance_size(n) ? 8 : 4;
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

struct DeterministicRestartDeduper {
    std::vector<EliteSolution> seen_partitions;
    int duplicates = 0;

    void reserve(size_t count) {
        seen_partitions.reserve(count);
    }

    bool remember_if_new(const std::vector<std::vector<int>>& comps, int n) {
        if (comps.empty()) return true;

        EliteSolution signature = build_elite_solution(comps, n);
        for (const auto& cur : seen_partitions) {
            if (cur.hash == signature.hash &&
                cur.comp_of_leaf == signature.comp_of_leaf) {
                ++duplicates;
                return false;
            }
        }

        seen_partitions.push_back(std::move(signature));
        return true;
    }
};

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

enum class GreedyPolicy {
    Balanced,
    PreferDifferentComponent,
    PreferLowConflictMass,
    PreferFewPendants,
    PreferImmediateGain,
    ConservativeSingleCut,
    AggressiveMultiCut
};

struct DeterministicRunConfig {
    bool swapped = false;
    GreedyPolicy policy = GreedyPolicy::Balanced;
    std::vector<int> discrepancy;
    bool elite_guided = false;
    int cherry_sample_cap = 0;
    long long budget_ms = 0;
    uint64_t tie_salt = 0;
    int id = 0;
};

const char* greedy_policy_name(GreedyPolicy policy) {
    switch (policy) {
        case GreedyPolicy::Balanced: return "balanced";
        case GreedyPolicy::PreferDifferentComponent: return "diff_component";
        case GreedyPolicy::PreferLowConflictMass: return "low_conflict";
        case GreedyPolicy::PreferFewPendants: return "few_pendants";
        case GreedyPolicy::PreferImmediateGain: return "immediate_gain";
        case GreedyPolicy::ConservativeSingleCut: return "single_cut";
        case GreedyPolicy::AggressiveMultiCut: return "multi_cut";
    }
    return "unknown";
}

uint64_t discrepancy_hash(const std::vector<int>& script) {
    uint64_t h = 0x9e3779b97f4a7c15ULL ^ static_cast<uint64_t>(script.size());
    for (int x : script) {
        h = mix64(h ^ (static_cast<uint64_t>(static_cast<unsigned>(x + 17)) + 0x9e3779b97f4a7c15ULL));
    }
    return h;
}

std::string discrepancy_string(const std::vector<int>& script) {
    if (script.empty()) return "-";
    std::string out;
    for (size_t i = 0; i < script.size(); ++i) {
        if (i) out.push_back('.');
        out += std::to_string(script[i]);
    }
    return out;
}

uint64_t deterministic_pair_key(
    uint64_t salt,
    uint64_t step,
    int a,
    int b,
    uint64_t tag = 0
) {
    uint64_t lo = static_cast<uint64_t>(std::min(a, b));
    uint64_t hi = static_cast<uint64_t>(std::max(a, b));
    return mix64(
        salt ^
        (step * 0x9e3779b97f4a7c15ULL) ^
        (lo << 32) ^
        hi ^
        (tag * 0xbf58476d1ce4e5b9ULL)
    );
}

uint64_t deterministic_plan_key(uint64_t salt, const std::vector<int>& plan) {
    uint64_t h = salt ^ (static_cast<uint64_t>(plan.size()) * 0x94d049bb133111ebULL);
    for (int x : plan) {
        h = mix64(h ^ static_cast<uint64_t>(static_cast<unsigned>(x + 31)));
    }
    return h;
}

bool deterministic_profile_enabled() {
    static int enabled = []() {
        const char* env = std::getenv("STRIDE_DET_PROFILE");
        return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

void emit_deterministic_profile(
    const char* phase,
    const DeterministicRunConfig& cfg,
    int reduced_n,
    bool complete,
    int components,
    bool accepted,
    bool improved,
    long long elapsed_ms,
    bool duplicate = false
) {
    if (!deterministic_profile_enabled()) return;
    std::cerr
        << "DET"
        << " phase=" << phase
        << " id=" << cfg.id
        << " n=" << reduced_n
        << " policy=" << greedy_policy_name(cfg.policy)
        << " swapped=" << (cfg.swapped ? 1 : 0)
        << " elite=" << (cfg.elite_guided ? 1 : 0)
        << " script=" << discrepancy_string(cfg.discrepancy)
        << " sample=" << cfg.cherry_sample_cap
        << " budget_ms=" << cfg.budget_ms
        << " complete=" << (complete ? 1 : 0)
        << " components=" << components
        << " accepted=" << (accepted ? 1 : 0)
        << " improved=" << (improved ? 1 : 0)
        << " duplicate=" << (duplicate ? 1 : 0)
        << " ms=" << elapsed_ms
        << '\n';
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

struct LightCherryCandidate {
    int a = -1;
    int b = -1;
    int na = -1;
    int nb = -1;
    int ra = -1;
    int rb = -1;
    bool common = false;
    bool same_component = false;
    int distance = 0;
    int pendant_count = 0;
    int conflict_mass = 0;
    int component_size = 0;
    int immediate_gain = 0;
    double score = -std::numeric_limits<double>::infinity();
};

double score_cherry_candidate_normalized(const CherryCandidate& cand, int total_leaves) {
    if (total_leaves <= 0) total_leaves = 1;
    double inv = 1.0 / static_cast<double>(total_leaves);

    double norm_dist       = cand.distance * inv;
    double norm_pendants   = cand.pendant_count * inv;
    double norm_conflict   = cand.conflict_mass * inv;
    double norm_comp_size  = cand.component_size * inv;

    // Base score – common cherries are enormously valuable
    double score = (cand.common ? 12.0 : 0.0)
                + (cand.same_component ? 0.0 : 5.0)
                - 20.0 * norm_dist
                - 25.0 * norm_pendants
                - 12.0 * norm_conflict
                + 3.0 * static_cast<double>(cand.immediate_gain)
                - 2.0 * norm_comp_size;
    return score;
}   

double score_cherry_candidate_policy(
    const CherryCandidate& cand,
    int total_leaves,
    GreedyPolicy policy)
{
    double s = score_cherry_candidate_normalized(cand, total_leaves);

    switch (policy) {
        case GreedyPolicy::PreferDifferentComponent:
            if (!cand.same_component) s += 5.0;
            break;

        case GreedyPolicy::PreferLowConflictMass:
            s -= 8.0 * (cand.conflict_mass / static_cast<double>(total_leaves > 0 ? total_leaves : 1));
            break;

        case GreedyPolicy::PreferFewPendants:
            s -= 4.0 * (cand.pendant_count / static_cast<double>(total_leaves > 0 ? total_leaves : 1));
            break;

        case GreedyPolicy::PreferImmediateGain:
            s += 4.0 * static_cast<double>(cand.immediate_gain);
            break;

        case GreedyPolicy::ConservativeSingleCut:
            s -= 4.0 * (cand.pendant_count / static_cast<double>(total_leaves > 0 ? total_leaves : 1));
            s -= 6.0 * (cand.conflict_mass / static_cast<double>(total_leaves > 0 ? total_leaves : 1));
            if (cand.pendant_count <= 1) s += 2.0;
            break;

        case GreedyPolicy::AggressiveMultiCut:
            s += 3.0 * static_cast<double>(cand.immediate_gain);
            if (cand.pendant_count >= 2) s += 1.5;
            break;

        default:
            break;
    }
    return s;
}

CherryCandidate build_cherry_candidate(
    const DynamicTree& t1,
    const DynamicTree& f,
    const std::vector<int>& f_mass,
    int a,
    int b,
    int total_leaves,
    PathScratch* scratch = nullptr
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
        PathScratch local_scratch;
        PathScratch& ps = scratch ? *scratch : local_scratch;

        const auto& path_ref = path_nodes_fast(f, cand.na, cand.nb, ps);
        cand.path.assign(path_ref.begin(), path_ref.end());

        cand.distance = cand.path.empty() ? 0 : static_cast<int>(cand.path.size()) - 1;
        fill_pendant_children_on_path(f, cand.path, cand.pendants);
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

    cand.score = score_cherry_candidate_normalized(cand, total_leaves);

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
    PathScratch path_scratch;
    std::vector<int> pendants_scratch;
    int comp_count = f.root_component_count;
    int sampled_pendants = 0;
    int sampled_conflict_mass = 0;
    int active_leaves = t1.active_leaf_count;

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

        const auto& path = path_nodes_fast(f, na, nb, path_scratch);
        fill_pendant_children_on_path(f, path, pendants_scratch);
        sampled_pendants += static_cast<int>(pendants_scratch.size());
        for (int u : pendants_scratch) sampled_conflict_mass += f_mass[u];
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
    int total_leaves,
    const std::vector<int>& f_mass,
    GreedyPolicy policy,
    uint64_t tie_salt
) {
    std::vector<std::vector<int>> plans;
    if (!cand.same_component) {
        plans.push_back({cand.na});
        plans.push_back({cand.nb});
    } else {
        std::vector<int> sorted = cand.pendants;
        std::sort(sorted.begin(), sorted.end(), [&](int x, int y) {
            if (f_mass[x] != f_mass[y]) return f_mass[x] < f_mass[y];
            return x < y;
        });

        size_t individual_limit = is_medium_instance_size(total_leaves)
            ? std::min<size_t>(10, sorted.size())
            : std::min<size_t>(4, sorted.size());
        for (size_t i = 0; i < individual_limit; ++i) {
            plans.push_back({sorted[i]});
        }

        plans.push_back({cand.na});
        plans.push_back({cand.nb});

        if (policy != GreedyPolicy::ConservativeSingleCut) {
            if (sorted.size() >= 2) {
                plans.push_back({sorted[0], sorted[1]});
            }
            if (sorted.size() >= 3) {
                plans.push_back({sorted[0], sorted[1], sorted[2]});
            }
        }

        if (policy != GreedyPolicy::ConservativeSingleCut && cand.pendants.size() > 1) {
            std::vector<int> bundle;
            size_t bundle_limit = std::min<size_t>(5, sorted.size());
            for (size_t i = 0; i < bundle_limit; ++i) {
                bundle.push_back(sorted[i]);
            }
            plans.push_back(std::move(bundle));

            if (sorted.size() <= 10) {
                plans.push_back(sorted);
            }

            if (cand.pendants.size() <= 6 ||
                (is_medium_instance_size(total_leaves) && cand.pendants.size() <= 8)) {
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

    int chosen = best_indices.front();

    for (int idx : best_indices) {
        const auto& p = plans[static_cast<size_t>(idx)];
        const auto& q = plans[static_cast<size_t>(chosen)];

        // Prefer fewer cuts unless aggressive policy explicitly allows larger plans.
        if (policy != GreedyPolicy::AggressiveMultiCut && p.size() != q.size()) {
            if (p.size() < q.size()) chosen = idx;
            continue;
        }

        // In aggressive mode, allow larger plans if they were tied by evaluation,
        // but still make the choice stable.
        if (policy == GreedyPolicy::AggressiveMultiCut && p.size() != q.size()) {
            if (p.size() > q.size()) chosen = idx;
            continue;
        }

        uint64_t hp = deterministic_plan_key(tie_salt, p);
        uint64_t hq = deterministic_plan_key(tie_salt, q);
        if (hp != hq) {
            if (hp < hq) chosen = idx;
            continue;
        }
        if (p < q) chosen = idx;
    }

    return plans[static_cast<size_t>(chosen)];
}

bool use_expensive_same_component_plan(
    int total_leaves,
    const std::vector<int>& path,
    const std::vector<int>& pendants
) {
    if (total_leaves > 4000) return false;
    if (path.size() > 96) return false;

    if (is_medium_instance_size(total_leaves)) {
        return pendants.size() <= 32;
    }

    if (is_large_instance_size(total_leaves)) {
        return pendants.size() <= 14;
    }

    return pendants.size() <= 24;
}

struct ThreeApproxResult {
    std::vector<std::vector<int>> comps;
    bool complete = false;
};

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
    roots.reserve(static_cast<size_t>(std::max(0, t.root_component_count)));
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
    key.active_leaves = t1.active_leaf_count;
    key.forest_components = f.root_component_count;
    return key;
}

struct ExactKernelResult {
    bool solved = false;
    std::vector<std::vector<int>> comps;
};

struct ExactSearchStats {
    int reduced_n = 0;
    int gap_cap = 0;
    int root_lower_bound = 0;
    int incumbent_before = 0;
    int incumbent_after = 0;
    long long elapsed_ms = 0;
    uint64_t nodes = 0;
    uint64_t memo_hits = 0;
    uint64_t bound_prunes = 0;
    uint64_t lb_prunes = 0;
    uint64_t deadline_prunes = 0;
    uint64_t node_budget = 0;
    uint64_t node_budget_prunes = 0;
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
        << " gap_cap=" << stats.gap_cap
        << " root_lb=" << stats.root_lower_bound
        << " triggered=" << (stats.triggered ? 1 : 0)
        << " incumbent_before=" << stats.incumbent_before
        << " incumbent_after=" << stats.incumbent_after
        << " improved=" << (stats.improved ? 1 : 0)
        << " ms=" << stats.elapsed_ms
        << " nodes=" << stats.nodes
        << " node_budget=" << stats.node_budget
        << " memo_hits=" << stats.memo_hits
        << " bound_prunes=" << stats.bound_prunes
        << " lb_prunes=" << stats.lb_prunes
        << " deadline_prunes=" << stats.deadline_prunes
        << " node_budget_prunes=" << stats.node_budget_prunes
        << '\n';
}

struct ExactSearchControl {
    uint64_t node_budget = 0;
    uint64_t nodes = 0;
    bool aborted = false;
};

struct ExactRootBounds {
    int current_components = 0;
    int additional_lb = 0;
    int lower_bound_total = 0;
    int active_leaves = 0;
};

int exact_branch_plan_count(const CherryCandidate& cand) {
    if (!cand.same_component) return 2;
    if (cand.pendant_count == 1) return 1;
    return cand.pendants.empty() ? 2 : 3;
}

std::vector<std::vector<int>> build_exact_branch_plans(const CherryCandidate& cand) {
    std::vector<std::vector<int>> plans;
    if (!cand.same_component) {
        plans.push_back({cand.na});
        plans.push_back({cand.nb});
        return plans;
    }

    if (cand.pendant_count == 1) {
        plans.push_back({cand.pendants[0]});
        return plans;
    }

    plans.push_back({cand.na});
    plans.push_back({cand.nb});
    if (!cand.pendants.empty()) {
        plans.push_back(cand.pendants);
    }
    return plans;
}

int exact_branch_candidate_scan_limit(int active_leaves, int cherry_count) {
    if (cherry_count <= 0) return 0;
    if (cherry_count <= 6) return cherry_count;
    if (active_leaves <= 24) return std::min(cherry_count, 10);
    if (active_leaves <= 40) return std::min(cherry_count, 8);
    return std::min(cherry_count, 6);
}

struct ExactBranchPlanEval {
    std::vector<int> plan;
    double order_score = -std::numeric_limits<double>::infinity();
    int projected_components = 0;
    int projected_lb = 0;
    int projected_total = 0;
    int projected_active_leaves = 0;
    ExactStateKey state_key;
};

struct ExactBranchChoice {
    bool valid = false;
    CherryCandidate cand;
    std::vector<ExactBranchPlanEval> live_plans;
    int total_plans = 0;
    int pruned_plans = 0;
    int min_projected_total = std::numeric_limits<int>::max();
    int max_projected_total = 0;
    int sum_projected_total = 0;
    int sum_active_leaves = 0;
};

bool exact_branch_prefilter_better(const CherryCandidate& a, const CherryCandidate& b) {
    int ba = exact_branch_plan_count(a);
    int bb = exact_branch_plan_count(b);
    if (ba != bb) return ba < bb;
    if (a.immediate_gain != b.immediate_gain) return a.immediate_gain > b.immediate_gain;
    if (a.pendant_count != b.pendant_count) return a.pendant_count < b.pendant_count;
    if (a.conflict_mass != b.conflict_mass) return a.conflict_mass > b.conflict_mass;
    if (a.component_size != b.component_size) return a.component_size < b.component_size;
    if (std::abs(a.score - b.score) > 1e-9) return a.score > b.score;
    if (a.a != b.a) return a.a < b.a;
    return a.b < b.b;
}

bool exact_branch_choice_better(const ExactBranchChoice& a, const ExactBranchChoice& b) {
    if (a.live_plans.size() != b.live_plans.size()) {
        return a.live_plans.size() < b.live_plans.size();
    }
    if (a.pruned_plans != b.pruned_plans) {
        return a.pruned_plans > b.pruned_plans;
    }
    if (a.min_projected_total != b.min_projected_total) {
        return a.min_projected_total > b.min_projected_total;
    }
    if (a.sum_projected_total != b.sum_projected_total) {
        return a.sum_projected_total > b.sum_projected_total;
    }
    if (a.sum_active_leaves != b.sum_active_leaves) {
        return a.sum_active_leaves < b.sum_active_leaves;
    }
    int ba = exact_branch_plan_count(a.cand);
    int bb = exact_branch_plan_count(b.cand);
    if (ba != bb) return ba < bb;
    if (std::abs(a.cand.score - b.cand.score) > 1e-9) {
        return a.cand.score > b.cand.score;
    }
    if (a.cand.a != b.cand.a) return a.cand.a < b.cand.a;
    return a.cand.b < b.cand.b;
}

ExactBranchChoice choose_exact_branch(
    DynamicTree& t1,
    DynamicTree& f,
    int& next_label,
    int best_components,
    UndoLog& undo
) {
    ExactBranchChoice best_choice;
    PathScratch path_scratch;
    auto cherries = cherries_by_label(t1);
    if (cherries.empty()) return best_choice;

    auto f_mass = compute_active_leaf_masses(f);
    std::vector<CherryCandidate> candidates;
    candidates.reserve(cherries.size());
    for (auto [a, b] : cherries) {
        CherryCandidate cand = build_cherry_candidate(
            t1,
            f,
            f_mass,
            a,
            b,
            t1.active_leaf_count,
            &path_scratch
        );
        if (cand.na == -1 || cand.nb == -1) continue;
        candidates.push_back(std::move(cand));
    }
    if (candidates.empty()) return best_choice;

    std::sort(candidates.begin(), candidates.end(), exact_branch_prefilter_better);
    int scan_limit = exact_branch_candidate_scan_limit(
        t1.active_leaf_count,
        static_cast<int>(candidates.size())
    );

    for (int ci = 0; ci < scan_limit; ++ci) {
        const CherryCandidate& cand = candidates[static_cast<size_t>(ci)];
        auto plans = build_exact_branch_plans(cand);

        ExactBranchChoice choice;
        choice.valid = true;
        choice.cand = cand;
        choice.total_plans = static_cast<int>(plans.size());

        std::unordered_map<ExactStateKey, size_t, ExactStateKeyHash> seen_children;

        for (auto& plan : plans) {
            size_t mark = undo.mark();
            int saved_next_label = next_label;

            apply_cut_plan(f, plan, &undo);
            contract_all_common_cherries(t1, f, next_label, &undo);

            int current_components = f.root_component_count;
            int lb = exact_conflict_lower_bound(t1, f);
            int total = current_components + lb;

            if (current_components < best_components && total < best_components) {
                ExactBranchPlanEval eval;
                eval.plan = plan;
                eval.projected_components = current_components;
                eval.projected_lb = lb;
                eval.projected_total = total;
                eval.projected_active_leaves = t1.active_leaf_count;
                eval.order_score =
                    evaluate_reduced_state(t1, f, t1.active_leaf_count) -
                    0.05 * static_cast<double>(plan.size());
                eval.state_key = exact_state_key(t1, f);

                auto [it, inserted] = seen_children.emplace(
                    eval.state_key,
                    choice.live_plans.size()
                );
                if (inserted) {
                    choice.min_projected_total = std::min(choice.min_projected_total, total);
                    choice.max_projected_total = std::max(choice.max_projected_total, total);
                    choice.sum_projected_total += total;
                    choice.sum_active_leaves += eval.projected_active_leaves;
                    choice.live_plans.push_back(std::move(eval));
                } else {
                    auto& cur = choice.live_plans[it->second];
                    bool better =
                        eval.order_score > cur.order_score + 1e-9 ||
                        (std::abs(eval.order_score - cur.order_score) <= 1e-9 &&
                         eval.projected_components < cur.projected_components) ||
                        (std::abs(eval.order_score - cur.order_score) <= 1e-9 &&
                         eval.projected_components == cur.projected_components &&
                         eval.plan.size() < cur.plan.size()) ||
                        (std::abs(eval.order_score - cur.order_score) <= 1e-9 &&
                         eval.projected_components == cur.projected_components &&
                         eval.plan.size() == cur.plan.size() &&
                         eval.plan < cur.plan);
                    if (better) cur = std::move(eval);
                }
            } else {
                ++choice.pruned_plans;
            }

            next_label = saved_next_label;
            undo.undo_to(mark);
        }

        if (choice.live_plans.empty()) {
            return choice;
        }

        std::sort(choice.live_plans.begin(), choice.live_plans.end(), [](const ExactBranchPlanEval& x, const ExactBranchPlanEval& y) {
            if (std::abs(x.order_score - y.order_score) > 1e-9) return x.order_score > y.order_score;
            if (x.projected_components != y.projected_components) {
                return x.projected_components < y.projected_components;
            }
            if (x.projected_active_leaves != y.projected_active_leaves) {
                return x.projected_active_leaves < y.projected_active_leaves;
            }
            if (x.projected_total != y.projected_total) return x.projected_total < y.projected_total;
            if (x.plan.size() != y.plan.size()) return x.plan.size() < y.plan.size();
            return x.plan < y.plan;
        });

        if (!best_choice.valid || exact_branch_choice_better(choice, best_choice)) {
            best_choice = std::move(choice);
        }
    }

    return best_choice;
}

ExactRootBounds analyze_exact_root_bounds(
    const DynamicTree& dt1_base,
    const DynamicTree& dt2_base,
    int reduced_n
) {
    ExactRootBounds out;
    DynamicTree t1 = dt1_base;
    DynamicTree f = dt2_base;
    int next_label = reduced_n + 1;
    contract_all_common_cherries(t1, f, next_label);
    out.current_components = f.root_component_count;
    out.additional_lb = exact_conflict_lower_bound(t1, f);
    out.lower_bound_total = out.current_components + out.additional_lb;
    out.active_leaves = t1.active_leaf_count;
    return out;
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
    DynamicTree& t1,
    DynamicTree& f,
    int& next_label,
    Clock::time_point deadline,
    int& best_components,
    std::vector<std::vector<int>>& best_comps,
    std::unordered_map<ExactStateKey, int, ExactStateKeyHash>& memo,
    UndoLog& undo,
    ExactSearchControl& control,
    ExactSearchStats* stats = nullptr
) {
    size_t entry_mark = undo.mark();
    int entry_next_label = next_label;

    auto cleanup = [&]() {
        next_label = entry_next_label;
        undo.undo_to(entry_mark);
    };

    if (control.aborted) {
        cleanup();
        return;
    }

    if (control.node_budget != 0 && control.nodes >= control.node_budget) {
        control.aborted = true;
        if (stats) ++stats->node_budget_prunes;
        cleanup();
        return;
    }

    ++control.nodes;
    if (stats) ++stats->nodes;

    if (g_terminate || Clock::now() >= deadline) {
        if (stats) ++stats->deadline_prunes;
        cleanup();
        return;
    }

    contract_all_common_cherries(t1, f, next_label, &undo);

    int current_components = f.root_component_count;

    if (current_components >= best_components) {
        if (stats) ++stats->bound_prunes;
        cleanup();
        return;
    }

    int lb = exact_conflict_lower_bound(t1, f);
    if (current_components + lb >= best_components) {
        if (stats) ++stats->lb_prunes;
        cleanup();
        return;
    }

    ExactStateKey key = exact_state_key(t1, f);
    auto it = memo.find(key);
    if (it != memo.end() && it->second <= current_components) {
        if (stats) {
            ++stats->memo_hits;
            ++stats->bound_prunes;
        }
        cleanup();
        return;
    }

    memo[key] = current_components;

    if (t1.active_leaf_count <= 2) {
        best_components = current_components;
        collect_current_components(f, best_comps);
        cleanup();
        return;
    }

    ExactBranchChoice branch = choose_exact_branch(
        t1,
        f,
        next_label,
        best_components,
        undo
    );
    if (!branch.valid) {
        cleanup();
        return;
    }
    if (branch.live_plans.empty()) {
        if (stats) ++stats->lb_prunes;
        cleanup();
        return;
    }

    for (const auto& child : branch.live_plans) {
        const auto& plan = child.plan;
        size_t mark = undo.mark();
        int saved_next_label = next_label;

        apply_cut_plan(f, plan, &undo);
        exact_kernel_dfs(t1, f, next_label, deadline, best_components, best_comps, memo, undo, control, stats);

        next_label = saved_next_label;
        undo.undo_to(mark);
        if (control.aborted || g_terminate || Clock::now() >= deadline) {
            cleanup();
            return;
        }
    }
    cleanup();
}

ExactKernelResult solve_exact_kernel_bounded(
    const DynamicTree& dt1_base,
    const DynamicTree& dt2_base,
    int reduced_n,
    int incumbent_components,
    Clock::time_point exact_deadline,
    int gap_cap,
    int root_lower_bound,
    uint64_t node_budget,
    const char* phase
) {
    ExactKernelResult res;
    ExactSearchStats stats;
    stats.reduced_n = reduced_n;
    stats.gap_cap = gap_cap;
    stats.root_lower_bound = root_lower_bound;
    stats.incumbent_before = incumbent_components;
    stats.incumbent_after = incumbent_components;
    stats.node_budget = node_budget;
    if (incumbent_components <= root_lower_bound) {
        emit_exact_profile(phase, stats);
        return res;
    }
    if (Clock::now() + std::chrono::milliseconds(5) >= exact_deadline) {
        emit_exact_profile(phase, stats);
        return res;
    }

    int best_components = incumbent_components;
    std::vector<std::vector<int>> best_comps;
    std::unordered_map<ExactStateKey, int, ExactStateKeyHash> memo;
    stats.triggered = true;
    auto exact_start = Clock::now();
    ExactSearchStats* stats_ptr = exact_profile_enabled() ? &stats : nullptr;

    DynamicTree t1 = dt1_base;
    DynamicTree f = dt2_base;
    int next_label = reduced_n + 1;
    UndoLog undo;
    ExactSearchControl control;
    control.node_budget = node_budget;

    exact_kernel_dfs(
        t1,
        f,
        next_label,
        exact_deadline,
        best_components,
        best_comps,
        memo,
        undo,
        control,
        stats_ptr
    );
    stats.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - exact_start
    ).count();
    stats.incumbent_after = best_components;
    if (!best_comps.empty() && best_components < incumbent_components) {
        res.solved = true;
        res.comps = std::move(best_comps);
        stats.improved = true;
    }
    emit_exact_profile(phase, stats);
    return res;
}

ExactKernelResult maybe_solve_exact_kernel(
    const DynamicTree& dt1_base,
    const DynamicTree& dt2_base,
    int reduced_n,
    int heuristic_components,
    Clock::time_point soft_deadline,
    const char* phase = "initial",
    int gap_cap_override = 0,
    uint64_t node_budget_override = 0
) {
    ExactSearchStats stats;
    stats.reduced_n = reduced_n;
    stats.incumbent_before = heuristic_components;
    stats.incumbent_after = heuristic_components;
    if (heuristic_components <= 1) {
        emit_exact_profile(phase, stats);
        return {};
    }

    auto now = Clock::now();
    auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(soft_deadline - now).count();
    long long min_ms_needed = (gap_cap_override > 0 || node_budget_override > 0) ? 10 : 40;
    if (ms_left < min_ms_needed) {
        emit_exact_profile(phase, stats);
        return {};
    }

    ExactRootBounds root = analyze_exact_root_bounds(dt1_base, dt2_base, reduced_n);
    stats.root_lower_bound = root.lower_bound_total;

    int gap = heuristic_components - root.lower_bound_total;
    if (gap <= 0) {
        emit_exact_profile(phase, stats);
        return {};
    }

    int gap_cap = gap_cap_override;
    long long budget_ms = 0;
    uint64_t node_budget = node_budget_override;

    if (gap_cap <= 0) {
        if (gap <= 2) {
            gap_cap = 2;
            budget_ms = (reduced_n > 2000) ? 1700 : 1200;
            node_budget = 260000;
        } else if (gap <= 4) {
            gap_cap = 4;
            budget_ms = (reduced_n > 1200) ? 900 : 650;
            node_budget = 140000;
        } else if (gap <= 6 && heuristic_components <= 24) {
            gap_cap = 6;
            budget_ms = 420;
            node_budget = 70000;
        } else if (gap <= 8 && heuristic_components <= 14) {
            gap_cap = 8;
            budget_ms = 220;
            node_budget = 30000;
        } else if (gap <= 10 && heuristic_components <= 16) {
            gap_cap = 10;
            budget_ms = 260;
            node_budget = 45000;
        } else if (gap <= 12 && heuristic_components <= 10) {
            gap_cap = 12;
            budget_ms = 180;
            node_budget = 28000;
        } else {
            emit_exact_profile(phase, stats);
            return {};
        }
    } else if (node_budget == 0) {
        if (gap_cap <= 4) node_budget = 160000;
        else if (gap_cap <= 6) node_budget = 90000;
        else node_budget = 50000;
    }

    stats.gap_cap = gap_cap;
    stats.node_budget = node_budget;
    if (gap > gap_cap) {
        emit_exact_profile(phase, stats);
        return {};
    }

    if (budget_ms <= 0) {
        budget_ms = ms_left;
    }
    long long share_cap = std::max<long long>(40, ms_left / 3);
    budget_ms = std::max<long long>(40, std::min<long long>(budget_ms, std::min<long long>(2200, share_cap)));
    auto exact_deadline = std::min(soft_deadline, now + std::chrono::milliseconds(budget_ms));
    return solve_exact_kernel_bounded(
        dt1_base,
        dt2_base,
        reduced_n,
        heuristic_components,
        exact_deadline,
        gap_cap,
        root.lower_bound_total,
        node_budget,
        phase
    );
}

std::vector<std::vector<int>> reduced_identity_expansion(int n) {
    std::vector<std::vector<int>> expansion(static_cast<size_t>(n + 1));
    for (int i = 1; i <= n; ++i) expansion[static_cast<size_t>(i)] = {i};
    return expansion;
}

void normalize_partition(std::vector<std::vector<int>>& comps) {
    for (auto& comp : comps) {
        std::sort(comp.begin(), comp.end());
        comp.erase(std::unique(comp.begin(), comp.end()), comp.end());
    }
    comps.erase(
        std::remove_if(comps.begin(), comps.end(), [](const std::vector<int>& comp) {
            return comp.empty();
        }),
        comps.end()
    );
    std::sort(comps.begin(), comps.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        if (a.empty() || b.empty()) return a.size() < b.size();
        if (a.front() != b.front()) return a.front() < b.front();
        return a.size() < b.size();
    });
}

bool map_original_partition_to_reduced(
    const std::vector<std::vector<int>>& original_comps,
    const std::vector<std::vector<int>>& expansion,
    int original_n,
    std::vector<std::vector<int>>& reduced_comps
) {
    if (original_comps.empty() || original_n <= 0) return false;

    std::vector<int> comp_of_leaf(static_cast<size_t>(original_n + 1), -1);
    int seen = 0;
    for (size_t cid = 0; cid < original_comps.size(); ++cid) {
        for (int x : original_comps[cid]) {
            if (x < 1 || x > original_n) return false;
            int& slot = comp_of_leaf[static_cast<size_t>(x)];
            if (slot != -1) return false;
            slot = static_cast<int>(cid);
            ++seen;
        }
    }
    if (seen != original_n) return false;

    std::vector<std::vector<int>> mapped(original_comps.size());
    for (size_t lbl = 1; lbl < expansion.size(); ++lbl) {
        if (expansion[lbl].empty()) continue;
        int cid = -1;
        for (int x : expansion[lbl]) {
            if (x < 1 || x > original_n) return false;
            int owner = comp_of_leaf[static_cast<size_t>(x)];
            if (owner < 0) return false;
            if (cid < 0) {
                cid = owner;
            } else if (cid != owner) {
                return false;
            }
        }
        mapped[static_cast<size_t>(cid)].push_back(static_cast<int>(lbl));
    }

    normalize_partition(mapped);
    if (mapped.empty()) return false;
    reduced_comps = std::move(mapped);
    return true;
}

std::string restricted_key_simple_dfs(
    const SimpleTree& st,
    int u,
    const std::vector<char>& in_comp
) {
    if (st.children[static_cast<size_t>(u)].empty()) {
        int lbl = st.leaf_label[static_cast<size_t>(u)];
        return (lbl > 0 && lbl < static_cast<int>(in_comp.size()) && in_comp[static_cast<size_t>(lbl)])
            ? ("L" + std::to_string(lbl))
            : std::string();
    }

    std::string a = restricted_key_simple_dfs(st, st.children[static_cast<size_t>(u)][0], in_comp);
    std::string b = restricted_key_simple_dfs(st, st.children[static_cast<size_t>(u)][1], in_comp);

    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a > b) std::swap(a, b);
    return "(" + a + "," + b + ")";
}

int mark_component_edges_dfs(
    const SimpleTree& st,
    int u,
    const std::vector<char>& in_comp,
    int comp_size,
    int comp_id,
    std::vector<int>& edge_owner,
    bool& ok
) {
    if (!ok) return 0;

    if (st.children[static_cast<size_t>(u)].empty()) {
        int lbl = st.leaf_label[static_cast<size_t>(u)];
        return (lbl > 0 && lbl < static_cast<int>(in_comp.size()) && in_comp[static_cast<size_t>(lbl)]) ? 1 : 0;
    }

    int total = 0;
    for (int v : st.children[static_cast<size_t>(u)]) {
        int child_count = mark_component_edges_dfs(
            st,
            v,
            in_comp,
            comp_size,
            comp_id,
            edge_owner,
            ok
        );

        if (child_count > 0 && child_count < comp_size) {
            if (edge_owner[static_cast<size_t>(v)] != -1 &&
                edge_owner[static_cast<size_t>(v)] != comp_id) {
                ok = false;
                return 0;
            }
            edge_owner[static_cast<size_t>(v)] = comp_id;
        }

        total += child_count;
    }

    return total;
}

bool partition_is_valid_agreement_forest(
    const SimpleTree& t1,
    const SimpleTree& t2,
    const std::vector<std::vector<int>>& input_comps,
    int n
) {
    std::vector<std::vector<int>> comps = input_comps;
    normalize_partition(comps);

    std::vector<int> seen(static_cast<size_t>(n + 1), 0);
    int total_seen = 0;

    for (const auto& comp : comps) {
        if (comp.empty()) return false;
        for (int x : comp) {
            if (x < 1 || x > n) return false;
            if (++seen[static_cast<size_t>(x)] != 1) return false;
            ++total_seen;
        }
    }

    if (total_seen != n) return false;

    std::vector<int> edge_owner_t1(t1.children.size(), -1);
    std::vector<int> edge_owner_t2(t2.children.size(), -1);

    for (int cid = 0; cid < static_cast<int>(comps.size()); ++cid) {
        const auto& comp = comps[static_cast<size_t>(cid)];
        std::vector<char> in_comp(static_cast<size_t>(n + 1), 0);

        for (int x : comp) {
            in_comp[static_cast<size_t>(x)] = 1;
        }

        std::string k1 = restricted_key_simple_dfs(t1, t1.root, in_comp);
        std::string k2 = restricted_key_simple_dfs(t2, t2.root, in_comp);

        if (k1.empty() || k2.empty() || k1 != k2) {
            return false;
        }

        bool ok1 = true;
        bool ok2 = true;

        int c1 = mark_component_edges_dfs(
            t1,
            t1.root,
            in_comp,
            static_cast<int>(comp.size()),
            cid,
            edge_owner_t1,
            ok1
        );

        int c2 = mark_component_edges_dfs(
            t2,
            t2.root,
            in_comp,
            static_cast<int>(comp.size()),
            cid,
            edge_owner_t2,
            ok2
        );

        if (!ok1 || !ok2) return false;
        if (c1 != static_cast<int>(comp.size())) return false;
        if (c2 != static_cast<int>(comp.size())) return false;
    }

    return true;
}

bool partition_matches_union(
    const std::vector<std::vector<int>>& comps,
    const std::vector<int>& union_labels
) {
    std::vector<int> seen;
    for (const auto& comp : comps) {
        seen.insert(seen.end(), comp.begin(), comp.end());
    }
    std::sort(seen.begin(), seen.end());
    seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
    return seen == union_labels;
}

void collect_simple_leaf_order_dfs(const SimpleTree& st, int u, std::vector<int>& order) {
    if (u < 0) return;
    if (st.children[static_cast<size_t>(u)].empty()) {
        int lbl = st.leaf_label[static_cast<size_t>(u)];
        if (lbl > 0) order.push_back(lbl);
        return;
    }
    collect_simple_leaf_order_dfs(st, st.children[static_cast<size_t>(u)][0], order);
    collect_simple_leaf_order_dfs(st, st.children[static_cast<size_t>(u)][1], order);
}

std::vector<int> compute_simple_leaf_positions(const SimpleTree& st, int max_label) {
    std::vector<int> pos(static_cast<size_t>(max_label + 1), -1);
    std::vector<int> order;
    order.reserve(static_cast<size_t>(max_label));
    collect_simple_leaf_order_dfs(st, st.root, order);
    for (int i = 0; i < static_cast<int>(order.size()); ++i) {
        int lbl = order[static_cast<size_t>(i)];
        if (lbl >= 0 && lbl < static_cast<int>(pos.size())) pos[static_cast<size_t>(lbl)] = i;
    }
    return pos;
}

struct RepairComponentSummary {
    int comp_index = -1;
    int size = 0;
    int min_t1 = std::numeric_limits<int>::max();
    int max_t1 = std::numeric_limits<int>::min();
    int min_t2 = std::numeric_limits<int>::max();
    int max_t2 = std::numeric_limits<int>::min();
};

struct RepairCandidateGroup {
    std::vector<int> component_indices;
    int union_size = 0;
    int score = 0;
    int elite_merge_votes = 0;
    int elite_full_merge_votes = 0;
    int elite_instability = 0;
    bool from_elite = false;
};

struct LocalExactSearchPlan {
    int gap_cap = 0;
    std::chrono::milliseconds budget{0};
    uint64_t node_budget = 0;
};

std::string repair_group_key(const std::vector<int>& group) {
    std::string key;
    key.reserve(group.size() * 4);
    for (int x : group) {
        key += std::to_string(x);
        key.push_back(',');
    }
    return key;
}

void add_repair_candidates_from_order(
    const std::vector<int>& order,
    const std::vector<RepairComponentSummary>& summaries,
    int max_group_size,
    int leaf_limit,
    std::unordered_set<std::string>& seen,
    std::vector<RepairCandidateGroup>& out
) {
    for (size_t i = 0; i < order.size(); ++i) {
        std::vector<int> group;
        int union_size = 0;
        int min_t1 = std::numeric_limits<int>::max();
        int max_t1 = std::numeric_limits<int>::min();
        int min_t2 = std::numeric_limits<int>::max();
        int max_t2 = std::numeric_limits<int>::min();
        for (size_t len = 0; len < static_cast<size_t>(max_group_size) && i + len < order.size(); ++len) {
            const auto& summary = summaries[static_cast<size_t>(order[i + len])];
            group.push_back(summary.comp_index);
            union_size += summary.size;
            min_t1 = std::min(min_t1, summary.min_t1);
            max_t1 = std::max(max_t1, summary.max_t1);
            min_t2 = std::min(min_t2, summary.min_t2);
            max_t2 = std::max(max_t2, summary.max_t2);
            if (union_size > leaf_limit) break;
            if (group.size() < 2) continue;

            std::vector<int> key_group = group;
            std::sort(key_group.begin(), key_group.end());
            std::string key = repair_group_key(key_group);
            if (!seen.insert(key).second) continue;

            int gap_t1 = std::max(0, (max_t1 - min_t1 + 1) - union_size);
            int gap_t2 = std::max(0, (max_t2 - min_t2 + 1) - union_size);
            RepairCandidateGroup cand;
            cand.component_indices = std::move(key_group);
            cand.union_size = union_size;
            cand.score = union_size + 2 * (gap_t1 + gap_t2) + 3 * static_cast<int>(group.size() - 2);
            out.push_back(std::move(cand));
        }
    }
}

int repair_group_union_size(
    const std::vector<std::vector<int>>& comps,
    const std::vector<int>& group
) {
    int union_size = 0;
    for (int idx : group) {
        if (idx >= 0 && idx < static_cast<int>(comps.size())) {
            union_size += static_cast<int>(comps[static_cast<size_t>(idx)].size());
        }
    }
    return union_size;
}

std::vector<int> build_component_of_leaf(
    const std::vector<std::vector<int>>& comps,
    int n
) {
    std::vector<int> comp_of_leaf(static_cast<size_t>(n + 1), -1);
    for (int i = 0; i < static_cast<int>(comps.size()); ++i) {
        for (int x : comps[static_cast<size_t>(i)]) {
            if (x >= 1 && x <= n) {
                comp_of_leaf[static_cast<size_t>(x)] = i;
            }
        }
    }
    return comp_of_leaf;
}

LocalExactSearchPlan plan_local_exact_search(
    int reduced_n,
    int incumbent_components,
    bool final_phase,
    long long ms_left
) {
    LocalExactSearchPlan plan;
    int incumbent = std::max(1, incumbent_components);

    if (incumbent <= 2) {
        plan.gap_cap = final_phase ? 12 : 10;
        plan.budget = std::chrono::milliseconds(final_phase ? 120 : 80);
        plan.node_budget = final_phase ? 160000 : 120000;
    } else if (incumbent == 3) {
        plan.gap_cap = final_phase ? 10 : 8;
        plan.budget = std::chrono::milliseconds(final_phase ? 95 : 60);
        plan.node_budget = final_phase ? 120000 : 90000;
    } else if (incumbent == 4) {
        plan.gap_cap = final_phase ? 8 : 7;
        plan.budget = std::chrono::milliseconds(final_phase ? 75 : 50);
        plan.node_budget = final_phase ? 90000 : 65000;
    } else {
        plan.gap_cap = final_phase ? 7 : 6;
        plan.budget = std::chrono::milliseconds(final_phase ? 60 : 40);
        plan.node_budget = final_phase ? 70000 : 50000;
    }

    if (reduced_n <= 24) {
        plan.gap_cap += 2;
        plan.budget += std::chrono::milliseconds(final_phase ? 50 : 30);
        plan.node_budget += final_phase ? 30000 : 20000;
    } else if (reduced_n <= 32 && incumbent <= 5) {
        plan.gap_cap += 1;
        plan.budget += std::chrono::milliseconds(final_phase ? 20 : 10);
        plan.node_budget += final_phase ? 15000 : 10000;
    } else if (reduced_n >= 40) {
        plan.budget -= std::chrono::milliseconds(10);
        plan.node_budget = static_cast<uint64_t>(plan.node_budget * 85 / 100);
    }

    long long budget_ms = plan.budget.count();
    long long max_cap = final_phase ? 220 : 140;
    long long share_cap = std::max<long long>(15, ms_left / (final_phase ? 3 : 5));
    budget_ms = std::max<long long>(15, std::min<long long>(budget_ms, std::min(max_cap, share_cap)));
    plan.budget = std::chrono::milliseconds(budget_ms);
    return plan;
}

LocalExactSearchPlan plan_polish_exact_search(
    int reduced_n,
    int incumbent_components,
    long long ms_left
) {
    LocalExactSearchPlan plan;
    int incumbent = std::max(1, incumbent_components);

    if (reduced_n <= 24) {
        plan.gap_cap = 12;
        plan.budget = std::chrono::milliseconds(1200);
        plan.node_budget = 320000;
    } else if (incumbent <= 6) {
        plan.gap_cap = 10;
        plan.budget = std::chrono::milliseconds(900);
        plan.node_budget = 240000;
    } else if (incumbent <= 8) {
        plan.gap_cap = 8;
        plan.budget = std::chrono::milliseconds(650);
        plan.node_budget = 170000;
    } else if (incumbent <= 10) {
        plan.gap_cap = 6;
        plan.budget = std::chrono::milliseconds(450);
        plan.node_budget = 110000;
    } else if (incumbent <= 12) {
        plan.gap_cap = 5;
        plan.budget = std::chrono::milliseconds(300);
        plan.node_budget = 80000;
    } else {
        plan.gap_cap = 4;
        plan.budget = std::chrono::milliseconds(220);
        plan.node_budget = 55000;
    }

    if (reduced_n <= 32 && incumbent <= 5) {
        plan.budget = std::chrono::milliseconds(std::max<long long>(plan.budget.count(), 850));
        plan.node_budget = std::max<uint64_t>(plan.node_budget, 220000);
    }

    long long budget_ms = plan.budget.count();
    long long share_cap = std::max<long long>(40, ms_left / 4);
    budget_ms = std::max<long long>(40, std::min<long long>(budget_ms, std::min<long long>(1400, share_cap)));
    plan.budget = std::chrono::milliseconds(budget_ms);
    return plan;
}

void annotate_repair_candidate_with_elite_signal(
    RepairCandidateGroup& cand,
    const std::vector<std::vector<int>>& comps,
    const std::vector<EliteSolution>& elite_pool
) {
    if (elite_pool.empty() || cand.component_indices.empty()) return;

    std::vector<int> union_labels;
    union_labels.reserve(static_cast<size_t>(cand.union_size));
    for (int idx : cand.component_indices) {
        if (idx < 0 || idx >= static_cast<int>(comps.size())) continue;
        union_labels.insert(
            union_labels.end(),
            comps[static_cast<size_t>(idx)].begin(),
            comps[static_cast<size_t>(idx)].end()
        );
    }
    std::sort(union_labels.begin(), union_labels.end());
    union_labels.erase(std::unique(union_labels.begin(), union_labels.end()), union_labels.end());

    int merge_votes = 0;
    int full_merge_votes = 0;
    for (const auto& elite : elite_pool) {
        std::unordered_set<int> elite_ids;
        for (int x : union_labels) {
            if (x >= 1 && x < static_cast<int>(elite.comp_of_leaf.size())) {
                int id = elite.comp_of_leaf[static_cast<size_t>(x)];
                if (id != 0) elite_ids.insert(id);
            }
        }
        int parts = static_cast<int>(elite_ids.size());
        if (parts < static_cast<int>(cand.component_indices.size())) ++merge_votes;
        if (parts == 1) ++full_merge_votes;
    }

    cand.elite_merge_votes = merge_votes;
    cand.elite_full_merge_votes = full_merge_votes;
    cand.elite_instability = merge_votes * static_cast<int>(elite_pool.size() - merge_votes);
}

void add_elite_repair_candidates(
    const std::vector<std::vector<int>>& comps,
    const std::vector<EliteSolution>& elite_pool,
    int max_group_size,
    int leaf_limit,
    std::unordered_set<std::string>& seen,
    std::vector<RepairCandidateGroup>& out
) {
    if (elite_pool.empty()) return;

    for (const auto& elite : elite_pool) {
        std::unordered_map<int, std::vector<int>> buckets;
        buckets.reserve(comps.size());
        for (int i = 0; i < static_cast<int>(comps.size()); ++i) {
            std::unordered_set<int> elite_ids;
            elite_ids.reserve(comps[static_cast<size_t>(i)].size());
            for (int x : comps[static_cast<size_t>(i)]) {
                if (x >= 1 && x < static_cast<int>(elite.comp_of_leaf.size())) {
                    int id = elite.comp_of_leaf[static_cast<size_t>(x)];
                    if (id != 0) elite_ids.insert(id);
                }
            }
            for (int id : elite_ids) {
                buckets[id].push_back(i);
            }
        }

        for (auto& [elite_id, group] : buckets) {
            (void)elite_id;
            std::sort(group.begin(), group.end());
            group.erase(std::unique(group.begin(), group.end()), group.end());
            if (group.size() < 2 || static_cast<int>(group.size()) > max_group_size) continue;
            int union_size = repair_group_union_size(comps, group);
            if (union_size > leaf_limit) continue;

            std::string key = repair_group_key(group);
            if (!seen.insert(key).second) continue;

            RepairCandidateGroup cand;
            cand.component_indices = std::move(group);
            cand.union_size = union_size;
            cand.score = union_size + 2 * static_cast<int>(cand.component_indices.size() - 2);
            cand.from_elite = true;
            out.push_back(std::move(cand));
        }
    }
}

void add_cherry_conflict_repair_candidates(
    const SimpleTree& t1,
    const std::vector<std::vector<int>>& comps,
    int n,
    int max_group_size,
    int leaf_limit,
    std::unordered_set<std::string>& seen,
    std::vector<RepairCandidateGroup>& out
) {
    if (max_group_size < 2) return;

    auto comp_of_leaf = build_component_of_leaf(comps, n);

    std::function<void(int)> dfs = [&](int u) {
        if (u < 0 || u >= static_cast<int>(t1.children.size())) return;
        if (t1.children[static_cast<size_t>(u)].empty()) return;

        int a = t1.children[static_cast<size_t>(u)][0];
        int b = t1.children[static_cast<size_t>(u)][1];

        bool a_leaf = t1.children[static_cast<size_t>(a)].empty();
        bool b_leaf = t1.children[static_cast<size_t>(b)].empty();

        if (a_leaf && b_leaf) {
            int la = t1.leaf_label[static_cast<size_t>(a)];
            int lb = t1.leaf_label[static_cast<size_t>(b)];

            if (la >= 1 && la <= n && lb >= 1 && lb <= n) {
                int ca = comp_of_leaf[static_cast<size_t>(la)];
                int cb = comp_of_leaf[static_cast<size_t>(lb)];

                if (ca != -1 && cb != -1 && ca != cb) {
                    std::vector<int> group{ca, cb};
                    std::sort(group.begin(), group.end());

                    int union_size = repair_group_union_size(comps, group);
                    if (union_size <= leaf_limit) {
                        std::string key = repair_group_key(group);
                        if (seen.insert(key).second) {
                            RepairCandidateGroup cand;
                            cand.component_indices = std::move(group);
                            cand.union_size = union_size;
                            cand.score = union_size - 8; // bonus: directly repairs split cherry
                            out.push_back(std::move(cand));
                        }
                    }
                }
            }
        }

        dfs(a);
        dfs(b);
    };

    dfs(t1.root);
}

void add_witness_pair_repair_candidates(
    const SimpleTree& tree,
    const std::vector<std::vector<int>>& comps,
    int n,
    int leaf_limit,
    int append_limit,
    std::unordered_set<std::string>& seen,
    std::vector<RepairCandidateGroup>& out
) {
    if (tree.root < 0 || comps.size() < 2 || append_limit <= 0) return;

    auto comp_of_leaf = build_component_of_leaf(comps, n);

    struct SubtreeComponentSummary {
        int leaf_count = 0;
        int pure_comp = -1;
        std::vector<int> comps;
    };

    std::vector<SubtreeComponentSummary> summary(tree.children.size());
    std::vector<RepairCandidateGroup> local;
    local.reserve(static_cast<size_t>(std::max(8, append_limit * 2)));

    auto trim_component_list = [&](std::vector<int>& ids) {
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        std::sort(ids.begin(), ids.end(), [&](int a, int b) {
            int sa = (a >= 0 && a < static_cast<int>(comps.size()))
                ? static_cast<int>(comps[static_cast<size_t>(a)].size())
                : std::numeric_limits<int>::max();
            int sb = (b >= 0 && b < static_cast<int>(comps.size()))
                ? static_cast<int>(comps[static_cast<size_t>(b)].size())
                : std::numeric_limits<int>::max();
            if (sa != sb) return sa < sb;
            return a < b;
        });
        if (ids.size() > 10) ids.resize(10);
    };

    auto add_witness_pair = [&](
        int w_comp,
        int neighbor_comp,
        int w_leaf_count,
        int sibling_part_count,
        bool sibling_pure
    ) {
        if (w_comp == -1 || neighbor_comp == -1 || w_comp == neighbor_comp) return;

        std::vector<int> group{w_comp, neighbor_comp};
        std::sort(group.begin(), group.end());

        int union_size = repair_group_union_size(comps, group);
        if (union_size > leaf_limit) return;

        std::string key = repair_group_key(group);
        if (!seen.insert(key).second) return;

        RepairCandidateGroup cand;
        cand.component_indices = std::move(group);
        cand.union_size = union_size;

        // Witness pairs are less certain than split cherries, but a small
        // pure W-subtree beside a simple sibling side is a strong signal.
        cand.score = union_size - 5
                   - std::min(4, w_leaf_count)
                   + std::min(6, std::max(0, sibling_part_count - 1));
        if (sibling_pure) cand.score -= 2;

        local.push_back(std::move(cand));
    };

    std::function<void(int)> dfs = [&](int u) {
        if (u < 0 || u >= static_cast<int>(tree.children.size())) return;

        auto& cur = summary[static_cast<size_t>(u)];
        const auto& children = tree.children[static_cast<size_t>(u)];

        if (children.empty()) {
            int label = tree.leaf_label[static_cast<size_t>(u)];
            if (label >= 1 && label <= n) {
                int comp = comp_of_leaf[static_cast<size_t>(label)];
                if (comp != -1) {
                    cur.leaf_count = 1;
                    cur.pure_comp = comp;
                    cur.comps = {comp};
                }
            }
            return;
        }

        for (int child : children) dfs(child);

        cur.leaf_count = 0;
        cur.pure_comp = -1;
        cur.comps.clear();

        for (int child : children) {
            if (child < 0 || child >= static_cast<int>(summary.size())) continue;
            const auto& child_summary = summary[static_cast<size_t>(child)];
            cur.leaf_count += child_summary.leaf_count;
            cur.comps.insert(
                cur.comps.end(),
                child_summary.comps.begin(),
                child_summary.comps.end()
            );
        }

        if (cur.comps.empty()) return;

        std::sort(cur.comps.begin(), cur.comps.end());
        cur.comps.erase(std::unique(cur.comps.begin(), cur.comps.end()), cur.comps.end());
        if (cur.comps.size() == 1) cur.pure_comp = cur.comps.front();

        if (children.size() == 2) {
            int a = children[0];
            int b = children[1];
            if (a >= 0 && a < static_cast<int>(summary.size()) &&
                b >= 0 && b < static_cast<int>(summary.size())) {
                const auto& left = summary[static_cast<size_t>(a)];
                const auto& right = summary[static_cast<size_t>(b)];

                auto right_neighbors = right.comps;
                auto left_neighbors = left.comps;
                trim_component_list(right_neighbors);
                trim_component_list(left_neighbors);

                if (left.pure_comp != -1) {
                    for (int neighbor : right_neighbors) {
                        add_witness_pair(
                            left.pure_comp,
                            neighbor,
                            left.leaf_count,
                            static_cast<int>(right.comps.size()),
                            right.pure_comp != -1
                        );
                    }
                }

                if (right.pure_comp != -1) {
                    for (int neighbor : left_neighbors) {
                        add_witness_pair(
                            right.pure_comp,
                            neighbor,
                            right.leaf_count,
                            static_cast<int>(left.comps.size()),
                            left.pure_comp != -1
                        );
                    }
                }
            }
        }

        trim_component_list(cur.comps);
    };

    dfs(tree.root);

    std::sort(local.begin(), local.end(), [](const RepairCandidateGroup& a, const RepairCandidateGroup& b) {
        if (a.score != b.score) return a.score < b.score;
        if (a.union_size != b.union_size) return a.union_size < b.union_size;
        return a.component_indices < b.component_indices;
    });

    int limit = std::min<int>(
        static_cast<int>(local.size()),
        append_limit
    );
    for (int i = 0; i < limit; ++i) {
        out.push_back(std::move(local[static_cast<size_t>(i)]));
    }
}

std::vector<RepairCandidateGroup> build_repair_candidates(
    const ReducedInstance& reduced,
    const std::vector<std::vector<int>>& comps,
    const std::vector<EliteSolution>& elite_pool,
    int max_group_size,
    int leaf_limit,
    int max_candidates,
    bool append_witness_tail = false
) {
    if (comps.size() < 2 || max_group_size < 2) return {};

    auto pos_t1 = compute_simple_leaf_positions(reduced.t1, reduced.reduced_leaf_count);
    auto pos_t2 = compute_simple_leaf_positions(reduced.t2, reduced.reduced_leaf_count);

    std::vector<RepairComponentSummary> summaries(comps.size());
    for (int i = 0; i < static_cast<int>(comps.size()); ++i) {
        auto& summary = summaries[static_cast<size_t>(i)];
        summary.comp_index = i;
        summary.size = static_cast<int>(comps[static_cast<size_t>(i)].size());
        for (int x : comps[static_cast<size_t>(i)]) {
            if (x >= 0 && x < static_cast<int>(pos_t1.size()) && pos_t1[static_cast<size_t>(x)] != -1) {
                summary.min_t1 = std::min(summary.min_t1, pos_t1[static_cast<size_t>(x)]);
                summary.max_t1 = std::max(summary.max_t1, pos_t1[static_cast<size_t>(x)]);
            }
            if (x >= 0 && x < static_cast<int>(pos_t2.size()) && pos_t2[static_cast<size_t>(x)] != -1) {
                summary.min_t2 = std::min(summary.min_t2, pos_t2[static_cast<size_t>(x)]);
                summary.max_t2 = std::max(summary.max_t2, pos_t2[static_cast<size_t>(x)]);
            }
        }
        if (summary.min_t1 == std::numeric_limits<int>::max()) summary.min_t1 = summary.max_t1 = 0;
        if (summary.min_t2 == std::numeric_limits<int>::max()) summary.min_t2 = summary.max_t2 = 0;
    }

    std::vector<int> order_t1(comps.size());
    std::vector<int> order_t2(comps.size());
    std::iota(order_t1.begin(), order_t1.end(), 0);
    std::iota(order_t2.begin(), order_t2.end(), 0);
    std::sort(order_t1.begin(), order_t1.end(), [&](int a, int b) {
        if (summaries[static_cast<size_t>(a)].min_t1 != summaries[static_cast<size_t>(b)].min_t1) {
            return summaries[static_cast<size_t>(a)].min_t1 < summaries[static_cast<size_t>(b)].min_t1;
        }
        return summaries[static_cast<size_t>(a)].size < summaries[static_cast<size_t>(b)].size;
    });
    std::sort(order_t2.begin(), order_t2.end(), [&](int a, int b) {
        if (summaries[static_cast<size_t>(a)].min_t2 != summaries[static_cast<size_t>(b)].min_t2) {
            return summaries[static_cast<size_t>(a)].min_t2 < summaries[static_cast<size_t>(b)].min_t2;
        }
        return summaries[static_cast<size_t>(a)].size < summaries[static_cast<size_t>(b)].size;
    });

    std::unordered_set<std::string> seen;
    std::vector<RepairCandidateGroup> candidates;
    candidates.reserve(static_cast<size_t>(max_candidates * 3));
    add_repair_candidates_from_order(order_t1, summaries, max_group_size, leaf_limit, seen, candidates);
    add_repair_candidates_from_order(order_t2, summaries, max_group_size, leaf_limit, seen, candidates);

    add_cherry_conflict_repair_candidates(
        reduced.t1,
        comps,
        reduced.reduced_leaf_count,
        max_group_size,
        leaf_limit,
        seen,
        candidates
    );

    add_cherry_conflict_repair_candidates(
        reduced.t2,
        comps,
        reduced.reduced_leaf_count,
        max_group_size,
        leaf_limit,
        seen,
        candidates
    );

    add_elite_repair_candidates(comps, elite_pool, max_group_size, leaf_limit, seen, candidates);

    for (auto& cand : candidates) {
        annotate_repair_candidate_with_elite_signal(cand, comps, elite_pool);
        cand.score -= 2 * cand.elite_merge_votes + cand.elite_full_merge_votes;
    }

    std::sort(candidates.begin(), candidates.end(), [](const RepairCandidateGroup& a, const RepairCandidateGroup& b) {
        if (a.elite_instability != b.elite_instability) {
            return a.elite_instability > b.elite_instability;
        }
        if (a.elite_merge_votes != b.elite_merge_votes) {
            return a.elite_merge_votes > b.elite_merge_votes;
        }
        if (a.elite_full_merge_votes != b.elite_full_merge_votes) {
            return a.elite_full_merge_votes > b.elite_full_merge_votes;
        }
        if (a.from_elite != b.from_elite) {
            return a.from_elite > b.from_elite;
        }
        if (a.score != b.score) return a.score < b.score;
        if (a.union_size != b.union_size) return a.union_size < b.union_size;
        if (a.component_indices.size() != b.component_indices.size()) {
            return a.component_indices.size() < b.component_indices.size();
        }
        return a.component_indices < b.component_indices;
    });
    if (static_cast<int>(candidates.size()) > max_candidates) {
        candidates.resize(static_cast<size_t>(max_candidates));
    }

    if (append_witness_tail && max_group_size == 2) {
        int witness_limit = std::min<int>(6, std::max<int>(2, max_candidates / 4));

        add_witness_pair_repair_candidates(
            reduced.t1,
            comps,
            reduced.reduced_leaf_count,
            leaf_limit,
            witness_limit,
            seen,
            candidates
        );

        add_witness_pair_repair_candidates(
            reduced.t2,
            comps,
            reduced.reduced_leaf_count,
            leaf_limit,
            witness_limit,
            seen,
            candidates
        );
    }
    return candidates;
}

// -----------------------------------------------------------------------------
// Greedy TryAndMerge-style component merge pass.
// Conceptual port of the Java TryAndMerge idea:
// if two currently separated components can be merged while remaining a valid
// agreement forest in both trees, merge them immediately.
// -----------------------------------------------------------------------------

int greedy_merge_leaf_limit_for_size(int n, bool final_phase) {
    if (n > 4000) return final_phase ? 384 : 256;
    if (n > 1000) return final_phase ? 768 : 512;
    if (n > 600)  return final_phase ? 1024 : 768;
    if (n > 300)  return final_phase ? 1200 : 900;
    return final_phase ? 2000 : 1200;
}

int greedy_merge_candidate_limit_for_size(int n, bool final_phase) {
    if (n > 4000) return final_phase ? 80 : 48;
    if (n > 1000) return final_phase ? 96 : 64;
    if (n > 600)  return final_phase ? 120 : 80;
    return final_phase ? 160 : 96;
}

bool apply_greedy_merge_group(
    const ReducedInstance& reduced,
    const std::vector<std::vector<int>>& current_partition,
    const std::vector<int>& group_indices,
    std::vector<std::vector<int>>& next_partition
) {
    if (group_indices.size() < 2) return false;

    std::vector<int> sorted_indices = group_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    sorted_indices.erase(std::unique(sorted_indices.begin(), sorted_indices.end()), sorted_indices.end());

    if (sorted_indices.size() < 2) return false;

    std::vector<int> merged;
    for (int idx : sorted_indices) {
        if (idx < 0 || idx >= static_cast<int>(current_partition.size())) return false;
        merged.insert(
            merged.end(),
            current_partition[static_cast<size_t>(idx)].begin(),
            current_partition[static_cast<size_t>(idx)].end()
        );
    }

    std::sort(merged.begin(), merged.end());
    merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
    if (merged.empty()) return false;

    next_partition = current_partition;

    std::sort(sorted_indices.begin(), sorted_indices.end(), std::greater<int>());
    for (int idx : sorted_indices) {
        next_partition.erase(next_partition.begin() + idx);
    }

    next_partition.push_back(std::move(merged));
    normalize_partition(next_partition);

    if (static_cast<int>(next_partition.size()) >= static_cast<int>(current_partition.size())) {
        return false;
    }

    return partition_is_valid_agreement_forest(
        reduced.t1,
        reduced.t2,
        next_partition,
        reduced.reduced_leaf_count
    );
}

bool run_greedy_merge_pass(
    const ReducedInstance& reduced,
    std::vector<std::vector<int>>& best_reduced,
    int& best_components,
    std::vector<EliteSolution>& elite_pool,
    int elite_limit,
    Clock::time_point deadline,
    bool final_phase,
    int max_group_size
) {
    if (best_reduced.size() < 2) return false;
    if (g_terminate || Clock::now() + std::chrono::milliseconds(5) >= deadline) return false;

    int leaf_limit = greedy_merge_leaf_limit_for_size(reduced.reduced_leaf_count, final_phase);
    int max_candidates = greedy_merge_candidate_limit_for_size(reduced.reduced_leaf_count, final_phase);

    auto candidates = build_repair_candidates(
        reduced,
        best_reduced,
        elite_pool,
        max_group_size,
        leaf_limit,
        max_candidates,
        final_phase && max_group_size == 2
    );

    for (const auto& cand : candidates) {
        if (g_terminate || Clock::now() + std::chrono::milliseconds(5) >= deadline) break;

        std::vector<std::vector<int>> merged_partition;
        if (!apply_greedy_merge_group(
                reduced,
                best_reduced,
                cand.component_indices,
                merged_partition)) {
            continue;
        }

        int merged_components = static_cast<int>(merged_partition.size());
        if (merged_components < best_components) {
            best_reduced = std::move(merged_partition);
            best_components = merged_components;

            maybe_add_elite_solution(
                elite_pool,
                best_reduced,
                reduced.reduced_leaf_count,
                elite_limit
            );

            return true;
        }
    }

    return false;
}

bool run_greedy_merge_portfolio(
    const ReducedInstance& reduced,
    std::vector<std::vector<int>>& best_reduced,
    int& best_components,
    std::vector<EliteSolution>& elite_pool,
    int elite_limit,
    Clock::time_point deadline,
    bool final_phase
) {
    auto now = Clock::now();
    auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    if (ms_left < 20) return false;

    long long budget_ms = final_phase
        ? std::min<long long>(500, std::max<long long>(80, ms_left / 8))
        : std::min<long long>(160, std::max<long long>(30, ms_left / 18));

    auto merge_deadline = std::min(deadline, now + std::chrono::milliseconds(budget_ms));

    bool improved_any = false;

    while (!g_terminate && Clock::now() + std::chrono::milliseconds(5) < merge_deadline) {
        bool improved = false;

        improved = run_greedy_merge_pass(
            reduced,
            best_reduced,
            best_components,
            elite_pool,
            elite_limit,
            merge_deadline,
            final_phase,
            2
        );

        if (!improved && final_phase) {
            improved = run_greedy_merge_pass(
                reduced,
                best_reduced,
                best_components,
                elite_pool,
                elite_limit,
                merge_deadline,
                final_phase,
                3
            );
        }

        if (!improved) break;
        improved_any = true;
    }

    return improved_any;
}

bool apply_exact_repair_group(
    const ReducedInstance& reduced,
    const std::vector<std::vector<int>>& current_partition,
    const std::vector<int>& group_indices,
    int repair_leaf_limit,
    Clock::time_point deadline,
    bool final_phase,
    std::vector<std::vector<int>>& next_partition
) {
    if (group_indices.size() < 2) return false;
    if (Clock::now() + std::chrono::milliseconds(5) >= deadline) return false;

    std::vector<int> union_labels;
    for (int idx : group_indices) {
        if (idx < 0 || idx >= static_cast<int>(current_partition.size())) return false;
        union_labels.insert(
            union_labels.end(),
            current_partition[static_cast<size_t>(idx)].begin(),
            current_partition[static_cast<size_t>(idx)].end()
        );
    }
    std::sort(union_labels.begin(), union_labels.end());
    union_labels.erase(std::unique(union_labels.begin(), union_labels.end()), union_labels.end());
    if (static_cast<int>(union_labels.size()) > repair_leaf_limit) return false;

    std::vector<char> keep(reduced.expansion.size(), 0);
    for (int x : union_labels) {
        if (x > 0 && x < static_cast<int>(keep.size())) keep[static_cast<size_t>(x)] = 1;
    }

    SimpleTree local_t1, local_t2;
    if (!restrict_simple_tree(reduced.t1, keep, local_t1) ||
        !restrict_simple_tree(reduced.t2, keep, local_t2)) {
        return false;
    }

    auto identity_expansion = filter_expansion(reduced_identity_expansion(reduced.reduced_leaf_count), keep);
    ReducedInstance local_reduced = build_reduced_instance(local_t1, local_t2, identity_expansion);
    if (local_reduced.reduced_leaf_count <= 0 || local_reduced.reduced_leaf_count > repair_leaf_limit) {
        return false;
    }
    if (local_reduced.reduced_leaf_count == 1) {
        next_partition = current_partition;
        std::vector<int> erase_indices = group_indices;
        std::sort(erase_indices.begin(), erase_indices.end(), std::greater<int>());
        for (int idx : erase_indices) {
            next_partition.erase(next_partition.begin() + idx);
        }
        next_partition.push_back(union_labels);
        normalize_partition(next_partition);

        if (static_cast<int>(union_labels.size()) == 0 ||
            static_cast<int>(group_indices.size()) <= 1 ||
            !partition_is_valid_agreement_forest(
                reduced.t1,
                reduced.t2,
                next_partition,
                reduced.reduced_leaf_count)) {
            return false;
        }
        return true;
    }

    auto now = Clock::now();
    auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    LocalExactSearchPlan plan = plan_local_exact_search(
        local_reduced.reduced_leaf_count,
        static_cast<int>(group_indices.size()),
        final_phase,
        ms_left
    );
    auto exact_deadline = std::min(deadline, now + plan.budget);
    DynamicTree dt1_base = build_dynamic_tree(local_reduced.t1);
    DynamicTree dt2_base = build_dynamic_tree(local_reduced.t2);
    auto exact = maybe_solve_exact_kernel(
        dt1_base,
        dt2_base,
        local_reduced.reduced_leaf_count,
        static_cast<int>(group_indices.size()),
        exact_deadline,
        "repair",
        plan.gap_cap,
        plan.node_budget
    );
    if (!exact.solved || exact.comps.empty()) return false;

    auto local_expanded = expand_reduced_components(exact.comps, local_reduced.expansion);
    normalize_partition(local_expanded);
    if (static_cast<int>(local_expanded.size()) >= static_cast<int>(group_indices.size())) return false;
    if (!partition_matches_union(local_expanded, union_labels)) return false;

    next_partition = current_partition;
    std::vector<int> erase_indices = group_indices;
    std::sort(erase_indices.begin(), erase_indices.end(), std::greater<int>());
    for (int idx : erase_indices) {
        next_partition.erase(next_partition.begin() + idx);
    }
    next_partition.insert(next_partition.end(), local_expanded.begin(), local_expanded.end());
    normalize_partition(next_partition);

    if (!partition_is_valid_agreement_forest(
            reduced.t1,
            reduced.t2,
            next_partition,
            reduced.reduced_leaf_count)) {
        return false;
    }

    return true;
}

bool run_exact_repair_pass(
    const ReducedInstance& reduced,
    std::vector<std::vector<int>>& best_reduced,
    int& best_components,
    const std::vector<EliteSolution>& elite_pool,
    Clock::time_point deadline,
    int max_group_size,
    int repair_leaf_limit,
    bool final_phase,
    int max_candidates
) {
    auto candidates = build_repair_candidates(
        reduced,
        best_reduced,
        elite_pool,
        max_group_size,
        repair_leaf_limit,
        max_candidates,
        final_phase && max_group_size == 2
    );
    for (const auto& cand : candidates) {
        if (g_terminate || Clock::now() + std::chrono::milliseconds(5) >= deadline) break;
        std::vector<std::vector<int>> improved_partition;
        if (!apply_exact_repair_group(
                reduced,
                best_reduced,
                cand.component_indices,
                repair_leaf_limit,
                deadline,
                final_phase,
                improved_partition)) {
            continue;
        }
        int improved_components = static_cast<int>(improved_partition.size());
        if (improved_components < best_components) {
            best_reduced = std::move(improved_partition);
            best_components = improved_components;
            return true;
        }
    }
    return false;
}

bool run_exact_repair_portfolio(
    const ReducedInstance& reduced,
    std::vector<std::vector<int>>& best_reduced,
    int& best_components,
    const std::vector<EliteSolution>& elite_pool,
    Clock::time_point deadline,
    bool final_phase
) {
    auto now = Clock::now();
    auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    if (ms_left < 20) return false;

    long long budget_ms = final_phase
        ? std::min<long long>(300, std::max<long long>(50, ms_left / 6))
        : std::min<long long>(120, std::max<long long>(20, ms_left / 10));
    auto repair_deadline = std::min(deadline, now + std::chrono::milliseconds(budget_ms));

    bool improved_any = false;
    while (!g_terminate && Clock::now() + std::chrono::milliseconds(5) < repair_deadline) {
        bool improved = false;
        improved = run_exact_repair_pass(
            reduced,
            best_reduced,
            best_components,
            elite_pool,
            repair_deadline,
            2,
            40,
            final_phase,
            final_phase ? 24 : 16
        );
        if (!improved) {
            improved = run_exact_repair_pass(
                reduced,
                best_reduced,
                best_components,
                elite_pool,
                repair_deadline,
                3,
                44,
                final_phase,
                final_phase ? 20 : 12
            );
        }
        if (!improved && final_phase) {
            improved = run_exact_repair_pass(
                reduced,
                best_reduced,
                best_components,
                elite_pool,
                repair_deadline,
                4,
                48,
                final_phase,
                12
            );
        }
        if (!improved) break;
        improved_any = true;
    }
    return improved_any;
}

ThreeApproxResult run_three_approx(
    DynamicTree t1,
    DynamicTree f,
    Clock::time_point deadline,
    int n,
    int incumbent_components,
    GreedyPolicy policy,
    const std::vector<int>* elite_comp_map = nullptr,
    double elite_bonus = 0.0,
    const std::vector<int>* discrepancy_choices = nullptr,
    uint64_t tie_salt = 0,
    int cherry_sample_cap = 0
);

// -----------------------------------------------------------------------------
// Duality-inspired 2-approx portfolio seed.
//
// This is intentionally conservative:
// - It does NOT claim to be a faithful LP-dual port of the Java implementation.
// - It gives us the integration hooks now:
//   1) deterministic portfolio candidate after reductions,
//   2) early incumbent candidate,
//   3) elite-pool seed even if not best,
//   4) lower-bound diagnostic,
//   5) TryAndMerge-style greedy merge on the result.
// -----------------------------------------------------------------------------

struct DualitySeedResult {
    bool complete = false;
    bool valid = false;
    int components = 0;
    int lb_proxy = 0;
    long long elapsed_ms = 0;
    std::vector<std::vector<int>> comps;
};

bool dual2_profile_enabled() {
    static int enabled = []() {
        const char* env = std::getenv("STRIDE_DUAL2_PROFILE");
        return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

void emit_dual2_profile(
    const char* phase,
    int reduced_n,
    int lb_proxy,
    int raw_components,
    int final_components,
    bool complete,
    bool valid,
    bool improved,
    long long elapsed_ms
) {
    if (!dual2_profile_enabled()) return;

    std::cerr
        << "DUAL2"
        << " phase=" << phase
        << " n=" << reduced_n
        << " lb_proxy=" << lb_proxy
        << " raw_components=" << raw_components
        << " final_components=" << final_components
        << " complete=" << (complete ? 1 : 0)
        << " valid=" << (valid ? 1 : 0)
        << " improved=" << (improved ? 1 : 0)
        << " ms=" << elapsed_ms
        << '\n';
}

int dual2_seed_sample_cap_for_size(int n) {
    if (n > 4000) return 192;
    if (n > 1000) return 256;
    if (n > 600)  return 0;
    return 0;
}

std::chrono::milliseconds dual2_seed_budget_for_size(int n, long long ms_left) {
    long long budget_ms = 0;

    if (n > 4000) {
        budget_ms = std::min<long long>(10000, std::max<long long>(1800, ms_left / 12));
    } else if (n > 1000) {
        budget_ms = std::min<long long>(4500, std::max<long long>(900, ms_left / 18));
    } else if (n > 600) {
        budget_ms = std::min<long long>(2200, std::max<long long>(500, ms_left / 25));
    } else {
        budget_ms = std::min<long long>(1000, std::max<long long>(180, ms_left / 35));
    }

    return std::chrono::milliseconds(std::max<long long>(50, budget_ms));
}

int elite_acceptance_slack_for_dual2(int n) {
    if (n > 4000) return 16;
    if (n > 1000) return 12;
    if (n > 600)  return 8;
    if (n > 300)  return 6;
    return 4;
}

DualitySeedResult run_duality_seed_candidate(
    const ReducedInstance& reduced,
    const DynamicTree& dt1_base,
    const DynamicTree& dt2_base,
    Clock::time_point global_deadline,
    uint64_t seed_base,
    std::vector<EliteSolution>& elite_pool,
    int elite_limit,
    bool final_phase
) {
    DualitySeedResult out;

    const int n = reduced.reduced_leaf_count;
    if (n <= 0) return out;
    if (g_terminate || Clock::now() + std::chrono::milliseconds(20) >= global_deadline) return out;

    auto started = Clock::now();
    auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(
        global_deadline - started
    ).count();

    auto local_deadline = std::min(
        global_deadline,
        started + dual2_seed_budget_for_size(n, ms_left)
    );

    // Lower-bound diagnostic proxy.
    // This is not the exact LP-dual value from the Java paper/implementation.
    // Keep it diagnostic only.
    {
        DynamicTree t1_lb = dt1_base;
        DynamicTree t2_lb = dt2_base;
        int next_label = n + 1;
        contract_all_common_cherries(t1_lb, t2_lb, next_label);
        out.lb_proxy = exact_conflict_lower_bound(t1_lb, t2_lb);
    }

    uint64_t salt = mix64(seed_base ^ 0x2a6f9d7c1b3e5a91ULL ^ static_cast<uint64_t>(n));

    // Use a conservative deterministic policy as the "duality-style" seed.
    // It is deliberately different from the aggressive/default portfolio.
    auto res = run_three_approx(
        dt1_base,
        dt2_base,
        local_deadline,
        n,
        std::numeric_limits<int>::max(),   // do not prune this seed by incumbent
        GreedyPolicy::ConservativeSingleCut,
        nullptr,
        0.0,
        nullptr,
        salt,
        dual2_seed_sample_cap_for_size(n)
    );

    out.complete = res.complete;
    if (res.comps.empty()) {
        out.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - started
        ).count();
        emit_dual2_profile(
            final_phase ? "final_seed" : "seed",
            n,
            out.lb_proxy,
            -1,
            -1,
            out.complete,
            false,
            false,
            out.elapsed_ms
        );
        return out;
    }

    out.comps = std::move(res.comps);
    normalize_partition(out.comps);
    int raw_components = static_cast<int>(out.comps.size());

    out.valid = partition_is_valid_agreement_forest(
        reduced.t1,
        reduced.t2,
        out.comps,
        reduced.reduced_leaf_count
    );

    if (!out.valid) {
        out.comps.clear();
        out.components = 0;
        out.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - started
        ).count();
        emit_dual2_profile(
            final_phase ? "final_seed" : "seed",
            n,
            out.lb_proxy,
            raw_components,
            -1,
            out.complete,
            false,
            false,
            out.elapsed_ms
        );
        return out;
    }

    // Add the raw valid seed to the elite pool even if it is not best.
    maybe_add_elite_solution(
        elite_pool,
        out.comps,
        reduced.reduced_leaf_count,
        elite_limit
    );

    // Now apply the cheap TryAndMerge-style pass to the seed itself.
    int seed_components = static_cast<int>(out.comps.size());
    std::vector<std::vector<int>> merged = out.comps;
    int merged_components = seed_components;

    std::vector<EliteSolution> temp_elite = elite_pool;
    run_greedy_merge_portfolio(
        reduced,
        merged,
        merged_components,
        temp_elite,
        elite_limit,
        local_deadline,
        final_phase
    );

    if (merged_components < seed_components &&
        partition_is_valid_agreement_forest(
            reduced.t1,
            reduced.t2,
            merged,
            reduced.reduced_leaf_count)) {
        out.comps = std::move(merged);
        out.components = merged_components;
        maybe_add_elite_solution(
            elite_pool,
            out.comps,
            reduced.reduced_leaf_count,
            elite_limit
        );
    } else {
        out.components = seed_components;
    }

    out.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - started
    ).count();

    emit_dual2_profile(
        final_phase ? "final_seed" : "seed",
        n,
        out.lb_proxy,
        raw_components,
        out.components,
        out.complete,
        out.valid,
        false,
        out.elapsed_ms
    );

    return out;
}

// -----------------------------------------------------------------------------
// ResolveSet-style 2-approx-inspired portfolio runner.
//
// This is the first real structural use of the 2-approx idea:
// instead of only running run_three_approx() with a conservative policy,
// this repeatedly selects an active sibling/cherry pair from T1 and applies
// ResolvePair-like rules against the current forest F.
//
// It is intentionally safe:
// - candidate is validated outside before acceptance;
// - it never replaces the existing portfolio;
// - it is deterministic;
// - it can be disabled simply by removing the call sites.
// -----------------------------------------------------------------------------

enum class ResolveSetMode {
    SmallConflict,
    LargeConflict,
    PreferDifferentComponent,
    PreferPendantBatch,
    PreferFinalCut
};

const char* resolve_set_mode_name(ResolveSetMode mode) {
    switch (mode) {
        case ResolveSetMode::SmallConflict: return "small_conflict";
        case ResolveSetMode::LargeConflict: return "large_conflict";
        case ResolveSetMode::PreferDifferentComponent: return "diff_component";
        case ResolveSetMode::PreferPendantBatch: return "pendant_batch";
        case ResolveSetMode::PreferFinalCut: return "final_cut";
    }
    return "unknown";
}

bool resolve_set_profile_enabled() {
    static int enabled = []() {
        const char* env = std::getenv("STRIDE_RESOLVE_PROFILE");
        return (env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0) ? 1 : 0;
    }();
    return enabled != 0;
}

void emit_resolve_set_profile(
    const char* phase,
    int n,
    ResolveSetMode mode,
    bool complete,
    int components,
    int steps,
    int contractions,
    int cuts,
    long long elapsed_ms
) {
    if (!resolve_set_profile_enabled()) return;

    std::cerr
        << "RESOLVESET"
        << " phase=" << phase
        << " n=" << n
        << " mode=" << resolve_set_mode_name(mode)
        << " complete=" << (complete ? 1 : 0)
        << " components=" << components
        << " steps=" << steps
        << " contractions=" << contractions
        << " cuts=" << cuts
        << " ms=" << elapsed_ms
        << '\n';
}

double score_resolve_set_candidate(
    const CherryCandidate& cand,
    int n,
    ResolveSetMode mode,
    uint64_t tie
) {
    double inv = 1.0 / static_cast<double>(std::max(1, n));

    // Common cherries should almost always be resolved immediately.
    double s = cand.common ? 1000000.0 : 0.0;

    switch (mode) {
        case ResolveSetMode::SmallConflict:
            s += cand.same_component ? 20.0 : 30.0;
            s -= 4.0 * cand.conflict_mass * inv;
            s -= 2.0 * cand.pendant_count;
            s -= 0.05 * cand.distance;
            break;

        case ResolveSetMode::LargeConflict:
            s += cand.same_component ? 25.0 : 20.0;
            s += 5.0 * cand.conflict_mass * inv;
            s += 2.0 * cand.pendant_count;
            s -= 0.03 * cand.distance;
            break;

        case ResolveSetMode::PreferDifferentComponent:
            s += cand.same_component ? 0.0 : 100.0;
            s -= 2.0 * cand.conflict_mass * inv;
            s -= 0.03 * cand.distance;
            break;

        case ResolveSetMode::PreferPendantBatch:
            s += cand.same_component ? 80.0 : 10.0;
            s += 4.0 * cand.pendant_count;
            s += 2.0 * cand.immediate_gain;
            s -= 0.03 * cand.distance;
            break;

        case ResolveSetMode::PreferFinalCut:
            s += cand.same_component ? 40.0 : 45.0;
            if (cand.pendant_count <= 1) s += 80.0;
            s += 3.0 * cand.immediate_gain;
            s -= 0.02 * cand.distance;
            break;
    }

    // Stable deterministic tie-breaker.
    s -= static_cast<double>(tie & 0xffffULL) * 1e-9;
    return s;
}

int resolve_set_sample_cap_for_size(int n) {
    if (n > 4000) return 192;
    if (n > 1000) return 256;
    if (n > 600)  return 384;
    return 0;
}

std::chrono::milliseconds resolve_set_budget_for_size(int n, long long ms_left) {
    long long budget_ms = 0;

    if (n > 4000) {
        budget_ms = std::min<long long>(9000, std::max<long long>(1500, ms_left / 14));
    } else if (n > 1000) {
        budget_ms = std::min<long long>(4200, std::max<long long>(800, ms_left / 20));
    } else if (n > 600) {
        budget_ms = std::min<long long>(2200, std::max<long long>(450, ms_left / 28));
    } else if (n > 300) {
        budget_ms = std::min<long long>(1200, std::max<long long>(250, ms_left / 35));
    } else {
        budget_ms = std::min<long long>(700, std::max<long long>(120, ms_left / 45));
    }

    return std::chrono::milliseconds(std::max<long long>(40, budget_ms));
}

std::vector<int> choose_resolve_set_cut_plan(
    DynamicTree& t1,
    DynamicTree& f,
    const CherryCandidate& cand,
    int next_label,
    int n,
    const std::vector<int>& f_mass,
    ResolveSetMode mode,
    uint64_t tie_salt
) {
    std::vector<int> plan;

    if (cand.na == -1 || cand.nb == -1) return plan;

    // Different components: this mimics the ResolvePair behaviour where
    // one endpoint is detached. Prefer the smaller component because it
    // usually causes less downstream damage.
    if (!cand.same_component) {
        int ma = (cand.ra >= 0 && cand.ra < static_cast<int>(f_mass.size())) ? f_mass[cand.ra] : 1;
        int mb = (cand.rb >= 0 && cand.rb < static_cast<int>(f_mass.size())) ? f_mass[cand.rb] : 1;

        if (mode == ResolveSetMode::LargeConflict) {
            // Diversification variant: sometimes detach the heavier side.
            if (ma >= mb) plan.push_back(cand.na);
            else plan.push_back(cand.nb);
        } else {
            if (ma <= mb) plan.push_back(cand.na);
            else plan.push_back(cand.nb);
        }
        return plan;
    }

    // Same component: if there are pendant subtrees on the path, ResolvePair
    // style behaviour is to cut pendant material from the path.
    if (!cand.pendants.empty()) {
        std::vector<int> sorted = cand.pendants;
        std::sort(sorted.begin(), sorted.end(), [&](int x, int y) {
            int mx = (x >= 0 && x < static_cast<int>(f_mass.size())) ? f_mass[x] : 1;
            int my = (y >= 0 && y < static_cast<int>(f_mass.size())) ? f_mass[y] : 1;
            if (mx != my) return mx < my;
            return x < y;
        });

        if (mode == ResolveSetMode::PreferPendantBatch) {
            int cap = 0;
            if (n > 4000) cap = 2;
            else if (n > 1000) cap = 3;
            else if (n > 600) cap = 4;
            else cap = 5;

            cap = std::min<int>(cap, static_cast<int>(sorted.size()));
            for (int i = 0; i < cap; ++i) {
                plan.push_back(sorted[static_cast<size_t>(i)]);
            }
            return plan;
        }

        // Conservative/final-cut variants cut one pendant only.
        plan.push_back(sorted.front());
        return plan;
    }

    // No pendant available. Fall back to cutting one endpoint.
    int ma = (cand.na >= 0 && cand.na < static_cast<int>(f_mass.size())) ? f_mass[cand.na] : 1;
    int mb = (cand.nb >= 0 && cand.nb < static_cast<int>(f_mass.size())) ? f_mass[cand.nb] : 1;

    if (ma != mb) {
        if (ma <= mb) plan.push_back(cand.na);
        else plan.push_back(cand.nb);
    } else {
        uint64_t h = deterministic_pair_key(tie_salt, 0, cand.a, cand.b, 777);
        if ((h & 1ULL) == 0ULL) plan.push_back(cand.na);
        else plan.push_back(cand.nb);
    }

    return plan;
}

ThreeApproxResult run_resolve_set_approx(
    DynamicTree t1,
    DynamicTree f,
    Clock::time_point deadline,
    int n,
    int incumbent_components,
    ResolveSetMode mode,
    uint64_t tie_salt,
    int cherry_sample_cap = 0,
    const char* phase = "main"
) {
    auto started = Clock::now();

    int next_label = n + 1;
    int steps = 0;
    int contractions = 0;
    int cuts = 0;
    bool timed_out_or_terminated = false;

    PathScratch path_scratch;
    std::vector<std::pair<int, int>> cherries;
    cherries.reserve(static_cast<size_t>(n));

    if (cherry_sample_cap <= 0) {
        cherry_sample_cap = resolve_set_sample_cap_for_size(n);
    }

    contract_all_common_cherries(t1, f, next_label);

    while (!g_terminate && Clock::now() < deadline && t1.active_leaf_count > 2) {
        ++steps;

        // Remove singleton roots in F from T1 as in the existing heuristic.
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

        if (incumbent_components < std::numeric_limits<int>::max() &&
            f.root_component_count >= incumbent_components) {
            break;
        }

        contract_all_common_cherries(t1, f, next_label);

        fill_cherries_by_label(t1, cherries);
        if (cherries.empty()) break;

        bool deadline_near = false;
        auto f_mass = compute_active_leaf_masses(f);

        struct RankedResolveCandidate {
            double score = -std::numeric_limits<double>::infinity();
            size_t idx = 0;
            int a = -1;
            int b = -1;
            uint64_t tie = 0;
            CherryCandidate cand;
        };

        std::vector<size_t> candidate_indices;

        if (cherry_sample_cap > 0 &&
            cherries.size() > static_cast<size_t>(cherry_sample_cap)) {

            size_t m = cherries.size();
            size_t cap = static_cast<size_t>(cherry_sample_cap);
            candidate_indices.reserve(cap);

            uint64_t start_seed = deterministic_pair_key(
                tie_salt,
                static_cast<uint64_t>(steps),
                static_cast<int>(m),
                n,
                901
            );

            size_t start = static_cast<size_t>(start_seed % m);
            size_t stride = static_cast<size_t>(
                (mix64(start_seed ^ 0x4cf5ad432745937fULL) % m) | 1ULL
            );

            std::vector<char> seen(m, 0);
            for (size_t attempts = 0;
                 candidate_indices.size() < cap && attempts < m + cap * 4ULL;
                 ++attempts) {
                size_t idx = (start + attempts * stride) % m;
                if (seen[idx]) continue;
                seen[idx] = 1;
                candidate_indices.push_back(idx);
            }

            for (size_t offset = 0;
                 candidate_indices.size() < cap && offset < m;
                 ++offset) {
                size_t idx = (start + offset) % m;
                if (seen[idx]) continue;
                seen[idx] = 1;
                candidate_indices.push_back(idx);
            }
        } else {
            candidate_indices.resize(cherries.size());
            std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
        }

        std::vector<RankedResolveCandidate> ranked;
        ranked.reserve(candidate_indices.size());

        for (size_t k = 0; k < candidate_indices.size(); ++k) {
            if ((k & 63ULL) == 0ULL &&
                (g_terminate || Clock::now() + std::chrono::milliseconds(2) >= deadline)) {
                deadline_near = true;
                break;
            }

            size_t idx = candidate_indices[k];
            auto [a, b] = cherries[idx];

            CherryCandidate cand = build_cherry_candidate(
                t1,
                f,
                f_mass,
                a,
                b,
                n,
                &path_scratch
            );

            if (cand.na == -1 || cand.nb == -1) continue;

            uint64_t tie = deterministic_pair_key(
                tie_salt,
                static_cast<uint64_t>(steps),
                a,
                b,
                902
            );

            double score = score_resolve_set_candidate(cand, n, mode, tie);

            ranked.push_back({
                score,
                idx,
                a,
                b,
                tie,
                std::move(cand)
            });
        }

        if (deadline_near || ranked.empty()) break;

        std::sort(ranked.begin(), ranked.end(), [](const RankedResolveCandidate& x, const RankedResolveCandidate& y) {
            if (x.score != y.score) return x.score > y.score;
            if (x.tie != y.tie) return x.tie < y.tie;
            if (x.a != y.a) return x.a < y.a;
            if (x.b != y.b) return x.b < y.b;
            return x.idx < y.idx;
        });

        CherryCandidate cand = ranked.front().cand;

        // ResolvePair case 1: common cherry.
        if (cand.common) {
            int nl = next_label++;
            if (contract_cherry(t1, cand.a, cand.b, nl) &&
                contract_cherry(f, cand.a, cand.b, nl)) {
                ++contractions;
                continue;
            }
        }

        // ResolvePair cases 2/3: cut endpoint or pendant path material.
        auto plan = choose_resolve_set_cut_plan(
            t1,
            f,
            cand,
            next_label,
            n,
            f_mass,
            mode,
            tie_salt ^ static_cast<uint64_t>(steps)
        );

        if (plan.empty()) {
            if (cand.na != -1) plan.push_back(cand.na);
            else break;
        }

        // Keep batch cuts bounded. This avoids the classic heuristic mistake:
        // one heroic multi-cut that destroys a good forest.
        int max_cuts = 1;
        if (mode == ResolveSetMode::PreferPendantBatch) {
            if (n > 4000) max_cuts = 2;
            else if (n > 1000) max_cuts = 3;
            else max_cuts = 4;
        }

        if (static_cast<int>(plan.size()) > max_cuts) {
            plan.resize(static_cast<size_t>(max_cuts));
        }

        int before_components = f.root_component_count;
        apply_cut_plan(f, plan);
        cuts += std::max(0, f.root_component_count - before_components);

        contract_all_common_cherries(t1, f, next_label);
    }

    if (g_terminate || Clock::now() >= deadline) {
        timed_out_or_terminated = true;
    }

    ThreeApproxResult res;
    collect_current_components(f, res.comps);
    normalize_partition(res.comps);
    res.complete = !timed_out_or_terminated;

    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - started
    ).count();

    emit_resolve_set_profile(
        phase,
        n,
        mode,
        res.complete,
        static_cast<int>(res.comps.size()),
        steps,
        contractions,
        cuts,
        elapsed_ms
    );

    return res;
}

bool run_discrepancy_polish(
    const ReducedInstance& reduced,
    const DynamicTree& dt1_base,
    const DynamicTree& dt2_base,
    int reduced_n,
    std::vector<std::vector<int>>& best_reduced,
    int& best_components,
    Clock::time_point deadline,
    uint64_t seed_base,
    const std::vector<EliteSolution>& elite_pool
) {
    (void)seed_base;
    auto now = Clock::now();
    auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    if (ms_left < 30) return false;

    const std::vector<int>* elite_comp_map = nullptr;
    double elite_bonus = 0.0;
    if (!elite_pool.empty()) {
        elite_comp_map = &elite_pool.front().comp_of_leaf;
        elite_bonus = 5.0;
    }

    std::array<std::vector<int>, 2> discrepancy_runs{{{1}, {2}}};
    bool improved = false;
    for (size_t i = 0; i < discrepancy_runs.size(); ++i) {
        if (g_terminate || Clock::now() + std::chrono::milliseconds(10) >= deadline) break;
        auto local_deadline = std::min(deadline, Clock::now() + std::chrono::milliseconds(20));
        auto res = run_three_approx(
            dt1_base,
            dt2_base,
            local_deadline,
            reduced_n,
            best_components,
            GreedyPolicy::Balanced,
            elite_comp_map,
            elite_bonus,
            &discrepancy_runs[i]
        );
        if (!res.complete || res.comps.empty()) continue;
        if (static_cast<int>(res.comps.size()) < best_components) {
            auto candidate = res.comps;
            normalize_partition(candidate);

            if (partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate,
                    reduced.reduced_leaf_count)) {
                best_reduced = std::move(candidate);
                best_components = static_cast<int>(best_reduced.size());
                improved = true;
            }
        }
    }
    return improved;
}

bool run_final_polishing_portfolio(
    const ReducedInstance& reduced,
    const DynamicTree& dt1_base,
    const DynamicTree& dt2_base,
    std::vector<std::vector<int>>& best_reduced,
    int& best_components,
    std::vector<EliteSolution>& elite_pool,
    int elite_limit,
    Clock::time_point deadline,
    uint64_t seed_base
) {
    bool improved = false;
    auto now = Clock::now();
    auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
    LocalExactSearchPlan polish_plan = plan_polish_exact_search(
        reduced.reduced_leaf_count,
        best_components,
        ms_left
    );
    if (now + std::chrono::milliseconds(20) < deadline) {
        auto exact_deadline = std::min(deadline, now + polish_plan.budget);
        auto exact = maybe_solve_exact_kernel(
            dt1_base,
            dt2_base,
            reduced.reduced_leaf_count,
            best_components,
            exact_deadline,
            "polish",
            polish_plan.gap_cap,
            polish_plan.node_budget
        );
        if (exact.solved && !exact.comps.empty() &&
            static_cast<int>(exact.comps.size()) < best_components) {

            auto candidate = exact.comps;
            normalize_partition(candidate);

            if (partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate,
                    reduced.reduced_leaf_count)) {
                best_reduced = std::move(candidate);
                best_components = static_cast<int>(best_reduced.size());
                improved = true;
            }
        }
    }

    if (run_exact_repair_portfolio(reduced, best_reduced, best_components, elite_pool, deadline, true)) {
        improved = true;
    }

    // Cheap TryAndMerge-style pass. This is intentionally placed after exact
    // repair because exact repair may create mergeable neighboring components.
    if (run_greedy_merge_portfolio(
            reduced,
            best_reduced,
            best_components,
            elite_pool,
            elite_limit,
            deadline,
            true)) {
        improved = true;
    }

    if (run_discrepancy_polish(
            reduced,
            dt1_base,
            dt2_base,
            reduced.reduced_leaf_count,
            best_reduced,
            best_components,
            deadline,
            seed_base,
            elite_pool)) {
        improved = true;
    }

    if (improved && static_cast<int>(best_reduced.size()) <= best_components + 2) {
        maybe_add_elite_solution(elite_pool, best_reduced, reduced.reduced_leaf_count, elite_limit);
    }
    return improved;
}

std::vector<std::vector<int>> reduced_singleton_components(int reduced_n) {
    std::vector<std::vector<int>> comps;
    comps.reserve(reduced_n);
    for (int i = 1; i <= reduced_n; ++i) comps.push_back({i});
    return comps;
}

void generate_lds_scripts_rec(
    int max_rank,
    int max_depth,
    int remaining_cost,
    std::vector<int>& cur,
    std::vector<std::vector<int>>& out,
    size_t cap
) {
    if (out.size() >= cap) return;
    if (remaining_cost == 0) {
        if (!cur.empty()) out.push_back(cur);
        return;
    }
    if (static_cast<int>(cur.size()) >= max_depth) return;
    for (int rank = 1; rank <= max_rank && rank <= remaining_cost; ++rank) {
        cur.push_back(rank);
        generate_lds_scripts_rec(max_rank, max_depth, remaining_cost - rank, cur, out, cap);
        cur.pop_back();
        if (out.size() >= cap) return;
    }
}

std::vector<std::vector<int>> deterministic_lds_scripts(int n) {
    std::vector<std::vector<int>> scripts;
    scripts.push_back({});

    int max_rank = n <= 600 ? 3 : 2;
    int max_depth = n <= 300 ? 6 : (n <= 600 ? 5 : 4);
    int max_cost = n <= 300 ? 9 : (n <= 600 ? 7 : 5);
    size_t cap = n <= 300 ? 96 : (n <= 600 ? 64 : 32);

    std::vector<int> cur;
    for (int cost = 1; cost <= max_cost && scripts.size() < cap; ++cost) {
        generate_lds_scripts_rec(max_rank, max_depth, cost, cur, scripts, cap);
    }
    return scripts;
}

void add_deterministic_config(
    std::vector<DeterministicRunConfig>& configs,
    std::unordered_set<std::string>& seen,
    bool swapped,
    GreedyPolicy policy,
    std::vector<int> discrepancy,
    bool elite_guided,
    int cherry_sample_cap = 0,
    long long budget_ms = 0
) {
    std::string key;
    key.reserve(64 + discrepancy.size() * 3);
    key += swapped ? "1:" : "0:";
    key += std::to_string(static_cast<int>(policy));
    key += elite_guided ? ":1:" : ":0:";
    key += std::to_string(cherry_sample_cap);
    key += ':';
    for (int x : discrepancy) {
        key += std::to_string(x);
        key.push_back(',');
    }
    if (!seen.insert(key).second) return;

    DeterministicRunConfig cfg;
    cfg.swapped = swapped;
    cfg.policy = policy;
    cfg.discrepancy = std::move(discrepancy);
    cfg.elite_guided = elite_guided;
    cfg.cherry_sample_cap = cherry_sample_cap;
    cfg.budget_ms = budget_ms;
    configs.push_back(std::move(cfg));
}

void finalize_deterministic_configs(std::vector<DeterministicRunConfig>& configs, int n) {
    int default_sample = 0;
    long long default_budget = 0;
    if (n > 4000) {
        default_sample = 96;
        default_budget = 7000;
    } else if (n > 1000) {
        default_sample = 160;
        default_budget = 4500;
    }

    for (size_t i = 0; i < configs.size(); ++i) {
        auto& cfg = configs[i];
        cfg.id = static_cast<int>(i);
        if (cfg.cherry_sample_cap == 0 && default_sample > 0) {
            cfg.cherry_sample_cap = default_sample;
        }
        if (cfg.budget_ms == 0 && default_budget > 0) {
            cfg.budget_ms = (i == 0 && n > 4000) ? 12000 : default_budget;
        }
        uint64_t h = 0x6a09e667f3bcc909ULL;
        h ^= static_cast<uint64_t>(n) * 0x9e3779b97f4a7c15ULL;
        h ^= static_cast<uint64_t>(i + 1) * 0xbf58476d1ce4e5b9ULL;
        h ^= static_cast<uint64_t>(static_cast<int>(cfg.policy) + 1) * 0x94d049bb133111ebULL;
        h ^= cfg.swapped ? 0x243f6a8885a308d3ULL : 0x13198a2e03707344ULL;
        h ^= cfg.elite_guided ? 0xd1b54a32d192ed03ULL : 0x853c49e6748fea9bULL;
        h ^= discrepancy_hash(cfg.discrepancy);
        cfg.tie_salt = mix64(h);
    }
}

Clock::time_point deterministic_config_deadline(
    const DeterministicRunConfig& cfg,
    Clock::time_point global_deadline
) {
    if (cfg.budget_ms <= 0) return global_deadline;
    return std::min(
        global_deadline,
        Clock::now() + std::chrono::milliseconds(cfg.budget_ms)
    );
}

std::vector<DeterministicRunConfig> deterministic_configs_for_size(int n) {
    std::vector<DeterministicRunConfig> configs;
    std::unordered_set<std::string> seen;

    auto add_core_policy_set = [&](bool swapped, const std::vector<int>& script, bool elite) {
        add_deterministic_config(configs, seen, swapped, GreedyPolicy::Balanced, script, elite);
        add_deterministic_config(configs, seen, swapped, GreedyPolicy::PreferDifferentComponent, script, elite);
        add_deterministic_config(configs, seen, swapped, GreedyPolicy::PreferLowConflictMass, script, elite);
        add_deterministic_config(configs, seen, swapped, GreedyPolicy::PreferImmediateGain, script, elite);
    };

    /*
        VERY LARGE INSTANCES:
        We cannot afford a deep LDS portfolio. Instead, run multiple shallow,
        sampled passes with different policies and different sample caps.

        The idea:
        - one fast first pass to get a publishable incumbent quickly
        - several medium-budget passes with different cherry sampling
        - a few swapped-tree and discrepancy variants
        - avoid exhaustive scripts because n is too large
    */
    if (n > 4000) {
        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {}, false, 64, 9000);
        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {}, false, 96, 7000);
        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {}, false, 160, 6500);

        add_deterministic_config(configs, seen, true, GreedyPolicy::Balanced, {}, false, 96, 6500);
        add_deterministic_config(configs, seen, true, GreedyPolicy::PreferDifferentComponent, {}, false, 96, 6000);

        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferDifferentComponent, {}, false, 128, 6500);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferLowConflictMass, {}, false, 128, 6500);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferFewPendants, {}, false, 128, 6000);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferImmediateGain, {}, false, 128, 6000);
        add_deterministic_config(configs, seen, false, GreedyPolicy::AggressiveMultiCut, {}, false, 96, 6000);
        add_deterministic_config(configs, seen, false, GreedyPolicy::ConservativeSingleCut, {}, false, 128, 5500);

        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {1}, false, 128, 5500);
        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {2}, false, 128, 5500);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferDifferentComponent, {1}, false, 128, 5000);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferLowConflictMass, {1}, false, 128, 5000);
        add_deterministic_config(configs, seen, true, GreedyPolicy::Balanced, {1}, false, 96, 5000);

        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {}, true, 128, 5000);
        add_deterministic_config(configs, seen, true, GreedyPolicy::Balanced, {}, true, 96, 4500);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferImmediateGain, {}, true, 128, 4500);

        finalize_deterministic_configs(configs, n);
        return configs;
    }

    /*
        LARGE INSTANCES:
        More room for LDS than n > 4000, but still not enough for the full
        medium-instance portfolio. Use shallow scripts and policy diversity.
    */
    if (n > 1000) {
        std::vector<std::vector<int>> scripts = {
            {},
            {1},
            {2},
            {1, 1},
            {1, 2},
            {2, 1},
            {3}
        };

        for (const auto& script : scripts) {
            add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, script, false, 160, script.empty() ? 4500 : 3500);
            add_deterministic_config(configs, seen, true,  GreedyPolicy::Balanced, script, false, 160, 3300);

            if (script.size() <= 1) {
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferDifferentComponent, script, false, 192, 3500);
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferLowConflictMass, script, false, 192, 3500);
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferFewPendants, script, false, 192, 3200);
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferImmediateGain, script, false, 160, 3200);
            }
        }

        add_deterministic_config(configs, seen, false, GreedyPolicy::AggressiveMultiCut, {}, false, 160, 3500);
        add_deterministic_config(configs, seen, false, GreedyPolicy::ConservativeSingleCut, {}, false, 192, 3200);

        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {}, true, 160, 3000);
        add_deterministic_config(configs, seen, true,  GreedyPolicy::Balanced, {}, true, 160, 3000);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferDifferentComponent, {}, true, 192, 2800);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferImmediateGain, {1}, true, 160, 2800);

        finalize_deterministic_configs(configs, n);
        return configs;
    }

    /*
        UPPER-MEDIUM INSTANCES:
        Here the current solver is too timid. We can afford a bigger LDS set.
    */
    if (n > 600) {
        for (const auto& script : deterministic_lds_scripts(n)) {
            add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, script, false);

            if (script.size() <= 3) {
                add_deterministic_config(configs, seen, true, GreedyPolicy::Balanced, script, false);
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferDifferentComponent, script, false);
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferLowConflictMass, script, false);
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferImmediateGain, script, false);
            }

            if (script.size() <= 1) {
                add_deterministic_config(configs, seen, true, GreedyPolicy::PreferDifferentComponent, script, false);
                add_deterministic_config(configs, seen, false, GreedyPolicy::PreferFewPendants, script, false);
                add_deterministic_config(configs, seen, false, GreedyPolicy::ConservativeSingleCut, script, false);
                add_deterministic_config(configs, seen, false, GreedyPolicy::AggressiveMultiCut, script, false);
            }
        }

        add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {}, true);
        add_deterministic_config(configs, seen, true,  GreedyPolicy::Balanced, {}, true);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferDifferentComponent, {}, true);
        add_deterministic_config(configs, seen, false, GreedyPolicy::PreferImmediateGain, {1}, true);

        finalize_deterministic_configs(configs, n);
        return configs;
    }

    /*
        SMALL / MEDIUM INSTANCES:
        Here exhaustive portfolio diversity is useful, because each run is cheap.
    */
    for (const auto& script : deterministic_lds_scripts(n)) {
        add_core_policy_set(false, script, false);
        add_core_policy_set(true, script, false);

        if (script.size() <= 3) {
            add_deterministic_config(configs, seen, false, GreedyPolicy::PreferFewPendants, script, false);
            add_deterministic_config(configs, seen, false, GreedyPolicy::ConservativeSingleCut, script, false);
            add_deterministic_config(configs, seen, false, GreedyPolicy::AggressiveMultiCut, script, false);
        }

        if (script.size() <= 2) {
            add_core_policy_set(false, script, true);
            add_deterministic_config(configs, seen, true, GreedyPolicy::Balanced, script, true);
        }
    }

    add_deterministic_config(configs, seen, false, GreedyPolicy::Balanced, {}, true);
    add_deterministic_config(configs, seen, true,  GreedyPolicy::Balanced, {}, true);
    add_deterministic_config(configs, seen, false, GreedyPolicy::PreferImmediateGain, {1}, true);
    add_deterministic_config(configs, seen, false, GreedyPolicy::PreferLowConflictMass, {1}, true);
    add_deterministic_config(configs, seen, true,  GreedyPolicy::PreferDifferentComponent, {}, true);

    finalize_deterministic_configs(configs, n);
    return configs;
}

std::vector<std::vector<int>> solve_direct_reduced(
    const ReducedInstance& reduced,
    Clock::time_point deadline,
    uint64_t seed_base,
    int incumbent_limit = std::numeric_limits<int>::max()
) {
    int reduced_n = reduced.reduced_leaf_count;
    if (reduced_n <= 0) return {};
    if (reduced_n == 1) return {{1}};
    if (reduced_n == 2) return {{1, 2}};

    DynamicTree dt1_base = build_dynamic_tree(reduced.t1);
    DynamicTree dt2_base = build_dynamic_tree(reduced.t2);

    std::vector<std::vector<int>> best_reduced = reduced_singleton_components(reduced_n);
    int best_components = static_cast<int>(best_reduced.size());
    int prune_limit = std::min(best_components, incumbent_limit);

    auto exact = maybe_solve_exact_kernel(
        dt1_base,
        dt2_base,
        reduced_n,
        prune_limit,
        deadline,
        "initial"
    );
    if (exact.solved && !exact.comps.empty()) {
        best_reduced = exact.comps;
        best_components = static_cast<int>(best_reduced.size());
        prune_limit = std::min(prune_limit, best_components);
    }

    std::vector<EliteSolution> elite_pool;
    int elite_limit = elite_pool_limit_for_size(reduced_n);

    // Deterministic duality-inspired seed after reductions.
    // Useful especially on large instances, and useful as elite-pool material
    // even when it does not beat the current best.
    {
        auto dual2 = run_duality_seed_candidate(
            reduced,
            dt1_base,
            dt2_base,
            deadline,
            mix64(seed_base ^ 0x7c3a21d9e8f04b55ULL),
            elite_pool,
            elite_limit,
            false
        );

        if (dual2.valid && !dual2.comps.empty()) {
            int dual_components = static_cast<int>(dual2.comps.size());

            if (dual_components <= best_components + elite_acceptance_slack_for_dual2(reduced_n)) {
                maybe_add_elite_solution(
                    elite_pool,
                    dual2.comps,
                    reduced_n,
                    elite_limit
                );
            }

            if (dual_components < best_components) {
                best_reduced = std::move(dual2.comps);
                best_components = static_cast<int>(best_reduced.size());
                prune_limit = std::min(prune_limit, best_components);

                run_greedy_merge_portfolio(
                    reduced,
                    best_reduced,
                    best_components,
                    elite_pool,
                    elite_limit,
                    deadline,
                    false
                );

                prune_limit = std::min(prune_limit, best_components);
            }
        }
    }

    if (!best_reduced.empty() &&
        static_cast<int>(best_reduced.size()) < reduced_n) {
        maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
    }
    if (static_cast<int>(best_reduced.size()) <= prune_limit + 2) {
        maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
    }
    // ResolveSet-style 2-approx-inspired candidate.
    if (!g_terminate && Clock::now() + std::chrono::milliseconds(30) < deadline) {
        auto now = Clock::now();
        auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        auto resolve_deadline = std::min(
            deadline,
            now + resolve_set_budget_for_size(reduced_n, ms_left)
        );

        auto resolve_res = run_resolve_set_approx(
            dt1_base,
            dt2_base,
            resolve_deadline,
            reduced_n,
            prune_limit,
            ResolveSetMode::PreferFinalCut,
            mix64(seed_base ^ 0xa8b31f4c72d95e11ULL),
            resolve_set_sample_cap_for_size(reduced_n),
            "direct"
        );

        if (!resolve_res.comps.empty()) {
            auto candidate = resolve_res.comps;
            normalize_partition(candidate);

            int cand_components = static_cast<int>(candidate.size());

            if (cand_components <= best_components + elite_acceptance_slack_for_dual2(reduced_n) &&
                partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate,
                    reduced.reduced_leaf_count)) {

                maybe_add_elite_solution(
                    elite_pool,
                    candidate,
                    reduced_n,
                    elite_limit
                );

                if (cand_components < best_components) {
                    best_reduced = std::move(candidate);
                    best_components = static_cast<int>(best_reduced.size());
                    prune_limit = std::min(prune_limit, best_components);

                    run_greedy_merge_portfolio(
                        reduced,
                        best_reduced,
                        best_components,
                        elite_pool,
                        elite_limit,
                        deadline,
                        false
                    );

                    prune_limit = std::min(prune_limit, best_components);
                }
            }
        }
    }
    if (best_components < reduced_n) {
        if (run_exact_repair_portfolio(reduced, best_reduced, best_components, elite_pool, deadline, false)) {
            prune_limit = std::min(prune_limit, best_components);
            if (static_cast<int>(best_reduced.size()) <= prune_limit + 2) {
                maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
            }
        }
    }
    if (!g_terminate &&
        best_components > 1 &&
        best_components < reduced_n &&
        Clock::now() + std::chrono::milliseconds(40) < deadline) {
        auto late_exact = maybe_solve_exact_kernel(
            dt1_base,
            dt2_base,
            reduced_n,
            best_components,
            deadline,
            "mid"
        );
        if (late_exact.solved && !late_exact.comps.empty()) {
            auto candidate = late_exact.comps;
            normalize_partition(candidate);

            if (static_cast<int>(candidate.size()) < best_components &&
                partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate,
                    reduced.reduced_leaf_count)) {
                best_reduced = std::move(candidate);
                best_components = static_cast<int>(best_reduced.size());
                prune_limit = std::min(prune_limit, best_components);
                maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
                run_greedy_merge_portfolio(
                    reduced,
                    best_reduced,
                    best_components,
                    elite_pool,
                    elite_limit,
                    deadline,
                    false
                );
                prune_limit = std::min(prune_limit, best_components);
            }
        }
    }
    auto configs = deterministic_configs_for_size(reduced_n);   
    bool final_polish_done = false;
    DeterministicRestartDeduper restart_deduper;
    restart_deduper.reserve(configs.size() + 4);

    for (const auto& cfg : configs) {
        if (g_terminate || Clock::now() + std::chrono::milliseconds(10) >= deadline) {
            break;
        }

        if (!final_polish_done &&
            Clock::now() + std::chrono::milliseconds(80) < deadline &&
            best_components < reduced_n) {
            run_exact_repair_portfolio(
                reduced,
                best_reduced,
                best_components,
                elite_pool,
                deadline,
                false
            );
            prune_limit = std::min(prune_limit, best_components);
            final_polish_done = true;
        }

        const std::vector<int>* elite_comp_map = nullptr;
        double elite_bonus = 0.0;

        if (cfg.elite_guided && !elite_pool.empty()) {
            elite_comp_map = &elite_pool.front().comp_of_leaf;
            elite_bonus = 5.0;
        }

        const std::vector<int>* discrepancy_ptr =
            cfg.discrepancy.empty() ? nullptr : &cfg.discrepancy;

        auto cfg_start = Clock::now();
        auto cfg_deadline = deterministic_config_deadline(cfg, deadline);
        auto res = run_three_approx(
            cfg.swapped ? dt2_base : dt1_base,
            cfg.swapped ? dt1_base : dt2_base,
            cfg_deadline,
            reduced_n,
            prune_limit,
            cfg.policy,
            elite_comp_map,
            elite_bonus,
            discrepancy_ptr,
            cfg.tie_salt,
            cfg.cherry_sample_cap
        );
        long long cfg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - cfg_start
        ).count();

        if (res.comps.empty()) {
            emit_deterministic_profile("direct", cfg, reduced_n, res.complete, -1, false, false, cfg_ms);
            continue;
        }

        std::vector<std::vector<int>> candidate = res.comps;
        normalize_partition(candidate);

        int cand_components = static_cast<int>(candidate.size());
        bool duplicate_restart = !restart_deduper.remember_if_new(candidate, reduced_n);
        if (duplicate_restart) {
            emit_deterministic_profile("direct", cfg, reduced_n, res.complete, cand_components, false, false, cfg_ms, true);
            continue;
        }

        bool accepted = false;
        bool improved = false;

        bool checked_valid = false;
        bool candidate_valid = false;

        auto ensure_candidate_valid = [&]() -> bool {
            if (!checked_valid) {
                candidate_valid = partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate,
                    reduced.reduced_leaf_count
                );
                checked_valid = true;
            }
            return candidate_valid;
        };

        if (cand_components <= best_components + 2) {
            if (ensure_candidate_valid()) {
                maybe_add_elite_solution(elite_pool, candidate, reduced_n, elite_limit);
                accepted = true;
            }
        }

        if (cand_components < best_components) {
            if (!ensure_candidate_valid()) {
                emit_deterministic_profile("direct", cfg, reduced_n, res.complete, cand_components, false, false, cfg_ms);
                continue;
            }

            best_reduced = std::move(candidate);
            best_components = static_cast<int>(best_reduced.size());
            prune_limit = std::min(prune_limit, best_components);
            accepted = true;
            improved = true;

            if (run_exact_repair_portfolio(
                    reduced,
                    best_reduced,
                    best_components,
                    elite_pool,
                    deadline,
                    false)) {
                prune_limit = std::min(prune_limit, best_components);
            }

            if (static_cast<int>(best_reduced.size()) <= prune_limit + 2) {
                maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
            }
        }
        emit_deterministic_profile("direct", cfg, reduced_n, res.complete, cand_components, accepted, improved, cfg_ms);
    }

    if (!g_terminate && Clock::now() + std::chrono::milliseconds(10) < deadline) {
        run_final_polishing_portfolio(
            reduced,
            dt1_base,
            dt2_base,
            best_reduced,
            best_components,
            elite_pool,
            elite_limit,
            deadline,
            mix64(seed_base ^ 0x8f4d7c1259b3abf1ULL)
        );
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

    auto reduced_comps = solve_direct_reduced(reduced, deadline, seed_base, incumbent_components);
    return expand_reduced_components(reduced_comps, reduced.expansion);
}

double score_after_single_cut(
    const DynamicTree& t1,
    DynamicTree& f,
    int cut_node,
    int n,
    UndoLog& undo
) {
    size_t mark = undo.mark();

    cut_edge_above(f, cut_node, &undo);

    double score = 4.0 * static_cast<double>(count_common_cherries(t1, f))
                 - 1.0 * static_cast<double>(f.root_component_count);

    undo.undo_to(mark);
    return score;
}

ThreeApproxResult run_three_approx(
    DynamicTree t1,
    DynamicTree f,
    Clock::time_point deadline,
    int n,
    int incumbent_components,
    GreedyPolicy policy,
    const std::vector<int>* elite_comp_map,
    double elite_bonus,
    const std::vector<int>* discrepancy_choices,
    uint64_t tie_salt,
    int cherry_sample_cap
) {
    int next_label = n + 1;
    bool timed_out_or_terminated = false;
    size_t discrepancy_step = 0;
    uint64_t decision_step = 0; 
    PathScratch path_scratch;
    std::vector<int> pendants_scratch;
    std::vector<std::pair<int, int>> cherries;
    cherries.reserve(static_cast<size_t>(n));

    // Reusable scratch for O(1) duplicate checks while sampling cherry candidates.
    // This avoids repeated std::find(candidate_indices.begin(), candidate_indices.end(), idx).
    std::vector<int> sampled_seen;
    int sampled_seen_stamp = 1;

    struct RankedCherry {
        double score = -std::numeric_limits<double>::infinity();
        size_t idx = 0;
        int a = -1;
        int b = -1;
        uint64_t tie = 0;
    };

    std::vector<RankedCherry> ranked;
    std::vector<size_t> candidate_indices;

    // Exhaust common cherries before branching.
    contract_all_common_cherries(t1, f, next_label);

    while (!g_terminate && Clock::now() < deadline && t1.active_leaf_count > 2) {
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
            f.root_component_count >= incumbent_components) {
            break;
        }

        fill_cherries_by_label(t1, cherries);
        if (cherries.empty()) break;

        int common_idx = -1;
        bool deadline_near = false;
        size_t common_start = static_cast<size_t>(
            deterministic_pair_key(
                tie_salt,
                decision_step,
                static_cast<int>(cherries.size()),
                n,
                1
            ) % cherries.size()
        );
        for (size_t offset = 0; offset < cherries.size(); ++offset) {
            if ((offset & 511ULL) == 0ULL &&
                (g_terminate || Clock::now() + std::chrono::milliseconds(2) >= deadline)) {
                deadline_near = true;
                break;
            }
            size_t i = (common_start + offset) % cherries.size();
            auto [a, b] = cherries[i];
            if (are_siblings_by_label(f, a, b)) {
                common_idx = static_cast<int>(i);
                break;
            }
        }
        if (deadline_near) break;

        if (common_idx != -1) {
            auto [a, b] = cherries[static_cast<size_t>(common_idx)];
            int nl = next_label++;
            if (contract_cherry(t1, a, b, nl) && contract_cherry(f, a, b, nl)) {
                continue;
            }
        }

        size_t chosen_idx = 0;
        uint64_t step_key = decision_step++;
        std::vector<int> f_mass = compute_active_leaf_masses(f);

        {
            ranked.clear();
            candidate_indices.clear();

            if (cherry_sample_cap > 0 &&
                cherries.size() > static_cast<size_t>(cherry_sample_cap)) {

                const size_t cap = static_cast<size_t>(cherry_sample_cap);
                const size_t m = cherries.size();

                candidate_indices.reserve(cap);

                // Ensure sampled_seen is large enough for this decision step.
                if (sampled_seen.size() < m) {
                    sampled_seen.assign(m, 0);
                    sampled_seen_stamp = 1;
                }

                // New stamp for this sampling round.
                ++sampled_seen_stamp;
                if (sampled_seen_stamp == std::numeric_limits<int>::max()) {
                    std::fill(sampled_seen.begin(), sampled_seen.end(), 0);
                    sampled_seen_stamp = 1;
                }

                auto add_candidate_index = [&](size_t idx) {
                    if (sampled_seen[idx] == sampled_seen_stamp) return false;
                    sampled_seen[idx] = sampled_seen_stamp;
                    candidate_indices.push_back(idx);
                    return true;
                };

                uint64_t start_seed = deterministic_pair_key(
                    tie_salt,
                    step_key,
                    static_cast<int>(m),
                    n,
                    2
                );

                size_t start = static_cast<size_t>(start_seed % m);

                size_t stride = static_cast<size_t>(
                    (mix64(start_seed ^ 0xd1b54a32d192ed03ULL) % m) | 1ULL
                );

                for (size_t attempts = 0;
                    candidate_indices.size() < cap &&
                    attempts < m + cap * 4ULL;
                    ++attempts) {

                    size_t idx = (start + attempts * stride) % m;
                    add_candidate_index(idx);
                }

                // Fallback linear pass in case the strided walk did not collect enough unique indices.
                for (size_t offset = 0;
                    candidate_indices.size() < cap && offset < m;
                    ++offset) {

                    size_t idx = (start + offset) % m;
                    add_candidate_index(idx);
                }

            } else {
                candidate_indices.resize(cherries.size());
                std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
            }

            if (ranked.capacity() < candidate_indices.size()) {
                ranked.reserve(candidate_indices.size());
            }

            for (size_t k = 0; k < candidate_indices.size(); ++k) {
                if ((k & 63ULL) == 0ULL &&
                    (g_terminate || Clock::now() + std::chrono::milliseconds(2) >= deadline)) {
                    deadline_near = true;
                    break;
                }
                size_t i = candidate_indices[k];
                auto [ca, cb] = cherries[i];

                auto cand = build_cherry_candidate(t1, f, f_mass, ca, cb, n, &path_scratch);
                double score = score_cherry_candidate_policy(cand, n, policy);

                if (elite_comp_map && elite_comp_map->size() > static_cast<size_t>(n)) {
                    int ida = (*elite_comp_map)[static_cast<size_t>(ca)];
                    int idb = (*elite_comp_map)[static_cast<size_t>(cb)];

                    if (ida != 0 && idb != 0) {
                        if (ida == idb) score += elite_bonus;
                        else score -= 0.5 * elite_bonus;
                    }
                }

                uint64_t tie = deterministic_pair_key(tie_salt, step_key, ca, cb, 3);
                ranked.push_back({score, i, ca, cb, tie});
            }
            if (deadline_near || ranked.empty()) break;

            size_t top_k = std::min<size_t>(4, ranked.size());

            if (n > 4000) {
                top_k = std::min<size_t>(3, ranked.size());
            } else if (n > 1000) {
                top_k = std::min<size_t>(4, ranked.size());
            } else if (is_large_instance_size(n)) {
                top_k = std::min<size_t>(4, ranked.size());
            }

            auto ranked_better = [](const RankedCherry& x, const RankedCherry& y) {
                if (x.score != y.score) return x.score > y.score;
                if (x.tie != y.tie) return x.tie < y.tie;
                if (x.a != y.a) return x.a < y.a;
                if (x.b != y.b) return x.b < y.b;
                return x.idx < y.idx;
            };

            std::partial_sort(
                ranked.begin(),
                ranked.begin() + static_cast<std::ptrdiff_t>(top_k),
                ranked.end(),
                ranked_better
            );

            size_t rank_choice = 0;

            if (discrepancy_choices && discrepancy_step < discrepancy_choices->size()) {
                int want = (*discrepancy_choices)[discrepancy_step++];
                if (want < 0) want = 0;
                rank_choice = static_cast<size_t>(
                    std::min<int>(want, static_cast<int>(top_k) - 1)
                );
            }

            chosen_idx = ranked[rank_choice].idx;
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
            double ma = score_after_single_cut(t1, f, na, n, undo);
            double mb = score_after_single_cut(t1, f, nb, n, undo);

            if (ma > mb) {
                cut_edge_above(f, na);
            } else if (mb > ma) {
                cut_edge_above(f, nb);
            } else {
                int mass_a = (ra >= 0 && ra < static_cast<int>(f_mass.size())) ? f_mass[ra] : 1;
                int mass_b = (rb >= 0 && rb < static_cast<int>(f_mass.size())) ? f_mass[rb] : 1;

                if (mass_a != mass_b) {
                    if (mass_a <= mass_b) cut_edge_above(f, na);
                    else cut_edge_above(f, nb);
                } else {
                    uint64_t ha = deterministic_pair_key(tie_salt, step_key, a, b, 4);
                    if ((ha & 1ULL) == 0ULL) cut_edge_above(f, na);
                    else cut_edge_above(f, nb);
                }
            }

            continue;
        }

        const auto& path = path_nodes_fast(f, na, nb, path_scratch);
        fill_pendant_children_on_path(f, path, pendants_scratch);
        if (pendants_scratch.size() == 1) {
            cut_edge_above(f, pendants_scratch[0]);
        } else {
            if (use_expensive_same_component_plan(n, path, pendants_scratch)) {
                auto cand = build_cherry_candidate(t1, f, f_mass, a, b, n, &path_scratch);
                auto plan = choose_cut_plan(t1, f, cand, next_label, n, f_mass, policy, tie_salt ^ step_key);

                int max_plan_cuts = is_medium_instance_size(n) ? 5 : 4;

                if (!plan.empty() && static_cast<int>(plan.size()) <= max_plan_cuts) {
                    apply_cut_plan(f, plan);
                    continue;
                }
            }

            {
                int best_cut = -1;
                int best_mass = std::numeric_limits<int>::max();

                for (int c : pendants_scratch) {
                    if (c >= 0 && c < static_cast<int>(f_mass.size()) && f_mass[c] < best_mass) {
                        best_mass = f_mass[c];
                        best_cut = c;
                    }
                }

                int mass_a = (na >= 0 && na < static_cast<int>(f_mass.size())) ? f_mass[na] : 1;
                int mass_b = (nb >= 0 && nb < static_cast<int>(f_mass.size())) ? f_mass[nb] : 1;

                if (mass_a < best_mass) {
                    best_mass = mass_a;
                    best_cut = na;
                }
                if (mass_b < best_mass) {
                    best_mass = mass_b;
                    best_cut = nb;
                }

                if (best_cut != -1) {
                    cut_edge_above(f, best_cut);
                } else {
                    cut_edge_above(f, na);
                }
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
    std::vector<std::vector<int>> decomp_seed_original;

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
    bool decomp_attempted = false;
    auto decomp_profile_start = Clock::now();
    int decomp_score_before = best_components;
    const char* decomp_reason = "skipped_gate";
    if (!g_terminate && Clock::now() + std::chrono::milliseconds(20) < decomp_deadline) {
        decomp_attempted = true;
        auto decomp_comps = solve_decomposed_candidate(
            parsed[0],
            parsed[1],
            identity_expansion,
            decomp_deadline,
            mix64(base_seed ^ 0x243f6a8885a308d3ULL),
            best_components
        );

        normalize_partition(decomp_comps);

        if (partition_is_valid_agreement_forest(parsed[0], parsed[1], decomp_comps, n)) {
            decomp_reason = "valid_no_improve";
            decomp_seed_original = decomp_comps;
            auto decomp_out = forest_from_partition(decomp_comps, n, original_t1);
            if (!decomp_out.empty() && static_cast<int>(decomp_out.size()) < best_components) {
                best_out = decomp_out;
                best_components = static_cast<int>(best_out.size());
                publish_best_solution(best_out);
                decomp_reason = "improved";
            }
        } else {
            decomp_reason = "invalid";
        }
    }
    emit_pace_profile(
        "cluster_decomposition_seed",
        n,
        -1,
        decomp_score_before,
        best_components,
        decomp_attempted ? elapsed_ms_since(decomp_profile_start) : 0,
        decomp_reason
    );

    auto reduction_profile_start = Clock::now();
    ReducedInstance reduced = build_reduced_instance(parsed[0], parsed[1], n);
    emit_pace_profile(
        "initial_reductions",
        n,
        reduced.reduced_leaf_count,
        best_components,
        best_components,
        elapsed_ms_since(reduction_profile_start),
        reduced.reduced_leaf_count <= 0 ? "invalid_reduced" : "ok"
    );
    if (reduced.reduced_leaf_count <= 0) return best_out;

    DynamicTree dt1_base = build_dynamic_tree(reduced.t1);
    DynamicTree dt2_base = build_dynamic_tree(reduced.t2);
    int reduced_n = reduced.reduced_leaf_count;
    std::vector<std::vector<int>> best_reduced = reduced_singleton_components(reduced_n);
    int best_reduced_components = static_cast<int>(best_reduced.size());
    bool have_mapped_decomp_seed = false;

    auto map_profile_start = Clock::now();
    int map_score_before = best_reduced_components;
    const char* map_reason = decomp_seed_original.empty() ? "skipped_no_decomp_seed" : "not_mapped";
    if (!decomp_seed_original.empty()) {
        std::vector<std::vector<int>> mapped_decomp;
        if (map_original_partition_to_reduced(
                decomp_seed_original,
                reduced.expansion,
                n,
                mapped_decomp) &&
            static_cast<int>(mapped_decomp.size()) < best_reduced_components &&
            partition_is_valid_agreement_forest(
                reduced.t1,
                reduced.t2,
                mapped_decomp,
                reduced.reduced_leaf_count)) {
            best_reduced = std::move(mapped_decomp);
            best_reduced_components = static_cast<int>(best_reduced.size());
            have_mapped_decomp_seed = true;
            map_reason = "mapped";
        } else {
            map_reason = "rejected";
        }
    }
    emit_pace_profile(
        "map_original_partition_to_reduced",
        n,
        reduced_n,
        map_score_before,
        best_reduced_components,
        elapsed_ms_since(map_profile_start),
        map_reason
    );

    int exact0_incumbent = std::min(best_components, best_reduced_components);

    auto exact0_profile_start = Clock::now();
    int exact0_score_before = std::min(best_components, best_reduced_components);
    const char* exact0_reason = "no_solution";
    auto exact0 = maybe_solve_exact_kernel(
        dt1_base,
        dt2_base,
        reduced_n,
        exact0_incumbent,
        soft_deadline,
        "main_initial"
    );

    if (exact0.solved && !exact0.comps.empty()) {
        exact0_reason = "solved";
        auto candidate = exact0.comps;
        normalize_partition(candidate);

        if (partition_is_valid_agreement_forest(
                reduced.t1,
                reduced.t2,
                candidate,
                reduced.reduced_leaf_count)) {

            best_reduced = candidate;
            best_reduced_components = static_cast<int>(candidate.size());
            have_mapped_decomp_seed = false;

            auto expanded = expand_reduced_components(candidate, reduced.expansion);
            auto out = forest_from_partition(expanded, n, original_t1);

            if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                best_out = std::move(out);
                best_components = static_cast<int>(best_out.size());
                publish_best_solution(best_out);
            }
        } else {
            exact0_reason = "invalid";
        }
    } else if (exact0.solved) {
        exact0_reason = "empty";
    }
    emit_pace_profile(
        "exact_kernel",
        n,
        reduced_n,
        exact0_score_before,
        std::min(best_components, best_reduced_components),
        elapsed_ms_since(exact0_profile_start),
        exact0_reason
    );
    std::vector<EliteSolution> elite_pool;
    int elite_limit = elite_pool_limit_for_size(reduced_n);
    auto elite_profile_start = Clock::now();
    int elite_score_before = std::min(best_components, best_reduced_components);
    if (!best_reduced.empty() &&
        static_cast<int>(best_reduced.size()) < reduced_n &&
        (!have_mapped_decomp_seed || reduced_n <= 4000)) {
        maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
    }
    emit_pace_profile(
        "elite_update",
        n,
        reduced_n,
        elite_score_before,
        std::min(best_components, best_reduced_components),
        elapsed_ms_since(elite_profile_start),
        have_mapped_decomp_seed && reduced_n > 4000 ? "skipped_large_decomp_seed" : "initial_seed"
    );

    auto immediate_profile_start = Clock::now();
    int immediate_score_before = std::min(best_components, best_reduced_components);
    const char* immediate_reason = "skipped_gate";
    if (have_mapped_decomp_seed &&
        best_reduced_components < reduced_n &&
        reduced_n <= 1000 &&
        !g_terminate &&
        Clock::now() + std::chrono::milliseconds(60) < soft_deadline) {

        immediate_reason = "ran_no_improve";
        bool repaired_or_merged = false;
        if (run_exact_repair_portfolio(
                reduced,
                best_reduced,
                best_reduced_components,
                elite_pool,
                soft_deadline,
                false)) {
            repaired_or_merged = true;
        }

        if (run_greedy_merge_portfolio(
                reduced,
                best_reduced,
                best_reduced_components,
                elite_pool,
                elite_limit,
                soft_deadline,
                false)) {
            repaired_or_merged = true;
        }

        if (repaired_or_merged &&
            best_reduced_components < best_components &&
            partition_is_valid_agreement_forest(
                reduced.t1,
                reduced.t2,
                best_reduced,
                reduced.reduced_leaf_count)) {

            auto expanded = expand_reduced_components(best_reduced, reduced.expansion);
            auto out = forest_from_partition(expanded, n, original_t1);
            if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                best_out = std::move(out);
                best_components = static_cast<int>(best_out.size());
                publish_best_solution(best_out);
                immediate_reason = "improved";
            }
        }
    }
    emit_pace_profile(
        "immediate_repair",
        n,
        reduced_n,
        immediate_score_before,
        std::min(best_components, best_reduced_components),
        elapsed_ms_since(immediate_profile_start),
        immediate_reason
    );

    // Deterministic duality-inspired candidate after reductions.
    // This can become an early incumbent and is also fed to the elite pool
    // even if it is only near-best.
    {
        auto greedy_profile_start = Clock::now();
        int greedy_score_before = std::min(best_components, best_reduced_components);
        const char* greedy_reason = "no_candidate";
        auto dual2 = run_duality_seed_candidate(
            reduced,
            dt1_base,
            dt2_base,
            soft_deadline,
            mix64(base_seed ^ 0x6d0f27bb35a91e43ULL),
            elite_pool,
            elite_limit,
            false
        );

        if (dual2.valid && !dual2.comps.empty()) {
            greedy_reason = "valid_no_improve";
            int dual_components = static_cast<int>(dual2.comps.size());

            int elite_reference = std::min(best_components, best_reduced_components);
            if (dual_components <= elite_reference + elite_acceptance_slack_for_dual2(reduced_n)) {
                maybe_add_elite_solution(
                    elite_pool,
                    dual2.comps,
                    reduced_n,
                    elite_limit
                );
            }

            if (dual_components < best_reduced_components) {
                best_reduced = dual2.comps;
                best_reduced_components = dual_components;
                have_mapped_decomp_seed = false;
            }

            if (dual_components < best_components) {
                auto expanded = expand_reduced_components(dual2.comps, reduced.expansion);
                auto out = forest_from_partition(expanded, n, original_t1);

                if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                    best_out = std::move(out);
                    best_components = static_cast<int>(best_out.size());
                    publish_best_solution(best_out);
                    greedy_reason = "improved";
                }
            }

            // Try cheap merge immediately after accepting/recording the seed.
            if (best_reduced_components < reduced_n &&
                (!have_mapped_decomp_seed || reduced_n <= 4000) &&
                run_greedy_merge_portfolio(
                    reduced,
                    best_reduced,
                    best_reduced_components,
                    elite_pool,
                    elite_limit,
                    soft_deadline,
                    false) &&
                best_reduced_components < best_components &&
                partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    best_reduced,
                    reduced.reduced_leaf_count)) {

                auto expanded = expand_reduced_components(best_reduced, reduced.expansion);
                auto out = forest_from_partition(expanded, n, original_t1);

                if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                    best_out = std::move(out);
                    best_components = static_cast<int>(best_out.size());
                    publish_best_solution(best_out);
                    greedy_reason = "merge_improved";
                }
            }
        } else if (dual2.valid) {
            greedy_reason = "empty";
        } else {
            greedy_reason = "invalid";
        }
        emit_pace_profile(
            "greedy_seed",
            n,
            reduced_n,
            greedy_score_before,
            std::min(best_components, best_reduced_components),
            elapsed_ms_since(greedy_profile_start),
            greedy_reason
        );
    }

    // ResolveSet-style 2-approx-inspired candidate.
    auto resolve_profile_start = Clock::now();
    int resolve_score_before = std::min(best_components, best_reduced_components);
    const char* resolve_reason = "skipped_gate";
    if (!g_terminate && Clock::now() + std::chrono::milliseconds(40) < soft_deadline) {
        resolve_reason = "no_candidate";
        auto now = Clock::now();
        auto ms_left = std::chrono::duration_cast<std::chrono::milliseconds>(soft_deadline - now).count();
        auto resolve_deadline = std::min(
            soft_deadline,
            now + resolve_set_budget_for_size(reduced_n, ms_left)
        );

        auto resolve_res = run_resolve_set_approx(
            dt1_base,
            dt2_base,
            resolve_deadline,
            reduced_n,
            std::min(best_components, best_reduced_components),
            ResolveSetMode::PreferFinalCut,
            mix64(base_seed ^ 0x3459a71ccf023b9dULL),
            resolve_set_sample_cap_for_size(reduced_n),
            "main"
        );

        if (!resolve_res.comps.empty()) {
            resolve_reason = "candidate_rejected";
            auto candidate = resolve_res.comps;
            normalize_partition(candidate);

            int cand_components = static_cast<int>(candidate.size());

            if (cand_components <= std::min(best_components, best_reduced_components) + elite_acceptance_slack_for_dual2(reduced_n) &&
                partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate,
                    reduced.reduced_leaf_count)) {

                resolve_reason = "valid_no_improve";
                maybe_add_elite_solution(
                    elite_pool,
                    candidate,
                    reduced_n,
                    elite_limit
                );

                if (cand_components < best_reduced_components) {
                    best_reduced = candidate;
                    best_reduced_components = cand_components;
                    have_mapped_decomp_seed = false;
                    resolve_reason = "reduced_improved";
                }

                if (cand_components < best_components) {
                    auto expanded = expand_reduced_components(candidate, reduced.expansion);
                    auto out = forest_from_partition(expanded, n, original_t1);

                    if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                        best_out = std::move(out);
                        best_components = static_cast<int>(best_out.size());
                        publish_best_solution(best_out);
                        resolve_reason = "improved";
                    }
                }

                if (best_reduced_components < reduced_n &&
                    run_greedy_merge_portfolio(
                        reduced,
                        best_reduced,
                        best_reduced_components,
                        elite_pool,
                        elite_limit,
                        soft_deadline,
                        false) &&
                    best_reduced_components < best_components &&
                    partition_is_valid_agreement_forest(
                        reduced.t1,
                        reduced.t2,
                        best_reduced,
                        reduced.reduced_leaf_count)) {

                    auto expanded = expand_reduced_components(best_reduced, reduced.expansion);
                    auto out = forest_from_partition(expanded, n, original_t1);

                    if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                        best_out = std::move(out);
                        best_components = static_cast<int>(best_out.size());
                        publish_best_solution(best_out);
                        resolve_reason = "merge_improved";
                    }
                }
            }
        }
    }
    emit_pace_profile(
        "resolveset_seed",
        n,
        reduced_n,
        resolve_score_before,
        std::min(best_components, best_reduced_components),
        elapsed_ms_since(resolve_profile_start),
        resolve_reason
    );

    int exact_mid_incumbent = std::min(best_components, best_reduced_components);
    auto exact_mid_profile_start = Clock::now();
    int exact_mid_score_before = std::min(best_components, best_reduced_components);
    const char* exact_mid_reason = "skipped_gate";
    if (!g_terminate &&
        exact_mid_incumbent > 1 &&
        best_reduced_components < reduced_n &&
        (!have_mapped_decomp_seed || reduced_n <= 4000) &&
        Clock::now() + std::chrono::milliseconds(50) < soft_deadline) {
        exact_mid_reason = "no_solution";
        auto exact_mid = maybe_solve_exact_kernel(
            dt1_base,
            dt2_base,
            reduced_n,
            exact_mid_incumbent,
            soft_deadline,
            "main_mid"
        );

        if (exact_mid.solved && !exact_mid.comps.empty()) {
            exact_mid_reason = "solved_no_improve";
            auto candidate = exact_mid.comps;
            normalize_partition(candidate);

            if (partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate,
                    reduced.reduced_leaf_count)) {

                int cand_components = static_cast<int>(candidate.size());
                if (cand_components < best_reduced_components) {
                    best_reduced = candidate;
                    best_reduced_components = cand_components;
                    have_mapped_decomp_seed = false;
                    maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
                    exact_mid_reason = "reduced_improved";

                    if (run_greedy_merge_portfolio(
                            reduced,
                            best_reduced,
                            best_reduced_components,
                            elite_pool,
                            elite_limit,
                            soft_deadline,
                            false) &&
                        partition_is_valid_agreement_forest(
                            reduced.t1,
                            reduced.t2,
                            best_reduced,
                            reduced.reduced_leaf_count)) {
                        best_reduced_components = static_cast<int>(best_reduced.size());
                    }
                }

                if (best_reduced_components < best_components) {
                    auto expanded = expand_reduced_components(best_reduced, reduced.expansion);
                    auto out = forest_from_partition(expanded, n, original_t1);
                    if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                        best_out = std::move(out);
                        best_components = static_cast<int>(best_out.size());
                        publish_best_solution(best_out);
                        exact_mid_reason = "improved";
                    }
                }
            }
        } else if (exact_mid.solved) {
            exact_mid_reason = "empty";
        }
    }
    emit_pace_profile(
        "exact_kernel_mid",
        n,
        reduced_n,
        exact_mid_score_before,
        std::min(best_components, best_reduced_components),
        elapsed_ms_since(exact_mid_profile_start),
        exact_mid_reason
    );

    auto configs = deterministic_configs_for_size(reduced_n);

    if (!best_reduced.empty() &&
        static_cast<int>(best_reduced.size()) < reduced_n &&
        (!have_mapped_decomp_seed || reduced_n <= 4000)) {
        maybe_add_elite_solution(elite_pool, best_reduced, reduced_n, elite_limit);
    }

    DeterministicRestartDeduper restart_deduper;
    restart_deduper.reserve(configs.size() + elite_pool.size() + 4);

    auto greedy_portfolio_profile_start = Clock::now();
    int greedy_portfolio_score_before = std::min(best_components, best_reduced_components);
    int greedy_runs = 0;
    int greedy_improvements = 0;
    for (const auto& cfg : configs) {
        if (g_terminate || Clock::now() + std::chrono::milliseconds(10) >= soft_deadline) {
            break;
        }
        ++greedy_runs;

        const std::vector<int>* elite_comp_map = nullptr;
        double elite_bonus = 0.0;

        if (cfg.elite_guided && !elite_pool.empty()) {
            elite_comp_map = &elite_pool.front().comp_of_leaf;
            elite_bonus = 5.0;
        }

        const std::vector<int>* discrepancy =
            cfg.discrepancy.empty() ? nullptr : &cfg.discrepancy;

        auto cfg_start = Clock::now();
        auto cfg_deadline = deterministic_config_deadline(cfg, soft_deadline);
        auto res = run_three_approx(
            cfg.swapped ? dt2_base : dt1_base,
            cfg.swapped ? dt1_base : dt2_base,
            cfg_deadline,
            reduced_n,
            best_components,
            cfg.policy,
            elite_comp_map,
            elite_bonus,
            discrepancy,
            cfg.tie_salt,
            cfg.cherry_sample_cap
        );
        long long cfg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - cfg_start
        ).count();

        if (res.comps.empty()) {
            emit_deterministic_profile("main", cfg, reduced_n, res.complete, -1, false, false, cfg_ms);
            continue;
        }

        std::vector<std::vector<int>> candidate_reduced = res.comps;
        normalize_partition(candidate_reduced);

        int cand_components = static_cast<int>(candidate_reduced.size());
        bool duplicate_restart = !restart_deduper.remember_if_new(candidate_reduced, reduced_n);
        if (duplicate_restart) {
            emit_deterministic_profile("main", cfg, reduced_n, res.complete, cand_components, false, false, cfg_ms, true);
            continue;
        }

        bool accepted = false;
        bool improved = false;

        bool checked_valid = false;
        bool candidate_valid = false;

        auto ensure_candidate_valid = [&]() -> bool {
            if (!checked_valid) {
                candidate_valid = partition_is_valid_agreement_forest(
                    reduced.t1,
                    reduced.t2,
                    candidate_reduced,
                    reduced.reduced_leaf_count
                );
                checked_valid = true;
            }
            return candidate_valid;
        };

        int elite_reference = std::min(best_components, best_reduced_components);

        if (cand_components < best_reduced_components) {
            if (ensure_candidate_valid()) {
                best_reduced = candidate_reduced;
                best_reduced_components = cand_components;
                have_mapped_decomp_seed = false;
                accepted = true;
            } else {
                emit_deterministic_profile("main", cfg, reduced_n, res.complete, cand_components, false, false, cfg_ms);
                continue;
            }
        }

        if (cand_components <= elite_reference + 2) {
            if (ensure_candidate_valid()) {
                maybe_add_elite_solution(elite_pool, candidate_reduced, reduced_n, elite_limit);
                accepted = true;
            }
        }

        if (cand_components < best_components) {
            if (!ensure_candidate_valid()) {
                emit_deterministic_profile("main", cfg, reduced_n, res.complete, cand_components, false, false, cfg_ms);
                continue;
            }

            auto expanded = expand_reduced_components(candidate_reduced, reduced.expansion);
            auto out = forest_from_partition(expanded, n, original_t1);

            if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                best_out = out;
                best_components = static_cast<int>(out.size());
                best_reduced = std::move(candidate_reduced);
                best_reduced_components = static_cast<int>(best_reduced.size());
                have_mapped_decomp_seed = false;
                accepted = true;
                improved = true;
                ++greedy_improvements;

                bool repaired_or_merged = false;

                if (run_exact_repair_portfolio(
                        reduced,
                        best_reduced,
                        best_reduced_components,
                        elite_pool,
                        soft_deadline,
                        false)) {
                    repaired_or_merged = true;
                }

                if (run_greedy_merge_portfolio(
                        reduced,
                        best_reduced,
                        best_reduced_components,
                        elite_pool,
                        elite_limit,
                        soft_deadline,
                        false)) {
                    repaired_or_merged = true;
                }

                if (repaired_or_merged &&
                    best_reduced_components < best_components &&
                    partition_is_valid_agreement_forest(
                        reduced.t1,
                        reduced.t2,
                        best_reduced,
                        reduced.reduced_leaf_count)) {
                    auto repaired_expanded = expand_reduced_components(best_reduced, reduced.expansion);
                    auto repaired_out = forest_from_partition(repaired_expanded, n, original_t1);

                    if (!repaired_out.empty() &&
                        static_cast<int>(repaired_out.size()) < best_components) {
                        best_out = std::move(repaired_out);
                        best_components = static_cast<int>(best_out.size());
                    }
                }

                publish_best_solution(best_out);
            }
        }
        emit_deterministic_profile("main", cfg, reduced_n, res.complete, cand_components, accepted, improved, cfg_ms);
    }
    emit_pace_profile(
        "greedy_portfolio",
        n,
        reduced_n,
        greedy_portfolio_score_before,
        std::min(best_components, best_reduced_components),
        elapsed_ms_since(greedy_portfolio_profile_start),
        greedy_runs == 0 ? "skipped_gate" : (greedy_improvements > 0 ? "improved" : "no_improve")
    );

    auto final_polish_profile_start = Clock::now();
    int final_polish_score_before = std::min(best_components, best_reduced_components);
    const char* final_polish_reason = "skipped_gate";
    if (!g_terminate &&
        best_reduced_components < reduced_n &&
        (!have_mapped_decomp_seed || reduced_n <= 4000) &&
        Clock::now() + std::chrono::milliseconds(200) < soft_deadline) {

        final_polish_reason = "ran_no_improve";
        run_final_polishing_portfolio(
            reduced,
            dt1_base,
            dt2_base,
            best_reduced,
            best_reduced_components,
            elite_pool,
            elite_limit,
            soft_deadline,
            mix64(base_seed ^ 0xfedc987654321234ULL)
        );

        if (best_reduced_components < best_components &&
            partition_is_valid_agreement_forest(
                reduced.t1,
                reduced.t2,
                best_reduced,
                reduced.reduced_leaf_count)) {

            auto expanded = expand_reduced_components(best_reduced, reduced.expansion);
            auto out = forest_from_partition(expanded, n, original_t1);

            if (!out.empty() && static_cast<int>(out.size()) < best_components) {
                best_out = std::move(out);
                best_components = static_cast<int>(best_out.size());
                publish_best_solution(best_out);
                final_polish_reason = "improved";
            }
        }
    }
    emit_pace_profile(
        "merge_polish",
        n,
        reduced_n,
        final_polish_score_before,
        std::min(best_components, best_reduced_components),
        elapsed_ms_since(final_polish_profile_start),
        final_polish_reason
    );
    emit_pace_profile(
        "final_publish",
        n,
        reduced_n,
        best_components,
        best_components,
        elapsed_ms_since(start),
        g_terminate ? "terminated" : "complete"
    );
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
