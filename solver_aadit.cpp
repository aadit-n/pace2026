#include "genesis/genesis.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using genesis::tree::Tree;

struct PaceInstance {
    int tree_count = 0;
    int leaf_count = 0;
    std::vector<std::string> newick_lines;
    std::vector<std::string> x_lines;
};

struct Signature {
    uint64_t hash = 0;
    int size = 0;

    bool operator==(const Signature& other) const {
        return hash == other.hash && size == other.size;
    }
};

struct SignatureHasher {
    size_t operator()(const Signature& s) const {
        uint64_t x = s.hash ^ (static_cast<uint64_t>(s.size) * 0x9e3779b97f4a7c15ULL);
        x ^= (x >> 33);
        x *= 0xff51afd7ed558ccdULL;
        x ^= (x >> 33);
        return static_cast<size_t>(x);
    }
};

struct SimpleTree {
    int root = -1;
    std::vector<std::vector<int>> children;
    std::vector<int> parent;
    std::vector<int> leaf_label;  // 0 if internal
};

struct TreeData {
    int n_nodes = 0;
    int root = -1;
    std::vector<int> parent;
    std::vector<int> child0;
    std::vector<int> child1;
    std::vector<char> is_leaf;
    std::vector<int> leaf_label;
    std::vector<int> node_of_leaf;   // indexed by leaf label

    std::vector<int> subtree_leaf_count;
    std::vector<uint64_t> subtree_hash;
    std::vector<uint64_t> topo_hash_a;
    std::vector<uint64_t> topo_hash_b;

    std::unordered_set<Signature, SignatureHasher> cluster_signatures;
    std::unordered_map<Signature, std::pair<uint64_t, uint64_t>, SignatureHasher> cluster_topology;
    std::vector<std::pair<int, int>> cherries;
};

struct Solution {
    std::vector<int> comp_of;              // label -> component id
    std::vector<std::vector<int>> members; // component id -> leaves
    std::vector<uint64_t> comp_hash;
    std::vector<int> comp_size;
    std::vector<char> active;
    int next_id = 1;
    int k = 0;
};

volatile sig_atomic_t g_terminate = 0;
char* g_best_output = nullptr;
size_t g_best_output_len = 0;
int g_last_leaf_count = 1;

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    return s;
}

void sigterm_handler(int) {
    g_terminate = 1;
    if (g_best_output && g_best_output_len > 0) {
        const ssize_t unused = write(STDOUT_FILENO, g_best_output, g_best_output_len);
        (void)unused;
    }
    _Exit(0);
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
    char* old = g_best_output;
    g_best_output = new_buffer;
    g_best_output_len = new_len;
    delete[] old;

    sigprocmask(SIG_SETMASK, &old_set, nullptr);
}

std::vector<std::string> build_singleton_forest(int n) {
    std::vector<std::string> out;
    out.reserve(n);
    for (int i = 1; i <= n; ++i) out.push_back(std::to_string(i) + ";");
    return out;
}

std::string normalize_forest_output(const std::vector<std::string>& forest_lines) {
    std::string out;
    for (const std::string& raw : forest_lines) {
        std::string line = trim(raw);
        if (line.empty()) continue;
        if (line.back() != ';') line.push_back(';');
        out += line;
        out.push_back('\n');
    }
    return out;
}

void publish_best_solution(const std::vector<std::string>& forest_lines) {
    set_best_output_buffer(normalize_forest_output(forest_lines));
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
    explicit NewickParser(std::string text) : s_(std::move(text)) {}

    bool parse(SimpleTree& tree) {
        try {
            skip_ws();
            int root = parse_subtree(tree);
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ';') return false;
            ++pos_;
            skip_ws();
            if (pos_ != s_.size()) return false;
            tree.root = root;
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    std::string s_;
    size_t pos_ = 0;

    static bool is_delim(char c) {
        return c == ',' || c == '(' || c == ')' || c == ';';
    }

    int add_node(SimpleTree& t, int leaf_label) {
        int id = static_cast<int>(t.children.size());
        t.children.push_back({});
        t.parent.push_back(-1);
        t.leaf_label.push_back(leaf_label);
        return id;
    }

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) ++pos_;
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

    static int parse_leaf_label(const std::string& tok) {
        size_t i = 0;
        while (i < tok.size() && !std::isdigit(static_cast<unsigned char>(tok[i]))) ++i;
        size_t j = i;
        while (j < tok.size() && std::isdigit(static_cast<unsigned char>(tok[j]))) ++j;
        if (i == j) throw std::runtime_error("no label");
        return std::stoi(tok.substr(i, j - i));
    }

    void skip_optional_token() {
        (void)read_token();
    }

    int parse_subtree(SimpleTree& t) {
        skip_ws();
        if (pos_ >= s_.size()) throw std::runtime_error("eof");

        if (s_[pos_] == '(') {
            ++pos_;
            int u = add_node(t, 0);
            while (true) {
                int c = parse_subtree(t);
                t.children[u].push_back(c);
                t.parent[c] = u;
                skip_ws();
                if (pos_ >= s_.size()) throw std::runtime_error("eof children");
                if (s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (s_[pos_] == ')') {
                    ++pos_;
                    break;
                }
                throw std::runtime_error("expected , or )");
            }
            skip_optional_token();
            return u;
        }

        const std::string token = read_token();
        if (token.empty()) throw std::runtime_error("empty leaf token");
        int label = parse_leaf_label(token);
        return add_node(t, label);
    }
};

bool read_instance_from_stdin(PaceInstance& inst) {
    std::string line;
    bool have_p = false;

    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (!line.empty() && line[0] == '#') {
            if (line.rfind("#p", 0) == 0) {
                std::istringstream ss(line);
                std::string tag;
                ss >> tag >> inst.tree_count >> inst.leaf_count;
                if (inst.tree_count > 0 && inst.leaf_count > 0) {
                    have_p = true;
                    g_last_leaf_count = inst.leaf_count;
                }
            } else if (line.rfind("#x", 0) == 0) {
                inst.x_lines.push_back(line.substr(2));
            }
            continue;
        }

        if (!have_p) continue;
        if (static_cast<int>(inst.newick_lines.size()) < inst.tree_count) {
            inst.newick_lines.push_back(line);
        }
    }

    if (!have_p) return false;
    return static_cast<int>(inst.newick_lines.size()) == inst.tree_count;
}

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

bool convert_to_tree_data(const SimpleTree& st, int n, const std::vector<uint64_t>& zobrist, TreeData& out) {
    out.n_nodes = static_cast<int>(st.children.size());
    out.root = st.root;
    out.parent = st.parent;
    out.child0.assign(out.n_nodes, -1);
    out.child1.assign(out.n_nodes, -1);
    out.is_leaf.assign(out.n_nodes, 0);
    out.leaf_label = st.leaf_label;
    out.node_of_leaf.assign(n + 1, -1);

    int leaf_count = 0;
    std::vector<int> seen(n + 1, 0);

    for (int u = 0; u < out.n_nodes; ++u) {
        if (st.children[u].empty()) {
            out.is_leaf[u] = 1;
            int lbl = st.leaf_label[u];
            if (lbl < 1 || lbl > n || seen[lbl]) return false;
            seen[lbl] = 1;
            out.node_of_leaf[lbl] = u;
            ++leaf_count;
        } else {
            if (st.children[u].size() != 2) return false;
            out.child0[u] = st.children[u][0];
            out.child1[u] = st.children[u][1];
        }
    }
    if (leaf_count != n) return false;
    for (int lbl = 1; lbl <= n; ++lbl) {
        if (!seen[lbl]) return false;
    }

    out.subtree_leaf_count.assign(out.n_nodes, 0);
    out.subtree_hash.assign(out.n_nodes, 0);
    out.topo_hash_a.assign(out.n_nodes, 0);
    out.topo_hash_b.assign(out.n_nodes, 0);
    out.cluster_signatures.clear();
    out.cluster_topology.clear();
    out.cherries.clear();

    std::vector<std::pair<int, bool>> stck;
    stck.reserve(out.n_nodes * 2);
    stck.push_back({out.root, false});

    while (!stck.empty()) {
        auto [u, done] = stck.back();
        stck.pop_back();
        if (!done) {
            stck.push_back({u, true});
            if (!out.is_leaf[u]) {
                stck.push_back({out.child0[u], false});
                stck.push_back({out.child1[u], false});
            }
            continue;
        }

        if (out.is_leaf[u]) {
            int lbl = out.leaf_label[u];
            out.subtree_leaf_count[u] = 1;
            out.subtree_hash[u] = zobrist[lbl];
            out.topo_hash_a[u] = mix64(static_cast<uint64_t>(lbl) + 0x123456789abcdefULL);
            out.topo_hash_b[u] = mix64(static_cast<uint64_t>(lbl) + 0xfedcba987654321ULL);
        } else {
            int a = out.child0[u];
            int b = out.child1[u];
            out.subtree_leaf_count[u] = out.subtree_leaf_count[a] + out.subtree_leaf_count[b];
            out.subtree_hash[u] = out.subtree_hash[a] ^ out.subtree_hash[b];

            uint64_t a1 = out.topo_hash_a[a];
            uint64_t b1 = out.topo_hash_a[b];
            if (a1 > b1) std::swap(a1, b1);
            uint64_t a2 = out.topo_hash_b[a];
            uint64_t b2 = out.topo_hash_b[b];
            if (a2 > b2) std::swap(a2, b2);
            out.topo_hash_a[u] = mix64(a1 ^ (b1 + 0x9e3779b97f4a7c15ULL));
            out.topo_hash_b[u] = mix64(a2 ^ (b2 + 0x517cc1b727220a95ULL));

            if (out.is_leaf[a] && out.is_leaf[b]) {
                int x = out.leaf_label[a];
                int y = out.leaf_label[b];
                if (x > y) std::swap(x, y);
                out.cherries.push_back({x, y});
            }
        }

        Signature sig{out.subtree_hash[u], out.subtree_leaf_count[u]};
        out.cluster_signatures.insert(sig);
        out.cluster_topology.emplace(sig, std::make_pair(out.topo_hash_a[u], out.topo_hash_b[u]));
    }

    return true;
}

uint64_t pair_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
}

std::string comb_newick(std::vector<int> leaves) {
    if (leaves.empty()) return "";
    std::sort(leaves.begin(), leaves.end());
    std::string cur = std::to_string(leaves[0]);
    for (size_t i = 1; i < leaves.size(); ++i) {
        cur = "(" + cur + "," + std::to_string(leaves[i]) + ")";
    }
    cur.push_back(';');
    return cur;
}

std::string restricted_newick_dfs(int node, const TreeData& base, const std::vector<char>& in_comp) {
    if (base.is_leaf[node]) {
        const int lbl = base.leaf_label[node];
        return in_comp[lbl] ? std::to_string(lbl) : std::string();
    }
    std::string s0 = restricted_newick_dfs(base.child0[node], base, in_comp);
    std::string s1 = restricted_newick_dfs(base.child1[node], base, in_comp);
    if (s0.empty() && s1.empty()) return {};
    if (s0.empty()) return s1;
    if (s1.empty()) return s0;
    return "(" + s0 + "," + s1 + ")";
}

std::vector<std::string> build_forest_from_comp_of(const std::vector<int>& comp_of, int n, const TreeData& base_tree) {
    std::map<int, std::vector<int>> mp;
    for (int lbl = 1; lbl <= n; ++lbl) mp[comp_of[lbl]].push_back(lbl);

    std::vector<std::pair<int, std::string>> keyed;
    keyed.reserve(mp.size());
    for (auto& kv : mp) {
        auto& leaves = kv.second;
        if (leaves.size() == 1) {
            keyed.push_back({leaves[0], std::to_string(leaves[0]) + ";"});
        } else {
            std::vector<char> in_comp(n + 1, 0);
            for (int x : leaves) in_comp[x] = 1;
            std::string nw = restricted_newick_dfs(base_tree.root, base_tree, in_comp);
            if (nw.empty()) nw = comb_newick(leaves);
            if (!nw.empty() && nw.back() != ';') nw.push_back(';');
            keyed.push_back({*std::min_element(leaves.begin(), leaves.end()), nw});
        }
    }
    std::sort(keyed.begin(), keyed.end(), [](const auto& a, const auto& b){ return a.first < b.first; });

    std::vector<std::string> out;
    out.reserve(keyed.size());
    for (auto& kv : keyed) out.push_back(std::move(kv.second));
    return out;
}

void init_singleton_solution(int n, const std::vector<uint64_t>& zobrist, Solution& sol) {
    sol.comp_of.assign(n + 1, 0);
    sol.members.assign(n + 1, {});
    sol.comp_hash.assign(n + 1, 0);
    sol.comp_size.assign(n + 1, 0);
    sol.active.assign(n + 1, 0);
    sol.next_id = n + 1;
    sol.k = n;
    for (int lbl = 1; lbl <= n; ++lbl) {
        int cid = lbl;
        sol.comp_of[lbl] = cid;
        sol.members[cid].push_back(lbl);
        sol.comp_hash[cid] = zobrist[lbl];
        sol.comp_size[cid] = 1;
        sol.active[cid] = 1;
    }
}

bool is_sig_valid_all_trees(const Signature& sig, const std::vector<TreeData>& trees) {
    for (const TreeData& td : trees) {
        if (td.cluster_signatures.find(sig) == td.cluster_signatures.end()) return false;
    }
    return true;
}

bool is_sig_topology_valid_all_trees(const Signature& sig, const std::vector<TreeData>& trees) {
    bool have = false;
    std::pair<uint64_t, uint64_t> ref{0, 0};
    for (const TreeData& td : trees) {
        auto it = td.cluster_topology.find(sig);
        if (it == td.cluster_topology.end()) return false;
        if (!have) {
            ref = it->second;
            have = true;
        } else if (it->second != ref) {
            return false;
        }
    }
    return true;
}

bool is_component_valid_all_trees(int cid, const Solution& sol, const std::vector<TreeData>& trees) {
    Signature sig{sol.comp_hash[cid], sol.comp_size[cid]};
    return is_sig_topology_valid_all_trees(sig, trees);
}

bool is_solution_valid(const Solution& sol, const std::vector<TreeData>& trees) {
    for (int cid = 1; cid < sol.next_id; ++cid) {
        if (!sol.active[cid]) continue;
        if (!is_component_valid_all_trees(cid, sol, trees)) return false;
    }
    return true;
}

bool merge_components(Solution& sol, int a, int b) {
    if (a == b) return false;
    if (a <= 0 || b <= 0 || a >= sol.next_id || b >= sol.next_id) return false;
    if (!sol.active[a] || !sol.active[b]) return false;

    int nid = sol.next_id++;
    if (nid >= static_cast<int>(sol.members.size())) {
        sol.members.push_back({});
        sol.comp_hash.push_back(0);
        sol.comp_size.push_back(0);
        sol.active.push_back(0);
    }

    std::vector<int>& ma = sol.members[a];
    std::vector<int>& mb = sol.members[b];
    std::vector<int>& mn = sol.members[nid];
    mn.clear();
    mn.reserve(ma.size() + mb.size());
    mn.insert(mn.end(), ma.begin(), ma.end());
    mn.insert(mn.end(), mb.begin(), mb.end());

    for (int lbl : mn) sol.comp_of[lbl] = nid;

    sol.comp_hash[nid] = sol.comp_hash[a] ^ sol.comp_hash[b];
    sol.comp_size[nid] = sol.comp_size[a] + sol.comp_size[b];
    sol.active[nid] = 1;

    sol.active[a] = 0;
    sol.active[b] = 0;
    ma.clear();
    mb.clear();
    sol.k -= 1;
    return true;
}

bool move_leaf_between_components(Solution& sol, int leaf, int from_cid, int to_cid, const std::vector<uint64_t>& zobrist) {
    if (from_cid == to_cid) return false;
    if (!sol.active[from_cid] || !sol.active[to_cid]) return false;
    if (sol.comp_size[from_cid] <= 1) return false;

    auto& from = sol.members[from_cid];
    auto it = std::find(from.begin(), from.end(), leaf);
    if (it == from.end()) return false;

    from.erase(it);
    sol.members[to_cid].push_back(leaf);
    sol.comp_of[leaf] = to_cid;

    sol.comp_hash[from_cid] ^= zobrist[leaf];
    sol.comp_hash[to_cid] ^= zobrist[leaf];
    --sol.comp_size[from_cid];
    ++sol.comp_size[to_cid];
    return true;
}

bool split_off_leaf(Solution& sol, int from_cid, int leaf, const std::vector<uint64_t>& zobrist) {
    if (!sol.active[from_cid]) return false;
    if (sol.comp_size[from_cid] <= 1) return false;

    auto& from = sol.members[from_cid];
    auto it = std::find(from.begin(), from.end(), leaf);
    if (it == from.end()) return false;

    int nid = sol.next_id++;
    if (nid >= static_cast<int>(sol.members.size())) {
        sol.members.push_back({});
        sol.comp_hash.push_back(0);
        sol.comp_size.push_back(0);
        sol.active.push_back(0);
    }

    from.erase(it);
    sol.members[nid].clear();
    sol.members[nid].push_back(leaf);
    sol.comp_of[leaf] = nid;

    sol.comp_hash[from_cid] ^= zobrist[leaf];
    --sol.comp_size[from_cid];

    sol.comp_hash[nid] = zobrist[leaf];
    sol.comp_size[nid] = 1;
    sol.active[nid] = 1;
    sol.k += 1;
    return true;
}

std::vector<int> active_components(const Solution& sol) {
    std::vector<int> ids;
    ids.reserve(sol.k);
    for (int cid = 1; cid < sol.next_id; ++cid) if (sol.active[cid]) ids.push_back(cid);
    return ids;
}

int count_potential_merges_probe(const Solution& sol, const std::vector<TreeData>& trees, std::mt19937_64& rng, int trials) {
    std::vector<int> act = active_components(sol);
    if (act.size() < 2) return 0;
    int ok = 0;
    for (int t = 0; t < trials; ++t) {
        int i = static_cast<int>(rng() % act.size());
        int j = static_cast<int>(rng() % act.size());
        if (i == j) continue;
        int a = act[i], b = act[j];
        Signature sig{sol.comp_hash[a] ^ sol.comp_hash[b], sol.comp_size[a] + sol.comp_size[b]};
        if (is_sig_valid_all_trees(sig, trees)) ++ok;
    }
    return ok;
}

void greedy_merge_closure(
    Solution& sol,
    const std::vector<TreeData>& trees,
    const std::vector<std::pair<int, int>>& scored_pairs,
    std::mt19937_64& rng,
    std::chrono::steady_clock::time_point deadline
) {
    int stall = 0;
    while (!g_terminate && std::chrono::steady_clock::now() < deadline) {
        bool improved = false;

        for (size_t i = 0; i < scored_pairs.size() && i < 20000; ++i) {
            if ((i & 127) == 0 && (g_terminate || std::chrono::steady_clock::now() >= deadline)) break;
            int a_leaf = scored_pairs[i].first;
            int b_leaf = scored_pairs[i].second;
            int a = sol.comp_of[a_leaf];
            int b = sol.comp_of[b_leaf];
            if (a == b) continue;
            Signature sig{sol.comp_hash[a] ^ sol.comp_hash[b], sol.comp_size[a] + sol.comp_size[b]};
            if (!is_sig_valid_all_trees(sig, trees)) continue;
            if (merge_components(sol, a, b)) improved = true;
        }

        for (int trial = 0; trial < 4000 && !g_terminate && std::chrono::steady_clock::now() < deadline; ++trial) {
            int la = 1 + static_cast<int>(rng() % static_cast<uint64_t>(sol.comp_of.size() - 1));
            int lb = 1 + static_cast<int>(rng() % static_cast<uint64_t>(sol.comp_of.size() - 1));
            int a = sol.comp_of[la], b = sol.comp_of[lb];
            if (a == b) continue;
            Signature sig{sol.comp_hash[a] ^ sol.comp_hash[b], sol.comp_size[a] + sol.comp_size[b]};
            if (!is_sig_valid_all_trees(sig, trees)) continue;
            if (merge_components(sol, a, b)) improved = true;
        }

        if (!improved) {
            ++stall;
            if (stall >= 4) break;
        } else {
            stall = 0;
        }
    }
}

void local_search_ils(
    Solution& sol,
    const std::vector<TreeData>& trees,
    const std::vector<uint64_t>& zobrist,
    std::mt19937_64& rng,
    std::chrono::steady_clock::time_point deadline,
    const std::vector<std::pair<int, int>>& scored_pairs
) {
    int no_improve_iters = 0;
    int last_k = sol.k;

    while (!g_terminate && std::chrono::steady_clock::now() < deadline) {
        bool changed = false;

        std::vector<int> act = active_components(sol);
        if (act.size() < 2) break;

        // Boundary shift neighborhood.
        for (int it = 0; it < 300 && !g_terminate && std::chrono::steady_clock::now() < deadline; ++it) {
            int ia = static_cast<int>(rng() % act.size());
            int ib = static_cast<int>(rng() % act.size());
            if (ia == ib) continue;
            int a = act[ia], b = act[ib];
            if (sol.comp_size[a] <= 1) continue;
            auto& A = sol.members[a];
            int leaf = A[static_cast<int>(rng() % A.size())];

            Signature sigA{sol.comp_hash[a] ^ zobrist[leaf], sol.comp_size[a] - 1};
            Signature sigB{sol.comp_hash[b] ^ zobrist[leaf], sol.comp_size[b] + 1};
            if (!is_sig_valid_all_trees(sigA, trees)) continue;
            if (!is_sig_valid_all_trees(sigB, trees)) continue;

            if (move_leaf_between_components(sol, leaf, a, b, zobrist)) {
                changed = true;
                greedy_merge_closure(sol, trees, scored_pairs, rng, deadline);
                break;
            }
        }

        if (!changed) {
            ++no_improve_iters;
        }

        if (sol.k < last_k) {
            last_k = sol.k;
            no_improve_iters = 0;
        }

        // Perturbation kick when stalled.
        if (no_improve_iters >= 4) {
            int y = std::max(1, static_cast<int>(active_components(sol).size() / 10));
            std::vector<int> ids = active_components(sol);
            std::shuffle(ids.begin(), ids.end(), rng);
            y = std::min(y, static_cast<int>(ids.size()));

            for (int i = 0; i < y && !g_terminate && std::chrono::steady_clock::now() < deadline; ++i) {
                int cid = ids[i];
                if (!sol.active[cid] || sol.comp_size[cid] <= 1) continue;
                int leaf = sol.members[cid][static_cast<int>(rng() % sol.members[cid].size())];
                (void)split_off_leaf(sol, cid, leaf, zobrist);
            }

            greedy_merge_closure(sol, trees, scored_pairs, rng, deadline);
            no_improve_iters = 0;
        }

        if (!changed && no_improve_iters == 0) {
            // no-op; allow next round after perturbation/closure
        }
    }
}

std::vector<std::string> solve(const PaceInstance& instance) {
    const int n = instance.leaf_count;
    const int t = instance.tree_count;

    if (n <= 0 || t <= 0) return build_singleton_forest(std::max(1, n));

    std::vector<uint64_t> zobrist(n + 1, 0);
    for (int lbl = 1; lbl <= n; ++lbl) {
        zobrist[lbl] = mix64(static_cast<uint64_t>(lbl) * 0x9e3779b97f4a7c15ULL + 0x123456789abcdefULL);
    }

    std::vector<TreeData> trees;
    trees.reserve(t);

    for (const std::string& line : instance.newick_lines) {
        if (!parse_with_genesis(line)) {
            return build_singleton_forest(n);
        }

        SimpleTree st;
        if (!NewickParser(line).parse(st)) {
            return build_singleton_forest(n);
        }

        TreeData td;
        if (!convert_to_tree_data(st, n, zobrist, td)) {
            return build_singleton_forest(n);
        }
        trees.push_back(std::move(td));
    }

    // Build cherry score map for Phase A.
    std::unordered_map<uint64_t, int> cherry_score;
    cherry_score.reserve(static_cast<size_t>(n) * 4);
    for (const TreeData& td : trees) {
        for (const auto& p : td.cherries) {
            ++cherry_score[pair_key(p.first, p.second)];
        }
    }

    std::vector<std::pair<int, int>> scored_pairs;
    scored_pairs.reserve(cherry_score.size());
    for (const auto& kv : cherry_score) {
        int a = static_cast<int>(kv.first >> 32);
        int b = static_cast<int>(kv.first & 0xffffffffu);
        scored_pairs.push_back({a, b});
    }
    std::sort(scored_pairs.begin(), scored_pairs.end(), [&](const auto& x, const auto& y) {
        return cherry_score[pair_key(x.first, x.second)] > cherry_score[pair_key(y.first, y.second)];
    });

    const auto start = std::chrono::steady_clock::now();
    const char* env = std::getenv("STRIDE_TIMEOUT");
    double timeout_sec = 30.0;
    if (env) {
        char* end = nullptr;
        double val = std::strtod(env, &end);
        if (end != env && val > 0.0) timeout_sec = val;
    }
    const auto deadline = start + std::chrono::milliseconds(static_cast<long long>(timeout_sec * 850.0));

    // Best-so-far valid solution.
    std::vector<int> best_comp_of(n + 1, 0);
    for (int i = 1; i <= n; ++i) best_comp_of[i] = i;
    int best_k = n;
    publish_best_solution(build_singleton_forest(n));

    const int restarts = (n <= 1200 ? 6 : (n <= 3500 ? 3 : 1));

    for (int r = 0; r < restarts && !g_terminate && std::chrono::steady_clock::now() < deadline; ++r) {
        const auto now = std::chrono::steady_clock::now();
        auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count();
        if (remaining_ms <= 0) break;

        std::mt19937_64 rng(static_cast<uint64_t>(start.time_since_epoch().count()) + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(r + 1));

        Solution sol;
        init_singleton_solution(n, zobrist, sol);

        // Phase A: constructive merges, cherry prioritized.
        const auto phaseA_deadline = now + std::chrono::milliseconds(std::max<long long>(10, remaining_ms / 6));
        greedy_merge_closure(sol, trees, scored_pairs, rng, phaseA_deadline);

        // Phase B: main merge closure.
        auto rem_after_A = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        const auto phaseB_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max<long long>(10, rem_after_A * 2 / 3));
        greedy_merge_closure(sol, trees, scored_pairs, rng, phaseB_deadline);

        // Phase C: ILS local search / perturbation.
        local_search_ils(sol, trees, zobrist, rng, deadline, scored_pairs);

        if (!g_terminate && is_solution_valid(sol, trees) && sol.k < best_k) {
            best_k = sol.k;
            best_comp_of = sol.comp_of;
            publish_best_solution(build_forest_from_comp_of(best_comp_of, n, trees[0]));
        }
    }

    if (best_k >= n) return build_singleton_forest(n);
    return build_forest_from_comp_of(best_comp_of, n, trees[0]);
}

}  // namespace

int main() {
    std::signal(SIGTERM, sigterm_handler);

    PaceInstance instance;
    if (!read_instance_from_stdin(instance)) {
        const auto fallback = build_singleton_forest(std::max(1, g_last_leaf_count));
        const std::string out = normalize_forest_output(fallback);
        std::cout << out;
        return 0;
    }

    publish_best_solution(build_singleton_forest(instance.leaf_count));

    std::vector<std::string> final_forest = solve(instance);
    publish_best_solution(final_forest);

    if (g_best_output && g_best_output_len > 0) {
        std::cout.write(g_best_output, static_cast<std::streamsize>(g_best_output_len));
    }
    return 0;
}
