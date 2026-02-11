#include "genesis/genesis.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cctype>
#include <csignal>
#include <unistd.h>
#include <cstring>

using namespace std;
using genesis::tree::Tree;

static char* g_best_output = nullptr;
static size_t g_best_output_len = 0;

static void sigterm_handler(int) {
    if (g_best_output && g_best_output_len > 0) {
        ssize_t unused = write(STDOUT_FILENO, g_best_output, g_best_output_len);
        (void)unused;
    }
    _Exit(0);
}

static inline string trim(string s) {
    while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && isspace((unsigned char)s.back()))  s.pop_back();
    return s;
}

static Tree parse_newick_line(const string& newick_line) {
    genesis::tree::CommonTreeNewickReader reader;
    auto src = genesis::utils::from_string(newick_line);
    return reader.read(src);
}

vector<Tree> read_input_from_stdin(int& k_out, int& n_out) {
    vector<Tree> trees;
    string line;
    int k = -1, n = -1;

    // Find #p line
    while (getline(cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (line[0] == '#') {
            if (line.rfind("#p", 0) == 0) {
                istringstream ss(line);
                string tag;
                ss >> tag >> k >> n;
                break;
            }
            continue;
        }
    }

    if (k <= 0 || n <= 0) {
        throw runtime_error("Invalid or missing #p line");
    }

    while ((int)trees.size() < k && getline(cin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        trees.push_back(parse_newick_line(line));
    }

    if ((int)trees.size() != k) {
        throw runtime_error("Wrong number of trees");
    }

    k_out = k;
    n_out = n;
    return trees;
}

static void build_singleton_forest_output(int n) {
    std::string out;
    out.reserve(n * 4);
    for (int i = 1; i <= n; ++i) {
        out += std::to_string(i);
        out += ";\n";
    }

    g_best_output_len = out.size();
    g_best_output = new char[g_best_output_len];
    std::memcpy(g_best_output, out.data(), g_best_output_len);
}
int main() {
    int k = 0, n = 0;
    auto trees = read_input_from_stdin(k, n);
    (void)trees;
    build_singleton_forest_output(n);
    std::signal(SIGTERM, sigterm_handler);
    if (g_best_output && g_best_output_len > 0) {
        std::cout.write(g_best_output, g_best_output_len);
    }
    return 0;
}
