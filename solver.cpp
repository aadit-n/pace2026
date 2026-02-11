#include "genesis/genesis.hpp"

#include <cctype>
#include <csignal>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using genesis::tree::Tree;

namespace {

struct PaceInstance {
    int tree_count = 0;
    int leaf_count = 0;
    std::vector<Tree> trees;
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

    while (static_cast<int>(instance.trees.size()) < instance.tree_count && std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        instance.trees.push_back(parse_newick_line(line));
    }

    if (static_cast<int>(instance.trees.size()) != instance.tree_count) {
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

// Plug your algorithm here.
// Input: parsed PACE instance (tree_count, leaf_count, and parsed trees).
// Output: forest as one Newick tree per vector entry.
// Currently, the strategy in place is to print the signleton leaf tree forest.
std::vector<std::string> solve(const PaceInstance& instance) {
    std::vector<std::string> forest;
    forest.reserve(instance.leaf_count);
    for (int leaf = 1; leaf <= instance.leaf_count; ++leaf) {
        forest.push_back(std::to_string(leaf) + ";");
    }
    return forest;
}

}  // namespace

int main() {
    std::signal(SIGTERM, sigterm_handler);

    try {
        const PaceInstance instance = read_instance_from_stdin();

        // Always keep at least one valid fallback in case SIGTERM arrives mid-computation.
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
