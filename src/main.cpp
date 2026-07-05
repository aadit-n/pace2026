#include "io/PaceParser.hpp"
#include "core/Tree.hpp"
#include "core/Forest.hpp"
#include "core/Timer.hpp"
#include "core/Random.hpp"

#include "reductions/CommonSubtreeReduction.hpp"
#include "reductions/ChainReduction.hpp"
#include "reductions/ThreeTwoChainReduction.hpp"
#include "reductions/ClusterDecomposition.hpp"
#include "reductions/ReductionStack.hpp"

#include "heuristics/FastCherryApprox.hpp"
#include "heuristics/CherryPairApprox.hpp"
#include "heuristics/TwoApprox.hpp"
#include "heuristics/LocalImprove.hpp"
#include "heuristics/RandomRestart.hpp"
#include "heuristics/ActiveCherryGreedyApprox.hpp"
#include "heuristics/AgreementComponentPacking.hpp"
#include "heuristics/ForestCrossover.hpp"

#include "exact/InProcessExactMaf.hpp"
#include "exact/TinyExactOracle.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <functional>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <iomanip>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sstream>

#if defined(__unix__) || defined(__APPLE__)
#include <csignal>
#include <signal.h>
#include <unistd.h>
#endif

namespace {

using pace26::io::NewickTree;
using pace26::core::Tree;
using pace26::core::Timer;
using pace26::core::Random;

using CoreForest = pace26::core::LabelForest;
using CoreComponent = pace26::core::LabelComponent;
using ForestPublishCallback = std::function<void(const CoreForest&)>;

using RedForest = pace26::reductions::LabelForest;
using RedComponent = pace26::reductions::LabelComponent;

#if defined(__unix__) || defined(__APPLE__)
std::string g_emergency_output;
const char* g_emergency_output_data = nullptr;
std::size_t g_emergency_output_size = 0;

extern "C" void emergency_alarm_handler(int) {
    if (g_emergency_output_data != nullptr && g_emergency_output_size != 0) {
        const char* data = g_emergency_output_data;
        std::size_t left = g_emergency_output_size;

        while (left > 0) {
            ssize_t written = ::write(STDOUT_FILENO, data, left);
            if (written <= 0) {
                break;
            }
            data += written;
            left -= static_cast<std::size_t>(written);
        }
    }

    _exit(0);
}

void install_emergency_alarm(double seconds) {
    if (seconds < 1.0) {
        seconds = 1.0;
    }

    std::signal(SIGALRM, emergency_alarm_handler);
    std::signal(SIGTERM, emergency_alarm_handler);
    ::alarm(static_cast<unsigned int>(seconds));
}

void cancel_emergency_alarm() {
    ::alarm(0);
}

void set_emergency_output(std::string output) {
    sigset_t block;
    sigset_t previous;
    sigemptyset(&block);
    sigaddset(&block, SIGALRM);
    sigaddset(&block, SIGTERM);
    sigprocmask(SIG_BLOCK, &block, &previous);

    g_emergency_output = std::move(output);
    g_emergency_output_data = g_emergency_output.data();
    g_emergency_output_size = g_emergency_output.size();

    sigprocmask(SIG_SETMASK, &previous, nullptr);
}
#else
std::string g_emergency_output;
void install_emergency_alarm(double) {}
void cancel_emergency_alarm() {}
void set_emergency_output(std::string output) {
    g_emergency_output = std::move(output);
}
#endif

std::string make_singleton_output(const NewickTree& tree) {
    std::vector<std::uint32_t> labels;

    for (const auto& node : tree.nodes) {
        if (node.is_leaf()) {
            labels.push_back(node.label);
        }
    }

    std::sort(labels.begin(), labels.end());

    std::ostringstream out;
    for (std::uint32_t label : labels) {
        out << label << ";\n";
    }

    return out.str();
}


constexpr double kDefaultExternalTimeLimitSeconds = 300.0;
constexpr double kBaselineStageLimitSeconds = 30.0;
constexpr double kOutputReserveSeconds = 5.0;
constexpr std::size_t kHugeLimit = std::numeric_limits<std::size_t>::max() / 4;
constexpr bool kEnableExactWindowRepair = false;

struct SolverConfig {
    double time_limit_seconds = kDefaultExternalTimeLimitSeconds;
    std::uint64_t seed = 123456789ULL;
    bool extended_search = false;

    // These are the proven 30-second baseline settings. The extended stage is
    // constructed separately after this baseline has produced a valid incumbent.
    std::size_t cluster_min_leaves = 1500;
    int max_cluster_depth = 24;
    bool enable_cluster_bridge_recovery = false;

    std::size_t two_approx_max_leaves = 0;
    std::size_t exact_max_leaves = 90;
    bool enable_legacy_exact = false;
    std::size_t tiny_exact_whole_max_leaves = 10;
    std::size_t tiny_exact_cluster_max_leaves = 12;

    std::size_t three_two_max_leaves = kHugeLimit;
    std::size_t chain_max_leaves = kHugeLimit;
    std::size_t random_restart_max_leaves = 0;
    std::size_t local_improve_max_leaves = 1500;
    bool extended_quality_recovery = false;
    bool packing_focused = false;
    bool enable_singleton_rescue = false;
    bool enable_cluster_open_profile = true;
    double cluster_open_profile_min_elapsed_seconds = kBaselineStageLimitSeconds + 1.0;
    std::size_t cluster_open_profile_max_leaves = 900;
    std::size_t cluster_open_profile_large_max_leaves = 2800;
    std::size_t cluster_open_profile_side_max_leaves = 420;
    int cluster_open_profile_max_depth = 5;

    std::size_t direct_cheap_output_min_leaves = 4000;

    int max_reduction_passes = 40;

    // Leave empty for Optil. Use --profile-dir only for local profiling.
    std::string profile_dir;
};

SolverConfig make_extended_config(const SolverConfig& baseline) {
    SolverConfig extended = baseline;
    extended.extended_search = true;
    extended.cluster_min_leaves = 700;
    extended.two_approx_max_leaves = 0;
    extended.exact_max_leaves = 120;
    extended.three_two_max_leaves = kHugeLimit;
    extended.random_restart_max_leaves = 0;
    extended.local_improve_max_leaves = 2500;
    extended.extended_quality_recovery = false;
    extended.direct_cheap_output_min_leaves = kHugeLimit;
    extended.cluster_open_profile_max_leaves = 1400;
    extended.cluster_open_profile_large_max_leaves = 4500;
    extended.cluster_open_profile_side_max_leaves = 650;
    extended.cluster_open_profile_max_depth = 8;
    if (extended.packing_focused) {
        extended.two_approx_max_leaves = 0;
        extended.exact_max_leaves = 0;
        extended.random_restart_max_leaves = 0;
    }
    return extended;
}

void apply_packing_focused_config(SolverConfig& config) {
    if (!config.packing_focused) {
        return;
    }

    config.two_approx_max_leaves = 0;
    config.exact_max_leaves = 0;
    config.random_restart_max_leaves = 0;
}

bool default_packing_focused_for_instance(
    std::size_t n,
    double time_limit_seconds
) {
    if (time_limit_seconds <= kBaselineStageLimitSeconds + kOutputReserveSeconds) {
        return n >= 1000;
    }

    return n >= 2000;
}

using ActiveCherryPolicy = pace26::heuristics::ActiveCherryGreedyApprox::Policy;

struct ActiveCherryPortfolioRun {
    ActiveCherryPolicy policy = ActiveCherryPolicy::Balanced;
    bool swapped = false;
    std::size_t max_steps_multiplier = 4;
    std::size_t candidate_sample_cap = 64;
    double local_time_limit_seconds = 0.0;
    std::uint64_t sample_salt = 0;
    std::vector<int> rank_choice_script;
};

std::uint64_t mix_portfolio_salt(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

void add_active_cherry_run(
    std::vector<ActiveCherryPortfolioRun>& runs,
    ActiveCherryPolicy policy,
    bool swapped,
    std::size_t max_steps_multiplier,
    std::size_t candidate_sample_cap,
    double local_time_limit_seconds,
    std::uint64_t salt_tag,
    std::vector<int> rank_choice_script = {}
) {
    ActiveCherryPortfolioRun run;
    run.policy = policy;
    run.swapped = swapped;
    run.max_steps_multiplier = max_steps_multiplier;
    run.candidate_sample_cap = candidate_sample_cap;
    run.local_time_limit_seconds = local_time_limit_seconds;
    run.sample_salt = salt_tag == 0 ? 0 : mix_portfolio_salt(salt_tag);
    run.rank_choice_script = std::move(rank_choice_script);
    runs.push_back(std::move(run));
}

const char* active_cherry_policy_name(ActiveCherryPolicy policy) {
    switch (policy) {
        case ActiveCherryPolicy::Balanced: return "Balanced";
        case ActiveCherryPolicy::PreferSmallCuts: return "PreferSmallCuts";
        case ActiveCherryPolicy::PreferBigProgress: return "PreferBigProgress";
        case ActiveCherryPolicy::PreferDifferentComponent: return "PreferDifferentComponent";
        case ActiveCherryPolicy::PreferLowConflictMass: return "PreferLowConflictMass";
        case ActiveCherryPolicy::PreferFewPendants: return "PreferFewPendants";
        case ActiveCherryPolicy::PreferImmediateGain: return "PreferImmediateGain";
        case ActiveCherryPolicy::ConservativeSingleCut: return "ConservativeSingleCut";
        case ActiveCherryPolicy::DualityConservative: return "DualityConservative";
        case ActiveCherryPolicy::ResolveFinalCut: return "ResolveFinalCut";
        case ActiveCherryPolicy::AggressiveMultiCut: return "AggressiveMultiCut";
    }

    return "Unknown";
}

std::vector<ActiveCherryPortfolioRun> active_cherry_portfolio_for_size(
    std::size_t n,
    bool extended_search
) {
    std::vector<ActiveCherryPortfolioRun> runs;

    if (n >= 6000) {
        // The first run is the large-instance workhorse. The continuation adds
        // only the one large secondary run that has remained productive in
        // profiles. Broader policy diversification is handled later by the
        // Balanced seed-bank phase.
        add_active_cherry_run(runs, ActiveCherryPolicy::Balanced, false, 3, 32, 0.0, 0);
        if (extended_search) {
            add_active_cherry_run(runs, ActiveCherryPolicy::Balanced, true, 3, 96, 2.0, 102);
        }
        return runs;
    }

    if (n >= 1000) {
        add_active_cherry_run(runs, ActiveCherryPolicy::Balanced, false, 5, 64, 0.0, 0);
        add_active_cherry_run(runs, ActiveCherryPolicy::PreferSmallCuts, false, 5, 64, 0.0, 0);
        add_active_cherry_run(runs, ActiveCherryPolicy::PreferLowConflictMass, false, 4, 160, 1.5, 13);
        add_active_cherry_run(runs, ActiveCherryPolicy::Balanced, true, 4, 128, 1.5, 16);
        add_active_cherry_run(runs, ActiveCherryPolicy::PreferFewPendants, false, 4, 160, 1.0, 14);
        add_active_cherry_run(runs, ActiveCherryPolicy::PreferLowConflictMass, true, 4, 160, 1.0, 18);
        if (extended_search) {
            add_active_cherry_run(runs, ActiveCherryPolicy::PreferDifferentComponent, false, 4, 192, 1.0, 31, {1});
            add_active_cherry_run(runs, ActiveCherryPolicy::PreferBigProgress, true, 4, 160, 1.0, 32, {1});
        }
        return runs;
    }

    add_active_cherry_run(runs, ActiveCherryPolicy::Balanced, false, 8, 96, 0.0, 0);
    add_active_cherry_run(runs, ActiveCherryPolicy::PreferSmallCuts, false, 8, 96, 0.0, 21);
    add_active_cherry_run(runs, ActiveCherryPolicy::PreferDifferentComponent, false, 8, 128, 0.0, 22);
    add_active_cherry_run(runs, ActiveCherryPolicy::PreferBigProgress, false, 8, 96, 0.0, 23);
    add_active_cherry_run(runs, ActiveCherryPolicy::PreferLowConflictMass, false, 6, 128, 0.75, 24);
    add_active_cherry_run(runs, ActiveCherryPolicy::PreferFewPendants, false, 6, 128, 0.75, 25);
    add_active_cherry_run(runs, ActiveCherryPolicy::PreferImmediateGain, false, 6, 96, 0.75, 26);
    add_active_cherry_run(runs, ActiveCherryPolicy::Balanced, true, 6, 96, 0.75, 27);
    add_active_cherry_run(runs, ActiveCherryPolicy::PreferLowConflictMass, true, 6, 128, 0.5, 28);
    if (extended_search && n >= 220) {
        add_active_cherry_run(runs, ActiveCherryPolicy::AggressiveMultiCut, false, 6, 160, 0.5, 29, {1});
        add_active_cherry_run(runs, ActiveCherryPolicy::DualityConservative, true, 6, 160, 0.5, 30, {1});
    }

    return runs;
}

CoreForest to_core_forest(const RedForest& forest) {
    CoreForest out;
    out.components.reserve(forest.components.size());

    for (const RedComponent& component : forest.components) {
        out.add_component(CoreComponent(component.labels));
    }

    out.normalize();
    out.validate_no_duplicates();
    return out;
}

RedForest to_reduction_forest(const CoreForest& forest) {
    RedForest out;
    out.components.reserve(forest.components.size());

    for (const CoreComponent& component : forest.components) {
        RedComponent c;
        c.labels = component.labels;
        out.components.push_back(std::move(c));
    }

    return out;
}

std::size_t count_leaves(const NewickTree& tree) {
    std::size_t count = 0;

    for (const auto& node : tree.nodes) {
        if (node.is_leaf()) {
            ++count;
        }
    }

    return count;
}

std::string json_quote(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    out.push_back('"');

    for (unsigned char ch : text) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (ch < 0x20) {
                    std::ostringstream hex;
                    hex << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch);
                    out += hex.str();
                } else {
                    out.push_back(static_cast<char>(ch));
                }
                break;
        }
    }

    out.push_back('"');
    return out;
}

std::uint64_t profile_mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

std::uint64_t hash_newick_tree_for_profile(const NewickTree& tree) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    h ^= profile_mix64(static_cast<std::uint64_t>(tree.root) + 1ULL);

    for (const auto& node : tree.nodes) {
        h ^= profile_mix64(static_cast<std::uint64_t>(node.id) + 0x10ULL);
        h *= 0x100000001b3ULL;
        h ^= profile_mix64(static_cast<std::uint64_t>(node.label) + 0x20ULL);
        h *= 0x100000001b3ULL;
        h ^= profile_mix64(static_cast<std::uint64_t>(node.left + 2) + 0x30ULL);
        h *= 0x100000001b3ULL;
        h ^= profile_mix64(static_cast<std::uint64_t>(node.right + 2) + 0x40ULL);
        h *= 0x100000001b3ULL;
    }

    return h;
}

std::string profile_hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

class ProfileLogger {
public:
    void open(
        const SolverConfig& config,
        const pace26::io::PaceInstance& instance
    ) {
        if (config.profile_dir.empty()) {
            return;
        }

        try {
            std::filesystem::create_directories(config.profile_dir);

            std::uint64_t h = 0x9e3779b97f4a7c15ULL;
            for (const NewickTree& tree : instance.trees) {
                h ^= profile_mix64(hash_newick_tree_for_profile(tree));
                h *= 0x100000001b3ULL;
            }
            h ^= profile_mix64(static_cast<std::uint64_t>(instance.num_leaves));

            instance_id_ = profile_hex64(h);
            const std::filesystem::path path =
                std::filesystem::path(config.profile_dir) /
                ("profile_n" + std::to_string(instance.num_leaves) + "_" + instance_id_ + ".jsonl");

            out_.open(path, std::ios::out | std::ios::trunc);
            if (!out_) {
                return;
            }

            enabled_ = true;
            started_ = Timer::Clock::now();

            event_raw(
                "instance_start",
                nullptr,
                {
                    {"num_trees", std::to_string(instance.num_trees)},
                    {"num_leaves", std::to_string(instance.num_leaves)},
                    {"time_limit_seconds", std::to_string(config.time_limit_seconds)},
                    {"direct_min", std::to_string(config.direct_cheap_output_min_leaves)},
                    {"two_max", std::to_string(config.two_approx_max_leaves)},
                    {"rr_max", std::to_string(config.random_restart_max_leaves)},
                    {"li_max", std::to_string(config.local_improve_max_leaves)},
                    {"exact_max", std::to_string(config.exact_max_leaves)},
                    {"tiny_exact_whole_max", std::to_string(config.tiny_exact_whole_max_leaves)},
                    {"tiny_exact_cluster_max", std::to_string(config.tiny_exact_cluster_max_leaves)},
                    {"packing_focused", config.packing_focused ? "true" : "false"}
                }
            );
        } catch (const std::exception&) {
            enabled_ = false;
        }
    }

    bool enabled() const noexcept {
        return enabled_;
    }

    double wall_ms() const {
        const auto now = Timer::Clock::now();
        const std::chrono::duration<double, std::milli> elapsed = now - started_;
        return elapsed.count();
    }

    void event_raw(
        const std::string& event,
        const Timer* timer,
        std::initializer_list<std::pair<std::string, std::string>> fields = {}
    ) {
        if (!enabled_) {
            return;
        }

        out_ << "{"
             << "\"event\":" << json_quote(event)
             << ",\"instance_id\":" << json_quote(instance_id_)
             << ",\"wall_ms\":" << wall_ms();

        if (timer != nullptr) {
            out_ << ",\"timer_elapsed_seconds\":" << timer->elapsed_seconds()
                 << ",\"timer_remaining_seconds\":" << timer->remaining_seconds();
        }

        for (const auto& [key, value] : fields) {
            out_ << "," << json_quote(key) << ":" << value;
        }

        out_ << "}\n";
        out_.flush();
    }

    void phase_result(
        const std::string& phase,
        const Timer& timer,
        double phase_start_seconds,
        std::size_t n,
        std::size_t before_components,
        std::size_t after_components,
        bool accepted,
        std::initializer_list<std::pair<std::string, std::string>> extra = {}
    ) {
        if (!enabled_) {
            return;
        }

        std::vector<std::pair<std::string, std::string>> fields;
        fields.reserve(extra.size() + 7);
        fields.push_back({"phase", json_quote(phase)});
        fields.push_back({"n", std::to_string(n)});
        fields.push_back({"duration_seconds", std::to_string(timer.elapsed_seconds() - phase_start_seconds)});
        fields.push_back({"before_components", std::to_string(before_components)});
        fields.push_back({"after_components", std::to_string(after_components)});
        fields.push_back({"delta_components", std::to_string(static_cast<long long>(after_components) - static_cast<long long>(before_components))});
        fields.push_back({"accepted", accepted ? "true" : "false"});
        for (const auto& kv : extra) {
            fields.push_back(kv);
        }

        event_vector("phase_result", &timer, fields);
    }

    void event_vector(
        const std::string& event,
        const Timer* timer,
        const std::vector<std::pair<std::string, std::string>>& fields
    ) {
        if (!enabled_) {
            return;
        }

        out_ << "{"
             << "\"event\":" << json_quote(event)
             << ",\"instance_id\":" << json_quote(instance_id_)
             << ",\"wall_ms\":" << wall_ms();

        if (timer != nullptr) {
            out_ << ",\"timer_elapsed_seconds\":" << timer->elapsed_seconds()
                 << ",\"timer_remaining_seconds\":" << timer->remaining_seconds();
        }

        for (const auto& [key, value] : fields) {
            out_ << "," << json_quote(key) << ":" << value;
        }

        out_ << "}\n";
        out_.flush();
    }

    void close(const Timer* timer = nullptr) {
        if (!enabled_) {
            return;
        }
        event_raw("instance_end", timer);
        out_.flush();
        out_.close();
        enabled_ = false;
    }

private:
    bool enabled_ = false;
    std::string instance_id_;
    Timer::Clock::time_point started_ = Timer::Clock::now();
    std::ofstream out_;
};

ProfileLogger g_profile;

std::uint32_t max_node_or_label_id(const NewickTree& tree) {
    std::uint32_t result = 0;

    for (const auto& node : tree.nodes) {
        result = std::max(result, node.id);
        result = std::max(result, node.label);
    }

    return result;
}

std::uint32_t choose_initial_placeholder(const NewickTree& t1, const NewickTree& t2) {
    std::uint32_t mx = std::max(max_node_or_label_id(t1), max_node_or_label_id(t2));

    if (mx == std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("cannot allocate placeholder label");
    }

    return mx + 1;
}

bool is_valid_partition(CoreForest forest, const std::vector<std::uint32_t>& expected_labels) {
    try {
        forest.normalize();
        forest.validate_partition_of(expected_labels);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

struct AgreementSignatureKey {
    std::uint64_t a = 0;
    std::uint64_t b = 0;

    bool operator==(const AgreementSignatureKey& other) const noexcept {
        return a == other.a && b == other.b;
    }
};

struct AgreementSignatureKeyHash {
    std::size_t operator()(const AgreementSignatureKey& key) const noexcept {
        std::uint64_t x = key.a + 0x9e3779b97f4a7c15ULL;
        x ^= key.b + 0xbf58476d1ce4e5b9ULL + (x << 6) + (x >> 2);
        return static_cast<std::size_t>(x);
    }
};

class AgreementSignatureInterner {
public:
    int leaf(std::uint32_t label) {
        return intern({0, static_cast<std::uint64_t>(label)});
    }

    int internal(int left, int right) {
        if (left <= 0 || right <= 0) {
            throw std::runtime_error("invalid agreement signature");
        }

        std::uint64_t a = static_cast<std::uint64_t>(left);
        std::uint64_t b = static_cast<std::uint64_t>(right);
        if (b < a) {
            std::swap(a, b);
        }

        return intern({a, b});
    }

private:
    std::unordered_map<AgreementSignatureKey, int, AgreementSignatureKeyHash> ids_;
    int next_id_ = 1;

    int intern(const AgreementSignatureKey& key) {
        auto it = ids_.find(key);
        if (it != ids_.end()) {
            return it->second;
        }

        const int id = next_id_++;
        ids_.emplace(key, id);
        return id;
    }
};

struct AgreementValidationWorkspace {
    std::vector<int> nodes;
    std::vector<int> parent_index;
    std::vector<int> stack;
    std::vector<int> signature;
};

int restricted_agreement_signature_virtual(
    const Tree& tree,
    const std::vector<std::uint32_t>& labels,
    AgreementSignatureInterner& interner,
    AgreementValidationWorkspace& workspace
) {
    if (labels.empty()) {
        return 0;
    }

    if (labels.size() == 1) {
        return interner.leaf(labels.front());
    }

    std::vector<int>& nodes = workspace.nodes;
    nodes.clear();
    nodes.reserve(labels.size() * 2);

    for (std::uint32_t label : labels) {
        nodes.push_back(tree.node_of_label(label));
    }

    std::sort(
        nodes.begin(),
        nodes.end(),
        [&](int a, int b) {
            return tree.tin[static_cast<std::size_t>(a)] <
                   tree.tin[static_cast<std::size_t>(b)];
        }
    );
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    const std::size_t leaf_node_count = nodes.size();
    for (std::size_t i = 1; i < leaf_node_count; ++i) {
        nodes.push_back(tree.lca(nodes[i - 1], nodes[i]));
    }

    std::sort(
        nodes.begin(),
        nodes.end(),
        [&](int a, int b) {
            return tree.tin[static_cast<std::size_t>(a)] <
                   tree.tin[static_cast<std::size_t>(b)];
        }
    );
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    const std::size_t k = nodes.size();
    std::vector<int>& parent_index = workspace.parent_index;
    parent_index.assign(k, -1);
    std::vector<int>& stack = workspace.stack;
    stack.clear();
    stack.reserve(k);

    for (std::size_t i = 0; i < k; ++i) {
        const int u = nodes[i];
        while (!stack.empty() &&
                   !tree.is_ancestor(nodes[static_cast<std::size_t>(stack.back())], u)) {
            stack.pop_back();
        }
        if (!stack.empty()) {
            parent_index[i] = stack.back();
        }
        stack.push_back(static_cast<int>(i));
    }

    std::vector<int>& signature = workspace.signature;
    signature.assign(k, 0);
    for (std::size_t rev = k; rev > 0; --rev) {
        const std::size_t i = rev - 1;
        const int u = nodes[i];
        const auto& node = tree.nodes[static_cast<std::size_t>(u)];

        int sig = signature[i];
        if (node.is_leaf()) {
            sig = interner.leaf(node.label);
        }

        signature[i] = sig;
        if (sig > 0 && parent_index[i] >= 0) {
            int& parent_sig =
                signature[static_cast<std::size_t>(parent_index[i])];
            if (parent_sig <= 0) {
                parent_sig = sig;
            } else {
                parent_sig = interner.internal(parent_sig, sig);
            }
        }
    }

    return signature.empty() ? 0 : signature.front();
}

bool component_has_same_restricted_topology(
    const CoreComponent& component,
    const Tree& t1,
    const Tree& t2,
    AgreementValidationWorkspace& workspace
) {
    if (component.size() <= 2) {
        return true;
    }

    AgreementSignatureInterner interner;
    const int sig1 =
        restricted_agreement_signature_virtual(t1, component.labels, interner, workspace);
    const int sig2 =
        restricted_agreement_signature_virtual(t2, component.labels, interner, workspace);

    return sig1 > 0 && sig1 == sig2;
}

bool components_are_edge_disjoint(
    const CoreForest& forest,
    const Tree& tree
) {
    std::vector<int> edge_owner(static_cast<std::size_t>(tree.node_count()), -1);
    std::vector<int> seen_in_component(static_cast<std::size_t>(tree.node_count()), 0);
    int stamp = 0;

    for (std::size_t i = 0; i < forest.components.size(); ++i) {
        const auto& labels = forest.components[i].labels;
        if (labels.size() <= 1) {
            continue;
        }

        int root = tree.node_of_label(labels.front());
        for (std::size_t j = 1; j < labels.size(); ++j) {
            root = tree.lca(root, tree.node_of_label(labels[j]));
        }

        ++stamp;
        if (stamp == std::numeric_limits<int>::max()) {
            std::fill(seen_in_component.begin(), seen_in_component.end(), 0);
            stamp = 1;
        }

        for (std::uint32_t label : labels) {
            int u = tree.node_of_label(label);

            while (u != root) {
                int& seen = seen_in_component[static_cast<std::size_t>(u)];
                if (seen != stamp) {
                    seen = stamp;

                    int& owner = edge_owner[static_cast<std::size_t>(u)];
                    if (owner != -1) {
                        return false;
                    }
                    owner = static_cast<int>(i);
                }

                u = tree.parent[static_cast<std::size_t>(u)];
                if (u < 0) {
                    throw std::runtime_error("invalid parent path in agreement validator");
                }
            }
        }
    }

    return true;
}

bool is_valid_agreement_forest(
    const CoreForest& forest,
    const Tree& t1,
    const Tree& t2
) {
    if (t1.leaf_labels != t2.leaf_labels) {
        return false;
    }

    try {
        for (const CoreComponent& component : forest.components) {
            if (component.empty()) {
                return false;
            }
        }
        forest.validate_partition_of(t1.leaf_labels);

        AgreementValidationWorkspace validation_workspace;
        for (const CoreComponent& component : forest.components) {
            if (!component_has_same_restricted_topology(
                    component,
                    t1,
                    t2,
                    validation_workspace)) {
                return false;
            }
        }

        return components_are_edge_disjoint(forest, t1) &&
               components_are_edge_disjoint(forest, t2);
    } catch (const std::exception&) {
        return false;
    }
}

std::uint64_t tie_score(const CoreForest& forest) {
    std::uint64_t score = 0;

    for (const CoreComponent& c : forest.components) {
        const std::uint64_t s = static_cast<std::uint64_t>(c.labels.size());
        score += s * s;
    }

    return score;
}

bool consider_candidate(
    CoreForest& incumbent,
    CoreForest candidate,
    const Tree& t1,
    const Tree& t2
) {
    candidate.normalize();
    incumbent.normalize();

    const std::size_t candidate_components = candidate.component_count();
    const std::size_t incumbent_components = incumbent.component_count();
    if (candidate_components > incumbent_components) {
        return false;
    }

    bool can_improve = candidate_components < incumbent_components;
    if (!can_improve) {
        can_improve = tie_score(candidate) > tie_score(incumbent);
    }
    if (!can_improve) {
        return false;
    }

    if (!is_valid_agreement_forest(candidate, t1, t2)) {
        return false;
    }

    incumbent = std::move(candidate);
    return true;
}

std::optional<CoreForest> run_tiny_exact_oracle(
    const Tree& t1,
    const Tree& t2,
    Timer& timer,
    std::size_t max_leaves,
    const std::string& phase,
    int depth
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    if (max_leaves == 0 || n == 0 || n > max_leaves || timer.should_stop(0.25)) {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote(phase)},
                {"n", std::to_string(n)},
                {"reason", json_quote(n > max_leaves ? "size_limit" : "timer_guard")},
                {"max_leaves", std::to_string(max_leaves)},
                {"depth", std::to_string(depth)}
            }
        );
        return std::nullopt;
    }

    const double phase_start = timer.elapsed_seconds();
    const std::size_t before = n;
    pace26::exact::TinyExactOracle::Stats stats;
    bool accepted = false;
    bool produced = false;
    bool valid = false;
    bool threw = false;
    std::size_t components = 0;

    try {
        pace26::exact::TinyExactOracle::Options opts;
        opts.max_leaves = max_leaves;
        opts.max_search_nodes = max_leaves <= 10 ? 2000000ULL : 8000000ULL;
        opts.guard_seconds = 0.05;

        pace26::exact::TinyExactOracle oracle(opts);
        std::optional<CoreForest> exact = oracle.solve(t1, t2, &timer, &stats);

        if (exact.has_value()) {
            produced = true;
            exact->normalize();
            components = exact->component_count();
            valid = is_valid_agreement_forest(*exact, t1, t2);
            accepted = valid;

            g_profile.phase_result(
                phase,
                timer,
                phase_start,
                n,
                before,
                valid ? components : before,
                accepted,
                {
                    {"depth", std::to_string(depth)},
                    {"max_leaves", std::to_string(max_leaves)},
                    {"produced", produced ? "true" : "false"},
                    {"valid", valid ? "true" : "false"},
                    {"cache_hit", stats.cache_hit ? "true" : "false"},
                    {"timed_out", stats.timed_out ? "true" : "false"},
                    {"node_limit_hit", stats.node_limit_hit ? "true" : "false"},
                    {"subsets_tested", std::to_string(stats.subsets_tested)},
                    {"valid_components", std::to_string(stats.valid_components)},
                    {"search_nodes", std::to_string(stats.search_nodes)},
                    {"cache_entries", std::to_string(stats.cache_entries)},
                    {"exception", "false"}
                }
            );

            if (valid) {
                return exact;
            }
            return std::nullopt;
        }
    } catch (const std::exception&) {
        threw = true;
    }

    g_profile.phase_result(
        phase,
        timer,
        phase_start,
        n,
        before,
        before,
        false,
        {
            {"depth", std::to_string(depth)},
            {"max_leaves", std::to_string(max_leaves)},
            {"produced", produced ? "true" : "false"},
            {"valid", valid ? "true" : "false"},
            {"cache_hit", stats.cache_hit ? "true" : "false"},
            {"timed_out", stats.timed_out ? "true" : "false"},
            {"node_limit_hit", stats.node_limit_hit ? "true" : "false"},
            {"subsets_tested", std::to_string(stats.subsets_tested)},
            {"valid_components", std::to_string(stats.valid_components)},
            {"search_nodes", std::to_string(stats.search_nodes)},
            {"cache_entries", std::to_string(stats.cache_entries)},
            {"exception", threw ? "true" : "false"}
        }
    );

    return std::nullopt;
}

CoreForest reconstruct_cluster_bridge(
    CoreForest top,
    CoreForest bottom,
    std::uint32_t x_up,
    std::uint32_t x_down
) {
    CoreForest merged;
    merged.components.reserve(top.components.size() + bottom.components.size());

    std::size_t top_idx = -1;
    for (std::size_t i = 0; i < top.components.size(); ++i) {
        auto it = std::find(top.components[i].labels.begin(), top.components[i].labels.end(), x_up);
        if (it != top.components[i].labels.end()) {
            top_idx = i;
            break;
        }
    }

    std::size_t bottom_idx = -1;
    for (std::size_t i = 0; i < bottom.components.size(); ++i) {
        auto it = std::find(bottom.components[i].labels.begin(), bottom.components[i].labels.end(), x_down);
        if (it != bottom.components[i].labels.end()) {
            bottom_idx = i;
            break;
        }
    }

    if (top_idx == (std::size_t)-1 || bottom_idx == (std::size_t)-1) {
        return CoreForest::unite(top, bottom);
    }

    CoreComponent merged_comp;
    for (std::uint32_t lbl : top.components[top_idx].labels) {
        if (lbl != x_up) {
            merged_comp.labels.push_back(lbl);
        }
    }
    for (std::uint32_t lbl : bottom.components[bottom_idx].labels) {
        if (lbl != x_down) {
            merged_comp.labels.push_back(lbl);
        }
    }
    merged_comp.normalize();

    for (std::size_t i = 0; i < top.components.size(); ++i) {
        if (i != top_idx) {
            merged.add_component(top.components[i]);
        }
    }

    for (std::size_t i = 0; i < bottom.components.size(); ++i) {
        if (i != bottom_idx) {
            merged.add_component(bottom.components[i]);
        }
    }

    if (!merged_comp.labels.empty()) {
        merged.add_component(std::move(merged_comp));
    }

    merged.normalize();
    return merged;
}

std::optional<CoreForest> try_cheap_cluster_bridge(
    const CoreForest& top,
    const CoreForest& bottom,
    const NewickTree& nt1,
    const NewickTree& nt2,
    const Timer& timer
) {
    if (timer.should_stop(1.5) || top.empty() || bottom.empty()) {
        return std::nullopt;
    }

    Tree t1 = Tree::from_newick(nt1);
    Tree t2 = Tree::from_newick(nt2);
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    const std::size_t max_top_component_size = n <= 1500 ? 256 : 160;
    const std::size_t max_merged_component_size = n <= 1500 ? 512 : 256;
    const std::size_t max_pairs_to_test = n <= 1500 ? 2048 : 768;

    struct MergeCandidate {
        std::size_t top_index = 0;
        std::size_t bottom_index = 0;
        std::size_t merged_size = 0;
    };

    std::vector<MergeCandidate> candidates;
    candidates.reserve(std::min<std::size_t>(max_pairs_to_test, top.components.size() * bottom.components.size()));
    for (std::size_t ti = 0; ti < top.components.size(); ++ti) {
        const std::size_t top_size = top.components[ti].labels.size();
        if (top_size == 0 || top_size > max_top_component_size) {
            continue;
        }
        for (std::size_t bi = 0; bi < bottom.components.size(); ++bi) {
            const std::size_t bottom_size = bottom.components[bi].labels.size();
            const std::size_t merged_size = top_size + bottom_size;
            if (bottom_size == 0 || merged_size < 3 || merged_size > max_merged_component_size) {
                continue;
            }
            candidates.push_back({ti, bi, merged_size});
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.merged_size != b.merged_size) {
            return a.merged_size > b.merged_size;
        }
        if (a.top_index != b.top_index) {
            return a.top_index < b.top_index;
        }
        return a.bottom_index < b.bottom_index;
    });
    if (candidates.size() > max_pairs_to_test) {
        candidates.resize(max_pairs_to_test);
    }

    for (const MergeCandidate& merge : candidates) {
        if (timer.should_stop(1.5)) {
            break;
        }

        CoreForest bridged;
        bridged.components.reserve(top.components.size() + bottom.components.size() - 1);
        for (std::size_t i = 0; i < top.components.size(); ++i) {
            if (i != merge.top_index) {
                bridged.add_component(top.components[i]);
            }
        }
        for (std::size_t i = 0; i < bottom.components.size(); ++i) {
            if (i != merge.bottom_index) {
                bridged.add_component(bottom.components[i]);
            }
        }

        CoreComponent merged = top.components[merge.top_index];
        merged.add_many(bottom.components[merge.bottom_index].labels);
        bridged.add_component(std::move(merged));
        bridged.normalize();

        if (bridged.component_count() + 1 != top.component_count() + bottom.component_count()) {
            continue;
        }
        if (is_valid_agreement_forest(bridged, t1, t2)) {
            return bridged;
        }
    }

    return std::nullopt;
}

struct StateSolution {
    CoreForest forest;
    int lb = 0;
    int ub = std::numeric_limits<int>::max();
    bool certified = false;
    int distinguished_component = -1;
    bool available = false;
};

struct ClusterProfile {
    StateSolution closed;
    StateSolution open;
};

struct ClusterProfileMemo {
    std::unordered_map<std::string, StateSolution> open_state_by_key;
    std::unordered_map<
        std::string,
        pace26::reductions::ClusterSplitTemplate
    > split_template_by_key;
};

int component_count_int(const CoreForest& forest) {
    return static_cast<int>(
        std::min<std::size_t>(
            forest.component_count(),
            static_cast<std::size_t>(std::numeric_limits<int>::max())
        )
    );
}

StateSolution make_state_solution(
    CoreForest forest,
    bool certified = false,
    int distinguished_component = -1
) {
    forest.normalize();
    StateSolution state;
    state.forest = std::move(forest);
    state.ub = component_count_int(state.forest);
    state.lb = certified ? state.ub : 0;
    state.certified = certified;
    state.distinguished_component = distinguished_component;
    state.available = true;
    return state;
}

bool state_better_than(
    const StateSolution& candidate,
    const StateSolution& incumbent
) {
    if (!candidate.available) {
        return false;
    }
    if (!incumbent.available) {
        return true;
    }
    if (candidate.ub != incumbent.ub) {
        return candidate.ub < incumbent.ub;
    }
    return tie_score(candidate.forest) > tie_score(incumbent.forest);
}

std::string cluster_hierarchy_key(
    const NewickTree& nt1,
    const NewickTree& nt2
) {
    std::string key;
    key.reserve(96);
    key += "pair:";
    key += std::to_string(nt1.root);
    key.push_back(':');
    key += std::to_string(nt1.nodes.size());
    key.push_back(':');
    key += profile_hex64(hash_newick_tree_for_profile(nt1));
    key.push_back('|');
    key += std::to_string(nt2.root);
    key.push_back(':');
    key += std::to_string(nt2.nodes.size());
    key.push_back(':');
    key += profile_hex64(hash_newick_tree_for_profile(nt2));
    return key;
}

std::string cluster_open_profile_key(
    const NewickTree& nt1,
    const NewickTree& nt2,
    const pace26::reductions::ClusterSplit& split,
    const SolverConfig& config
) {
    std::string key;
    key.reserve(192);
    key += "open:";
    key += config.extended_search ? "ext" : "base";
    key.push_back(':');
    key += std::to_string(config.cluster_min_leaves);
    key.push_back(':');
    key += std::to_string(config.max_cluster_depth);
    key.push_back(':');
    key += std::to_string(config.exact_max_leaves);
    key.push_back(':');
    key += std::to_string(config.tiny_exact_whole_max_leaves);
    key.push_back(':');
    key += std::to_string(config.tiny_exact_cluster_max_leaves);
    key.push_back(':');
    key += config.packing_focused ? "pf" : "std";
    key.push_back(':');
    key += std::to_string(split.record.x_up_label);
    key.push_back(':');
    key += std::to_string(split.record.x_down_label);
    key.push_back('|');
    key += cluster_hierarchy_key(nt1, nt2);
    return key;
}

struct ClusterSplitLookup {
    pace26::reductions::ClusterSplit split;
    bool hierarchy_cache_hit = false;
    bool stale_cache_entry = false;
};

ClusterSplitLookup find_cluster_split_with_hierarchy_cache(
    const NewickTree& nt1,
    const NewickTree& nt2,
    std::uint32_t& next_placeholder,
    ClusterProfileMemo* cluster_profile_memo
) {
    ClusterSplitLookup lookup;

    if (cluster_profile_memo == nullptr) {
        lookup.split = pace26::reductions::ClusterDecomposition::split_once(
            nt1,
            nt2,
            next_placeholder
        );
        return lookup;
    }

    const std::string key = cluster_hierarchy_key(nt1, nt2);
    auto cached = cluster_profile_memo->split_template_by_key.find(key);
    if (cached != cluster_profile_memo->split_template_by_key.end()) {
        if (pace26::reductions::ClusterDecomposition::template_matches(
                nt1,
                nt2,
                cached->second)) {
            lookup.hierarchy_cache_hit = true;
            lookup.split = pace26::reductions::ClusterDecomposition::materialize_split(
                nt1,
                nt2,
                cached->second,
                next_placeholder
            );
            return lookup;
        }

        lookup.stale_cache_entry = true;
        cluster_profile_memo->split_template_by_key.erase(cached);
    }

    pace26::reductions::ClusterSplitTemplate templ =
        pace26::reductions::ClusterDecomposition::find_best_common_cluster_template(
            nt1,
            nt2
        );
    if (templ.found) {
        cluster_profile_memo->split_template_by_key.emplace(key, templ);
    }
    lookup.split = pace26::reductions::ClusterDecomposition::materialize_split(
        nt1,
        nt2,
        templ,
        next_placeholder
    );
    return lookup;
}

bool should_attempt_cluster_open_profile(
    std::size_t leaves,
    std::size_t bottom_leaves,
    std::size_t top_leaves,
    const SolverConfig& config,
    const Timer& timer,
    int depth
) {
    if (!config.enable_cluster_open_profile ||
        depth > config.cluster_open_profile_max_depth ||
        timer.should_stop(4.0)) {
        return false;
    }

    if (leaves <= 150) {
        return true;
    }

    // Marker recursion is correctness-useful but too expensive for the
    // 30-second anytime prefix. Keep the old tiny-cluster behavior above, and
    // defer larger OPEN states until the baseline answer has already had time
    // to publish.
    if (timer.elapsed_seconds() < config.cluster_open_profile_min_elapsed_seconds) {
        return false;
    }

    if (leaves <= config.cluster_open_profile_max_leaves) {
        return true;
    }

    const std::size_t smaller_side = std::min(bottom_leaves, top_leaves);
    if (leaves <= config.cluster_open_profile_large_max_leaves &&
        smaller_side <= config.cluster_open_profile_side_max_leaves &&
        !timer.should_stop(config.extended_search ? 12.0 : 7.0)) {
        return true;
    }

    return false;
}

class ComponentForestArchive;

CoreForest solve_instance(
    NewickTree nt1,
    NewickTree nt2,
    const SolverConfig& config,
    Timer& timer,
    Random& rng,
    int depth,
    const ForestPublishCallback& publish_progress,
    ComponentForestArchive* archive,
    ClusterProfileMemo* cluster_profile_memo
);

std::optional<StateSolution> solve_cluster_open_state(
    const NewickTree& nt1,
    const NewickTree& nt2,
    const pace26::reductions::ClusterSplit& split,
    const SolverConfig& child_config,
    Timer& timer,
    const Random& parent_rng,
    int depth,
    ClusterProfileMemo* cluster_profile_memo
) {
    if (timer.should_stop(4.0)) {
        return std::nullopt;
    }

    std::string key;
    if (cluster_profile_memo != nullptr) {
        key = cluster_open_profile_key(nt1, nt2, split, child_config);
        auto found = cluster_profile_memo->open_state_by_key.find(key);
        if (found != cluster_profile_memo->open_state_by_key.end()) {
            return found->second;
        }
    }

    const double phase_start = timer.elapsed_seconds();
    bool valid = false;
    bool threw = false;
    StateSolution state;

    try {
        const std::uint64_t open_seed =
            profile_mix64(parent_rng.seed()) ^
            profile_mix64(hash_newick_tree_for_profile(nt1)) ^
            profile_mix64(hash_newick_tree_for_profile(nt2)) ^
            profile_mix64(static_cast<std::uint64_t>(split.record.x_up_label) << 32) ^
            profile_mix64(split.record.x_down_label);
        Random open_rng(open_seed);

        pace26::reductions::ClusterOpenTrees open_trees =
            pace26::reductions::ClusterDecomposition::materialize_open_trees(
                nt1,
                nt2,
                split.record
            );

        CoreForest bottom_with = solve_instance(
            std::move(open_trees.bottom1_with_marker),
            std::move(open_trees.bottom2_with_marker),
            child_config,
            timer,
            open_rng,
            depth + 1,
            ForestPublishCallback{},
            nullptr,
            cluster_profile_memo
        );
        CoreForest top_with = solve_instance(
            std::move(open_trees.top1_with_placeholder),
            std::move(open_trees.top2_with_placeholder),
            child_config,
            timer,
            open_rng,
            depth + 1,
            ForestPublishCallback{},
            nullptr,
            cluster_profile_memo
        );

        CoreForest bridged = reconstruct_cluster_bridge(
            std::move(top_with),
            std::move(bottom_with),
            split.record.x_up_label,
            split.record.x_down_label
        );

        Tree reduced_t1 = Tree::from_newick(nt1);
        Tree reduced_t2 = Tree::from_newick(nt2);
        valid = is_valid_agreement_forest(bridged, reduced_t1, reduced_t2);
        if (valid) {
            state = make_state_solution(std::move(bridged), false, -1);
            if (cluster_profile_memo != nullptr) {
                cluster_profile_memo->open_state_by_key.emplace(key, state);
            }
        }
    } catch (const std::exception&) {
        threw = true;
    }

    g_profile.event_raw(
        "cluster_open_profile",
        &timer,
        {
            {"depth", std::to_string(depth)},
            {"leaves", std::to_string(count_leaves(nt1))},
            {"valid", valid ? "true" : "false"},
            {"available", state.available ? "true" : "false"},
            {"components", std::to_string(state.available ? state.ub : -1)},
            {"duration_seconds", std::to_string(timer.elapsed_seconds() - phase_start)},
            {"exception", threw ? "true" : "false"}
        }
    );

    if (!state.available) {
        return std::nullopt;
    }
    return state;
}

enum class AgreementPackingMode {
    Normal,
    FinalGlobal,
    ReservedTail5,
    ReservedTail10,
    ReservedTail15
};

const char* agreement_packing_mode_name(AgreementPackingMode mode) {
    switch (mode) {
        case AgreementPackingMode::Normal: return "normal";
        case AgreementPackingMode::FinalGlobal: return "final_global";
        case AgreementPackingMode::ReservedTail5: return "reserved_tail_5";
        case AgreementPackingMode::ReservedTail10: return "reserved_tail_10";
        case AgreementPackingMode::ReservedTail15: return "reserved_tail_15";
    }
    return "unknown";
}

bool is_final_global_packing_mode(AgreementPackingMode mode) {
    return mode != AgreementPackingMode::Normal;
}

bool is_singleton_rescue_packing_mode(AgreementPackingMode mode) {
    return mode == AgreementPackingMode::Normal ||
           mode == AgreementPackingMode::FinalGlobal;
}

double reserved_tail_large_share(AgreementPackingMode mode) {
    switch (mode) {
        case AgreementPackingMode::ReservedTail5: return 0.05;
        case AgreementPackingMode::ReservedTail10: return 0.10;
        case AgreementPackingMode::ReservedTail15: return 0.15;
        default: return 0.0;
    }
}

void apply_reserved_tail_generated_buckets(
    pace26::heuristics::AgreementComponentPacking::Options& opts,
    double large_share
) {
    if (!(large_share > 0.0)) {
        return;
    }

    opts.generated_bucket_capacity.fill(0);
    const std::size_t budget = std::max<std::size_t>(
        1,
        opts.max_generated_candidates == 0
            ? opts.max_candidates
            : opts.max_generated_candidates
    );

    const double non_large_share = std::max(0.0, 1.0 - large_share);
    auto take = [&](double share) {
        return std::min<std::size_t>(
            budget,
            static_cast<std::size_t>(std::llround(static_cast<double>(budget) * share))
        );
    };

    std::size_t used = 0;
    opts.generated_bucket_capacity[2] = take(non_large_share * (45.0 / 85.0));
    used += opts.generated_bucket_capacity[2];
    opts.generated_bucket_capacity[3] = take(non_large_share * (25.0 / 85.0));
    used += opts.generated_bucket_capacity[3];
    opts.generated_bucket_capacity[4] = take(non_large_share * (15.0 / 85.0));
    used += opts.generated_bucket_capacity[4];

    const std::size_t large_budget = used >= budget ? 0 : budget - used;
    opts.generated_bucket_capacity[5] = static_cast<std::size_t>(
        std::llround(static_cast<double>(large_budget) * (2.0 / 3.0))
    );
    if (opts.generated_bucket_capacity[5] > large_budget) {
        opts.generated_bucket_capacity[5] = large_budget;
    }
    const std::size_t large_9plus =
        large_budget - opts.generated_bucket_capacity[5];
    opts.generated_bucket_capacity[6] =
        static_cast<std::size_t>(
            std::llround(static_cast<double>(large_9plus) * 0.80)
        );
    if (opts.generated_bucket_capacity[6] > large_9plus) {
        opts.generated_bucket_capacity[6] = large_9plus;
    }
    opts.generated_bucket_capacity[7] =
        large_9plus - opts.generated_bucket_capacity[6];

    std::size_t total = 0;
    for (std::size_t value : opts.generated_bucket_capacity) {
        total += value;
    }
    if (total < budget) {
        opts.generated_bucket_capacity[7] += budget - total;
    } else if (total > budget) {
        std::size_t excess = total - budget;
        for (std::size_t bucket = opts.generated_bucket_capacity.size(); bucket-- > 2 && excess > 0;) {
            const std::size_t remove =
                std::min(excess, opts.generated_bucket_capacity[bucket]);
            opts.generated_bucket_capacity[bucket] -= remove;
            excess -= remove;
        }
    }
}

pace26::heuristics::AgreementComponentPacking::Options
agreement_packing_options(
    std::size_t n,
    bool extended_search,
    const Timer& timer,
    AgreementPackingMode mode = AgreementPackingMode::Normal,
    bool packing_focused = false
) {
    pace26::heuristics::AgreementComponentPacking::Options opts;
    const bool final_global = extended_search && is_final_global_packing_mode(mode);
    const double reserved_large_share = reserved_tail_large_share(mode);
    const bool reserved_tail = reserved_large_share > 0.0;

    if (n >= 6000) {
        opts.max_candidates = final_global ? 360000 : (extended_search ? 180000 : 120000);
        opts.order_window = final_global ? 16 : (extended_search ? 12 : 9);
        opts.random_partners_per_leaf = final_global ? 6 : (extended_search ? 4 : 3);
        opts.extension_neighbors = 3;
        opts.max_generated_component_size = 3;
        opts.exchange_rounds = final_global ? 7 : (extended_search ? 5 : 3);
        opts.two_exchange_pool = final_global ? 420 : (extended_search ? 240 : 128);
        opts.enable_exact = false;
        opts.local_time_limit_seconds = final_global ? 45.0 : (extended_search ? 28.0 : 6.0);
    } else if (n >= 2500) {
        opts.max_candidates = final_global ? 320000 : (extended_search ? 180000 : 150000);
        opts.order_window = final_global ? 24 : (extended_search ? 18 : 12);
        opts.random_partners_per_leaf = final_global ? 7 : (extended_search ? 5 : 4);
        opts.extension_neighbors = 4;
        opts.max_generated_component_size = 3;
        opts.exchange_rounds = final_global ? 7 : (extended_search ? 5 : 4);
        opts.two_exchange_pool = final_global ? 440 : (extended_search ? 256 : 160);
        opts.enable_exact = false;
        opts.local_time_limit_seconds = final_global ? 38.0 : (extended_search ? 24.0 : 5.0);
    } else if (n >= 600) {
        opts.max_candidates = final_global ? 240000 : (extended_search ? 180000 : 150000);
        opts.order_window = final_global ? 30 : (extended_search ? 24 : 16);
        opts.random_partners_per_leaf = final_global ? 8 : (extended_search ? 6 : 4);
        opts.extension_neighbors = final_global ? 7 : (extended_search ? 6 : 5);
        opts.max_generated_component_size = 4;
        opts.exchange_rounds = final_global ? 7 : (extended_search ? 6 : 4);
        opts.two_exchange_pool = final_global ? 460 : (extended_search ? 300 : 200);
        opts.enable_exact = false;
        opts.local_time_limit_seconds = final_global ? 28.0 : (extended_search ? 20.0 : 4.0);
    } else {
        opts.max_candidates = final_global ? 220000 : (n <= 220 ? 140000 : 100000);
        opts.order_window = final_global ? 34 : (n <= 220 ? 28 : 20);
        opts.random_partners_per_leaf = final_global ? 8 : (n <= 220 ? 6 : 4);
        opts.extension_neighbors = final_global ? 8 : (n <= 220 ? 8 : 5);
        opts.all_pairs_max_leaves = final_global ? 500 : 220;
        opts.max_generated_component_size = 4;
        opts.exchange_rounds = final_global ? 8 : (n <= 220 ? (extended_search ? 8 : 5) : 3);
        opts.two_exchange_pool = final_global ? 420 : (n <= 220 ? (extended_search ? 360 : 220) : 128);
        opts.exact_max_leaves = 180;
        opts.exact_candidate_limit = extended_search ? 220 : 150;
        opts.enable_exact = n <= opts.exact_max_leaves;
        opts.local_time_limit_seconds = n <= 220
            ? (final_global ? 18.0 : (extended_search ? 12.0 : 1.5))
            : (final_global ? 14.0 : (extended_search ? 8.0 : 0.9));
    }

    if (packing_focused) {
        const bool large_final_global = final_global && n >= 2500;
        const bool huge_final_global = large_final_global && n >= 15000;
        const double candidate_multiplier = large_final_global
            ? (n >= 6000 && !huge_final_global ? 1.70 : 1.55)
            : (extended_search ? 1.35 : 1.20);
        const double time_multiplier = large_final_global
            ? (n >= 6000 && !huge_final_global ? 1.55 : 1.45)
            : (extended_search ? 1.30 : 1.15);
        const std::size_t candidate_cap =
            large_final_global && n >= 6000 && !huge_final_global
                ? 700000
                : 600000;

        opts.max_candidates = std::min<std::size_t>(
            candidate_cap,
            static_cast<std::size_t>(
                static_cast<double>(opts.max_candidates) * candidate_multiplier
            )
        );
        opts.order_window += large_final_global ? 6 : (extended_search ? 3 : 2);
        opts.random_partners_per_leaf += large_final_global ? 3 : (extended_search ? 2 : 1);
        opts.extension_neighbors += large_final_global ? 2 : 1;
        opts.exchange_rounds += large_final_global ? 3 : (extended_search ? 2 : 1);
        opts.two_exchange_pool += large_final_global ? 180 : (extended_search ? 120 : 64);
        opts.local_time_limit_seconds *= time_multiplier;
    }

    auto quota = [&](double fraction) {
        return static_cast<std::size_t>(
            std::max<double>(1.0, static_cast<double>(opts.max_candidates) * fraction)
        );
    };

    if (n >= 6000) {
        opts.generated_bucket_capacity[2] = quota(final_global ? 0.64 : (extended_search ? 0.62 : 0.70));
        opts.generated_bucket_capacity[3] = quota(final_global ? 0.30 : (extended_search ? 0.26 : 0.22));
        opts.generated_bucket_capacity[4] = quota(final_global ? 0.23 : (extended_search ? 0.18 : 0.13));
        opts.generated_bucket_capacity[5] = quota(final_global ? 0.20 : (extended_search ? 0.15 : 0.10));
        opts.max_generated_component_size = final_global ? 7 : (extended_search ? 6 : 5);
        opts.structural_subtree_max_size = final_global ? 12 : (extended_search ? 10 : 8);
        opts.structural_subtree_node_budget = final_global ? 42000 : (extended_search ? 26000 : 12000);
        opts.structure_growth_max_component_size = final_global ? 7 : (extended_search ? 6 : 5);
        opts.structure_growth_neighbors = final_global ? 9 : (extended_search ? 7 : 5);
        opts.structure_growth_branching = final_global ? 5 : (extended_search ? 4 : 3);
        opts.structure_growth_candidates_per_size =
            final_global ? 70000 : (extended_search ? 42000 : 16000);
    } else if (n >= 2500) {
        opts.generated_bucket_capacity[2] = quota(final_global ? 0.66 : (extended_search ? 0.64 : 0.72));
        opts.generated_bucket_capacity[3] = quota(final_global ? 0.30 : (extended_search ? 0.26 : 0.22));
        opts.generated_bucket_capacity[4] = quota(final_global ? 0.22 : (extended_search ? 0.17 : 0.12));
        opts.generated_bucket_capacity[5] = quota(final_global ? 0.17 : (extended_search ? 0.13 : 0.08));
        opts.max_generated_component_size = final_global ? 6 : (extended_search ? 5 : 5);
        opts.structural_subtree_max_size = final_global ? 10 : (extended_search ? 8 : 7);
        opts.structural_subtree_node_budget = final_global ? 30000 : (extended_search ? 18000 : 9000);
        opts.structure_growth_max_component_size = final_global ? 6 : (extended_search ? 5 : 5);
        opts.structure_growth_neighbors = final_global ? 8 : (extended_search ? 6 : 5);
        opts.structure_growth_branching = final_global ? 5 : (extended_search ? 4 : 3);
        opts.structure_growth_candidates_per_size =
            final_global ? 46000 : (extended_search ? 28000 : 12000);
    } else if (n >= 600) {
        opts.generated_bucket_capacity[2] = quota(final_global ? 0.78 : (extended_search ? 0.76 : 0.82));
        opts.generated_bucket_capacity[3] = quota(final_global ? 0.34 : (extended_search ? 0.30 : 0.24));
        opts.generated_bucket_capacity[4] = quota(final_global ? 0.24 : (extended_search ? 0.20 : 0.12));
        opts.generated_bucket_capacity[5] = quota(final_global ? 0.12 : (extended_search ? 0.10 : 0.06));
        opts.max_generated_component_size = final_global ? 7 : (extended_search ? 6 : 5);
        opts.structural_subtree_max_size = final_global ? 16 : (extended_search ? 12 : 10);
        opts.structural_subtree_node_budget = final_global ? 22000 : (extended_search ? 14000 : 7000);
        opts.structure_growth_max_component_size = final_global ? 7 : (extended_search ? 6 : 5);
        opts.structure_growth_neighbors = final_global ? 9 : (extended_search ? 7 : 5);
        opts.structure_growth_branching = final_global ? 5 : (extended_search ? 4 : 3);
        opts.structure_growth_candidates_per_size =
            final_global ? 30000 : (extended_search ? 18000 : 8000);
    } else if (n >= 220) {
        opts.generated_bucket_capacity[2] = quota(final_global ? 0.88 : (extended_search ? 0.86 : 0.90));
        opts.generated_bucket_capacity[3] = quota(final_global ? 0.34 : (extended_search ? 0.30 : 0.24));
        opts.generated_bucket_capacity[4] = quota(final_global ? 0.24 : (extended_search ? 0.18 : 0.12));
        opts.generated_bucket_capacity[5] = quota(final_global ? 0.12 : (extended_search ? 0.08 : 0.05));
        opts.max_generated_component_size = final_global ? 6 : (extended_search ? 6 : 5);
        opts.structural_subtree_max_size = final_global ? 18 : (extended_search ? 14 : 10);
        opts.structural_subtree_node_budget = final_global ? 8000 : (extended_search ? 5000 : 2500);
        opts.structure_growth_max_component_size = final_global ? 6 : (extended_search ? 6 : 5);
        opts.structure_growth_neighbors = final_global ? 10 : (extended_search ? 8 : 6);
        opts.structure_growth_branching = final_global ? 5 : (extended_search ? 4 : 3);
        opts.structure_growth_candidates_per_size =
            final_global ? 26000 : (extended_search ? 16000 : 7000);
    }

    if (n >= 220 && n < 600) {
        opts.enable_exact = true;
        opts.exact_max_leaves = 600;
        opts.exact_candidate_limit = final_global ? 300 : (extended_search ? 180 : 150);
        opts.exact_selected_candidate_limit = final_global ? 120 : (extended_search ? 72 : 56);
        opts.exact_pool_size_quota[5] = final_global ? 42 : (extended_search ? 24 : 18);
        opts.exact_pool_size_quota[4] = final_global ? 56 : (extended_search ? 34 : 26);
        opts.exact_pool_size_quota[3] = final_global ? 72 : (extended_search ? 46 : 34);
        opts.exact_node_limit = final_global ? 650000 : (extended_search ? 180000 : 90000);
        opts.max_refill_deficit = 1;
        opts.deficit_exchange_rounds = final_global ? 2 : 1;
        opts.deficit_exchange_pool = final_global ? 360 : (extended_search ? 220 : 160);
        opts.deficit_refill_pool = final_global ? 1600 : (extended_search ? 1100 : 800);
        opts.local_mwis_anchors = final_global ? 28 : (extended_search ? 18 : 12);
        opts.local_mwis_candidate_limit = final_global ? 260 : (extended_search ? 190 : 150);
        opts.local_mwis_node_limit = final_global ? 600000 : (extended_search ? 280000 : 140000);
    }

    if (final_global && n <= 500) {
        opts.seed_intersection_candidate_limit = std::max<std::size_t>(
            3000,
            opts.max_candidates / 20
        );
        opts.local_time_limit_seconds = n <= 220 ? 24.0 : 32.0;
        opts.local_mwis_anchors = n <= 220 ? 40 : 32;
        opts.local_mwis_candidate_limit = n <= 220 ? 360 : 420;
        opts.local_mwis_node_limit = n <= 220 ? 1200000 : 900000;
    }

    if (reserved_tail && n >= 6000) {
        opts.separate_seed_generated_budgets = true;
        opts.max_seed_candidates = std::min<std::size_t>(
            n >= 15000 ? 90000 : 120000,
            std::max<std::size_t>(n >= 15000 ? 24000 : 30000,
                                  n * (n >= 15000 ? 5 : 6))
        );
        opts.max_generated_candidates = opts.max_candidates;
        opts.max_generated_component_size = 16;
        opts.structural_subtree_max_size = 20;
        opts.structural_subtree_node_budget = 65000;
        opts.structure_growth_max_component_size = 16;
        opts.structure_growth_neighbors = 12;
        opts.structure_growth_branching = 6;
        opts.structure_growth_candidates_per_size = 90000;
        opts.exchange_rounds = std::max<std::size_t>(opts.exchange_rounds, 8);
        opts.two_exchange_pool = std::max<std::size_t>(opts.two_exchange_pool, 520);
        opts.local_time_limit_seconds = std::min(opts.local_time_limit_seconds, 24.0);
        apply_reserved_tail_generated_buckets(opts, reserved_large_share);
    }

    if (final_global && n >= 2500) {
        opts.max_refill_deficit = 1;
        opts.deficit_exchange_rounds = reserved_tail ? 2 : 1;
        if (n >= 6000) {
            opts.deficit_exchange_pool = reserved_tail ? 900 : 520;
            opts.deficit_refill_pool = reserved_tail ? 5000 : 3200;
            opts.local_mwis_anchors = reserved_tail ? 10 : 7;
            opts.local_mwis_candidate_limit = reserved_tail ? 120 : 96;
            opts.local_mwis_node_limit = reserved_tail ? 100000 : 60000;
        } else {
            opts.deficit_exchange_pool = reserved_tail ? 700 : 420;
            opts.deficit_refill_pool = reserved_tail ? 3600 : 2400;
            opts.local_mwis_anchors = reserved_tail ? 12 : 8;
            opts.local_mwis_candidate_limit = reserved_tail ? 150 : 120;
            opts.local_mwis_node_limit = reserved_tail ? 200000 : 120000;
        }
        if (reserved_tail) {
            opts.max_refill_deficit = 0;
            opts.deficit_exchange_rounds = 0;
            opts.deficit_exchange_pool = 0;
            opts.deficit_refill_pool = 0;
            opts.local_mwis_anchors = 0;
            opts.local_mwis_candidate_limit = 0;
            opts.local_mwis_node_limit = 0;
        }
    }

    if (n < 600) {
        opts.greedy_variant_count = final_global ? 10 : (extended_search ? 8 : 6);
        opts.greedy_polish_variants = final_global ? 5 : (extended_search ? 3 : 1);
        opts.greedy_polish_exchange_rounds = final_global ? 2 : 1;
        opts.greedy_polish_two_exchange_pool = final_global ? 220 : (extended_search ? 160 : 96);
    } else if (n < 2500) {
        opts.greedy_variant_count = final_global ? 9 : (extended_search ? 7 : 5);
        opts.greedy_polish_variants = final_global ? 4 : (extended_search ? 2 : 1);
        opts.greedy_polish_exchange_rounds = final_global ? 2 : 1;
        opts.greedy_polish_two_exchange_pool = final_global ? 240 : (extended_search ? 160 : 96);
    } else if (n < 6000) {
        opts.greedy_variant_count = final_global ? 8 : (extended_search ? 6 : 5);
        opts.greedy_polish_variants = final_global ? 3 : (extended_search ? 2 : 1);
        opts.greedy_polish_exchange_rounds = final_global ? 2 : 1;
        opts.greedy_polish_two_exchange_pool = final_global ? 240 : (extended_search ? 144 : 80);
    } else {
        opts.greedy_variant_count = final_global ? 6 : (extended_search ? 5 : 4);
        opts.greedy_polish_variants = final_global ? 2 : (extended_search ? 1 : 0);
        opts.greedy_polish_exchange_rounds = 1;
        opts.greedy_polish_two_exchange_pool = final_global ? 160 : 112;
    }

    if (packing_focused) {
        opts.greedy_variant_count = std::min<std::size_t>(
            n >= 10000 ? 7 : 10,
            opts.greedy_variant_count + 1
        );
        opts.greedy_polish_variants = std::min<std::size_t>(
            opts.greedy_variant_count,
            opts.greedy_polish_variants + (n >= 10000 ? 0 : 1)
        );
        opts.greedy_polish_two_exchange_pool += n >= 6000 ? 32 : 64;
    }

    if (reserved_tail && n >= 6000) {
        opts.greedy_variant_count = std::max<std::size_t>(opts.greedy_variant_count, 5);
        opts.greedy_polish_variants = std::min<std::size_t>(opts.greedy_polish_variants, 1);
        opts.greedy_polish_exchange_rounds = 1;
        opts.greedy_polish_two_exchange_pool =
            std::min<std::size_t>(opts.greedy_polish_two_exchange_pool, 128);
    }

    if (opts.local_mwis_anchors > 0) {
        if (n < 600) {
            opts.local_mwis_anchor_scan_limit =
                final_global ? 7000 : (extended_search ? 4500 : 2200);
        } else if (n < 2500) {
            opts.local_mwis_anchor_scan_limit =
                final_global ? 6500 : (extended_search ? 4200 : 1800);
        } else if (n < 6000) {
            opts.local_mwis_anchor_scan_limit =
                final_global ? 5200 : (extended_search ? 3200 : 1600);
        } else {
            opts.local_mwis_anchor_scan_limit =
                final_global ? 3000 : (extended_search ? 1600 : 900);
        }
        if (reserved_tail && n >= 6000) {
            opts.local_mwis_anchor_scan_limit =
                std::min<std::size_t>(opts.local_mwis_anchor_scan_limit, 1200);
        }
    }

    opts.local_time_limit_seconds = std::min(
        opts.local_time_limit_seconds,
        std::max(0.10, timer.remaining_seconds() - 1.25)
    );
    opts.guard_seconds = 0.75;
    return opts;
}

bool same_packing_seed_forest(
    const CoreForest& a,
    const CoreForest& b
) {
    if (a.components.size() != b.components.size()) {
        return false;
    }

    for (std::size_t i = 0; i < a.components.size(); ++i) {
        if (a.components[i].labels != b.components[i].labels) {
            return false;
        }
    }

    return true;
}

bool packing_seed_better(
    const CoreForest& a,
    const CoreForest& b
) {
    if (a.component_count() != b.component_count()) {
        return a.component_count() < b.component_count();
    }

    const std::uint64_t a_tie = tie_score(a);
    const std::uint64_t b_tie = tie_score(b);
    if (a_tie != b_tie) {
        return a_tie > b_tie;
    }

    return false;
}

bool remember_packing_seed(
    std::vector<CoreForest>& seeds,
    CoreForest candidate,
    std::size_t max_seeds = 16
) {
    if (max_seeds == 0) {
        return false;
    }

    candidate.normalize();
    if (candidate.empty()) {
        return false;
    }

    for (const CoreForest& existing : seeds) {
        if (same_packing_seed_forest(existing, candidate)) {
            return false;
        }
    }

    seeds.push_back(std::move(candidate));
    std::stable_sort(
        seeds.begin(),
        seeds.end(),
        [](const CoreForest& a, const CoreForest& b) {
            return packing_seed_better(a, b);
        }
    );

    if (seeds.size() > max_seeds) {
        seeds.resize(max_seeds);
    }

    return true;
}

std::size_t forest_singleton_count(const CoreForest& forest) {
    std::size_t count = 0;
    for (const CoreComponent& component : forest.components) {
        if (component.labels.size() == 1) {
            ++count;
        }
    }
    return count;
}

using SeedDiversitySignature = std::array<std::uint64_t, 8>;

SeedDiversitySignature seed_diversity_signature(const CoreForest& forest) {
    std::vector<std::uint32_t> labels;
    std::unordered_map<std::uint32_t, std::size_t> component_of;
    for (std::size_t component_index = 0;
         component_index < forest.components.size();
         ++component_index) {
        for (std::uint32_t label : forest.components[component_index].labels) {
            labels.push_back(label);
            component_of[label] = component_index;
        }
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());

    SeedDiversitySignature signature{};
    if (labels.size() < 2) {
        return signature;
    }

    for (std::size_t bit = 0; bit < 512; ++bit) {
        const std::uint64_t a_hash =
            profile_mix64(0x9e3779b97f4a7c15ULL ^ static_cast<std::uint64_t>(bit));
        const std::uint64_t b_hash =
            profile_mix64(0xbf58476d1ce4e5b9ULL ^ static_cast<std::uint64_t>(bit * 3 + 1));
        std::size_t a = static_cast<std::size_t>(a_hash % labels.size());
        std::size_t b = static_cast<std::size_t>(b_hash % labels.size());
        if (a == b) {
            b = (b + 1) % labels.size();
        }
        if (component_of[labels[a]] == component_of[labels[b]]) {
            signature[bit / 64] |= (1ULL << (bit % 64));
        }
    }
    return signature;
}

std::size_t signature_distance(
    const SeedDiversitySignature& a,
    const SeedDiversitySignature& b
) {
    std::size_t distance = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        distance += static_cast<std::size_t>(__builtin_popcountll(a[i] ^ b[i]));
    }
    return distance;
}

void select_diverse_packing_archive(
    std::vector<CoreForest>& seeds,
    std::size_t max_seeds
) {
    if (seeds.size() <= max_seeds) {
        return;
    }

    std::vector<SeedDiversitySignature> signatures;
    signatures.reserve(seeds.size());
    for (const CoreForest& seed : seeds) {
        signatures.push_back(seed_diversity_signature(seed));
    }

    std::vector<char> chosen(seeds.size(), 0);
    std::vector<std::size_t> chosen_indices;
    auto choose_index = [&](std::size_t index) {
        if (index >= seeds.size() || chosen[index] || chosen_indices.size() >= max_seeds) {
            return;
        }
        chosen[index] = 1;
        chosen_indices.push_back(index);
    };

    std::vector<std::size_t> order(seeds.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return packing_seed_better(seeds[a], seeds[b]);
    });
    for (std::size_t i = 0; i < order.size() && i < 4; ++i) {
        choose_index(order[i]);
    }

    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const std::size_t sa = forest_singleton_count(seeds[a]);
        const std::size_t sb = forest_singleton_count(seeds[b]);
        if (sa != sb) {
            return sa < sb;
        }
        return packing_seed_better(seeds[a], seeds[b]);
    });
    for (std::size_t index : order) {
        if (chosen_indices.size() >= std::min<std::size_t>(max_seeds, 6)) {
            break;
        }
        choose_index(index);
    }

    while (chosen_indices.size() < max_seeds) {
        std::size_t best_index = seeds.size();
        std::size_t best_distance = 0;
        for (std::size_t i = 0; i < seeds.size(); ++i) {
            if (chosen[i]) {
                continue;
            }
            std::size_t min_distance = std::numeric_limits<std::size_t>::max();
            for (std::size_t chosen_index : chosen_indices) {
                min_distance = std::min(
                    min_distance,
                    signature_distance(signatures[i], signatures[chosen_index])
                );
            }
            if (best_index == seeds.size() ||
                min_distance > best_distance ||
                (min_distance == best_distance &&
                 packing_seed_better(seeds[i], seeds[best_index]))) {
                best_index = i;
                best_distance = min_distance;
            }
        }
        if (best_index == seeds.size()) {
            break;
        }
        choose_index(best_index);
    }

    std::vector<CoreForest> selected;
    selected.reserve(chosen_indices.size());
    for (std::size_t index : chosen_indices) {
        selected.push_back(std::move(seeds[index]));
    }
    seeds = std::move(selected);
}

bool remember_diverse_packing_seed(
    std::vector<CoreForest>& seeds,
    CoreForest candidate,
    std::size_t max_seeds = 8
) {
    if (max_seeds == 0) {
        return false;
    }

    candidate.normalize();
    if (candidate.empty()) {
        return false;
    }

    for (const CoreForest& existing : seeds) {
        if (same_packing_seed_forest(existing, candidate)) {
            return false;
        }
    }

    seeds.push_back(std::move(candidate));
    select_diverse_packing_archive(seeds, max_seeds);
    return true;
}

std::size_t small_alternate_exactification_state_limit(std::size_t n) {
    if (n <= 120) {
        return 8;
    }
    if (n <= 220) {
        return 6;
    }
    if (n <= 350) {
        return 3;
    }
    if (n <= 500) {
        return 2;
    }
    return 0;
}

std::size_t small_alternate_exactification_extra_slack(std::size_t n) {
    if (n <= 220) {
        return 3;
    }
    if (n <= 350) {
        return 1;
    }
    return 0;
}

double small_alternate_exactification_local_budget(std::size_t n) {
    if (n <= 120) {
        return 2.0;
    }
    if (n <= 220) {
        return 2.5;
    }
    if (n <= 350) {
        return 3.0;
    }
    return 3.5;
}

std::size_t small_alternate_exactification_supplement_cap(std::size_t n) {
    if (n <= 120) {
        return 9000;
    }
    if (n <= 220) {
        return 14000;
    }
    return 0;
}

std::size_t small_alternate_exactification_archive_slack(std::size_t n) {
    if (n <= 120) {
        return 28;
    }
    if (n <= 220) {
        return 36;
    }
    if (n <= 350) {
        return 20;
    }
    return 12;
}

void enrich_small_alternate_exactification_archive(
    std::vector<CoreForest>& archive,
    std::size_t archive_limit,
    const CoreForest& incumbent,
    const Tree& t1,
    const Tree& t2,
    std::size_t n,
    Timer& timer
) {
    if (archive_limit == 0 || timer.should_stop(6.0)) {
        return;
    }

    const std::size_t run_limit =
        n <= 120 ? 8 : (n <= 220 ? 6 : (n <= 350 ? 3 : 2));
    const std::size_t component_slack =
        small_alternate_exactification_archive_slack(n);
    const std::size_t max_components =
        incumbent.component_count() + component_slack;
    const double local_budget =
        n <= 220 ? 0.45 : (n <= 350 ? 0.65 : 0.80);
    const double phase_start = timer.elapsed_seconds();
    const std::size_t states_before = archive.size();
    std::size_t produced = 0;
    std::size_t retained = 0;
    std::size_t invalid = 0;
    std::size_t too_large = 0;

    for (std::size_t i = 0; i < run_limit && !timer.should_stop(4.0); ++i) {
        try {
            pace26::heuristics::ActiveCherryGreedyApprox::Options opts;
            opts.policy = ActiveCherryPolicy::Balanced;
            opts.max_steps_multiplier = n <= 220 ? 8 : 6;
            opts.candidate_sample_cap = n <= 220 ? 192 : 128;
            opts.guard_seconds = 0.25;
            opts.local_time_limit_seconds = local_budget;
            opts.sample_salt = mix_portfolio_salt(
                0x91e10da5c79b4f13ULL + static_cast<std::uint64_t>(i * 0x9e37U + n)
            );
            switch (i % 4) {
                case 1:
                    opts.rank_choice_script = {1};
                    break;
                case 2:
                    opts.rank_choice_script = {2};
                    break;
                case 3:
                    opts.rank_choice_script = {1, 1};
                    break;
                default:
                    break;
            }

            const bool swapped = (i % 3) == 2;
            const Tree& first = swapped ? t2 : t1;
            const Tree& second = swapped ? t1 : t2;
            pace26::heuristics::ActiveCherryGreedyApprox greedy(opts);
            CoreForest candidate = greedy.solve(first, second, &timer);
            ++produced;

            if (!is_valid_agreement_forest(candidate, t1, t2)) {
                ++invalid;
                continue;
            }
            candidate.normalize();
            if (candidate.component_count() > max_components) {
                ++too_large;
                continue;
            }
            if (remember_diverse_packing_seed(
                    archive,
                    std::move(candidate),
                    archive_limit)) {
                ++retained;
            }
        } catch (const std::exception&) {
            ++invalid;
        }
    }

    if (g_profile.enabled()) {
        g_profile.phase_result(
            "main.small_exactification_alternate_state_archive_active_cherry",
            timer,
            phase_start,
            n,
            states_before,
            archive.size(),
            retained > 0,
            {
                {"runs", std::to_string(run_limit)},
                {"produced", std::to_string(produced)},
                {"retained", std::to_string(retained)},
                {"invalid", std::to_string(invalid)},
                {"too_large", std::to_string(too_large)},
                {"archive_limit", std::to_string(archive_limit)},
                {"component_slack", std::to_string(component_slack)},
                {"local_budget_seconds", std::to_string(local_budget)}
            }
        );
    }
}

struct LabelSetHash {
    std::size_t operator()(const std::vector<std::uint32_t>& labels) const noexcept {
        std::uint64_t h = 0x9e3779b97f4a7c15ULL;
        for (std::uint32_t label : labels) {
            h ^= profile_mix64(static_cast<std::uint64_t>(label) + h);
        }
        return static_cast<std::size_t>(
            profile_mix64(h ^ static_cast<std::uint64_t>(labels.size()))
        );
    }
};

class ComponentForestArchive {
public:
    explicit ComponentForestArchive(
        std::size_t max_components = 120000,
        std::size_t max_forests = 32
    )
        : max_components_(max_components),
          max_forests_(max_forests) {}

    bool remember_forest(
        CoreForest forest,
        const Tree* t1 = nullptr,
        const Tree* t2 = nullptr,
        bool validate = false
    ) {
        if (forest.empty()) {
            return false;
        }
        if (validate) {
            if (t1 == nullptr || t2 == nullptr ||
                !is_valid_agreement_forest(forest, *t1, *t2)) {
                return false;
            }
        }

        forest.normalize();
        ++epoch_;
        bool has_non_singleton_component = false;
        for (const CoreComponent& component : forest.components) {
            if (component.labels.size() > 1) {
                has_non_singleton_component = true;
            }
            remember_component_labels(component.labels);
        }
        if (!has_non_singleton_component) {
            return false;
        }

        return remember_diverse_packing_seed(
            forests_,
            std::move(forest),
            max_forests_
        );
    }

    bool remember_if_valid(
        CoreForest forest,
        const Tree& t1,
        const Tree& t2
    ) {
        return remember_forest(std::move(forest), &t1, &t2, true);
    }

    bool remember_component(std::vector<std::uint32_t> labels) {
        ++epoch_;
        return remember_component_labels(std::move(labels));
    }

    std::vector<CoreForest> seed_forests(std::size_t limit) const {
        if (limit == 0 || forests_.empty()) {
            return {};
        }
        std::vector<CoreForest> result = forests_;
        select_diverse_packing_archive(result, std::min(limit, result.size()));
        if (result.size() > limit) {
            result.resize(limit);
        }
        return result;
    }

    std::vector<std::vector<std::uint32_t>> extra_candidate_components(
        std::vector<std::uint32_t> allowed_labels,
        std::size_t limit,
        std::size_t max_component_size = 0
    ) const {
        if (limit == 0 || components_.empty()) {
            return {};
        }

        if (!std::is_sorted(allowed_labels.begin(), allowed_labels.end())) {
            std::sort(allowed_labels.begin(), allowed_labels.end());
        }

        std::vector<std::size_t> order;
        order.reserve(std::min(limit * 2, components_.size()));
        for (std::size_t i = 0; i < components_.size(); ++i) {
            const ComponentEntry& entry = components_[i];
            const std::size_t size = entry.labels.size();
            if (size <= 1 ||
                (max_component_size != 0 && size > max_component_size) ||
                !labels_subset_of(entry.labels, allowed_labels)) {
                continue;
            }
            order.push_back(i);
        }

        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            const ComponentEntry& ea = components_[a];
            const ComponentEntry& eb = components_[b];
            if (ea.support != eb.support) {
                return ea.support > eb.support;
            }
            if (ea.labels.size() != eb.labels.size()) {
                return ea.labels.size() > eb.labels.size();
            }
            if (ea.last_epoch != eb.last_epoch) {
                return ea.last_epoch > eb.last_epoch;
            }
            return ea.labels < eb.labels;
        });

        if (order.size() > limit) {
            order.resize(limit);
        }

        std::vector<std::vector<std::uint32_t>> result;
        result.reserve(order.size());
        for (std::size_t index : order) {
            result.push_back(components_[index].labels);
        }
        return result;
    }

    std::size_t component_count() const {
        return components_.size();
    }

    std::size_t forest_count() const {
        return forests_.size();
    }

private:
    struct ComponentEntry {
        std::vector<std::uint32_t> labels;
        std::size_t support = 0;
        std::size_t first_epoch = 0;
        std::size_t last_epoch = 0;
    };

    static bool labels_subset_of(
        const std::vector<std::uint32_t>& labels,
        const std::vector<std::uint32_t>& allowed_labels
    ) {
        for (std::uint32_t label : labels) {
            if (!std::binary_search(
                    allowed_labels.begin(),
                    allowed_labels.end(),
                    label)) {
                return false;
            }
        }
        return true;
    }

    bool remember_component_labels(std::vector<std::uint32_t> labels) {
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (labels.size() <= 1) {
            return false;
        }

        auto found = component_index_.find(labels);
        if (found != component_index_.end()) {
            ComponentEntry& entry = components_[found->second];
            ++entry.support;
            entry.last_epoch = epoch_;
            return false;
        }

        if (components_.size() >= max_components_) {
            return false;
        }

        ComponentEntry entry;
        entry.labels = std::move(labels);
        entry.support = 1;
        entry.first_epoch = epoch_;
        entry.last_epoch = epoch_;
        const std::size_t index = components_.size();
        components_.push_back(std::move(entry));
        component_index_.emplace(components_.back().labels, index);
        return true;
    }

    std::size_t max_components_ = 0;
    std::size_t max_forests_ = 0;
    std::size_t epoch_ = 0;
    std::vector<ComponentEntry> components_;
    std::unordered_map<std::vector<std::uint32_t>, std::size_t, LabelSetHash>
        component_index_;
    std::vector<CoreForest> forests_;
};

std::size_t archive_seed_limit_for_packing(
    std::size_t n,
    bool extended_search,
    AgreementPackingMode mode
) {
    if (extended_search || is_final_global_packing_mode(mode)) {
        if (n >= 2500) {
            return 18;
        }
        return 14;
    }
    return n >= 2500 ? 8 : 6;
}

std::size_t archive_extra_component_limit_for_packing(
    std::size_t n,
    bool extended_search,
    AgreementPackingMode mode
) {
    if (!extended_search && !is_final_global_packing_mode(mode)) {
        return 0;
    }
    if (n <= 220) {
        return 7000;
    }
    if (n <= 600) {
        return 9000;
    }
    if (n <= 2500) {
        return 12000;
    }
    if (n <= 6000) {
        return 18000;
    }
    return 24000;
}

std::size_t archive_capture_component_limit_for_packing(
    std::size_t n,
    bool extended_search,
    AgreementPackingMode mode
) {
    if (!extended_search && mode == AgreementPackingMode::Normal) {
        return 0;
    }
    if (n <= 220) {
        return 4000;
    }
    if (n <= 600) {
        return 6000;
    }
    if (n <= 2500) {
        return 9000;
    }
    if (n <= 6000) {
        return 12000;
    }
    return 15000;
}

std::size_t archive_extra_component_max_size(std::size_t n) {
    if (n <= 220) {
        return 64;
    }
    if (n <= 600) {
        return 96;
    }
    if (n <= 2500) {
        return 128;
    }
    if (n <= 6000) {
        return 192;
    }
    return 256;
}

struct SingletonRescueBaseComponent {
    std::vector<std::uint32_t> labels;
    int root1 = -1;
    int root2 = -1;
    std::size_t support = 0;
    bool from_incumbent = false;
};

struct SingletonRescueOptions {
    std::size_t max_candidates = 0;
    std::size_t max_singletons = 0;
    std::size_t max_base_components = 0;
    std::size_t base_components_per_singleton = 0;
    std::size_t max_base_component_size = 0;
    std::size_t eject_labels_per_component = 0;
    std::size_t second_singletons_per_target = 0;
    bool enable_two_singleton_growth = false;
};

SingletonRescueOptions singleton_rescue_options(
    std::size_t n,
    bool extended_search,
    AgreementPackingMode mode,
    bool packing_focused
) {
    SingletonRescueOptions opts;
    if (!extended_search || !is_singleton_rescue_packing_mode(mode)) {
        return opts;
    }

    if (n <= 120) {
        opts.max_candidates = 18000;
        opts.max_singletons = 120;
        opts.max_base_components = 1400;
        opts.base_components_per_singleton = 48;
        opts.max_base_component_size = 96;
        opts.eject_labels_per_component = 10;
        opts.second_singletons_per_target = 8;
        opts.enable_two_singleton_growth = true;
    } else if (n <= 220) {
        opts.max_candidates = 22000;
        opts.max_singletons = 140;
        opts.max_base_components = 1800;
        opts.base_components_per_singleton = 44;
        opts.max_base_component_size = 112;
        opts.eject_labels_per_component = 8;
        opts.second_singletons_per_target = 6;
        opts.enable_two_singleton_growth = true;
    } else if (n <= 700) {
        opts.max_candidates = 18000;
        opts.max_singletons = 150;
        opts.max_base_components = 1700;
        opts.base_components_per_singleton = 28;
        opts.max_base_component_size = 96;
        opts.eject_labels_per_component = 6;
        opts.second_singletons_per_target = 4;
        opts.enable_two_singleton_growth = true;
    } else if (n <= 2500) {
        opts.max_candidates = 22000;
        opts.max_singletons = 180;
        opts.max_base_components = 2200;
        opts.base_components_per_singleton = 18;
        opts.max_base_component_size = 88;
        opts.eject_labels_per_component = 4;
        opts.second_singletons_per_target = 3;
        opts.enable_two_singleton_growth = true;
    } else if (n <= 6000) {
        opts.max_candidates = 26000;
        opts.max_singletons = 220;
        opts.max_base_components = 2600;
        opts.base_components_per_singleton = 12;
        opts.max_base_component_size = 80;
        opts.eject_labels_per_component = 3;
        opts.second_singletons_per_target = 2;
        opts.enable_two_singleton_growth = true;
    } else {
        opts.max_candidates = 30000;
        opts.max_singletons = 260;
        opts.max_base_components = 3200;
        opts.base_components_per_singleton = 9;
        opts.max_base_component_size = 72;
        opts.eject_labels_per_component = 2;
        opts.second_singletons_per_target = 0;
        opts.enable_two_singleton_growth = false;
    }

    if (packing_focused) {
        opts.max_candidates = static_cast<std::size_t>(
            static_cast<double>(opts.max_candidates) * 1.25
        );
        opts.base_components_per_singleton += 6;
        opts.max_base_components += 400;
    }

    return opts;
}

int component_root_in_tree(
    const Tree& tree,
    const std::vector<std::uint32_t>& labels
) {
    int root = tree.node_of_label(labels.front());
    for (std::size_t i = 1; i < labels.size(); ++i) {
        root = tree.lca(root, tree.node_of_label(labels[i]));
    }
    return root;
}

bool label_set_contains(
    const std::vector<std::uint32_t>& labels,
    std::uint32_t label
) {
    return std::binary_search(labels.begin(), labels.end(), label);
}

std::vector<std::vector<std::uint32_t>> generate_singleton_rescue_components(
    const CoreForest& incumbent,
    const std::vector<CoreForest>& seeds,
    const Tree& t1,
    const Tree& t2,
    Timer& timer,
    bool extended_search,
    AgreementPackingMode mode,
    bool packing_focused
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    const SingletonRescueOptions opts =
        singleton_rescue_options(n, extended_search, mode, packing_focused);
    if (opts.max_candidates == 0 || timer.should_stop(1.5)) {
        return {};
    }

    CoreForest normalized_incumbent = incumbent;
    normalized_incumbent.normalize();

    std::unordered_map<std::uint32_t, std::size_t> singleton_frequency;
    singleton_frequency.reserve(n * 2 + 1);
    std::vector<std::uint32_t> incumbent_singletons;
    incumbent_singletons.reserve(normalized_incumbent.components.size());
    for (const CoreComponent& component : normalized_incumbent.components) {
        if (component.labels.size() == 1) {
            incumbent_singletons.push_back(component.labels.front());
        }
    }
    if (incumbent_singletons.empty()) {
        return {};
    }

    for (const CoreForest& seed : seeds) {
        for (const CoreComponent& component : seed.components) {
            if (component.labels.size() == 1) {
                ++singleton_frequency[component.labels.front()];
            }
        }
    }

    std::sort(incumbent_singletons.begin(), incumbent_singletons.end());
    std::stable_sort(
        incumbent_singletons.begin(),
        incumbent_singletons.end(),
        [&](std::uint32_t a, std::uint32_t b) {
            const std::size_t fa = singleton_frequency[a];
            const std::size_t fb = singleton_frequency[b];
            if (fa != fb) {
                return fa > fb;
            }
            const int da = t1.depth[static_cast<std::size_t>(t1.node_of_label(a))] +
                           t2.depth[static_cast<std::size_t>(t2.node_of_label(a))];
            const int db = t1.depth[static_cast<std::size_t>(t1.node_of_label(b))] +
                           t2.depth[static_cast<std::size_t>(t2.node_of_label(b))];
            if (da != db) {
                return da > db;
            }
            return a < b;
        }
    );
    if (incumbent_singletons.size() > opts.max_singletons) {
        incumbent_singletons.resize(opts.max_singletons);
    }

    std::unordered_map<
        std::vector<std::uint32_t>,
        std::size_t,
        LabelSetHash
    > base_index;
    base_index.reserve(opts.max_base_components * 3 + 1);
    std::vector<SingletonRescueBaseComponent> base_components;
    base_components.reserve(opts.max_base_components * 2 + 1);

    auto add_base_component = [&](const CoreComponent& component, bool from_incumbent) {
        if (component.labels.size() <= 1 ||
            component.labels.size() > opts.max_base_component_size) {
            return;
        }

        std::vector<std::uint32_t> labels = component.labels;
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (labels.size() <= 1 || labels.size() > opts.max_base_component_size) {
            return;
        }

        auto found = base_index.find(labels);
        if (found != base_index.end()) {
            SingletonRescueBaseComponent& existing =
                base_components[found->second];
            ++existing.support;
            existing.from_incumbent = existing.from_incumbent || from_incumbent;
            return;
        }

        SingletonRescueBaseComponent out;
        out.labels = std::move(labels);
        out.support = 1;
        out.from_incumbent = from_incumbent;
        const std::size_t index = base_components.size();
        base_index.emplace(out.labels, index);
        base_components.push_back(std::move(out));
    };

    for (const CoreComponent& component : normalized_incumbent.components) {
        add_base_component(component, true);
    }
    for (const CoreForest& seed : seeds) {
        for (const CoreComponent& component : seed.components) {
            add_base_component(component, false);
        }
    }
    if (base_components.empty()) {
        return {};
    }

    std::stable_sort(
        base_components.begin(),
        base_components.end(),
        [](const SingletonRescueBaseComponent& a, const SingletonRescueBaseComponent& b) {
            const long long as =
                static_cast<long long>(a.support) * 100000LL +
                (a.from_incumbent ? 30000LL : 0LL) +
                static_cast<long long>(std::min<std::size_t>(a.labels.size(), 64)) * 100LL -
                static_cast<long long>(a.labels.size() > 96 ? a.labels.size() : 0);
            const long long bs =
                static_cast<long long>(b.support) * 100000LL +
                (b.from_incumbent ? 30000LL : 0LL) +
                static_cast<long long>(std::min<std::size_t>(b.labels.size(), 64)) * 100LL -
                static_cast<long long>(b.labels.size() > 96 ? b.labels.size() : 0);
            if (as != bs) {
                return as > bs;
            }
            if (a.labels.size() != b.labels.size()) {
                return a.labels.size() > b.labels.size();
            }
            return a.labels < b.labels;
        }
    );
    if (base_components.size() > opts.max_base_components) {
        base_components.resize(opts.max_base_components);
    }

    for (SingletonRescueBaseComponent& component : base_components) {
        component.root1 = component_root_in_tree(t1, component.labels);
        component.root2 = component_root_in_tree(t2, component.labels);
    }

    std::vector<std::vector<std::uint32_t>> result;
    result.reserve(opts.max_candidates);
    std::unordered_set<std::vector<std::uint32_t>, LabelSetHash> seen;
    seen.reserve(opts.max_candidates * 2 + 1);

    auto add_component_labels = [&](std::vector<std::uint32_t> labels) {
        if (result.size() >= opts.max_candidates) {
            return;
        }
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (labels.size() <= 1) {
            return;
        }
        if (seen.insert(labels).second) {
            result.push_back(std::move(labels));
        }
    };

    auto pair_closeness = [&](std::uint32_t a, std::uint32_t b) {
        const int a1 = t1.node_of_label(a);
        const int b1 = t1.node_of_label(b);
        const int a2 = t2.node_of_label(a);
        const int b2 = t2.node_of_label(b);
        return t1.depth[static_cast<std::size_t>(t1.lca(a1, b1))] +
               t2.depth[static_cast<std::size_t>(t2.lca(a2, b2))];
    };

    for (std::uint32_t x : incumbent_singletons) {
        if (result.size() >= opts.max_candidates || timer.should_stop(1.5)) {
            break;
        }

        const int x1 = t1.node_of_label(x);
        const int x2 = t2.node_of_label(x);
        std::vector<std::pair<long long, std::size_t>> ranked;
        ranked.reserve(base_components.size());
        for (std::size_t i = 0; i < base_components.size(); ++i) {
            const SingletonRescueBaseComponent& base = base_components[i];
            if (label_set_contains(base.labels, x)) {
                continue;
            }

            const int lca1 = t1.lca(x1, base.root1);
            const int lca2 = t2.lca(x2, base.root2);
            const long long score =
                static_cast<long long>(t1.depth[static_cast<std::size_t>(lca1)] +
                                       t2.depth[static_cast<std::size_t>(lca2)]) * 18LL +
                static_cast<long long>(base.support) * 120LL +
                (base.from_incumbent ? 50LL : 0LL) +
                static_cast<long long>(std::min<std::size_t>(base.labels.size(), 32)) * 5LL;
            ranked.push_back({score, i});
        }
        std::stable_sort(
            ranked.begin(),
            ranked.end(),
            [&](const auto& a, const auto& b) {
                if (a.first != b.first) {
                    return a.first > b.first;
                }
                const auto& ca = base_components[a.second];
                const auto& cb = base_components[b.second];
                if (ca.labels.size() != cb.labels.size()) {
                    return ca.labels.size() > cb.labels.size();
                }
                return ca.labels < cb.labels;
            }
        );
        if (ranked.size() > opts.base_components_per_singleton) {
            ranked.resize(opts.base_components_per_singleton);
        }

        std::vector<std::uint32_t> paired_singletons;
        if (opts.enable_two_singleton_growth &&
            opts.second_singletons_per_target > 0 &&
            incumbent_singletons.size() > 1) {
            std::vector<std::pair<int, std::uint32_t>> choices;
            choices.reserve(incumbent_singletons.size());
            for (std::uint32_t z : incumbent_singletons) {
                if (z != x) {
                    choices.push_back({pair_closeness(x, z), z});
                }
            }
            std::stable_sort(choices.begin(), choices.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                    return a.first > b.first;
                }
                return a.second < b.second;
            });
            const std::size_t take =
                std::min(opts.second_singletons_per_target, choices.size());
            for (std::size_t i = 0; i < take; ++i) {
                paired_singletons.push_back(choices[i].second);
            }
        }

        for (const auto& entry : ranked) {
            if (result.size() >= opts.max_candidates || timer.should_stop(1.5)) {
                break;
            }

            const SingletonRescueBaseComponent& base = base_components[entry.second];

            std::vector<std::uint32_t> attached = base.labels;
            attached.push_back(x);
            add_component_labels(std::move(attached));

            if (!paired_singletons.empty()) {
                for (std::uint32_t z : paired_singletons) {
                    if (label_set_contains(base.labels, z)) {
                        continue;
                    }
                    std::vector<std::uint32_t> grown = base.labels;
                    grown.push_back(x);
                    grown.push_back(z);
                    add_component_labels(std::move(grown));
                    if (result.size() >= opts.max_candidates) {
                        break;
                    }
                }
            }

            if (opts.eject_labels_per_component == 0) {
                continue;
            }

            std::vector<std::pair<int, std::uint32_t>> eject_order;
            eject_order.reserve(base.labels.size());
            for (std::uint32_t y : base.labels) {
                eject_order.push_back({pair_closeness(x, y), y});
            }
            std::stable_sort(eject_order.begin(), eject_order.end(), [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                    return a.first < b.first;
                }
                return a.second < b.second;
            });
            const std::size_t eject_take =
                std::min(opts.eject_labels_per_component, eject_order.size());
            for (std::size_t ei = 0; ei < eject_take; ++ei) {
                const std::uint32_t y = eject_order[ei].second;
                std::vector<std::uint32_t> swapped;
                swapped.reserve(base.labels.size());
                for (std::uint32_t label : base.labels) {
                    if (label != y) {
                        swapped.push_back(label);
                    }
                }
                swapped.push_back(x);
                add_component_labels(std::move(swapped));

                if (!paired_singletons.empty()) {
                    for (std::uint32_t z : paired_singletons) {
                        if (z == y || label_set_contains(base.labels, z)) {
                            continue;
                        }
                        std::vector<std::uint32_t> double_swapped;
                        double_swapped.reserve(base.labels.size() + 1);
                        for (std::uint32_t label : base.labels) {
                            if (label != y) {
                                double_swapped.push_back(label);
                            }
                        }
                        double_swapped.push_back(x);
                        double_swapped.push_back(z);
                        add_component_labels(std::move(double_swapped));
                        if (result.size() >= opts.max_candidates) {
                            break;
                        }
                    }
                }

                if (result.size() >= opts.max_candidates) {
                    break;
                }
            }
        }
    }

    return result;
}

struct SeedShatterOptions {
    std::size_t max_candidates = 0;
    std::size_t max_seed_components = 0;
    std::size_t max_direct_component_size = 0;
    std::size_t bad_group_pool = 0;
    std::size_t max_prefix_bad_groups = 0;
    std::size_t max_window_components = 0;
    std::array<std::size_t, 3> window_sizes{};
};

SeedShatterOptions seed_shatter_options(
    std::size_t n,
    AgreementPackingMode mode,
    bool packing_focused
) {
    SeedShatterOptions opts;
    if (!is_singleton_rescue_packing_mode(mode)) {
        return opts;
    }

    if (n <= 700) {
        opts.max_candidates = 14000;
        opts.max_seed_components = 1000;
        opts.max_direct_component_size = 160;
        opts.bad_group_pool = 7;
        opts.max_prefix_bad_groups = 3;
        opts.max_window_components = 4;
        opts.window_sizes = {16, 32, 64};
    } else if (n <= 2500) {
        opts.max_candidates = 22000;
        opts.max_seed_components = 1600;
        opts.max_direct_component_size = 220;
        opts.bad_group_pool = 7;
        opts.max_prefix_bad_groups = 3;
        opts.max_window_components = 5;
        opts.window_sizes = {24, 48, 96};
    } else if (n <= 6000) {
        opts.max_candidates = 28000;
        opts.max_seed_components = 2200;
        opts.max_direct_component_size = 280;
        opts.bad_group_pool = 6;
        opts.max_prefix_bad_groups = 2;
        opts.max_window_components = 6;
        opts.window_sizes = {32, 64, 128};
    } else {
        opts.max_candidates = 32000;
        opts.max_seed_components = 3000;
        opts.max_direct_component_size = 360;
        opts.bad_group_pool = 6;
        opts.max_prefix_bad_groups = 2;
        opts.max_window_components = 6;
        opts.window_sizes = {48, 96, 192};
    }

    if (packing_focused) {
        opts.max_candidates = static_cast<std::size_t>(
            static_cast<double>(opts.max_candidates) * 1.20
        );
        opts.max_seed_components += n <= 2500 ? 300 : 600;
        opts.max_window_components += 2;
    }

    return opts;
}

std::vector<std::uint32_t> labels_without_sorted(
    const std::vector<std::uint32_t>& labels,
    std::vector<std::uint32_t> removed
) {
    if (removed.empty()) {
        return labels;
    }
    std::sort(removed.begin(), removed.end());
    removed.erase(std::unique(removed.begin(), removed.end()), removed.end());

    std::vector<std::uint32_t> out;
    out.reserve(labels.size());
    std::size_t ri = 0;
    for (std::uint32_t label : labels) {
        while (ri < removed.size() && removed[ri] < label) {
            ++ri;
        }
        if (ri < removed.size() && removed[ri] == label) {
            continue;
        }
        out.push_back(label);
    }
    return out;
}

std::vector<std::uint32_t> tree_ordered_subset(
    const std::vector<std::uint32_t>& labels,
    const Tree& tree
) {
    std::vector<std::uint32_t> ordered = labels;
    std::sort(ordered.begin(), ordered.end(), [&](std::uint32_t a, std::uint32_t b) {
        return tree.tin[static_cast<std::size_t>(tree.node_of_label(a))] <
               tree.tin[static_cast<std::size_t>(tree.node_of_label(b))];
    });
    return ordered;
}

std::vector<std::vector<std::uint32_t>> generate_blocked_seed_shatter_components(
    const CoreForest& incumbent,
    const std::vector<CoreForest>& seeds,
    const Tree& t1,
    const Tree& t2,
    Timer& timer,
    AgreementPackingMode mode,
    bool packing_focused,
    std::size_t hard_cap = 0
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    SeedShatterOptions opts = seed_shatter_options(n, mode, packing_focused);
    if (opts.max_candidates == 0 || seeds.empty() || timer.should_stop(2.0)) {
        return {};
    }
    if (hard_cap > 0) {
        opts.max_candidates = std::min(opts.max_candidates, hard_cap);
    }

    CoreForest normalized_incumbent = incumbent;
    normalized_incumbent.normalize();

    std::uint32_t max_label = 0;
    for (std::uint32_t label : t1.leaf_labels) {
        max_label = std::max(max_label, label);
    }
    std::vector<int> owner_flat(static_cast<std::size_t>(max_label) + 1, -1);
    std::vector<std::size_t> owner_size;
    owner_size.reserve(normalized_incumbent.components.size());
    for (std::size_t cid = 0; cid < normalized_incumbent.components.size(); ++cid) {
        const CoreComponent& component = normalized_incumbent.components[cid];
        owner_size.push_back(component.labels.size());
        for (std::uint32_t label : component.labels) {
            if (label < owner_flat.size()) {
                owner_flat[static_cast<std::size_t>(label)] = static_cast<int>(cid);
            }
        }
    }

    struct SeedComponent {
        std::vector<std::uint32_t> labels;
        long long score = 0;
        std::size_t singleton_overlap = 0;
    };

    std::unordered_set<std::vector<std::uint32_t>, LabelSetHash> seen_seed_components;
    seen_seed_components.reserve(opts.max_seed_components * 2 + 1);
    std::vector<SeedComponent> seed_components;
    seed_components.reserve(opts.max_seed_components + 1);

    for (const CoreForest& seed : seeds) {
        for (const CoreComponent& component : seed.components) {
            if (component.labels.size() < 4) {
                continue;
            }
            std::vector<std::uint32_t> labels = component.labels;
            std::sort(labels.begin(), labels.end());
            labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
            if (labels.size() < 4 ||
                !seen_seed_components.insert(labels).second) {
                continue;
            }

            std::size_t singleton_overlap = 0;
            std::size_t touched = 0;
            std::unordered_set<int> owners;
            owners.reserve(labels.size() * 2 + 1);
            for (std::uint32_t label : labels) {
                const int owner = label < owner_flat.size()
                    ? owner_flat[static_cast<std::size_t>(label)]
                    : -1;
                if (owner >= 0) {
                    owners.insert(owner);
                    if (owner_size[static_cast<std::size_t>(owner)] == 1) {
                        ++singleton_overlap;
                    }
                }
            }
            touched = owners.size();
            if (singleton_overlap < 2 && labels.size() < 8) {
                continue;
            }

            SeedComponent out;
            out.singleton_overlap = singleton_overlap;
            out.score =
                static_cast<long long>(singleton_overlap) * 120LL +
                static_cast<long long>(labels.size()) * 8LL -
                static_cast<long long>(touched) * 10LL;
            out.labels = std::move(labels);
            seed_components.push_back(std::move(out));
        }
    }

    if (seed_components.empty()) {
        return {};
    }
    std::stable_sort(
        seed_components.begin(),
        seed_components.end(),
        [](const SeedComponent& a, const SeedComponent& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            if (a.singleton_overlap != b.singleton_overlap) {
                return a.singleton_overlap > b.singleton_overlap;
            }
            if (a.labels.size() != b.labels.size()) {
                return a.labels.size() > b.labels.size();
            }
            return a.labels < b.labels;
        }
    );
    if (seed_components.size() > opts.max_seed_components) {
        seed_components.resize(opts.max_seed_components);
    }

    std::vector<std::vector<std::uint32_t>> result;
    result.reserve(opts.max_candidates);
    std::unordered_set<std::vector<std::uint32_t>, LabelSetHash> seen;
    seen.reserve(opts.max_candidates * 2 + 1);

    auto add_candidate_labels = [&](std::vector<std::uint32_t> labels) {
        if (result.size() >= opts.max_candidates) {
            return;
        }
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (labels.size() < 3) {
            return;
        }
        if (seen.insert(labels).second) {
            result.push_back(std::move(labels));
        }
    };

    struct OwnerGroup {
        int owner = -1;
        std::vector<std::uint32_t> labels;
        std::size_t owner_size = 0;
        long long badness = 0;
    };

    auto add_windows = [&](
        const std::vector<std::uint32_t>& labels,
        const Tree& tree,
        std::size_t& windows_added
    ) {
        if (labels.size() < 6 || windows_added >= opts.max_window_components) {
            return;
        }
        std::vector<std::uint32_t> ordered = tree_ordered_subset(labels, tree);
        for (std::size_t window : opts.window_sizes) {
            if (window < 3 || window >= ordered.size() ||
                windows_added >= opts.max_window_components) {
                continue;
            }
            const std::size_t stride = std::max<std::size_t>(1, window / 2);
            for (std::size_t start = 0;
                 start + window <= ordered.size() &&
                 windows_added < opts.max_window_components;
                 start += stride) {
                std::vector<std::uint32_t> chunk(
                    ordered.begin() + static_cast<std::ptrdiff_t>(start),
                    ordered.begin() + static_cast<std::ptrdiff_t>(start + window)
                );
                add_candidate_labels(std::move(chunk));
                ++windows_added;
            }
        }
    };

    for (const SeedComponent& seed_component : seed_components) {
        if (result.size() >= opts.max_candidates || timer.should_stop(2.0)) {
            break;
        }

        std::unordered_map<int, std::size_t> group_index;
        group_index.reserve(seed_component.labels.size() * 2 + 1);
        std::vector<OwnerGroup> groups;
        groups.reserve(seed_component.labels.size());
        std::vector<std::uint32_t> singleton_core;
        singleton_core.reserve(seed_component.labels.size());

        for (std::uint32_t label : seed_component.labels) {
            const int owner = label < owner_flat.size()
                ? owner_flat[static_cast<std::size_t>(label)]
                : -1;
            if (owner < 0) {
                continue;
            }
            auto [it, inserted] = group_index.emplace(owner, groups.size());
            if (inserted) {
                OwnerGroup group;
                group.owner = owner;
                group.owner_size = owner_size[static_cast<std::size_t>(owner)];
                groups.push_back(std::move(group));
            }
            groups[it->second].labels.push_back(label);
        }

        for (OwnerGroup& group : groups) {
            if (group.owner_size == 1 ||
                (group.labels.size() == group.owner_size &&
                 group.owner_size <= 6)) {
                singleton_core.insert(
                    singleton_core.end(),
                    group.labels.begin(),
                    group.labels.end()
                );
            }
            const std::size_t outside =
                group.owner_size > group.labels.size()
                    ? group.owner_size - group.labels.size()
                    : 0;
            group.badness =
                static_cast<long long>(outside) * 120LL +
                static_cast<long long>(group.owner_size) * 9LL -
                static_cast<long long>(group.labels.size()) * 18LL;
            if (group.labels.size() <= 2 && group.owner_size > group.labels.size()) {
                group.badness += 160;
            }
        }

        std::stable_sort(groups.begin(), groups.end(), [](const OwnerGroup& a, const OwnerGroup& b) {
            if (a.badness != b.badness) {
                return a.badness > b.badness;
            }
            if (a.owner_size != b.owner_size) {
                return a.owner_size > b.owner_size;
            }
            return a.labels < b.labels;
        });

        if (singleton_core.size() >= 3 &&
            singleton_core.size() <= opts.max_direct_component_size) {
            add_candidate_labels(singleton_core);
        } else if (singleton_core.size() > opts.max_direct_component_size) {
            std::size_t windows_added = 0;
            add_windows(singleton_core, t1, windows_added);
            add_windows(singleton_core, t2, windows_added);
        }

        const std::size_t pool = std::min(opts.bad_group_pool, groups.size());
        if (seed_component.labels.size() <= opts.max_direct_component_size) {
            std::vector<std::uint32_t> removed;
            for (std::size_t k = 0;
                 k < std::min(opts.max_prefix_bad_groups, pool);
                 ++k) {
                if (groups[k].badness <= 0) {
                    break;
                }
                removed.insert(
                    removed.end(),
                    groups[k].labels.begin(),
                    groups[k].labels.end()
                );
                add_candidate_labels(labels_without_sorted(seed_component.labels, removed));
            }

            for (std::size_t i = 0; i < pool; ++i) {
                if (groups[i].badness <= 0) {
                    continue;
                }
                add_candidate_labels(
                    labels_without_sorted(seed_component.labels, groups[i].labels)
                );
                for (std::size_t j = i + 1; j < pool; ++j) {
                    if (groups[j].badness <= 0 ||
                        result.size() >= opts.max_candidates) {
                        break;
                    }
                    std::vector<std::uint32_t> pair_removed = groups[i].labels;
                    pair_removed.insert(
                        pair_removed.end(),
                        groups[j].labels.begin(),
                        groups[j].labels.end()
                    );
                    add_candidate_labels(
                        labels_without_sorted(seed_component.labels, std::move(pair_removed))
                    );
                }
            }
        }

        std::size_t windows_added = 0;
        add_windows(seed_component.labels, t1, windows_added);
        add_windows(seed_component.labels, t2, windows_added);
    }

    return result;
}

struct SeedUnionOptions {
    std::size_t max_candidates = 0;
    std::size_t max_seed_forests = 0;
    std::size_t max_items_per_forest = 0;
    std::size_t max_union_labels = 0;
    std::size_t max_piece_labels = 0;
    std::size_t neighbor_window = 0;
    bool enable_triples = false;
};

SeedUnionOptions seed_union_options(
    std::size_t n,
    AgreementPackingMode mode,
    bool packing_focused
) {
    SeedUnionOptions opts;
    if (!is_singleton_rescue_packing_mode(mode)) {
        return opts;
    }

    if (n <= 700) {
        opts.max_candidates = 18000;
        opts.max_seed_forests = 14;
        opts.max_items_per_forest = 900;
        opts.max_union_labels = 80;
        opts.max_piece_labels = 48;
        opts.neighbor_window = 9;
        opts.enable_triples = true;
    } else if (n <= 2500) {
        opts.max_candidates = 24000;
        opts.max_seed_forests = 16;
        opts.max_items_per_forest = 1300;
        opts.max_union_labels = 96;
        opts.max_piece_labels = 56;
        opts.neighbor_window = 8;
        opts.enable_triples = true;
    } else if (n <= 6000) {
        opts.max_candidates = 30000;
        opts.max_seed_forests = 18;
        opts.max_items_per_forest = 1700;
        opts.max_union_labels = 112;
        opts.max_piece_labels = 64;
        opts.neighbor_window = 7;
        opts.enable_triples = false;
    } else {
        opts.max_candidates = 36000;
        opts.max_seed_forests = 18;
        opts.max_items_per_forest = 2200;
        opts.max_union_labels = 128;
        opts.max_piece_labels = 72;
        opts.neighbor_window = 6;
        opts.enable_triples = false;
    }

    if (packing_focused) {
        opts.max_candidates = static_cast<std::size_t>(
            static_cast<double>(opts.max_candidates) * 1.20
        );
        opts.max_items_per_forest += n <= 2500 ? 300 : 500;
        opts.neighbor_window += 1;
    }

    return opts;
}

std::vector<std::vector<std::uint32_t>> generate_seed_union_components(
    const std::vector<CoreForest>& seeds,
    const Tree& t1,
    const Tree& t2,
    Timer& timer,
    AgreementPackingMode mode,
    bool packing_focused,
    std::size_t hard_cap = 0
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    SeedUnionOptions opts = seed_union_options(n, mode, packing_focused);
    if (opts.max_candidates == 0 || seeds.empty() || timer.should_stop(2.0)) {
        return {};
    }
    if (hard_cap > 0) {
        opts.max_candidates = std::min(opts.max_candidates, hard_cap);
    }

    struct Item {
        std::vector<std::uint32_t> labels;
        int root1 = -1;
        int root2 = -1;
        int tin1 = 0;
        int tin2 = 0;
        long long score = 0;
    };

    std::vector<std::vector<std::uint32_t>> result;
    result.reserve(opts.max_candidates);
    std::unordered_set<std::vector<std::uint32_t>, LabelSetHash> seen;
    seen.reserve(opts.max_candidates * 2 + 1);

    auto add_union_candidate = [&](std::vector<std::uint32_t> labels) {
        if (result.size() >= opts.max_candidates) {
            return;
        }
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (labels.size() < 3 || labels.size() > opts.max_union_labels) {
            return;
        }
        if (seen.insert(labels).second) {
            result.push_back(std::move(labels));
        }
    };

    auto merged_labels = [](const std::vector<std::uint32_t>& a,
                            const std::vector<std::uint32_t>& b) {
        std::vector<std::uint32_t> merged;
        merged.reserve(a.size() + b.size());
        merged.insert(merged.end(), a.begin(), a.end());
        merged.insert(merged.end(), b.begin(), b.end());
        return merged;
    };

    const std::size_t seed_limit = std::min(opts.max_seed_forests, seeds.size());
    for (std::size_t seed_index = 0;
         seed_index < seed_limit &&
         result.size() < opts.max_candidates &&
         !timer.should_stop(2.0);
         ++seed_index) {
        CoreForest forest = seeds[seed_index];
        forest.normalize();

        std::vector<Item> items;
        items.reserve(std::min(opts.max_items_per_forest, forest.components.size()));
        for (const CoreComponent& component : forest.components) {
            if (component.labels.empty() ||
                component.labels.size() > opts.max_piece_labels) {
                continue;
            }

            Item item;
            item.labels = component.labels;
            std::sort(item.labels.begin(), item.labels.end());
            item.labels.erase(
                std::unique(item.labels.begin(), item.labels.end()),
                item.labels.end()
            );
            if (item.labels.empty() ||
                item.labels.size() > opts.max_piece_labels) {
                continue;
            }
            item.root1 = component_root_in_tree(t1, item.labels);
            item.root2 = component_root_in_tree(t2, item.labels);
            item.tin1 = t1.tin[static_cast<std::size_t>(item.root1)];
            item.tin2 = t2.tin[static_cast<std::size_t>(item.root2)];
            item.score =
                static_cast<long long>(std::min<std::size_t>(item.labels.size(), 16)) * 100LL +
                static_cast<long long>(t1.depth[static_cast<std::size_t>(item.root1)] +
                                       t2.depth[static_cast<std::size_t>(item.root2)]);
            items.push_back(std::move(item));
        }

        if (items.size() < 2) {
            continue;
        }
        std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            if (a.score != b.score) {
                return a.score > b.score;
            }
            if (a.labels.size() != b.labels.size()) {
                return a.labels.size() > b.labels.size();
            }
            return a.labels < b.labels;
        });
        if (items.size() > opts.max_items_per_forest) {
            items.resize(opts.max_items_per_forest);
        }

        std::vector<int> order1(items.size());
        std::vector<int> order2(items.size());
        std::iota(order1.begin(), order1.end(), 0);
        std::iota(order2.begin(), order2.end(), 0);
        std::sort(order1.begin(), order1.end(), [&](int ai, int bi) {
            const Item& a = items[static_cast<std::size_t>(ai)];
            const Item& b = items[static_cast<std::size_t>(bi)];
            if (a.tin1 != b.tin1) {
                return a.tin1 < b.tin1;
            }
            return a.tin2 < b.tin2;
        });
        std::sort(order2.begin(), order2.end(), [&](int ai, int bi) {
            const Item& a = items[static_cast<std::size_t>(ai)];
            const Item& b = items[static_cast<std::size_t>(bi)];
            if (a.tin2 != b.tin2) {
                return a.tin2 < b.tin2;
            }
            return a.tin1 < b.tin1;
        });

        auto process_order = [&](const std::vector<int>& order) {
            for (std::size_t pos = 0;
                 pos < order.size() &&
                 result.size() < opts.max_candidates &&
                 !timer.should_stop(2.0);
                 ++pos) {
                const int ai = order[pos];
                const Item& a = items[static_cast<std::size_t>(ai)];
                const std::size_t end =
                    std::min(order.size(), pos + opts.neighbor_window + 1);
                for (std::size_t next = pos + 1;
                     next < end && result.size() < opts.max_candidates;
                     ++next) {
                    const int bi = order[next];
                    if (ai == bi) {
                        continue;
                    }
                    const Item& b = items[static_cast<std::size_t>(bi)];
                    const std::size_t pair_size = a.labels.size() + b.labels.size();
                    if (pair_size > opts.max_union_labels) {
                        continue;
                    }
                    std::vector<std::uint32_t> pair = merged_labels(a.labels, b.labels);
                    add_union_candidate(pair);

                    if (!opts.enable_triples ||
                        pair_size >= opts.max_union_labels ||
                        next + 1 >= end ||
                        result.size() >= opts.max_candidates) {
                        continue;
                    }
                    for (std::size_t third = next + 1;
                         third < end && result.size() < opts.max_candidates;
                         ++third) {
                        const int ci = order[third];
                        if (ci == ai || ci == bi) {
                            continue;
                        }
                        const Item& c = items[static_cast<std::size_t>(ci)];
                        if (pair_size + c.labels.size() > opts.max_union_labels) {
                            continue;
                        }
                        std::vector<std::uint32_t> triple = pair;
                        triple.insert(triple.end(), c.labels.begin(), c.labels.end());
                        add_union_candidate(std::move(triple));
                        break;
                    }
                }
            }
        };

        process_order(order1);
        if (result.size() < opts.max_candidates && !timer.should_stop(2.0)) {
            process_order(order2);
        }
    }

    return result;
}

bool run_agreement_component_packing(
    CoreForest& best,
    const Tree& t1,
    const Tree& t2,
    std::vector<CoreForest> seeds,
    Timer& timer,
    bool extended_search,
    const std::string& profile_phase,
    const std::function<void(const CoreForest&)>& publish_progress = {},
    AgreementPackingMode mode = AgreementPackingMode::Normal,
    bool packing_focused = false,
    bool strict_component_improvement_only = false,
    std::vector<std::vector<std::uint32_t>> extra_candidate_components = {},
    ComponentForestArchive* archive = nullptr
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    if (timer.should_stop(1.5)) {
        return false;
    }

    std::size_t archive_seed_forests = 0;
    std::size_t archive_extra_candidates = 0;
    const std::size_t archive_forests_before =
        archive == nullptr ? 0 : archive->forest_count();
    const std::size_t archive_components_before =
        archive == nullptr ? 0 : archive->component_count();
    if (archive != nullptr) {
        std::vector<CoreForest> archive_seeds =
            archive->seed_forests(archive_seed_limit_for_packing(n, extended_search, mode));
        archive_seed_forests = archive_seeds.size();
        for (CoreForest& archive_seed : archive_seeds) {
            remember_diverse_packing_seed(seeds, std::move(archive_seed), 24);
        }
    }

    std::size_t invalid_seed_forests = 0;
    std::vector<CoreForest> valid_seeds;
    valid_seeds.reserve(seeds.size() + 1);
    for (CoreForest& seed : seeds) {
        if (is_valid_agreement_forest(seed, t1, t2)) {
            remember_packing_seed(valid_seeds, std::move(seed), 24);
        } else {
            ++invalid_seed_forests;
        }
    }
    seeds = std::move(valid_seeds);
    remember_packing_seed(seeds, best, 24);
    if (extra_candidate_components.empty() && strict_component_improvement_only) {
        extra_candidate_components = generate_singleton_rescue_components(
            best,
            seeds,
            t1,
            t2,
            timer,
            extended_search,
            mode,
            packing_focused
        );
    }
    const std::size_t singleton_rescue_candidates =
        extra_candidate_components.size();
    if (archive != nullptr) {
        std::vector<std::vector<std::uint32_t>> archived_components =
            archive->extra_candidate_components(
                t1.leaf_labels,
                archive_extra_component_limit_for_packing(n, extended_search, mode),
                archive_extra_component_max_size(n)
            );
        archive_extra_candidates = archived_components.size();
        if (!archived_components.empty()) {
            extra_candidate_components.reserve(
                extra_candidate_components.size() + archived_components.size()
            );
            for (std::vector<std::uint32_t>& component : archived_components) {
                extra_candidate_components.push_back(std::move(component));
            }
        }
    }

    const double phase_start = timer.elapsed_seconds();
    const std::size_t before = best.component_count();
    bool accepted = false;
    bool threw = false;
    pace26::heuristics::AgreementComponentPacking::Stats packing_stats;

    try {
        auto opts = agreement_packing_options(
            n,
            extended_search,
            timer,
            mode,
            packing_focused
        );
        if (g_profile.enabled()) {
            opts.stats = &packing_stats;
        }
        if (archive != nullptr) {
            const std::size_t capture_limit =
                archive_capture_component_limit_for_packing(n, extended_search, mode);
            if (capture_limit != 0) {
                opts.archive_component =
                    [archive](const std::vector<std::uint32_t>& labels) {
                        archive->remember_component(labels);
                    };
                opts.max_archived_candidate_components = capture_limit;
                opts.min_archived_candidate_size = 3;
            }
        }
        if (!extra_candidate_components.empty()) {
            const std::size_t extra_capacity =
                std::min<std::size_t>(extra_candidate_components.size(), 30000);
            opts.max_candidates = std::min<std::size_t>(
                700000,
                opts.max_candidates + extra_capacity
            );
        }
        opts.max_extra_candidate_components = extra_candidate_components.size();
        opts.extra_candidate_components = std::move(extra_candidate_components);
        pace26::heuristics::AgreementComponentPacking packing(opts);
        CoreForest candidate = packing.solve(t1, t2, seeds, &timer);
        if (archive != nullptr) {
            archive->remember_if_valid(candidate, t1, t2);
        }
        if (strict_component_improvement_only) {
            candidate.normalize();
            best.normalize();
            if (candidate.component_count() < best.component_count() &&
                is_valid_agreement_forest(candidate, t1, t2)) {
                best = std::move(candidate);
                accepted = true;
            }
        } else {
            accepted = consider_candidate(best, std::move(candidate), t1, t2);
        }
        if (accepted && publish_progress) {
            publish_progress(best);
        }
    } catch (const std::exception&) {
        threw = true;
    }

    if (g_profile.enabled()) {
        g_profile.phase_result(
            profile_phase,
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {
                {"seed_forests", std::to_string(seeds.size())},
                {"invalid_seed_forests", std::to_string(invalid_seed_forests)},
                {"extended", extended_search ? "true" : "false"},
                {"mode", json_quote(agreement_packing_mode_name(mode))},
                {"packing_focused", packing_focused ? "true" : "false"},
                {"strict_component_improvement_only", strict_component_improvement_only ? "true" : "false"},
                {"exception", threw ? "true" : "false"},
                {"singleton_rescue_candidates", std::to_string(singleton_rescue_candidates)},
                {"archive_seed_forests", std::to_string(archive_seed_forests)},
                {"archive_extra_candidates", std::to_string(archive_extra_candidates)},
                {"archive_forests_before", std::to_string(archive_forests_before)},
                {"archive_forests_after", std::to_string(archive == nullptr ? 0 : archive->forest_count())},
                {"archive_components_before", std::to_string(archive_components_before)},
                {"archive_components_after", std::to_string(archive == nullptr ? 0 : archive->component_count())},
                {"packing_seed_forests", std::to_string(packing_stats.seed_forests)},
                {"packing_seed_components", std::to_string(packing_stats.seed_components)},
                {"packing_seed_groups", std::to_string(packing_stats.seed_groups)},
                {"packing_extra_candidate_components", std::to_string(packing_stats.extra_candidate_components)},
                {"packing_extra_candidate_added", std::to_string(packing_stats.extra_candidate_added)},
                {"packing_archived_candidate_components", std::to_string(packing_stats.archived_candidate_components)},
                {"packing_add_attempts", std::to_string(packing_stats.add_attempts)},
                {"packing_candidates", std::to_string(packing_stats.added)},
                {"packing_candidates_from_seed", std::to_string(packing_stats.added_from_seed)},
                {"packing_duplicate_rejects", std::to_string(packing_stats.duplicate)},
                {"packing_capacity_rejects", std::to_string(packing_stats.capacity_reject)},
                {"packing_size_quota_rejects", std::to_string(packing_stats.size_quota_reject)},
                {"packing_topology_rejects", std::to_string(packing_stats.topology_reject)},
                {"packing_singleton_rejects", std::to_string(packing_stats.singleton_reject)},
                {"packing_unknown_label_rejects", std::to_string(packing_stats.unknown_label_reject)},
                {"packing_candidates_size2", std::to_string(packing_stats.added_by_size[2])},
                {"packing_candidates_size3", std::to_string(packing_stats.added_by_size[3])},
                {"packing_candidates_size4", std::to_string(packing_stats.added_by_size[4])},
                {"packing_candidates_size5_8", std::to_string(packing_stats.added_by_size[5])},
                {"packing_candidates_size9_16", std::to_string(packing_stats.added_by_size[6])},
                {"packing_candidates_size17plus", std::to_string(packing_stats.added_by_size[7])},
                {"packing_candidates_size5plus", std::to_string(
                    packing_stats.added_by_size[5] +
                    packing_stats.added_by_size[6] +
                    packing_stats.added_by_size[7])},
                {"packing_generated_size2", std::to_string(packing_stats.generated_added_by_size[2])},
                {"packing_generated_size3", std::to_string(packing_stats.generated_added_by_size[3])},
                {"packing_generated_size4", std::to_string(packing_stats.generated_added_by_size[4])},
                {"packing_generated_size5_8", std::to_string(packing_stats.generated_added_by_size[5])},
                {"packing_generated_size9_16", std::to_string(packing_stats.generated_added_by_size[6])},
                {"packing_generated_size17plus", std::to_string(packing_stats.generated_added_by_size[7])},
                {"packing_generated_size5plus", std::to_string(
                    packing_stats.generated_added_by_size[5] +
                    packing_stats.generated_added_by_size[6] +
                    packing_stats.generated_added_by_size[7])},
                {"packing_capacity_rejects_size2", std::to_string(packing_stats.capacity_reject_by_size[2])},
                {"packing_capacity_rejects_size3", std::to_string(packing_stats.capacity_reject_by_size[3])},
                {"packing_capacity_rejects_size4", std::to_string(packing_stats.capacity_reject_by_size[4])},
                {"packing_capacity_rejects_size5_8", std::to_string(packing_stats.capacity_reject_by_size[5])},
                {"packing_capacity_rejects_size9_16", std::to_string(packing_stats.capacity_reject_by_size[6])},
                {"packing_capacity_rejects_size17plus", std::to_string(packing_stats.capacity_reject_by_size[7])},
                {"packing_capacity_rejects_size5plus", std::to_string(
                    packing_stats.capacity_reject_by_size[5] +
                    packing_stats.capacity_reject_by_size[6] +
                    packing_stats.capacity_reject_by_size[7])},
                {"packing_size_quota_rejects_size2", std::to_string(packing_stats.size_quota_reject_by_size[2])},
                {"packing_size_quota_rejects_size3", std::to_string(packing_stats.size_quota_reject_by_size[3])},
                {"packing_size_quota_rejects_size4", std::to_string(packing_stats.size_quota_reject_by_size[4])},
                {"packing_size_quota_rejects_size5_8", std::to_string(packing_stats.size_quota_reject_by_size[5])},
                {"packing_size_quota_rejects_size9_16", std::to_string(packing_stats.size_quota_reject_by_size[6])},
                {"packing_size_quota_rejects_size17plus", std::to_string(packing_stats.size_quota_reject_by_size[7])},
                {"packing_size_quota_rejects_size5plus", std::to_string(
                    packing_stats.size_quota_reject_by_size[5] +
                    packing_stats.size_quota_reject_by_size[6] +
                    packing_stats.size_quota_reject_by_size[7])},
                {"packing_after_seeds", std::to_string(packing_stats.candidates_after_seeds)},
                {"packing_after_pairs", std::to_string(packing_stats.candidates_after_pairs)},
                {"packing_after_extensions", std::to_string(packing_stats.candidates_after_extensions)},
                {"packing_after_structure_growth", std::to_string(packing_stats.candidates_after_structure_growth)},
                {"packing_structure_growth_attempts", std::to_string(packing_stats.structure_growth_attempts)},
                {"packing_structure_growth_added", std::to_string(packing_stats.structure_growth_added)},
                {"packing_seed_intersection_attempts", std::to_string(packing_stats.seed_intersection_attempts)},
                {"packing_seed_intersection_added", std::to_string(packing_stats.seed_intersection_added)},
                {"packing_greedy_variants", std::to_string(packing_stats.greedy_variants)},
                {"packing_greedy_polished_variants", std::to_string(packing_stats.greedy_polished_variants)},
                {"packing_exchange_rounds", std::to_string(packing_stats.exchange_rounds)},
                {"packing_deficit_exchange_tests", std::to_string(packing_stats.deficit_exchange_tests)},
                {"packing_deficit_exchange_attempts", std::to_string(packing_stats.deficit_exchange_attempts)},
                {"packing_deficit_exchange_accepted", std::to_string(packing_stats.deficit_exchange_accepted)},
                {"packing_deficit_exchange_rolled_back", std::to_string(packing_stats.deficit_exchange_rolled_back)},
                {"packing_local_mwis_runs", std::to_string(packing_stats.local_mwis_runs)},
                {"packing_local_mwis_accepted", std::to_string(packing_stats.local_mwis_accepted)},
                {"packing_local_mwis_pool_total", std::to_string(packing_stats.local_mwis_pool_total)},
                {"packing_local_mwis_largest_pool", std::to_string(packing_stats.local_mwis_largest_pool)},
                {"packing_local_mwis_nodes", std::to_string(packing_stats.local_mwis_nodes)},
                {"packing_exact_ran", packing_stats.exact_ran ? "true" : "false"},
                {"packing_exact_pool", std::to_string(packing_stats.exact_pool_size)},
                {"packing_exact_pool_size2", std::to_string(packing_stats.exact_pool_by_size[2])},
                {"packing_exact_pool_size3", std::to_string(packing_stats.exact_pool_by_size[3])},
                {"packing_exact_pool_size4", std::to_string(packing_stats.exact_pool_by_size[4])},
                {"packing_exact_pool_size5_8", std::to_string(packing_stats.exact_pool_by_size[5])},
                {"packing_exact_pool_size9_16", std::to_string(packing_stats.exact_pool_by_size[6])},
                {"packing_exact_pool_size17plus", std::to_string(packing_stats.exact_pool_by_size[7])},
                {"packing_exact_pool_size5plus", std::to_string(
                    packing_stats.exact_pool_by_size[5] +
                    packing_stats.exact_pool_by_size[6] +
                    packing_stats.exact_pool_by_size[7])},
                {"packing_exact_nodes", std::to_string(packing_stats.exact_nodes)},
                {"packing_exact_conflict_edges", std::to_string(packing_stats.exact_conflict_edges)},
                {"packing_exact_components", std::to_string(packing_stats.exact_components)},
                {"packing_exact_largest_component", std::to_string(packing_stats.exact_largest_component)},
                {"packing_exact_forced", std::to_string(packing_stats.exact_forced)},
                {"packing_exact_dominated", std::to_string(packing_stats.exact_dominated)},
                {"packing_selected", std::to_string(packing_stats.final_selected)},
                {"packing_selected_size2", std::to_string(packing_stats.selected_by_size[2])},
                {"packing_selected_size3", std::to_string(packing_stats.selected_by_size[3])},
                {"packing_selected_size4", std::to_string(packing_stats.selected_by_size[4])},
                {"packing_selected_size5_8", std::to_string(packing_stats.selected_by_size[5])},
                {"packing_selected_size9_16", std::to_string(packing_stats.selected_by_size[6])},
                {"packing_selected_size17plus", std::to_string(packing_stats.selected_by_size[7])},
                {"packing_selected_size5plus", std::to_string(
                    packing_stats.selected_by_size[5] +
                    packing_stats.selected_by_size[6] +
                    packing_stats.selected_by_size[7])},
                {"packing_uncovered_leaves", std::to_string(packing_stats.final_uncovered_leaves)},
                {"packing_final_gain", std::to_string(packing_stats.final_gain)},
                {"packing_stopped", packing_stats.stopped ? "true" : "false"}
            }
        );
    }
    return accepted;
}

struct IncumbentUnionOptions {
    std::size_t max_candidates = 0;
    std::size_t max_window_components = 0;
    std::size_t max_union_labels = 0;
    std::size_t max_piece_labels = 0;
    std::size_t neighbor_window = 0;
};

IncumbentUnionOptions incumbent_union_options(
    std::size_t n,
    AgreementPackingMode mode,
    bool packing_focused
) {
    IncumbentUnionOptions opts;
    if (!is_singleton_rescue_packing_mode(mode)) {
        return opts;
    }

    if (n <= 700) {
        opts.max_candidates = 16000;
        opts.max_window_components = 8;
        opts.max_union_labels = 260;
        opts.max_piece_labels = 180;
        opts.neighbor_window = 10;
    } else if (n <= 2500) {
        opts.max_candidates = 24000;
        opts.max_window_components = 8;
        opts.max_union_labels = 420;
        opts.max_piece_labels = 220;
        opts.neighbor_window = 9;
    } else if (n <= 6000) {
        opts.max_candidates = 30000;
        opts.max_window_components = 9;
        opts.max_union_labels = 720;
        opts.max_piece_labels = 320;
        opts.neighbor_window = 8;
    } else if (n <= 12000) {
        opts.max_candidates = 34000;
        opts.max_window_components = 8;
        opts.max_union_labels = 1000;
        opts.max_piece_labels = 420;
        opts.neighbor_window = 7;
    } else {
        opts.max_candidates = 32000;
        opts.max_window_components = 7;
        opts.max_union_labels = 1200;
        opts.max_piece_labels = 520;
        opts.neighbor_window = 6;
    }

    if (packing_focused) {
        opts.max_candidates = static_cast<std::size_t>(
            static_cast<double>(opts.max_candidates) * 1.20
        );
        opts.max_union_labels = static_cast<std::size_t>(
            static_cast<double>(opts.max_union_labels) * 1.15
        );
        opts.neighbor_window += 1;
    }

    return opts;
}

std::vector<std::vector<std::uint32_t>> generate_incumbent_union_components(
    const CoreForest& incumbent,
    const Tree& t1,
    const Tree& t2,
    Timer& timer,
    AgreementPackingMode mode,
    bool packing_focused,
    std::size_t hard_cap = 0
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    IncumbentUnionOptions opts = incumbent_union_options(n, mode, packing_focused);
    if (opts.max_candidates == 0 ||
        incumbent.components.size() < 2 ||
        timer.should_stop(2.0)) {
        return {};
    }
    if (hard_cap > 0) {
        opts.max_candidates = std::min(opts.max_candidates, hard_cap);
    }

    struct Item {
        std::size_t index = 0;
        const std::vector<std::uint32_t>* labels = nullptr;
        int root1 = -1;
        int root2 = -1;
        int tin1 = 0;
        int tin2 = 0;
        std::size_t size = 0;
    };

    CoreForest normalized = incumbent;
    normalized.normalize();

    std::vector<Item> items;
    items.reserve(normalized.components.size());
    for (std::size_t i = 0; i < normalized.components.size(); ++i) {
        const CoreComponent& component = normalized.components[i];
        if (component.empty() || component.labels.size() > opts.max_piece_labels) {
            continue;
        }
        Item item;
        item.index = i;
        item.labels = &component.labels;
        item.size = component.labels.size();
        item.root1 = component_root_in_tree(t1, component.labels);
        item.root2 = component_root_in_tree(t2, component.labels);
        item.tin1 = t1.tin[static_cast<std::size_t>(item.root1)];
        item.tin2 = t2.tin[static_cast<std::size_t>(item.root2)];
        items.push_back(item);
    }
    if (items.size() < 2) {
        return {};
    }

    std::vector<std::vector<std::uint32_t>> result;
    result.reserve(opts.max_candidates);
    std::unordered_set<std::vector<std::uint32_t>, LabelSetHash> seen;
    seen.reserve(opts.max_candidates * 2 + 1);

    auto add_candidate = [&](std::vector<std::uint32_t> labels) {
        if (result.size() >= opts.max_candidates) {
            return false;
        }
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (labels.size() < 3 || labels.size() > opts.max_union_labels) {
            return false;
        }
        if (!seen.insert(labels).second) {
            return false;
        }
        result.push_back(std::move(labels));
        return true;
    };

    auto append_item_labels = [](std::vector<std::uint32_t>& labels, const Item& item) {
        labels.insert(labels.end(), item.labels->begin(), item.labels->end());
    };

    auto process_order = [&](std::vector<int> order, bool reversed) {
        if (reversed) {
            std::reverse(order.begin(), order.end());
        }
        for (std::size_t start = 0;
             start < order.size() &&
             result.size() < opts.max_candidates &&
             !timer.should_stop(2.0);
             ++start) {
            std::vector<std::uint32_t> window_labels;
            window_labels.reserve(std::min<std::size_t>(opts.max_union_labels, 256));
            std::size_t labels_in_window = 0;

            for (std::size_t offset = 0;
                 offset < opts.max_window_components &&
                 start + offset < order.size();
                 ++offset) {
                const Item& item = items[static_cast<std::size_t>(order[start + offset])];
                labels_in_window += item.size;
                if (labels_in_window > opts.max_union_labels) {
                    break;
                }
                append_item_labels(window_labels, item);
                if (offset >= 1) {
                    add_candidate(window_labels);
                }
            }

            const std::size_t end =
                std::min(order.size(), start + opts.neighbor_window + 1);
            const Item& anchor = items[static_cast<std::size_t>(order[start])];
            for (std::size_t next = start + 1;
                 next < end && result.size() < opts.max_candidates;
                 ++next) {
                const Item& partner = items[static_cast<std::size_t>(order[next])];
                if (anchor.size + partner.size > opts.max_union_labels) {
                    continue;
                }
                std::vector<std::uint32_t> pair;
                pair.reserve(anchor.size + partner.size);
                append_item_labels(pair, anchor);
                append_item_labels(pair, partner);
                add_candidate(std::move(pair));
            }
        }
    };

    std::vector<int> order1(items.size());
    std::vector<int> order2(items.size());
    std::iota(order1.begin(), order1.end(), 0);
    std::iota(order2.begin(), order2.end(), 0);

    std::sort(order1.begin(), order1.end(), [&](int ai, int bi) {
        const Item& a = items[static_cast<std::size_t>(ai)];
        const Item& b = items[static_cast<std::size_t>(bi)];
        if (a.tin1 != b.tin1) return a.tin1 < b.tin1;
        if (a.tin2 != b.tin2) return a.tin2 < b.tin2;
        if (a.size != b.size) return a.size < b.size;
        return a.index < b.index;
    });
    std::sort(order2.begin(), order2.end(), [&](int ai, int bi) {
        const Item& a = items[static_cast<std::size_t>(ai)];
        const Item& b = items[static_cast<std::size_t>(bi)];
        if (a.tin2 != b.tin2) return a.tin2 < b.tin2;
        if (a.tin1 != b.tin1) return a.tin1 < b.tin1;
        if (a.size != b.size) return a.size < b.size;
        return a.index < b.index;
    });

    process_order(order1, false);
    if (result.size() < opts.max_candidates && !timer.should_stop(2.0)) {
        process_order(order2, false);
    }
    if (result.size() < opts.max_candidates && !timer.should_stop(2.0)) {
        process_order(order1, true);
    }
    if (result.size() < opts.max_candidates && !timer.should_stop(2.0)) {
        process_order(order2, true);
    }

    return result;
}

bool run_incumbent_union_repacking(
    CoreForest& best,
    const Tree& t1,
    const Tree& t2,
    Timer& timer,
    bool extended_search,
    const std::string& profile_phase,
    const std::function<void(const CoreForest&)>& publish_progress = {},
    AgreementPackingMode mode = AgreementPackingMode::Normal,
    bool packing_focused = false,
    ComponentForestArchive* archive = nullptr,
    std::size_t hard_cap = 0,
    double guard_seconds = 4.0
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    if (timer.should_stop(guard_seconds)) {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote(profile_phase)},
                {"n", std::to_string(n)},
                {"reason", json_quote("timer_guard")}
            }
        );
        return false;
    }

    std::vector<std::vector<std::uint32_t>> union_components =
        generate_incumbent_union_components(
            best,
            t1,
            t2,
            timer,
            mode,
            packing_focused,
            hard_cap
        );

    if (union_components.empty()) {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote(profile_phase)},
                {"n", std::to_string(n)},
                {"reason", json_quote("no_incumbent_union_candidates")}
            }
        );
        return false;
    }

    g_profile.event_raw(
        "candidate_bank",
        &timer,
        {
            {"phase", json_quote(profile_phase)},
            {"n", std::to_string(n)},
            {"incumbent_components", std::to_string(best.component_count())},
            {"incumbent_union_candidates", std::to_string(union_components.size())}
        }
    );

    std::vector<CoreForest> seeds;
    remember_diverse_packing_seed(seeds, best, 12);
    if (archive != nullptr) {
        std::vector<CoreForest> archive_seeds =
            archive->seed_forests(n >= 6000 ? 8 : 12);
        for (CoreForest& archive_seed : archive_seeds) {
            remember_diverse_packing_seed(
                seeds,
                std::move(archive_seed),
                n >= 6000 ? 10 : 14
            );
        }
    }

    return run_agreement_component_packing(
        best,
        t1,
        t2,
        std::move(seeds),
        timer,
        extended_search,
        profile_phase,
        publish_progress,
        mode,
        packing_focused,
        true,
        std::move(union_components),
        archive
    );
}

bool run_singleton_ejection_repacking(
    CoreForest& best,
    const Tree& t1,
    const Tree& t2,
    std::vector<CoreForest> seeds,
    Timer& timer,
    bool extended_search,
    const std::string& profile_phase,
    const std::function<void(const CoreForest&)>& publish_progress = {},
    AgreementPackingMode mode = AgreementPackingMode::Normal,
    bool packing_focused = false,
    ComponentForestArchive* archive = nullptr,
    std::size_t max_extra_candidates = 0,
    double guard_seconds = 3.0,
    bool include_seed_shatter = false
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    if (timer.should_stop(guard_seconds)) {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote(profile_phase)},
                {"n", std::to_string(n)},
                {"reason", json_quote("timer_guard")}
            }
        );
        return false;
    }

    remember_diverse_packing_seed(seeds, best, 16);
    if (archive != nullptr) {
        std::vector<CoreForest> archive_seeds =
            archive->seed_forests(archive_seed_limit_for_packing(
                n,
                true,
                mode
            ));
        for (CoreForest& archive_seed : archive_seeds) {
            remember_diverse_packing_seed(seeds, std::move(archive_seed), 20);
        }
    }

    std::vector<std::vector<std::uint32_t>> ejection_components =
        generate_singleton_rescue_components(
            best,
            seeds,
            t1,
            t2,
            timer,
            true,
            mode,
            packing_focused
        );
    const std::size_t singleton_ejection_candidates = ejection_components.size();
    const std::size_t ejection_cap =
        include_seed_shatter && max_extra_candidates > 0
            ? std::max<std::size_t>(1, max_extra_candidates / 2)
            : max_extra_candidates;
    if (ejection_cap > 0 && ejection_components.size() > ejection_cap) {
        ejection_components.resize(ejection_cap);
    }
    std::size_t seed_shatter_candidates = 0;
    std::size_t seed_union_candidates = 0;
    if (include_seed_shatter &&
        (max_extra_candidates == 0 ||
         ejection_components.size() < max_extra_candidates) &&
        !timer.should_stop(guard_seconds)) {
        std::size_t remaining_cap =
            max_extra_candidates == 0
                ? 0
                : max_extra_candidates - ejection_components.size();
        const std::size_t shatter_cap =
            max_extra_candidates == 0
                ? 0
                : std::max<std::size_t>(1, max_extra_candidates / 4);
        if (remaining_cap > 0) {
            remaining_cap = std::min(remaining_cap, shatter_cap);
        }
        std::vector<std::vector<std::uint32_t>> shatter_components =
            generate_blocked_seed_shatter_components(
                best,
                seeds,
                t1,
                t2,
                timer,
                mode,
                packing_focused,
                remaining_cap
            );
        seed_shatter_candidates = shatter_components.size();
        if (!shatter_components.empty()) {
            ejection_components.reserve(
                ejection_components.size() + shatter_components.size()
            );
            for (std::vector<std::uint32_t>& component : shatter_components) {
                ejection_components.push_back(std::move(component));
            }
        }
    }
    if (include_seed_shatter &&
        (max_extra_candidates == 0 ||
         ejection_components.size() < max_extra_candidates) &&
        !timer.should_stop(guard_seconds)) {
        const std::size_t remaining_cap =
            max_extra_candidates == 0
                ? 0
                : max_extra_candidates - ejection_components.size();
        std::vector<std::vector<std::uint32_t>> union_components =
            generate_seed_union_components(
                seeds,
                t1,
                t2,
                timer,
                mode,
                packing_focused,
                remaining_cap
            );
        seed_union_candidates = union_components.size();
        if (!union_components.empty()) {
            ejection_components.reserve(
                ejection_components.size() + union_components.size()
            );
            for (std::vector<std::uint32_t>& component : union_components) {
                ejection_components.push_back(std::move(component));
            }
        }
    }

    if (ejection_components.empty()) {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote(profile_phase)},
                {"n", std::to_string(n)},
                {"reason", json_quote("no_singleton_ejection_candidates")},
                {"seed_forests", std::to_string(seeds.size())}
            }
        );
        return false;
    }

    g_profile.event_raw(
        "candidate_bank",
        &timer,
        {
            {"phase", json_quote(profile_phase)},
            {"n", std::to_string(n)},
            {"seed_forests", std::to_string(seeds.size())},
            {"singleton_ejection_candidates", std::to_string(singleton_ejection_candidates)},
            {"seed_shatter_candidates", std::to_string(seed_shatter_candidates)},
            {"seed_union_candidates", std::to_string(seed_union_candidates)},
            {"total_extra_candidates", std::to_string(ejection_components.size())}
        }
    );

    return run_agreement_component_packing(
        best,
        t1,
        t2,
        std::move(seeds),
        timer,
        extended_search,
        profile_phase,
        publish_progress,
        mode,
        packing_focused,
        true,
        std::move(ejection_components),
        archive
    );
}

bool run_forest_crossover_portfolio(
    CoreForest& best,
    const Tree& t1,
    const Tree& t2,
    const std::vector<CoreForest>& seeds,
    Timer& timer,
    const std::string& profile_phase,
    const std::function<void(const CoreForest&)>& publish_progress = {},
    std::size_t max_seeds = 12,
    std::size_t max_passes = 2,
    ComponentForestArchive* archive = nullptr,
    bool enable_parent_parent = false,
    std::size_t max_parent_pairs = 0
) {
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    if (max_seeds == 0 || max_passes == 0 || timer.should_stop(1.25)) {
        return false;
    }

    CoreForest best_signature = best;
    best_signature.normalize();

    std::size_t invalid_seed_forests = 0;
    std::size_t duplicate_seed_forests = 0;
    std::size_t archive_seed_forests = 0;
    const std::size_t archive_forests_before =
        archive == nullptr ? 0 : archive->forest_count();
    const std::size_t archive_components_before =
        archive == nullptr ? 0 : archive->component_count();
    std::vector<CoreForest> parent_sources = seeds;
    if (archive != nullptr) {
        std::vector<CoreForest> archived_parents =
            archive->seed_forests(std::max<std::size_t>(max_seeds, 16));
        archive_seed_forests = archived_parents.size();
        parent_sources.reserve(parent_sources.size() + archived_parents.size());
        for (CoreForest& parent : archived_parents) {
            parent_sources.push_back(std::move(parent));
        }
    }
    std::vector<CoreForest> parents;
    parents.reserve(std::min(max_seeds, parent_sources.size()));

    for (const CoreForest& seed : parent_sources) {
        CoreForest parent = seed;
        parent.normalize();
        if (!is_valid_agreement_forest(parent, t1, t2)) {
            ++invalid_seed_forests;
            continue;
        }
        if (same_packing_seed_forest(parent, best_signature)) {
            ++duplicate_seed_forests;
            continue;
        }
        const bool retained = enable_parent_parent
            ? remember_diverse_packing_seed(parents, std::move(parent), max_seeds)
            : remember_packing_seed(parents, std::move(parent), max_seeds);
        if (!retained) {
            ++duplicate_seed_forests;
        }
    }

    if (parents.empty()) {
        return false;
    }

    bool any_accepted = false;
    bool pass_improved = true;
    for (std::size_t pass = 0; pass < max_passes && pass_improved; ++pass) {
        pass_improved = false;

        for (std::size_t order_pos = 0; order_pos < parents.size(); ++order_pos) {
            if (timer.should_stop(1.0)) {
                return any_accepted;
            }

            const std::size_t seed_index = (pass % 2 == 0)
                ? order_pos
                : parents.size() - 1 - order_pos;

            CoreForest current = best;
            current.normalize();
            if (same_packing_seed_forest(current, parents[seed_index])) {
                continue;
            }

            const double phase_start = timer.elapsed_seconds();
            const std::size_t before = best.component_count();
            bool accepted = false;
            bool threw = false;
            pace26::heuristics::ForestCrossover::Stats crossover_stats;

            try {
                pace26::heuristics::ForestCrossover::Options opts;
                opts.guard_seconds = 0.75;
                if (g_profile.enabled()) {
                    opts.stats = &crossover_stats;
                }

                pace26::heuristics::ForestCrossover crossover(opts);
                CoreForest candidate =
                    crossover.solve(t1, t2, std::move(current), parents[seed_index], &timer);
                if (archive != nullptr) {
                    archive->remember_if_valid(candidate, t1, t2);
                }
                accepted = consider_candidate(best, std::move(candidate), t1, t2);
                if (accepted) {
                    if (archive != nullptr) {
                        archive->remember_forest(best);
                    }
                    any_accepted = true;
                    pass_improved = true;
                    if (publish_progress) {
                        publish_progress(best);
                    }
                }
            } catch (const std::exception&) {
                threw = true;
            }

            if (g_profile.enabled()) {
                g_profile.phase_result(
                    profile_phase,
                    timer,
                    phase_start,
                    n,
                    before,
                    best.component_count(),
                    accepted,
                    {
                        {"pass", std::to_string(pass)},
                        {"order_pos", std::to_string(order_pos)},
                        {"seed_index", std::to_string(seed_index)},
                        {"seed_forests", std::to_string(parents.size())},
                        {"invalid_seed_forests", std::to_string(invalid_seed_forests)},
                        {"duplicate_seed_forests", std::to_string(duplicate_seed_forests)},
                        {"archive_seed_forests", std::to_string(archive_seed_forests)},
                        {"archive_forests_before", std::to_string(archive_forests_before)},
                        {"archive_forests_after", std::to_string(archive == nullptr ? 0 : archive->forest_count())},
                        {"archive_components_before", std::to_string(archive_components_before)},
                        {"archive_components_after", std::to_string(archive == nullptr ? 0 : archive->component_count())},
                        {"seed_components", std::to_string(parents[seed_index].component_count())},
                        {"exception", threw ? "true" : "false"},
                        {"crossover_parent_a_components", std::to_string(crossover_stats.parent_a_components)},
                        {"crossover_parent_b_components", std::to_string(crossover_stats.parent_b_components)},
                        {"crossover_shared_components", std::to_string(crossover_stats.shared_components)},
                        {"crossover_left_candidates", std::to_string(crossover_stats.left_candidates)},
                        {"crossover_right_candidates", std::to_string(crossover_stats.right_candidates)},
                        {"crossover_conflict_edges", std::to_string(crossover_stats.conflict_edges)},
                        {"crossover_selected_shared", std::to_string(crossover_stats.selected_shared)},
                        {"crossover_selected_left", std::to_string(crossover_stats.selected_left)},
                        {"crossover_selected_right", std::to_string(crossover_stats.selected_right)},
                        {"crossover_final_components", std::to_string(crossover_stats.final_components)},
                        {"crossover_final_non_singletons", std::to_string(crossover_stats.final_non_singletons)},
                        {"crossover_uncovered_leaves", std::to_string(crossover_stats.uncovered_leaves)},
                        {"crossover_final_gain", std::to_string(crossover_stats.final_gain)},
                        {"crossover_total_weight", std::to_string(crossover_stats.total_weight)},
                        {"crossover_selected_weight", std::to_string(crossover_stats.selected_weight)},
                        {"crossover_min_cut_weight", std::to_string(crossover_stats.min_cut_weight)},
                        {"crossover_stopped", crossover_stats.stopped ? "true" : "false"}
                    }
                );
            }
        }
    }

    if (enable_parent_parent && parents.size() >= 2 && !timer.should_stop(1.0)) {
        const std::size_t pair_cap = max_parent_pairs == 0
            ? std::min<std::size_t>(24, parents.size() * (parents.size() - 1) / 2)
            : max_parent_pairs;
        std::vector<std::pair<std::size_t, std::size_t>> parent_pairs;
        parent_pairs.reserve(std::min<std::size_t>(
            pair_cap,
            parents.size() * (parents.size() - 1) / 2
        ));

        for (std::size_t gap = 1;
             gap < parents.size() && parent_pairs.size() < pair_cap;
             ++gap) {
            for (std::size_t i = 0;
                 i + gap < parents.size() && parent_pairs.size() < pair_cap;
                 ++i) {
                parent_pairs.emplace_back(i, i + gap);
            }
        }

        std::size_t pair_attempts = 0;
        for (const auto& [left_index, right_index] : parent_pairs) {
            if (timer.should_stop(1.0)) {
                break;
            }

            const double phase_start = timer.elapsed_seconds();
            const std::size_t before = best.component_count();
            bool accepted = false;
            bool archived = false;
            bool threw = false;
            pace26::heuristics::ForestCrossover::Stats crossover_stats;
            ++pair_attempts;

            try {
                pace26::heuristics::ForestCrossover::Options opts;
                opts.guard_seconds = 0.75;
                if (g_profile.enabled()) {
                    opts.stats = &crossover_stats;
                }

                pace26::heuristics::ForestCrossover crossover(opts);
                CoreForest candidate =
                    crossover.solve(t1, t2, parents[left_index], parents[right_index], &timer);
                if (archive != nullptr) {
                    archived = archive->remember_if_valid(candidate, t1, t2);
                }
                accepted = consider_candidate(best, std::move(candidate), t1, t2);
                if (accepted) {
                    if (archive != nullptr) {
                        archive->remember_forest(best);
                    }
                    any_accepted = true;
                    if (publish_progress) {
                        publish_progress(best);
                    }
                }
            } catch (const std::exception&) {
                threw = true;
            }

            if (g_profile.enabled()) {
                g_profile.phase_result(
                    profile_phase + ".parent_parent",
                    timer,
                    phase_start,
                    n,
                    before,
                    best.component_count(),
                    accepted,
                    {
                        {"pair_attempt", std::to_string(pair_attempts)},
                        {"pair_cap", std::to_string(pair_cap)},
                        {"left_index", std::to_string(left_index)},
                        {"right_index", std::to_string(right_index)},
                        {"seed_forests", std::to_string(parents.size())},
                        {"left_components", std::to_string(parents[left_index].component_count())},
                        {"right_components", std::to_string(parents[right_index].component_count())},
                        {"archived_child", archived ? "true" : "false"},
                        {"archive_seed_forests", std::to_string(archive_seed_forests)},
                        {"archive_forests_after", std::to_string(archive == nullptr ? 0 : archive->forest_count())},
                        {"archive_components_after", std::to_string(archive == nullptr ? 0 : archive->component_count())},
                        {"exception", threw ? "true" : "false"},
                        {"crossover_parent_a_components", std::to_string(crossover_stats.parent_a_components)},
                        {"crossover_parent_b_components", std::to_string(crossover_stats.parent_b_components)},
                        {"crossover_shared_components", std::to_string(crossover_stats.shared_components)},
                        {"crossover_conflict_edges", std::to_string(crossover_stats.conflict_edges)},
                        {"crossover_selected_shared", std::to_string(crossover_stats.selected_shared)},
                        {"crossover_selected_left", std::to_string(crossover_stats.selected_left)},
                        {"crossover_selected_right", std::to_string(crossover_stats.selected_right)},
                        {"crossover_final_components", std::to_string(crossover_stats.final_components)},
                        {"crossover_uncovered_leaves", std::to_string(crossover_stats.uncovered_leaves)},
                        {"crossover_final_gain", std::to_string(crossover_stats.final_gain)},
                        {"crossover_stopped", crossover_stats.stopped ? "true" : "false"}
                    }
                );
            }
        }
    }

    return any_accepted;
}

double final_global_repacking_reserve_seconds(
    std::size_t n,
    const Timer& timer,
    bool packing_focused = false
) {
    double desired = 16.0;
    if (n >= 10000) {
        desired = 60.0;
    } else if (n >= 6000) {
        desired = 55.0;
    } else if (n >= 5000) {
        desired = 45.0;
    } else if (n >= 2500) {
        desired = 32.0;
    } else if (n >= 600) {
        desired = 32.0;
    }

    const double reserve_floor = packing_focused ? 16.0 : 12.0;
    const double remaining_fraction = n >= 6000
        ? (packing_focused ? 0.42 : 0.36)
        : (packing_focused ? 0.60 : 0.45);
    if (packing_focused) {
        desired *= n >= 6000 ? 1.0 : 1.35;
    }

    return std::min(
        desired,
        std::max(reserve_floor, timer.remaining_seconds() * remaining_fraction)
    );
}

void run_active_cherry_portfolio(
    CoreForest& best,
    const Tree& t1,
    const Tree& t2,
    std::size_t n,
    bool extended_search,
    Timer& timer,
    const std::function<bool(CoreForest)>& accept_candidate,
    const std::function<void(const CoreForest&)>& publish_progress = {},
    const std::string& profile_context = "active_cherry"
) {
    const std::vector<ActiveCherryPortfolioRun> runs =
        active_cherry_portfolio_for_size(n, extended_search);

    int run_index = 0;
    for (const ActiveCherryPortfolioRun& run : runs) {
        ++run_index;
        const double guard = n >= 1000 ? 0.75 : 1.0;
        if (timer.should_stop(guard)) {
            g_profile.event_raw(
                "active_cherry_portfolio_stop",
                &timer,
                {
                    {"context", json_quote(profile_context)},
                    {"reason", json_quote("timer_guard")},
                    {"n", std::to_string(n)},
                    {"run_index", std::to_string(run_index)}
                }
            );
            break;
        }
        if (run.local_time_limit_seconds > 0.0 &&
            timer.remaining_seconds() <= run.local_time_limit_seconds + 2.0) {
            g_profile.event_raw(
                "active_cherry_portfolio_stop",
                &timer,
                {
                    {"context", json_quote(profile_context)},
                    {"reason", json_quote("insufficient_time_for_run")},
                    {"n", std::to_string(n)},
                    {"run_index", std::to_string(run_index)},
                    {"run_budget_seconds", std::to_string(run.local_time_limit_seconds)}
                }
            );
            break;
        }

        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;

        try {
            pace26::heuristics::ActiveCherryGreedyApprox::Options opts;
            opts.policy = run.policy;
            opts.max_steps_multiplier = run.max_steps_multiplier;
            opts.candidate_sample_cap = run.candidate_sample_cap;
            opts.guard_seconds = 0.25;
            opts.local_time_limit_seconds = run.local_time_limit_seconds;
            opts.sample_salt = run.sample_salt;
            opts.rank_choice_script = run.rank_choice_script;

            if (n >= 6000) {
                opts.publish_interval_steps = 2048;
            } else if (n >= 500) {
                opts.publish_interval_steps = 512;
            }

            if (publish_progress) {
                opts.publish_candidate = publish_progress;
            }

            const Tree& first = run.swapped ? t2 : t1;
            const Tree& second = run.swapped ? t1 : t2;

            pace26::heuristics::ActiveCherryGreedyApprox greedy(opts);
            CoreForest candidate = greedy.solve(first, second, &timer);
            accepted = accept_candidate(std::move(candidate));
        } catch (const std::exception&) {
            threw = true;
            // Keep the best candidate found by previous portfolio members.
        }

        g_profile.phase_result(
            profile_context + ".run",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {
                {"run_index", std::to_string(run_index)},
                {"policy", json_quote(active_cherry_policy_name(run.policy))},
                {"swapped", run.swapped ? "true" : "false"},
                {"max_steps_multiplier", std::to_string(run.max_steps_multiplier)},
                {"candidate_sample_cap", std::to_string(run.candidate_sample_cap)},
                {"local_time_limit_seconds", std::to_string(run.local_time_limit_seconds)},
                {"sample_salt", std::to_string(run.sample_salt)},
                {"script_length", std::to_string(run.rank_choice_script.size())},
                {"exception", threw ? "true" : "false"}
            }
        );
    }


    best.normalize();
}

std::vector<CoreForest> run_large_balanced_seed_bank(
    CoreForest& best,
    const Tree& t1,
    const Tree& t2,
    Timer& timer,
    const std::function<void(const CoreForest&)>& publish_progress,
    std::size_t max_seeds,
    ComponentForestArchive* archive = nullptr,
    const std::string& profile_context = "main.large_balanced_seed_bank"
) {
    std::vector<CoreForest> seeds;
    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());
    if (n < 5000 || max_seeds == 0 || timer.should_stop(18.0)) {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote(profile_context)},
                {"n", std::to_string(n)},
                {"reason", json_quote(n < 5000 ? "size_limit" : "timer_guard")}
            }
        );
        return seeds;
    }

    const std::uint64_t salt_tags[] = {
        0x5164b7c19e3779b9ULL,
        0x9a1f23d0b4c6e8f1ULL,
        0x2d7c4a8193bf56e0ULL,
        0xe35b9178ac04d26fULL,
        0x6f02c8d4a17b39e5ULL,
        0xbd946f1357a20c8eULL,
        0x41e2a7c90d6b538fULL,
        0xc87a391e54f206bdULL,
        0x75d31b86e4a902cfULL,
        0x1f4c8b63d0e7a295ULL
    };
    const std::size_t requested_runs =
        n >= 10000 ? 8 : (n >= 7000 ? 7 : 6);
    const std::size_t run_count = std::min<std::size_t>(
        requested_runs,
        sizeof(salt_tags) / sizeof(salt_tags[0])
    );

    std::size_t valid = 0;
    std::size_t retained = 0;
    std::size_t accepted_total = 0;
    for (std::size_t run_index = 0; run_index < run_count; ++run_index) {
        if (timer.should_stop(12.0)) {
            break;
        }

        const double remaining = timer.remaining_seconds();
        const double run_budget = std::min(
            n >= 10000 ? 8.0 : 5.5,
            std::max(1.0, remaining - 10.0)
        );
        if (run_budget <= 1.0) {
            break;
        }

        const bool swapped = (run_index % 3) == 1;
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool candidate_valid = false;
        bool seed_retained = false;
        bool accepted = false;
        bool threw = false;

        try {
            pace26::heuristics::ActiveCherryGreedyApprox::Options opts;
            opts.policy = ActiveCherryPolicy::Balanced;
            opts.max_steps_multiplier = 3;
            opts.candidate_sample_cap = n >= 10000 ? 128 : 96;
            opts.guard_seconds = 0.30;
            opts.local_time_limit_seconds = run_budget;
            opts.sample_salt = mix_portfolio_salt(
                salt_tags[run_index] ^
                (swapped ? 0xa0761d6478bd642fULL : 0x0000000000000000ULL)
            );
            opts.publish_interval_steps = n >= 6000 ? 2048 : 1024;

            const Tree& first = swapped ? t2 : t1;
            const Tree& second = swapped ? t1 : t2;
            pace26::heuristics::ActiveCherryGreedyApprox greedy(opts);
            CoreForest candidate = greedy.solve(first, second, &timer);
            candidate.normalize();
            candidate_valid = is_valid_agreement_forest(candidate, t1, t2);
            if (candidate_valid) {
                ++valid;
                if (archive != nullptr) {
                    archive->remember_forest(candidate);
                }
                seed_retained = remember_diverse_packing_seed(
                    seeds,
                    candidate,
                    max_seeds
                );
                if (seed_retained) {
                    ++retained;
                }
                accepted = consider_candidate(best, std::move(candidate), t1, t2);
                if (accepted) {
                    ++accepted_total;
                    if (archive != nullptr) {
                        archive->remember_forest(best);
                    }
                    if (publish_progress) {
                        publish_progress(best);
                    }
                }
            }
        } catch (const std::exception&) {
            threw = true;
        }

        g_profile.phase_result(
            profile_context,
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {
                {"run_index", std::to_string(run_index)},
                {"swapped", swapped ? "true" : "false"},
                {"run_budget_seconds", std::to_string(run_budget)},
                {"candidate_valid", candidate_valid ? "true" : "false"},
                {"seed_retained", seed_retained ? "true" : "false"},
                {"valid_seeds", std::to_string(valid)},
                {"retained_seeds", std::to_string(retained)},
                {"accepted_total", std::to_string(accepted_total)},
                {"exception", threw ? "true" : "false"}
            }
        );
    }

    remember_diverse_packing_seed(seeds, best, max_seeds);
    return seeds;
}

std::vector<int> single_discrepancy_script(std::size_t depth, int choice_rank) {
    std::vector<int> script(depth + 1, 0);
    script[depth] = choice_rank;
    return script;
}

std::vector<int> double_discrepancy_script(
    std::size_t d1,
    int r1,
    std::size_t d2,
    int r2
) {
    std::vector<int> script(std::max(d1, d2) + 1, 0);
    script[d1] = r1;
    script[d2] = r2;
    return script;
}

void apply_reductions_exhaustively(
    NewickTree& t1,
    NewickTree& t2,
    std::uint32_t& next_placeholder,
    pace26::reductions::ReductionStack& stack,
    const SolverConfig& config,
    const Timer& timer,
    double guard_seconds = 6.0
) {
    int passes = 0;

    while (!timer.should_stop(guard_seconds) && passes < config.max_reduction_passes) {
        ++passes;

        const std::size_t leaves = count_leaves(t1);

        const double common_start = timer.elapsed_seconds();
        auto sub = pace26::reductions::CommonSubtreeReduction::reduce(
            t1,
            t2,
            next_placeholder,
            2
        );

        if (sub.changed) {
            const std::size_t after_leaves = count_leaves(sub.tree1);
            g_profile.event_raw(
                "reduction",
                &timer,
                {
                    {"rule", json_quote("common_subtree")},
                    {"pass", std::to_string(passes)},
                    {"changed", "true"},
                    {"leaves_before", std::to_string(leaves)},
                    {"leaves_after", std::to_string(after_leaves)},
                    {"duration_seconds", std::to_string(timer.elapsed_seconds() - common_start)},
                    {"records", std::to_string(sub.records.size())}
                }
            );
            stack.push_common_subtrees(sub.records);
            t1 = std::move(sub.tree1);
            t2 = std::move(sub.tree2);
            continue;
        } else {
            g_profile.event_raw(
                "reduction",
                &timer,
                {
                    {"rule", json_quote("common_subtree")},
                    {"pass", std::to_string(passes)},
                    {"changed", "false"},
                    {"leaves_before", std::to_string(leaves)},
                    {"leaves_after", std::to_string(leaves)},
                    {"duration_seconds", std::to_string(timer.elapsed_seconds() - common_start)},
                    {"records", "0"}
                }
            );
        }

        // This implementation repeatedly deletes one taxon and rebuilds the tree,
        // so keep it away from full huge instances.
        if (leaves <= config.three_two_max_leaves) {
            const double tt_start = timer.elapsed_seconds();
            auto tt = pace26::reductions::ThreeTwoChainReduction::reduce(t1, t2);

            if (tt.changed) {
                const std::size_t after_leaves = count_leaves(tt.tree1);
                g_profile.event_raw(
                    "reduction",
                    &timer,
                    {
                        {"rule", json_quote("three_two_chain")},
                        {"pass", std::to_string(passes)},
                        {"changed", "true"},
                        {"leaves_before", std::to_string(leaves)},
                        {"leaves_after", std::to_string(after_leaves)},
                        {"duration_seconds", std::to_string(timer.elapsed_seconds() - tt_start)},
                        {"records", std::to_string(tt.records.size())}
                    }
                );
                stack.push_three_two_chains(tt.records);
                t1 = std::move(tt.tree1);
                t2 = std::move(tt.tree2);
                continue;
            } else {
                g_profile.event_raw(
                    "reduction",
                    &timer,
                    {
                        {"rule", json_quote("three_two_chain")},
                        {"pass", std::to_string(passes)},
                        {"changed", "false"},
                        {"leaves_before", std::to_string(leaves)},
                        {"leaves_after", std::to_string(leaves)},
                        {"duration_seconds", std::to_string(timer.elapsed_seconds() - tt_start)},
                        {"records", "0"}
                    }
                );
            }
        } else {
            g_profile.event_raw(
                "reduction_skipped",
                &timer,
                {
                    {"rule", json_quote("three_two_chain")},
                    {"pass", std::to_string(passes)},
                    {"leaves", std::to_string(leaves)},
                    {"reason", json_quote("size_limit")}
                }
            );
        }

        if (leaves <= config.chain_max_leaves) {
            const double chain_start = timer.elapsed_seconds();
            auto chain = pace26::reductions::ChainReduction::reduce(t1, t2);

            if (chain.changed) {
                const std::size_t after_leaves = count_leaves(chain.tree1);
                g_profile.event_raw(
                    "reduction",
                    &timer,
                    {
                        {"rule", json_quote("chain")},
                        {"pass", std::to_string(passes)},
                        {"changed", "true"},
                        {"leaves_before", std::to_string(leaves)},
                        {"leaves_after", std::to_string(after_leaves)},
                        {"duration_seconds", std::to_string(timer.elapsed_seconds() - chain_start)},
                        {"records", std::to_string(chain.records.size())}
                    }
                );
                stack.push_chains(chain.records);
                t1 = std::move(chain.tree1);
                t2 = std::move(chain.tree2);
                continue;
            } else {
                g_profile.event_raw(
                    "reduction",
                    &timer,
                    {
                        {"rule", json_quote("chain")},
                        {"pass", std::to_string(passes)},
                        {"changed", "false"},
                        {"leaves_before", std::to_string(leaves)},
                        {"leaves_after", std::to_string(leaves)},
                        {"duration_seconds", std::to_string(timer.elapsed_seconds() - chain_start)},
                        {"records", "0"}
                    }
                );
            }
        } else {
            g_profile.event_raw(
                "reduction_skipped",
                &timer,
                {
                    {"rule", json_quote("chain")},
                    {"pass", std::to_string(passes)},
                    {"leaves", std::to_string(leaves)},
                    {"reason", json_quote("size_limit")}
                }
            );
        }

        break;
    }
}

enum class ReducedGlobalCandidateKind {
    ResolveSet,
    DualitySeed
};

const char* reduced_global_candidate_name(ReducedGlobalCandidateKind kind) {
    switch (kind) {
        case ReducedGlobalCandidateKind::ResolveSet: return "resolve_set";
        case ReducedGlobalCandidateKind::DualitySeed: return "duality_seed";
    }

    return "unknown";
}

std::size_t reduced_global_candidate_sample_cap(
    ReducedGlobalCandidateKind kind,
    std::size_t n
) {
    if (kind == ReducedGlobalCandidateKind::DualitySeed) {
        if (n > 4000) return 192;
        if (n > 1000) return 256;
        return 0;
    }

    if (n > 4000) return 192;
    if (n > 1000) return 256;
    if (n > 600) return 384;
    return 0;
}

std::optional<CoreForest> run_reduced_global_candidate(
    const NewickTree& original_nt1,
    const NewickTree& original_nt2,
    const SolverConfig& config,
    Timer& global_timer,
    ReducedGlobalCandidateKind kind,
    bool swapped,
    double local_budget_seconds
) {
    if (local_budget_seconds <= 0.75 || global_timer.should_stop(1.0)) {
        return std::nullopt;
    }

    const double local_deadline = std::min(
        global_timer.limit_seconds(),
        global_timer.elapsed_seconds() + local_budget_seconds
    );
    Timer candidate_timer(local_deadline, global_timer.start_time());

    // Each candidate owns its reduced instance and expansion history. This
    // prevents one candidate's reductions or contractions from biasing another.
    NewickTree reduced_nt1 = original_nt1;
    NewickTree reduced_nt2 = original_nt2;
    pace26::reductions::ReductionStack reduction_stack;
    std::uint32_t next_placeholder =
        choose_initial_placeholder(reduced_nt1, reduced_nt2);

    apply_reductions_exhaustively(
        reduced_nt1,
        reduced_nt2,
        next_placeholder,
        reduction_stack,
        config,
        candidate_timer,
        0.50
    );

    if (candidate_timer.should_stop(0.35)) {
        return std::nullopt;
    }

    Tree reduced_t1 = Tree::from_newick(reduced_nt1);
    Tree reduced_t2 = Tree::from_newick(reduced_nt2);
    const std::size_t reduced_n =
        static_cast<std::size_t>(reduced_t1.leaf_count());

    pace26::heuristics::ActiveCherryGreedyApprox::Options opts;
    opts.policy = kind == ReducedGlobalCandidateKind::ResolveSet
        ? ActiveCherryPolicy::ResolveFinalCut
        : ActiveCherryPolicy::DualityConservative;
    opts.max_steps_multiplier = 4;
    opts.candidate_sample_cap =
        reduced_global_candidate_sample_cap(kind, reduced_n);
    opts.guard_seconds = 0.20;
    opts.local_time_limit_seconds =
        std::max(0.0, candidate_timer.remaining_seconds() - 0.25);
    opts.sample_salt = kind == ReducedGlobalCandidateKind::ResolveSet
        ? mix_portfolio_salt(0xa8b31f4c72d95e11ULL)
        : mix_portfolio_salt(0x2a6f9d7c1b3e5a91ULL);
    if (swapped) {
        opts.sample_salt = mix_portfolio_salt(
            opts.sample_salt ^ 0x6d0f27bb35a91e43ULL
        );
    }

    pace26::heuristics::ActiveCherryGreedyApprox candidate_solver(opts);
    CoreForest reduced_candidate =
        swapped
            ? candidate_solver.solve(reduced_t2, reduced_t1, &candidate_timer)
            : candidate_solver.solve(reduced_t1, reduced_t2, &candidate_timer);

    if (!candidate_timer.should_stop(0.35)) {
        pace26::heuristics::LocalImprove::Options merge_opts;
        merge_opts.max_rounds = 256;
        merge_opts.try_singleton_reinsertion = false;
        merge_opts.try_pair_merges = false;
        merge_opts.try_structural_pair_merges = true;
        merge_opts.max_structural_pair_tests_per_round = 160;
        merge_opts.structural_neighbor_window = 4;
        merge_opts.max_merged_component_size = reduced_n > 4000 ? 256 : 512;
        merge_opts.guard_seconds = 0.25;
        merge_opts.deterministic = true;

        pace26::heuristics::LocalImprove merger(merge_opts);
        reduced_candidate = merger.improve(
            reduced_t1,
            reduced_t2,
            std::move(reduced_candidate),
            &candidate_timer,
            nullptr
        );
    }

    RedForest expanded_reduction_forest =
        reduction_stack.expand(to_reduction_forest(reduced_candidate));
    CoreForest expanded = to_core_forest(expanded_reduction_forest);
    expanded.normalize();
    expanded.validate_no_duplicates();
    return expanded;
}

void run_reduced_global_candidate_portfolio(
    CoreForest& best,
    const NewickTree& nt1,
    const NewickTree& nt2,
    const SolverConfig& config,
    Timer& timer,
    bool extended_search,
    const std::function<bool(CoreForest)>& accept_candidate,
    const std::string& profile_context,
    std::vector<CoreForest>* packing_seeds = nullptr,
    bool packing_focused = false,
    ComponentForestArchive* archive = nullptr,
    const std::function<void(const CoreForest&)>& seed_ready = {}
) {
    const std::size_t n = count_leaves(nt1);
    if (n < 5000 || (extended_search && n >= 5000) || timer.should_stop(2.0)) {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote(profile_context)},
                {"n", std::to_string(n)},
                {"reason", json_quote(
                    n < 5000
                        ? "size_limit_pruned_low_roi"
                        : ((extended_search && n >= 5000)
                            ? "disabled_extended_low_roi"
                            : "timer_guard")
                )}
            }
        );
        return;
    }

    struct CandidateSpec {
        ReducedGlobalCandidateKind kind;
        bool swapped;
    };

    const CandidateSpec candidates[] = {
        {ReducedGlobalCandidateKind::ResolveSet, false},
        {ReducedGlobalCandidateKind::DualitySeed, false},
        {ReducedGlobalCandidateKind::ResolveSet, true},
        {ReducedGlobalCandidateKind::DualitySeed, true}
    };

    constexpr double kBaselineReserveSeconds = 1.50;
    constexpr double kExtendedReserveSeconds = 10.0;
    const double reserve_seconds =
        extended_search
            ? (packing_focused ? 16.0 : kExtendedReserveSeconds)
            : kBaselineReserveSeconds;
    const double desired_budget = extended_search
        ? (packing_focused
              ? (n > 10000 ? 14.0 : (n > 5000 ? 11.0 : 7.0))
              : (n > 4000 ? 16.0 : 10.0))
        : (n > 4000 ? 6.0 : 4.0);

    constexpr std::size_t kCandidateCount =
        sizeof(candidates) / sizeof(candidates[0]);

    for (std::size_t i = 0; i < kCandidateCount; ++i) {
        const CandidateSpec& spec = candidates[i];
        const double available = timer.remaining_seconds() - reserve_seconds;
        const std::size_t budget_peers = i < 2 ? 2 - i : kCandidateCount - i;
        const double fair_share = available / static_cast<double>(budget_peers);
        const double local_budget = std::min(desired_budget, fair_share);

        if (local_budget < 1.25 || timer.should_stop(reserve_seconds + 0.75)) {
            g_profile.event_raw(
                "reduced_global_candidate_skip",
                &timer,
                {
                    {"context", json_quote(profile_context)},
                    {"candidate", json_quote(reduced_global_candidate_name(spec.kind))},
                    {"swapped", spec.swapped ? "true" : "false"},
                    {"reason", json_quote("insufficient_time")},
                    {"local_budget_seconds", std::to_string(local_budget)},
                    {"packing_focused", packing_focused ? "true" : "false"}
                }
            );
            continue;
        }

        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool produced = false;
        bool threw = false;
        std::size_t candidate_components = 0;

        try {
            std::optional<CoreForest> candidate = run_reduced_global_candidate(
                nt1,
                nt2,
                config,
                timer,
                spec.kind,
                spec.swapped,
                local_budget
            );

            if (candidate.has_value()) {
                produced = true;
                candidate_components = candidate->component_count();
                if (packing_seeds != nullptr) {
                    remember_packing_seed(*packing_seeds, *candidate, 12);
                }
                if (archive != nullptr) {
                    archive->remember_forest(*candidate);
                }
                if (seed_ready) {
                    seed_ready(*candidate);
                }
                accepted = accept_candidate(std::move(*candidate));
            }
        } catch (const std::exception&) {
            threw = true;
        }

        g_profile.phase_result(
            profile_context + ".candidate",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {
                {"candidate", json_quote(reduced_global_candidate_name(spec.kind))},
                {"swapped", spec.swapped ? "true" : "false"},
                {"produced", produced ? "true" : "false"},
                {"candidate_components", std::to_string(candidate_components)},
                {"exception", threw ? "true" : "false"},
                {"local_budget_seconds", std::to_string(local_budget)},
                {"packing_focused", packing_focused ? "true" : "false"}
            }
        );
    }
}

void publish_progress_safely(
    const ForestPublishCallback& publish_progress,
    const CoreForest& forest
);

CoreForest solve_without_cluster_recursion(
    const NewickTree& nt1,
    const NewickTree& nt2,
    const SolverConfig& config,
    Timer& timer,
    Random& rng,
    const ForestPublishCallback& publish_progress = {},
    ComponentForestArchive* archive = nullptr
) {
    Tree t1 = Tree::from_newick(nt1);
    Tree t2 = Tree::from_newick(nt2);

    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());

    CoreForest best = CoreForest::singleton_forest_from_tree(t1);
    auto publish_best = [&]() {
        publish_progress_safely(publish_progress, best);
    };

    if (config.enable_legacy_exact &&
        n <= config.exact_max_leaves &&
        !timer.should_stop(1.0)) {
        try {
            pace26::exact::InProcessExactMaf::Options opts;
            opts.enabled = true;
            opts.max_leaves = config.exact_max_leaves;
            opts.local_time_limit_seconds = std::min(60.0, std::max(5.0, timer.remaining_seconds() * 0.50));
            opts.global_guard_seconds = 1.0;

            pace26::exact::InProcessExactMaf exact(opts);
            std::optional<CoreForest> exact_forest = exact.solve(t1, t2, &timer, &best);
            if (exact_forest.has_value() &&
                is_valid_agreement_forest(*exact_forest, t1, t2)) {
                if (archive != nullptr) {
                    archive->remember_forest(*exact_forest);
                }
                best = std::move(*exact_forest);
                best.normalize();
                best.validate_partition_of(t1.leaf_labels);
                publish_best();
                return best;
            }
        } catch (const std::exception&) {
            // Fall back to heuristics.
        }
    }

    std::vector<CoreForest> packing_seeds;
    std::optional<CoreForest> independent_small_packing_candidate;
    {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        try {
            pace26::heuristics::FastCherryApprox fast;
            CoreForest candidate = fast.solve(t1, t2, &timer, &rng);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            remember_packing_seed(packing_seeds, candidate);
            accepted = consider_candidate(best, std::move(candidate), t1, t2);
            if (accepted) {
                publish_best();
            }
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "solve.fast_cherry",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {{"exception", threw ? "true" : "false"}}
        );
    }

    {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        try {
            pace26::heuristics::CherryPairApprox pairs;
            CoreForest candidate = pairs.solve(t1, t2);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            remember_packing_seed(packing_seeds, candidate);
            accepted = consider_candidate(best, std::move(candidate), t1, t2);
            if (accepted) {
                publish_best();
            }
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "solve.cherry_pair",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {{"exception", threw ? "true" : "false"}}
        );
    }

    {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();

        run_active_cherry_portfolio(
            best,
            t1,
            t2,
            n,
            config.extended_search,
            timer,
            [&](CoreForest candidate) {
                if (archive != nullptr) {
                    archive->remember_if_valid(candidate, t1, t2);
                }
                remember_packing_seed(packing_seeds, candidate);
                const bool accepted = consider_candidate(best, std::move(candidate), t1, t2);
                if (accepted) {
                    publish_best();
                }
                return accepted;
            },
            {},
            "solve.active_cherry"
        );

        g_profile.phase_result(
            "solve.active_cherry.total",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            best.component_count() < before
        );
    }

    if (n >= 600 || n <= 120) {
        const bool keep_initial_packing_seeds =
            config.extended_search &&
            n >= 600 &&
            n <= config.local_improve_max_leaves;
        std::vector<CoreForest> initial_packing_seeds =
            keep_initial_packing_seeds
                ? packing_seeds
                : std::move(packing_seeds);
        run_agreement_component_packing(
            best,
            t1,
            t2,
            std::move(initial_packing_seeds),
            timer,
            config.extended_search,
            "solve.agreement_component_packing",
            publish_progress,
            AgreementPackingMode::Normal,
            config.packing_focused,
            config.extended_search && config.enable_singleton_rescue && n <= 220,
            {},
            archive
        );
        if (!keep_initial_packing_seeds) {
            packing_seeds.clear();
        }
    } else if (n <= 220) {
        CoreForest packing_best = best;
        run_agreement_component_packing(
            packing_best,
            t1,
            t2,
            packing_seeds,
            timer,
            config.extended_search,
            "solve.agreement_component_packing.independent",
            publish_progress,
            AgreementPackingMode::Normal,
            config.packing_focused,
            config.extended_search && config.enable_singleton_rescue,
            {},
            archive
        );
        if (packing_best.component_count() < best.component_count()) {
            independent_small_packing_candidate = std::move(packing_best);
        }
    }
    // The baseline avoids expensive local search on huge instances. The
    // continuation keeps going because it has a separately protected incumbent.
    if ((!config.extended_search && n > 6000) || timer.should_stop(8.0)) {
        g_profile.event_raw(
            "solve_without_cluster_return",
            &timer,
            {
                {"reason", json_quote((!config.extended_search && n > 6000) ? "large_instance_cutoff" : "timer_guard")},
                {"n", std::to_string(n)},
                {"components", std::to_string(best.component_count())}
            }
        );
        best.normalize();
        best.validate_partition_of(t1.leaf_labels);
        if (archive != nullptr) {
            archive->remember_forest(best);
        }
        return best;
    }

    if (!timer.should_stop(8.0) && n <= config.two_approx_max_leaves) {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        try {
            pace26::heuristics::TwoApprox::Options opts;
            opts.max_full_leaves = config.two_approx_max_leaves;
            opts.local_time_limit_seconds =
                std::min(10.0, std::max(1.0, timer.remaining_seconds() * 0.10));
            opts.run_fast_cherry_fallback = true;
            opts.run_safe_greedy_merge = true;

            pace26::heuristics::TwoApprox two(opts);
            CoreForest candidate = two.solve(t1, t2, &timer, &rng);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            accepted = consider_candidate(best, std::move(candidate), t1, t2);
            if (accepted) {
                publish_best();
            }
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "solve.two_approx",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {{"exception", threw ? "true" : "false"}}
        );
    } else {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote("solve.two_approx")},
                {"n", std::to_string(n)},
                {"reason", json_quote(n > config.two_approx_max_leaves ? "size_limit" : "timer_guard")}
            }
        );
    }

    if (!timer.should_stop(6.0) && n <= config.local_improve_max_leaves) {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        try {
            pace26::heuristics::LocalImprove::Options opts;

            if (n <= 1000) {
                opts.max_rounds = 6;
                opts.max_candidates_per_singleton = 384;
                opts.max_pair_tests_per_round = 40000;
            } else {
                opts.max_rounds = 4;
                opts.max_candidates_per_singleton = 128;
                opts.max_pair_tests_per_round = 15000;
                opts.max_merged_component_size = 0;
            }

            opts.guard_seconds = 4.0;
            opts.deterministic = false;

            pace26::heuristics::LocalImprove improver(opts);
            CoreForest candidate = improver.improve(t1, t2, best, &timer, &rng);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            accepted = consider_candidate(best, std::move(candidate), t1, t2);
            if (accepted) {
                publish_best();
            }
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "solve.local_improve.pre_rr",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {{"exception", threw ? "true" : "false"}}
        );
    } else {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote("solve.local_improve.pre_rr")},
                {"n", std::to_string(n)},
                {"reason", json_quote(n > config.local_improve_max_leaves ? "size_limit" : "timer_guard")}
            }
        );
    }

    if (!timer.should_stop(8.0) && n <= config.random_restart_max_leaves) {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        pace26::heuristics::RandomRestart::Stats rr_stats;
        try {
            pace26::heuristics::RandomRestart::Options opts;
            const bool extended_rr = config.extended_search;
            const bool quality_recovery_rr =
                extended_rr && config.extended_quality_recovery;

            opts.max_restarts = quality_recovery_rr
                ? (n <= 1000 ? 100 : 20)
                : (extended_rr ? (n <= 350 ? 28 : 10) : (n <= 1000 ? 100 : 20));
            opts.min_restarts = extended_rr && !quality_recovery_rr ? 0 : 1;
            opts.guard_seconds = extended_rr && !quality_recovery_rr ? 2.5 : 4.0;

            opts.two_approx_max_leaves = config.two_approx_max_leaves;
            opts.singleton_start_max_leaves = extended_rr && !quality_recovery_rr ? 700 : 1000;

            opts.two_options.max_full_leaves = config.two_approx_max_leaves;
            opts.two_options.local_time_limit_seconds =
                std::min(5.0, std::max(1.0, timer.remaining_seconds() * 0.05));

            if (extended_rr && !quality_recovery_rr) {
                opts.improve_options.max_rounds = n <= 350 ? 3 : 2;
                opts.improve_options.max_candidates_per_singleton = n <= 350 ? 96 : 48;
                opts.improve_options.max_pair_tests_per_round = n <= 350 ? 6000 : 2000;
                opts.improve_options.guard_seconds = 2.5;
            } else {
                opts.improve_options.max_rounds = (n <= 1000 ? 6 : 3);
                opts.improve_options.max_candidates_per_singleton = (n <= 1000 ? 256 : 64);
                opts.improve_options.max_pair_tests_per_round = (n <= 1000 ? 20000 : 3000);
                opts.improve_options.guard_seconds = 4.0;
            }
            if (g_profile.enabled()) {
                opts.stats = &rr_stats;
            }

            pace26::heuristics::RandomRestart rr(opts);
            CoreForest candidate = rr.solve(t1, t2, &timer, &rng, &best);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            accepted = consider_candidate(best, std::move(candidate), t1, t2);
            if (accepted) {
                publish_best();
            }
        } catch (const std::exception&) {
            threw = true;
        }
        if (g_profile.enabled()) {
            g_profile.phase_result(
                "solve.random_restart",
                timer,
                phase_start,
                n,
                before,
                best.component_count(),
                accepted,
                {
                    {"exception", threw ? "true" : "false"},
                    {"rr_initial_components", std::to_string(rr_stats.initial_components)},
                    {"rr_final_components", std::to_string(rr_stats.final_components)},
                    {"rr_best_updates", std::to_string(rr_stats.best_updates)},
                    {"rr_fast_attempted", rr_stats.fast_attempted ? "true" : "false"},
                    {"rr_fast_improved", rr_stats.fast_improved ? "true" : "false"},
                    {"rr_two_attempted", rr_stats.two_attempted ? "true" : "false"},
                    {"rr_two_improved", rr_stats.two_improved ? "true" : "false"},
                    {"rr_singleton_attempted", rr_stats.singleton_attempted ? "true" : "false"},
                    {"rr_singleton_improved", rr_stats.singleton_improved ? "true" : "false"},
                    {"rr_restarts_attempted", std::to_string(rr_stats.restarts_attempted)},
                    {"rr_restarts_completed", std::to_string(rr_stats.restarts_completed)},
                    {"rr_restart_improvements", std::to_string(rr_stats.restart_improvements)},
                    {"rr_restart_exceptions", std::to_string(rr_stats.restart_exceptions)},
                    {"rr_mode0", std::to_string(rr_stats.restart_mode_counts[0])},
                    {"rr_mode1", std::to_string(rr_stats.restart_mode_counts[1])},
                    {"rr_mode2", std::to_string(rr_stats.restart_mode_counts[2])},
                    {"rr_mode3", std::to_string(rr_stats.restart_mode_counts[3])},
                    {"rr_mode4", std::to_string(rr_stats.restart_mode_counts[4])},
                    {"rr_mode5", std::to_string(rr_stats.restart_mode_counts[5])},
                    {"rr_stopped", rr_stats.stopped ? "true" : "false"}
                }
            );
        }
    } else {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote("solve.random_restart")},
                {"n", std::to_string(n)},
                {"reason", json_quote(n > config.random_restart_max_leaves ? "size_limit" : "timer_guard")}
            }
        );
    }

    if (independent_small_packing_candidate.has_value()) {
        consider_candidate(
            best,
            std::move(*independent_small_packing_candidate),
            t1,
            t2
        );
        publish_best();
    } else if (n > 220 && n < 600) {
        run_agreement_component_packing(
            best,
            t1,
            t2,
            std::move(packing_seeds),
            timer,
            config.extended_search,
            "solve.agreement_component_packing.post_restart",
            publish_progress,
            AgreementPackingMode::Normal,
            config.packing_focused,
            config.extended_search && config.enable_singleton_rescue && n <= 220,
            {},
            archive
        );
    }

    const bool enable_post_rr_local_improve = false;

    if (enable_post_rr_local_improve &&
        !timer.should_stop(6.0) &&
        n <= config.local_improve_max_leaves) {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        try {
            pace26::heuristics::LocalImprove::Options opts;

            if (n <= 1000) {
                opts.max_rounds = 6;
                opts.max_candidates_per_singleton = 384;
                opts.max_pair_tests_per_round = 40000;
            } else {
                opts.max_rounds = 4;
                opts.max_candidates_per_singleton = 128;
                opts.max_pair_tests_per_round = 15000;
            }

            opts.guard_seconds = 4.0;

            pace26::heuristics::LocalImprove improver(opts);
            CoreForest candidate = improver.improve(t1, t2, best, &timer, &rng);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            accepted = consider_candidate(best, std::move(candidate), t1, t2);
            if (accepted) {
                publish_best();
            }
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "solve.local_improve",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {{"exception", threw ? "true" : "false"}}
        );
    } else {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote("solve.local_improve")},
                {"n", std::to_string(n)},
                {"reason", json_quote(!enable_post_rr_local_improve ? "disabled_low_roi" : (n > config.local_improve_max_leaves ? "size_limit" : "timer_guard"))}
            }
        );
    }

    if (config.enable_legacy_exact &&
        !timer.should_stop(3.0) &&
        n <= config.exact_max_leaves) {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool found = false;
        bool threw = false;
        try {
            pace26::exact::InProcessExactMaf::Options opts;
            opts.enabled = true;
            opts.max_leaves = config.exact_max_leaves;
            opts.local_time_limit_seconds =
                std::min(5.0, std::max(1.0, timer.remaining_seconds() * 0.20));
            opts.global_guard_seconds = 1.0;

            pace26::exact::InProcessExactMaf exact(opts);
            std::optional<CoreForest> exact_forest = exact.solve(t1, t2, &timer, &best);

            if (exact_forest.has_value()) {
                found = true;
                if (archive != nullptr) {
                    archive->remember_if_valid(*exact_forest, t1, t2);
                }
                accepted = consider_candidate(best, std::move(*exact_forest), t1, t2);
                if (accepted) {
                    publish_best();
                }
            }
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "solve.exact",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            accepted,
            {
                {"found_exact_candidate", found ? "true" : "false"},
                {"exception", threw ? "true" : "false"}
            }
        );
    } else {
        g_profile.event_raw(
            "phase_skipped",
            &timer,
            {
                {"phase", json_quote("solve.exact")},
                {"n", std::to_string(n)},
                {"reason", json_quote(!config.enable_legacy_exact ? "disabled_legacy_exact" : (n > config.exact_max_leaves ? "size_limit" : "timer_guard"))}
            }
        );
    }

    best.normalize();
    best.validate_partition_of(t1.leaf_labels);
    if (archive != nullptr) {
        archive->remember_forest(best);
    }
    publish_best();
    return best;
}

CoreForest solve_instance(
    NewickTree nt1,
    NewickTree nt2,
    const SolverConfig& config,
    Timer& timer,
    Random& rng,
    int depth,
    const ForestPublishCallback& publish_progress = {},
    ComponentForestArchive* archive = nullptr,
    ClusterProfileMemo* cluster_profile_memo = nullptr
) {
    const bool archive_top_level_solution =
        archive != nullptr && depth == 0;
    NewickTree archive_nt1;
    NewickTree archive_nt2;
    if (archive_top_level_solution) {
        archive_nt1 = nt1;
        archive_nt2 = nt2;
    }

    pace26::reductions::ReductionStack stack;

    std::uint32_t next_placeholder = choose_initial_placeholder(nt1, nt2);

    const std::size_t input_leaves_before_reduction = count_leaves(nt1);
    const double reductions_start = timer.elapsed_seconds();

    apply_reductions_exhaustively(
        nt1,
        nt2,
        next_placeholder,
        stack,
        config,
        timer
    );

    ForestPublishCallback publish_reduced_progress;
    if (publish_progress) {
        publish_reduced_progress = [&](const CoreForest& reduced_candidate) {
            try {
                RedForest expanded_reduction_forest =
                    stack.expand(to_reduction_forest(reduced_candidate));
                CoreForest expanded = to_core_forest(expanded_reduction_forest);
                expanded.normalize();
                expanded.validate_no_duplicates();
                publish_progress_safely(publish_progress, expanded);
            } catch (const std::exception&) {
                // Keep the previous valid emergency answer.
            }
        };
    }

    CoreForest solved_reduced;

    const std::size_t leaves = count_leaves(nt1);
    g_profile.event_raw(
        "reductions_complete",
        &timer,
        {
            {"depth", std::to_string(depth)},
            {"leaves_before", std::to_string(input_leaves_before_reduction)},
            {"leaves_after", std::to_string(leaves)},
            {"duration_seconds", std::to_string(timer.elapsed_seconds() - reductions_start)}
        }
    );

    std::optional<CoreForest> tiny_reduced_solution;
    const std::size_t tiny_exact_limit =
        depth == 0
            ? config.tiny_exact_whole_max_leaves
            : config.tiny_exact_cluster_max_leaves;
    if (leaves <= tiny_exact_limit && !timer.should_stop(0.25)) {
        try {
            Tree reduced_t1 = Tree::from_newick(nt1);
            Tree reduced_t2 = Tree::from_newick(nt2);
            tiny_reduced_solution = run_tiny_exact_oracle(
                reduced_t1,
                reduced_t2,
                timer,
                tiny_exact_limit,
                "solve.tiny_exact.reduced",
                depth
            );
            if (tiny_reduced_solution.has_value()) {
                tiny_reduced_solution->normalize();
                tiny_reduced_solution->validate_partition_of(reduced_t1.leaf_labels);
                publish_progress_safely(
                    publish_reduced_progress,
                    *tiny_reduced_solution
                );
            }
        } catch (const std::exception&) {
            tiny_reduced_solution.reset();
        }
    }

    if (tiny_reduced_solution.has_value()) {
        solved_reduced = std::move(*tiny_reduced_solution);
    } else if (!timer.should_stop(1.5) &&
        depth < config.max_cluster_depth) {
        try {
            const double cluster_lookup_start = timer.elapsed_seconds();
            ClusterSplitLookup split_lookup = find_cluster_split_with_hierarchy_cache(
                nt1,
                nt2,
                next_placeholder,
                cluster_profile_memo
            );
            auto& split = split_lookup.split;

            if (split.found) {
                const std::size_t bottom_sz =
                    split.bottom_leaf_count != 0
                        ? split.bottom_leaf_count
                        : count_leaves(split.bottom1);
                const std::size_t top_sz =
                    split.top_without_cluster_leaf_count != 0
                        ? split.top_without_cluster_leaf_count
                        : count_leaves(split.top1_without_cluster);
                const bool should_split = 
                    leaves >= config.cluster_min_leaves ||
                    bottom_sz <= config.exact_max_leaves ||
                    top_sz <= config.exact_max_leaves;

                if (should_split) {
                    g_profile.event_raw(
                        "cluster_split",
                        &timer,
                        {
                            {"depth", std::to_string(depth)},
                            {"leaves", std::to_string(leaves)},
                            {"bottom_leaves", std::to_string(bottom_sz)},
                            {"top_leaves", std::to_string(top_sz)},
                            {"bottom_open_leaves", std::to_string(bottom_sz + 1)},
                            {"top_open_leaves", std::to_string(top_sz + 1)},
                            {"found", "true"},
                            {"hierarchy_cache_hit", split_lookup.hierarchy_cache_hit ? "true" : "false"},
                            {"stale_hierarchy_cache_entry", split_lookup.stale_cache_entry ? "true" : "false"},
                            {"lookup_duration_seconds", std::to_string(timer.elapsed_seconds() - cluster_lookup_start)},
                            {"open_trees_materialized", split.open_trees_materialized ? "true" : "false"}
                        }
                    );

                    SolverConfig child_config = config;
                    child_config.enable_singleton_rescue = false;

                    // Solve Case 1: Cut the edge above the cluster (without marker/placeholder)
                    CoreForest bottom_without = solve_instance(
                        std::move(split.bottom1),
                        std::move(split.bottom2),
                        child_config,
                        timer,
                        rng,
                        depth + 1,
                        ForestPublishCallback{},
                        nullptr,
                        cluster_profile_memo
                    );
                    CoreForest top_without = solve_instance(
                        std::move(split.top1_without_cluster),
                        std::move(split.top2_without_cluster),
                        child_config,
                        timer,
                        rng,
                        depth + 1,
                        ForestPublishCallback{},
                        nullptr,
                        cluster_profile_memo
                    );
                    CoreForest solved_reduced_without = CoreForest::unite(top_without, bottom_without);

                    if (config.enable_cluster_bridge_recovery && !timer.should_stop(1.5)) {
                        const double bridge_start = timer.elapsed_seconds();
                        std::optional<CoreForest> bridged = try_cheap_cluster_bridge(
                            top_without,
                            bottom_without,
                            nt1,
                            nt2,
                            timer
                        );
                        const bool accepted =
                            bridged.has_value() &&
                            bridged->component_count() < solved_reduced_without.component_count();
                        if (accepted) {
                            solved_reduced_without = std::move(*bridged);
                        }
                        g_profile.event_raw(
                            "cluster_bridge_recovery",
                            &timer,
                            {
                                {"depth", std::to_string(depth)},
                                {"leaves", std::to_string(leaves)},
                                {"accepted", accepted ? "true" : "false"},
                                {"duration_seconds", std::to_string(timer.elapsed_seconds() - bridge_start)}
                            }
                        );
                    }

                    ClusterProfile profile;
                    profile.closed = make_state_solution(std::move(solved_reduced_without));

                    const bool attempt_open = should_attempt_cluster_open_profile(
                        leaves,
                        bottom_sz,
                        top_sz,
                        child_config,
                        timer,
                        depth
                    );
                    if (attempt_open) {
                        std::optional<StateSolution> open_state =
                            solve_cluster_open_state(
                                nt1,
                                nt2,
                                split,
                                child_config,
                                timer,
                                rng,
                                depth,
                                cluster_profile_memo
                            );
                        if (open_state.has_value()) {
                            profile.open = std::move(*open_state);
                        }
                    } else {
                        g_profile.event_raw(
                            "cluster_open_profile_skipped",
                            &timer,
                            {
                                {"depth", std::to_string(depth)},
                                {"leaves", std::to_string(leaves)},
                                {"bottom_leaves", std::to_string(bottom_sz)},
                                {"top_leaves", std::to_string(top_sz)},
                                {"reason", json_quote(
                                    !child_config.enable_cluster_open_profile
                                        ? "disabled"
                                        : (depth > child_config.cluster_open_profile_max_depth
                                            ? "depth_limit"
                                            : (timer.should_stop(4.0)
                                                ? "timer_guard"
                                                : "size_limit"))
                                )}
                            }
                        );
                    }

                    const bool selected_open = state_better_than(profile.open, profile.closed);
                    g_profile.event_raw(
                        "cluster_profile_decision",
                        &timer,
                        {
                            {"depth", std::to_string(depth)},
                            {"leaves", std::to_string(leaves)},
                            {"bottom_leaves", std::to_string(bottom_sz)},
                            {"top_leaves", std::to_string(top_sz)},
                            {"closed_components", std::to_string(profile.closed.available ? profile.closed.ub : -1)},
                            {"open_available", profile.open.available ? "true" : "false"},
                            {"open_components", std::to_string(profile.open.available ? profile.open.ub : -1)},
                            {"selected", json_quote(selected_open ? "open" : "closed")}
                        }
                    );

                    if (selected_open) {
                        solved_reduced = std::move(profile.open.forest);
                    } else {
                        solved_reduced = std::move(profile.closed.forest);
                    }
                } else {
                    solved_reduced = solve_without_cluster_recursion(
                        nt1,
                        nt2,
                        config,
                        timer,
                        rng,
                        publish_reduced_progress
                    );
                }
            } else {
                g_profile.event_raw(
                    "cluster_split",
                    &timer,
                    {
                        {"depth", std::to_string(depth)},
                        {"leaves", std::to_string(leaves)},
                        {"found", "false"},
                        {"hierarchy_cache_hit", split_lookup.hierarchy_cache_hit ? "true" : "false"},
                        {"stale_hierarchy_cache_entry", split_lookup.stale_cache_entry ? "true" : "false"},
                        {"lookup_duration_seconds", std::to_string(timer.elapsed_seconds() - cluster_lookup_start)}
                    }
                );
                solved_reduced = solve_without_cluster_recursion(
                    nt1,
                    nt2,
                    config,
                    timer,
                    rng,
                    publish_reduced_progress
                );
            }
        } catch (const std::exception&) {
            solved_reduced = solve_without_cluster_recursion(
                nt1,
                nt2,
                config,
                timer,
                rng,
                publish_reduced_progress
            );
        }
    } else {
        solved_reduced = solve_without_cluster_recursion(
            nt1,
            nt2,
            config,
            timer,
            rng,
            publish_reduced_progress
        );
    }

    pace26::reductions::ReductionExpansionReport reduction_expansion_report;
    RedForest expanded_reduction_forest =
        stack.expand(
            to_reduction_forest(solved_reduced),
            &reduction_expansion_report
        );

    CoreForest expanded = to_core_forest(expanded_reduction_forest);
    expanded.normalize();
    expanded.validate_no_duplicates();
    if (archive_top_level_solution) {
        try {
            Tree archive_t1 = Tree::from_newick(archive_nt1);
            Tree archive_t2 = Tree::from_newick(archive_nt2);
            archive->remember_if_valid(expanded, archive_t1, archive_t2);
        } catch (const std::exception&) {
            // Archive use must never affect the returned solution.
        }
    }
    publish_progress_safely(publish_progress, expanded);

    g_profile.event_raw(
        "solve_instance_return",
        &timer,
        {
            {"depth", std::to_string(depth)},
            {"reduced_leaves", std::to_string(leaves)},
            {"components", std::to_string(expanded.component_count())},
            {"reduction_certification_preserving", reduction_expansion_report.certification_preserving ? "true" : "false"},
            {"reduction_common_subtree_expansions", std::to_string(reduction_expansion_report.common_subtree_expansions)},
            {"reduction_chain_expansions", std::to_string(reduction_expansion_report.chain_expansions)},
            {"reduction_chain_suffix_labels_reattached", std::to_string(reduction_expansion_report.chain_suffix_labels_reattached)},
            {"reduction_chain_suffix_singleton_fallbacks", std::to_string(reduction_expansion_report.chain_suffix_singleton_fallbacks)},
            {"reduction_chain_suffix_singleton_labels", std::to_string(reduction_expansion_report.chain_suffix_singleton_labels)},
            {"reduction_three_two_chain_expansions", std::to_string(reduction_expansion_report.three_two_chain_expansions)}
        }
    );

    return expanded;
}

std::optional<std::string> restricted_subtree_newick(
    const NewickTree& tree,
    int node,
    const std::unordered_set<std::uint32_t>& selected
) {
    const auto& n = tree.nodes[static_cast<std::size_t>(node)];

    if (n.is_leaf()) {
        if (selected.find(n.label) == selected.end()) {
            return std::nullopt;
        }

        return std::to_string(n.label);
    }

    std::optional<std::string> left =
        restricted_subtree_newick(tree, n.left, selected);

    std::optional<std::string> right =
        restricted_subtree_newick(tree, n.right, selected);

    if (left.has_value() && right.has_value()) {
        return "(" + *left + "," + *right + ")";
    }

    if (left.has_value()) {
        return left;
    }

    if (right.has_value()) {
        return right;
    }

    return std::nullopt;
}

std::string component_to_newick(
    const NewickTree& original_tree,
    const CoreComponent& component
) {
    if (component.labels.size() == 1) {
        return std::to_string(component.labels.front());
    }

    if (component.labels.size() == 2) {
        return "(" + std::to_string(component.labels[0]) +
               "," + std::to_string(component.labels[1]) + ")";
    }

    std::unordered_set<std::uint32_t> selected;
    selected.reserve(component.labels.size() * 2);

    for (std::uint32_t label : component.labels) {
        selected.insert(label);
    }

    std::optional<std::string> result =
        restricted_subtree_newick(original_tree, original_tree.root, selected);

    if (!result.has_value()) {
        throw std::runtime_error("failed to induce output component");
    }

    return *result;
}

void write_forest(
    const NewickTree& original_tree,
    CoreForest forest,
    std::ostream& out
) {
    forest.normalize();

    for (const CoreComponent& component : forest.components) {
        out << component_to_newick(original_tree, component) << ";\n";
    }
}

std::string forest_to_output_string(
    const NewickTree& original_tree,
    CoreForest forest
) {
    std::ostringstream out;
    write_forest(original_tree, std::move(forest), out);
    return out.str();
}

struct ExactWindowRepairOptions {
    std::size_t max_subproblem_leaves = 24;
    std::size_t max_components_per_subproblem = 8;
    std::size_t max_attempts = 64;
    int max_rounds = 3;

    double total_time_limit_seconds = 3.0;
    double local_time_limit_seconds = 0.20;
    double guard_seconds = 2.0;

    std::uint64_t max_candidate_masks_tested = 1500000ULL;
};

ExactWindowRepairOptions exact_window_repair_options(
    std::size_t n,
    const Timer& timer,
    bool direct_stage,
    bool extended_search
) {
    ExactWindowRepairOptions opts;

    if (extended_search) {
        opts.max_subproblem_leaves = direct_stage ? 64 : 48;
        opts.max_components_per_subproblem = direct_stage ? 12 : 10;
        opts.max_attempts = direct_stage ? 420 : 300;
        opts.max_rounds = direct_stage ? 10 : 7;
        opts.total_time_limit_seconds = direct_stage ? 45.0 : 25.0;
        opts.local_time_limit_seconds = direct_stage ? 1.10 : 0.85;
        opts.guard_seconds = 7.0;
        opts.max_candidate_masks_tested = 12000000ULL;
        return opts;
    }

    if (n >= 7000) {
        opts.max_subproblem_leaves = 28;
        opts.max_components_per_subproblem = 8;
        opts.max_attempts = 0;
        opts.max_rounds = 0;
        opts.total_time_limit_seconds = 0.0;
        opts.local_time_limit_seconds = 0.16;
        opts.max_candidate_masks_tested = 900000ULL;
    } else if (n >= 4000) {
        opts.max_subproblem_leaves = 40;
        opts.max_components_per_subproblem = 9;
        opts.max_attempts = direct_stage ? 100 : 64;
        opts.max_rounds = direct_stage ? 5 : 3;
        opts.total_time_limit_seconds = direct_stage ? 5.0 : 2.5;
        opts.local_time_limit_seconds = 0.20;
        opts.max_candidate_masks_tested = 1500000ULL;
    } else {
        opts.max_subproblem_leaves = 45;
        opts.max_components_per_subproblem = 9;
        opts.max_attempts = 72;
        opts.max_rounds = 3;
        opts.total_time_limit_seconds = 2.5;
        opts.local_time_limit_seconds = 0.18;
        opts.max_candidate_masks_tested = 1200000ULL;
    }

    opts.total_time_limit_seconds = std::min(
        opts.total_time_limit_seconds,
        std::max(0.0, timer.remaining_seconds() - opts.guard_seconds)
    );

    return opts;
}

NewickTree induced_newick_tree(
    const NewickTree& tree,
    const std::vector<std::uint32_t>& labels
) {
    if (labels.empty()) {
        throw std::runtime_error("cannot induce empty tree");
    }

    std::unordered_set<std::uint32_t> selected;
    selected.reserve(labels.size() * 2);

    for (std::uint32_t label : labels) {
        selected.insert(label);
    }

    std::optional<std::string> body =
        restricted_subtree_newick(tree, tree.root, selected);

    if (!body.has_value()) {
        throw std::runtime_error("failed to induce exact-repair tree");
    }

    std::string text = *body + ";";
    return pace26::io::NewickParser::parse_string(text);
}

std::uint64_t exact_repair_label_hash(const std::vector<std::uint32_t>& labels) {
    std::uint64_t h = 0x84222325cbf29ce4ULL;

    for (std::uint32_t label : labels) {
        h ^= profile_mix64(static_cast<std::uint64_t>(label) + 0x9e3779b97f4a7c15ULL);
        h *= 0x100000001b3ULL;
    }

    h ^= profile_mix64(static_cast<std::uint64_t>(labels.size()));
    return h;
}

struct ExactRepairWindow {
    std::vector<std::size_t> component_indices;
    std::vector<std::uint32_t> labels;
    std::uint64_t hash = 0;
};

std::vector<std::size_t> component_order_by_tree(
    const CoreForest& forest,
    const Tree& tree
) {
    struct Entry {
        std::size_t index = 0;
        int min_tin = 0;
        int max_tin = 0;
        std::size_t size = 0;
    };

    std::vector<Entry> entries;
    entries.reserve(forest.components.size());

    for (std::size_t i = 0; i < forest.components.size(); ++i) {
        const CoreComponent& component = forest.components[i];
        if (component.empty()) {
            continue;
        }

        int min_tin = std::numeric_limits<int>::max();
        int max_tin = std::numeric_limits<int>::min();

        for (std::uint32_t label : component.labels) {
            const int node = tree.node_of_label(label);
            const int tin = tree.tin[static_cast<std::size_t>(node)];
            min_tin = std::min(min_tin, tin);
            max_tin = std::max(max_tin, tin);
        }

        entries.push_back({i, min_tin, max_tin, component.labels.size()});
    }

    std::sort(
        entries.begin(),
        entries.end(),
        [](const Entry& a, const Entry& b) {
            if (a.min_tin != b.min_tin) {
                return a.min_tin < b.min_tin;
            }
            if (a.max_tin != b.max_tin) {
                return a.max_tin < b.max_tin;
            }
            if (a.size != b.size) {
                return a.size < b.size;
            }
            return a.index < b.index;
        }
    );

    std::vector<std::size_t> order;
    order.reserve(entries.size());

    for (const Entry& entry : entries) {
        order.push_back(entry.index);
    }

    return order;
}

void collect_exact_repair_windows_from_order(
    const CoreForest& forest,
    const std::vector<std::size_t>& order,
    const ExactWindowRepairOptions& opts,
    std::vector<ExactRepairWindow>& windows,
    std::unordered_set<std::uint64_t>& seen_hashes
) {
    const std::size_t collection_cap = std::max<std::size_t>(opts.max_attempts * 6, 32);

    for (std::size_t start = 0; start < order.size() && windows.size() < collection_cap; ++start) {
        ExactRepairWindow window;

        for (std::size_t pos = start;
             pos < order.size() &&
             window.component_indices.size() < opts.max_components_per_subproblem;
             ++pos) {
            const std::size_t component_index = order[pos];
            const CoreComponent& component = forest.components[component_index];

            if (component.size() > opts.max_subproblem_leaves) {
                break;
            }

            if (window.labels.size() + component.labels.size() > opts.max_subproblem_leaves) {
                break;
            }

            window.component_indices.push_back(component_index);
            window.labels.insert(
                window.labels.end(),
                component.labels.begin(),
                component.labels.end()
            );

            if (window.component_indices.size() < 3 || window.labels.size() < 3) {
                continue;
            }

            ExactRepairWindow candidate = window;
            std::sort(candidate.labels.begin(), candidate.labels.end());
            candidate.hash = exact_repair_label_hash(candidate.labels);

            if (!seen_hashes.insert(candidate.hash).second) {
                continue;
            }

            windows.push_back(std::move(candidate));
            if (windows.size() >= collection_cap) {
                break;
            }
        }
    }
}

std::vector<ExactRepairWindow> collect_exact_repair_windows(
    const CoreForest& forest,
    const Tree& t1,
    const Tree& t2,
    const ExactWindowRepairOptions& opts
) {
    std::vector<ExactRepairWindow> windows;
    std::unordered_set<std::uint64_t> seen_hashes;

    seen_hashes.reserve(opts.max_attempts * 8 + 16);

    const std::vector<std::size_t> order1 = component_order_by_tree(forest, t1);
    collect_exact_repair_windows_from_order(forest, order1, opts, windows, seen_hashes);

    const std::vector<std::size_t> order2 = component_order_by_tree(forest, t2);
    collect_exact_repair_windows_from_order(forest, order2, opts, windows, seen_hashes);

    std::sort(
        windows.begin(),
        windows.end(),
        [](const ExactRepairWindow& a, const ExactRepairWindow& b) {
            if (a.component_indices.size() != b.component_indices.size()) {
                return a.component_indices.size() > b.component_indices.size();
            }
            if (a.labels.size() != b.labels.size()) {
                return a.labels.size() < b.labels.size();
            }
            return a.hash < b.hash;
        }
    );

    if (windows.size() > opts.max_attempts) {
        windows.resize(opts.max_attempts);
    }

    return windows;
}

bool try_exact_repair_window(
    CoreForest& best,
    const NewickTree& nt1,
    const NewickTree& nt2,
    const Tree& full_t1,
    const Tree& full_t2,
    const ExactRepairWindow& window,
    const ExactWindowRepairOptions& opts,
    Timer& timer,
    const std::string& context
) {
    const double attempt_start = timer.elapsed_seconds();
    auto log_attempt = [&](const char* reason,
                           bool accepted,
                           std::size_t sub_before,
                           std::size_t sub_after) {
        if (!g_profile.enabled()) {
            return;
        }
        g_profile.event_raw(
            "exact_window_repair_attempt",
            &timer,
            {
                {"context", json_quote(context)},
                {"hash", json_quote(profile_hex64(window.hash))},
                {"reason", json_quote(reason)},
                {"accepted", accepted ? "true" : "false"},
                {"labels", std::to_string(window.labels.size())},
                {"window_components", std::to_string(window.component_indices.size())},
                {"sub_components_before", std::to_string(sub_before)},
                {"sub_components_after", std::to_string(sub_after)},
                {"duration_seconds", std::to_string(timer.elapsed_seconds() - attempt_start)}
            }
        );
    };

    if (window.component_indices.size() < 3 ||
        window.labels.empty() ||
        window.labels.size() > opts.max_subproblem_leaves ||
        timer.should_stop(opts.guard_seconds)) {
        log_attempt("precheck_failed", false, 0, 0);
        return false;
    }

    CoreForest sub_incumbent;
    for (std::size_t component_index : window.component_indices) {
        if (component_index >= best.components.size()) {
            log_attempt("invalid_component_index", false, 0, 0);
            return false;
        }
        sub_incumbent.add_component(best.components[component_index]);
    }
    sub_incumbent.normalize();
    sub_incumbent.validate_partition_of(window.labels);

    try {
        NewickTree sub_nt1 = induced_newick_tree(nt1, window.labels);
        NewickTree sub_nt2 = induced_newick_tree(nt2, window.labels);
        Tree sub_t1 = Tree::from_newick(sub_nt1);
        Tree sub_t2 = Tree::from_newick(sub_nt2);

        pace26::exact::InProcessExactMaf::Options exact_opts;
        exact_opts.enabled = true;
        exact_opts.max_leaves = opts.max_subproblem_leaves;
        exact_opts.local_time_limit_seconds = opts.local_time_limit_seconds;
        exact_opts.global_guard_seconds = opts.guard_seconds;
        exact_opts.local_guard_seconds = 0.01;
        exact_opts.max_candidate_masks_tested = opts.max_candidate_masks_tested;

        pace26::exact::InProcessExactMaf exact(exact_opts);
        std::optional<CoreForest> exact_forest =
            exact.solve(sub_t1, sub_t2, &timer, &sub_incumbent);

        if (!exact_forest.has_value()) {
            log_attempt(
                "no_exact_candidate",
                false,
                sub_incumbent.component_count(),
                0
            );
            return false;
        }

        exact_forest->normalize();
        exact_forest->validate_partition_of(window.labels);

        if (exact_forest->component_count() >= sub_incumbent.component_count()) {
            log_attempt(
                "no_sub_improvement",
                false,
                sub_incumbent.component_count(),
                exact_forest->component_count()
            );
            return false;
        }

        std::vector<char> selected(best.components.size(), 0);
        for (std::size_t component_index : window.component_indices) {
            selected[component_index] = 1;
        }

        CoreForest candidate;
        candidate.components.reserve(
            best.components.size() -
            window.component_indices.size() +
            exact_forest->component_count()
        );

        for (std::size_t i = 0; i < best.components.size(); ++i) {
            if (selected[i] == 0) {
                candidate.add_component(best.components[i]);
            }
        }

        for (const CoreComponent& component : exact_forest->components) {
            candidate.add_component(component);
        }

        const std::size_t before = best.component_count();
        const bool accepted =
            consider_candidate(best, std::move(candidate), full_t1, full_t2);

        if (accepted) {
            log_attempt(
                "accepted",
                true,
                sub_incumbent.component_count(),
                exact_forest->component_count()
            );
            g_profile.event_raw(
                "exact_window_repair_accept",
                &timer,
                {
                    {"context", json_quote(context)},
                    {"labels", std::to_string(window.labels.size())},
                    {"components_before", std::to_string(before)},
                    {"components_after", std::to_string(best.component_count())},
                    {"sub_components_before", std::to_string(sub_incumbent.component_count())},
                    {"sub_components_after", std::to_string(exact_forest->component_count())}
                }
            );
        } else {
            log_attempt(
                "not_global_improvement",
                false,
                sub_incumbent.component_count(),
                exact_forest->component_count()
            );
        }

        return accepted;
    } catch (const std::exception&) {
        log_attempt("exception", false, 0, 0);
        return false;
    }
}

bool run_exact_window_repair(
    CoreForest& best,
    const NewickTree& nt1,
    const NewickTree& nt2,
    Timer& timer,
    const ExactWindowRepairOptions& opts,
    const std::string& context
) {
    if (best.component_count() < 3 ||
        opts.total_time_limit_seconds <= 0.0 ||
        timer.should_stop(opts.guard_seconds)) {
        return false;
    }

    const double phase_start = timer.elapsed_seconds();
    const std::size_t n = count_leaves(nt1);
    const std::size_t before = best.component_count();
    std::size_t attempts = 0;
    std::size_t accepted_windows = 0;
    std::size_t rounds_run = 0;
    std::size_t windows_collected_total = 0;
    std::size_t max_windows_in_round = 0;
    std::size_t no_window_rounds = 0;

    try {
        Tree full_t1 = Tree::from_newick(nt1);
        Tree full_t2 = Tree::from_newick(nt2);

        for (int round = 0; round < opts.max_rounds; ++round) {
            if (timer.should_stop(opts.guard_seconds) ||
                timer.elapsed_seconds() - phase_start >= opts.total_time_limit_seconds ||
                attempts >= opts.max_attempts) {
                break;
            }

            const std::vector<ExactRepairWindow> windows =
                collect_exact_repair_windows(best, full_t1, full_t2, opts);
            ++rounds_run;
            windows_collected_total += windows.size();
            max_windows_in_round = std::max(max_windows_in_round, windows.size());
            if (windows.empty()) {
                ++no_window_rounds;
            }

            bool accepted_in_round = false;

            for (const ExactRepairWindow& window : windows) {
                if (timer.should_stop(opts.guard_seconds) ||
                    timer.elapsed_seconds() - phase_start >= opts.total_time_limit_seconds ||
                    attempts >= opts.max_attempts) {
                    break;
                }

                ++attempts;

                if (try_exact_repair_window(
                        best,
                        nt1,
                        nt2,
                        full_t1,
                        full_t2,
                        window,
                        opts,
                        timer,
                        context)) {
                    ++accepted_windows;
                    accepted_in_round = true;
                    break;
                }
            }

            if (!accepted_in_round) {
                break;
            }
        }
    } catch (const std::exception&) {
        // Repair is opportunistic. Keep the incoming valid incumbent.
    }

    if (g_profile.enabled()) {
        g_profile.phase_result(
            context,
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            best.component_count() < before,
            {
                {"attempts", std::to_string(attempts)},
                {"accepted_windows", std::to_string(accepted_windows)},
                {"rounds_run", std::to_string(rounds_run)},
                {"windows_collected_total", std::to_string(windows_collected_total)},
                {"max_windows_in_round", std::to_string(max_windows_in_round)},
                {"no_window_rounds", std::to_string(no_window_rounds)},
                {"max_subproblem_leaves", std::to_string(opts.max_subproblem_leaves)},
                {"max_components_per_subproblem", std::to_string(opts.max_components_per_subproblem)},
                {"max_attempts", std::to_string(opts.max_attempts)},
                {"max_rounds", std::to_string(opts.max_rounds)},
                {"total_time_limit_seconds", std::to_string(opts.total_time_limit_seconds)}
            }
        );
    }

    best.normalize();
    return best.component_count() < before;
}

void publish_emergency_forest(
    const NewickTree& original_tree,
    CoreForest forest
) {
    try {
        set_emergency_output(forest_to_output_string(original_tree, std::move(forest)));
    } catch (const std::exception&) {
        // Keep the previous valid emergency answer.
    }
}

void publish_progress_safely(
    const ForestPublishCallback& publish_progress,
    const CoreForest& forest
) {
    if (!publish_progress) {
        return;
    }

    try {
        publish_progress(forest);
    } catch (const std::exception&) {
        // Keep the previous valid emergency answer.
    }
}

bool emergency_candidate_is_better(
    CoreForest candidate,
    CoreForest incumbent,
    const Tree& t1,
    const Tree& t2
) {
    candidate.normalize();
    incumbent.normalize();

    const std::size_t candidate_components = candidate.component_count();
    const std::size_t incumbent_components = incumbent.component_count();
    if (candidate_components > incumbent_components) {
        return false;
    }

    bool can_improve = candidate_components < incumbent_components;
    if (!can_improve) {
        can_improve = tie_score(candidate) > tie_score(incumbent);
    }
    if (!can_improve) {
        return false;
    }

    return is_valid_agreement_forest(candidate, t1, t2);
}

bool accept_initial_emergency_candidate(
    CoreForest& incumbent,
    CoreForest candidate,
    const NewickTree& original_tree,
    const Tree& t1,
    const Tree& t2
) {
    if (!emergency_candidate_is_better(candidate, incumbent, t1, t2)) {
        return false;
    }

    candidate.normalize();
    incumbent = std::move(candidate);
    publish_emergency_forest(original_tree, incumbent);
    return true;
}

struct InitialHeuristicResult {
    CoreForest forest;
    std::string output;
};

InitialHeuristicResult make_initial_heuristic_result(
    const NewickTree& nt1,
    const NewickTree& nt2,
    Timer& timer,
    Random& rng,
    std::size_t warm_lds_min_leaves,
    ComponentForestArchive* archive = nullptr
) {
    Tree t1 = Tree::from_newick(nt1);
    Tree t2 = Tree::from_newick(nt2);

    CoreForest best = CoreForest::singleton_forest_from_tree(t1);
    std::vector<CoreForest> packing_seeds;

    {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        try {
            pace26::heuristics::FastCherryApprox fast;
            CoreForest candidate = fast.solve(t1, t2, &timer, &rng);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            remember_packing_seed(packing_seeds, candidate);
            accepted = accept_initial_emergency_candidate(best, std::move(candidate), nt1, t1, t2);
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "warm.fast_cherry",
            timer,
            phase_start,
            static_cast<std::size_t>(t1.leaf_count()),
            before,
            best.component_count(),
            accepted,
            {{"exception", threw ? "true" : "false"}}
        );
    }

    {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();
        bool accepted = false;
        bool threw = false;
        try {
            pace26::heuristics::CherryPairApprox pairs;
            CoreForest candidate = pairs.solve(t1, t2);
            if (archive != nullptr) {
                archive->remember_if_valid(candidate, t1, t2);
            }
            remember_packing_seed(packing_seeds, candidate);
            accepted = accept_initial_emergency_candidate(best, std::move(candidate), nt1, t1, t2);
        } catch (const std::exception&) {
            threw = true;
        }
        g_profile.phase_result(
            "warm.cherry_pair",
            timer,
            phase_start,
            static_cast<std::size_t>(t1.leaf_count()),
            before,
            best.component_count(),
            accepted,
            {{"exception", threw ? "true" : "false"}}
        );
    }

    const std::size_t n = static_cast<std::size_t>(t1.leaf_count());

    {
        const double phase_start = timer.elapsed_seconds();
        const std::size_t before = best.component_count();

        run_active_cherry_portfolio(
            best,
            t1,
            t2,
            n,
            false,
            timer,
            [&](CoreForest candidate) {
                if (archive != nullptr) {
                    archive->remember_if_valid(candidate, t1, t2);
                }
                remember_packing_seed(packing_seeds, candidate);
                return accept_initial_emergency_candidate(
                    best,
                    std::move(candidate),
                    nt1,
                    t1,
                    t2
                );
            },
            [&](const CoreForest& candidate) {
                accept_initial_emergency_candidate(best, candidate, nt1, t1, t2);
            },
            "warm.active_cherry"
        );

        g_profile.phase_result(
            "warm.active_cherry.total",
            timer,
            phase_start,
            n,
            before,
            best.component_count(),
            best.component_count() < before
        );
    }

    run_agreement_component_packing(
        best,
        t1,
        t2,
        std::move(packing_seeds),
        timer,
        false,
        "warm.agreement_component_packing",
        [&](const CoreForest& candidate) {
            publish_emergency_forest(nt1, candidate);
        },
        AgreementPackingMode::Normal,
        false,
        false,
        {},
        archive
    );

    best.normalize();
    if (archive != nullptr) {
        archive->remember_forest(best);
    }
    InitialHeuristicResult result;
    result.forest = best;
    result.output = forest_to_output_string(nt1, best);
    return result;
}

SolverConfig parse_config_from_args(int argc, char** argv) {
    SolverConfig config;

    /*
     * Optil runs the default 300-second configuration. The environment override
     * only truncates this same binary for local iteration.
     */
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + name);
            }

            return std::string(argv[++i]);
        };
        if (arg == "--profile-dir") {
            config.profile_dir = value(arg);
        } 
    }

    return config;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);

        SolverConfig parsed_config = parse_config_from_args(argc, argv);
        const double external_time_limit_seconds = parsed_config.time_limit_seconds;

        pace26::io::PaceInstance instance =
            pace26::io::read_pace_instance(std::cin);

        if (instance.trees.size() != 2) {
            throw std::runtime_error("this solver currently expects exactly two rooted trees");
        }

        const std::size_t original_leaf_count = count_leaves(instance.trees[0]);

        SolverConfig baseline_config = parsed_config;
        baseline_config.time_limit_seconds = std::min(
            kBaselineStageLimitSeconds,
            external_time_limit_seconds
        );
        baseline_config.packing_focused = default_packing_focused_for_instance(
            original_leaf_count,
            baseline_config.time_limit_seconds
        );
        apply_packing_focused_config(baseline_config);

        SolverConfig extended_base_config = parsed_config;
        extended_base_config.time_limit_seconds = external_time_limit_seconds;
        extended_base_config.packing_focused = default_packing_focused_for_instance(
            original_leaf_count,
            external_time_limit_seconds
        );
        apply_packing_focused_config(extended_base_config);

        const NewickTree original_tree_for_output = instance.trees[0];

        // Always keep a valid emergency answer ready. If the process is close
        // to the external timeout, SIGALRM/SIGTERM writes this directly to stdout.
        set_emergency_output(make_singleton_output(original_tree_for_output));
        Timer process_timer(external_time_limit_seconds);
        Timer baseline_timer(
            baseline_config.time_limit_seconds,
            process_timer.start_time()
        );
        g_profile.open(extended_base_config, instance);
        Random rng(baseline_config.seed);
        install_emergency_alarm(external_time_limit_seconds - 1.0);

        Tree original_core_tree = Tree::from_newick(original_tree_for_output);
        Tree original_core_tree2 = Tree::from_newick(instance.trees[1]);
        CoreForest incumbent = CoreForest::singleton_forest_from_tree(original_core_tree);
        CoreForest published_incumbent = incumbent;
        const bool enable_component_archive =
            original_leaf_count >= 220 && original_leaf_count <= 5000;
        ComponentForestArchive component_archive;
        ComponentForestArchive* component_archive_ptr =
            enable_component_archive ? &component_archive : nullptr;
        ClusterProfileMemo cluster_profile_memo;
        if (component_archive_ptr != nullptr) {
            component_archive_ptr->remember_forest(incumbent);
        }

        auto publish_if_global_best = [&](const CoreForest& candidate) {
            try {
                CoreForest copy = candidate;
                if (consider_candidate(
                        published_incumbent,
                        std::move(copy),
                        original_core_tree,
                        original_core_tree2)) {
                    publish_emergency_forest(
                        original_tree_for_output,
                        published_incumbent
                    );
                    if (component_archive_ptr != nullptr) {
                        component_archive_ptr->remember_forest(published_incumbent);
                    }
                }
            } catch (const std::exception&) {
                // Keep the previous valid emergency answer.
            }
        };

        auto sync_incumbent_from_published = [&]() {
            try {
                consider_candidate(
                    incumbent,
                    published_incumbent,
                    original_core_tree,
                    original_core_tree2
                );
            } catch (const std::exception&) {
                // Keep the current incumbent.
            }
        };

        auto run_early_archive_crossover = [&](const std::string& phase) {
            if (component_archive_ptr == nullptr ||
                component_archive_ptr->forest_count() < 2 ||
                baseline_timer.should_stop(2.0)) {
                return;
            }

            const std::size_t seed_cap = original_leaf_count >= 6000 ? 8 : 10;
            const std::size_t pair_cap = original_leaf_count >= 6000 ? 8 : 14;
            const bool accepted = run_forest_crossover_portfolio(
                incumbent,
                original_core_tree,
                original_core_tree2,
                std::vector<CoreForest>{},
                baseline_timer,
                phase,
                [&](const CoreForest& candidate) {
                    publish_if_global_best(candidate);
                },
                seed_cap,
                1,
                component_archive_ptr,
                false,
                0
            );
            if (accepted) {
                publish_if_global_best(incumbent);
            }
            sync_incumbent_from_published();
        };

        std::optional<CoreForest> warm_forest;

        try {
            InitialHeuristicResult warm = make_initial_heuristic_result(
                instance.trees[0],
                instance.trees[1],
                baseline_timer,
                rng,
                baseline_config.direct_cheap_output_min_leaves,
                component_archive_ptr
            );

            warm_forest = warm.forest;
            publish_if_global_best(*warm_forest);
        } catch (const std::exception&) {
            // Keep singleton emergency output.
        }

        if (warm_forest.has_value()) {
            try {
                const bool accepted = consider_candidate(
                    incumbent,
                    *warm_forest,
                    original_core_tree,
                    original_core_tree2
                );
                if (accepted) {
                    publish_if_global_best(incumbent);
                }
            } catch (const std::exception&) {
                // Keep the singleton incumbent.
            }
        }

        const bool baseline_direct =
            original_leaf_count >= baseline_config.direct_cheap_output_min_leaves;

        if (baseline_direct) {
            run_early_archive_crossover("main.early_archive_forest_crossover");
        }

        if (kEnableExactWindowRepair &&
            !baseline_config.packing_focused &&
            baseline_direct &&
            warm_forest.has_value() &&
            !baseline_timer.should_stop(2.0)) {
            try {
                ExactWindowRepairOptions repair_opts =
                    exact_window_repair_options(
                        original_leaf_count,
                        baseline_timer,
                        true,
                        false
                    );

                if (run_exact_window_repair(
                        *warm_forest,
                        instance.trees[0],
                        instance.trees[1],
                        baseline_timer,
                        repair_opts,
                        "main.direct_exact_window_repair")) {
                    publish_if_global_best(*warm_forest);
                }

                const bool accepted = consider_candidate(
                    incumbent,
                    *warm_forest,
                    original_core_tree,
                    original_core_tree2
                );
                if (accepted) {
                    publish_if_global_best(incumbent);
                }
            } catch (const std::exception&) {
                // Keep the warm emergency incumbent.
            }
        } else if (!kEnableExactWindowRepair && baseline_direct) {
            g_profile.event_raw(
                "phase_skipped",
                &baseline_timer,
                {
                    {"phase", json_quote("main.direct_exact_window_repair")},
                    {"n", std::to_string(original_leaf_count)},
                    {"reason", json_quote("disabled_low_roi_pruning")}
                }
            );
        }

        if (baseline_direct) {
            g_profile.event_raw(
                "baseline_direct_complete",
                &baseline_timer,
                {
                    {"leaves", std::to_string(original_leaf_count)},
                    {"direct_min", std::to_string(baseline_config.direct_cheap_output_min_leaves)},
                    {"components", std::to_string(incumbent.component_count())}
                }
            );
        } else {
            try {
                CoreForest baseline_candidate = solve_instance(
                    instance.trees[0],
                    instance.trees[1],
                    baseline_config,
                    baseline_timer,
                    rng,
                    0,
                    {},
                    component_archive_ptr,
                    &cluster_profile_memo
                );

                const bool accepted = consider_candidate(
                    incumbent,
                    std::move(baseline_candidate),
                    original_core_tree,
                    original_core_tree2
                );
                if (accepted) {
                    publish_if_global_best(incumbent);
                }
            } catch (const std::exception&) {
                // Keep the validated warm incumbent.
            }
            sync_incumbent_from_published();

            if (kEnableExactWindowRepair &&
                !baseline_config.packing_focused &&
                original_leaf_count >= 1000 &&
                !baseline_timer.should_stop(2.0)) {
                try {
                    ExactWindowRepairOptions repair_opts =
                        exact_window_repair_options(
                            original_leaf_count,
                            baseline_timer,
                            false,
                            false
                        );

                    if (run_exact_window_repair(
                        incumbent,
                        instance.trees[0],
                        instance.trees[1],
                        baseline_timer,
                        repair_opts,
                        "main.final_exact_window_repair")) {
                        publish_if_global_best(incumbent);
                    }
                } catch (const std::exception&) {
                    // Keep the validated incumbent.
                }
            } else if (!kEnableExactWindowRepair && original_leaf_count >= 1000) {
                g_profile.event_raw(
                    "phase_skipped",
                    &baseline_timer,
                    {
                        {"phase", json_quote("main.final_exact_window_repair")},
                        {"n", std::to_string(original_leaf_count)},
                        {"reason", json_quote("disabled_low_roi_pruning")}
                    }
                );
            }
        }

        if (!baseline_direct) {
            run_early_archive_crossover("main.post_baseline_archive_forest_crossover");
        }

        std::vector<CoreForest> baseline_global_packing_seeds;
        if (!baseline_timer.should_stop(2.0)) {
            run_reduced_global_candidate_portfolio(
                incumbent,
                instance.trees[0],
                instance.trees[1],
                baseline_config,
                baseline_timer,
                false,
                [&](CoreForest candidate) {
                    const bool accepted = consider_candidate(
                        incumbent,
                        std::move(candidate),
                        original_core_tree,
                        original_core_tree2
                    );
                    if (accepted) {
                        publish_if_global_best(incumbent);
                    }
                    return accepted;
                },
                "main.baseline_reduced_global",
                &baseline_global_packing_seeds,
                baseline_config.packing_focused,
                component_archive_ptr,
                [&](const CoreForest& seed) {
                    if (!baseline_direct ||
                        original_leaf_count < 6000 ||
                        baseline_timer.should_stop(2.0)) {
                        return;
                    }

                    std::vector<CoreForest> immediate_seed;
                    immediate_seed.push_back(seed);
                    const bool accepted = run_forest_crossover_portfolio(
                        incumbent,
                        original_core_tree,
                        original_core_tree2,
                        immediate_seed,
                        baseline_timer,
                        "main.baseline_reduced_global_immediate_crossover",
                        [&](const CoreForest& candidate) {
                            publish_if_global_best(candidate);
                        },
                        1,
                        1,
                        component_archive_ptr,
                        false,
                        0
                    );
                    if (accepted) {
                        publish_if_global_best(incumbent);
                    }
                    sync_incumbent_from_published();
                }
            );
        }

        if (!baseline_global_packing_seeds.empty() && !baseline_timer.should_stop(1.5)) {
            run_forest_crossover_portfolio(
                incumbent,
                original_core_tree,
                original_core_tree2,
                baseline_global_packing_seeds,
                baseline_timer,
                "main.baseline_forest_crossover",
                [&](const CoreForest& candidate) {
                    publish_if_global_best(candidate);
                },
                12,
                2,
                component_archive_ptr,
                false,
                0
            );
        }

        if (!baseline_global_packing_seeds.empty() && !baseline_timer.should_stop(1.5)) {
            run_agreement_component_packing(
                incumbent,
                original_core_tree,
                original_core_tree2,
                std::move(baseline_global_packing_seeds),
                baseline_timer,
                false,
                "main.baseline_global_repacking",
                [&](const CoreForest& candidate) {
                    publish_if_global_best(candidate);
                },
                AgreementPackingMode::Normal,
                baseline_config.packing_focused,
                false,
                {},
                component_archive_ptr
            );
        }

        if (original_leaf_count < 2500 && !baseline_timer.should_stop(3.0)) {
            run_incumbent_union_repacking(
                incumbent,
                original_core_tree,
                original_core_tree2,
                baseline_timer,
                false,
                "main.baseline_incumbent_union_repacking",
                [&](const CoreForest& candidate) {
                    publish_if_global_best(candidate);
                },
                AgreementPackingMode::Normal,
                baseline_config.packing_focused,
                component_archive_ptr,
                original_leaf_count <= 700 ? 10000 : 14000,
                3.0
            );
        }

        const bool baseline_singleton_ejection_allowed =
            original_leaf_count < 2500 ||
            (original_leaf_count < 6000 &&
             baseline_timer.remaining_seconds() >= 8.0);
        if (baseline_singleton_ejection_allowed &&
            !baseline_timer.should_stop(original_leaf_count <= 700 ? 2.0 : 3.0)) {
            std::vector<CoreForest> ejection_seeds;
            remember_diverse_packing_seed(ejection_seeds, incumbent, 8);
            run_singleton_ejection_repacking(
                incumbent,
                original_core_tree,
                original_core_tree2,
                std::move(ejection_seeds),
                baseline_timer,
                original_leaf_count <= 700,
                "main.baseline_singleton_ejection_repacking",
                [&](const CoreForest& candidate) {
                    publish_if_global_best(candidate);
                },
                AgreementPackingMode::Normal,
                baseline_config.packing_focused,
                component_archive_ptr,
                original_leaf_count <= 220
                    ? 14000
                    : (original_leaf_count <= 700
                        ? 12000
                        : (original_leaf_count <= 2500 ? 16000 : 10000)),
                original_leaf_count <= 700 ? 2.0 : 3.0
            );
        }

        sync_incumbent_from_published();
        incumbent.normalize();
        incumbent.validate_partition_of(original_core_tree.leaf_labels);
        publish_if_global_best(incumbent);

        g_profile.event_raw(
            "baseline_complete",
            &baseline_timer,
            {
                {"components", std::to_string(incumbent.component_count())},
                {"valid", "true"},
                {"direct", baseline_direct ? "true" : "false"}
            }
        );

        const CoreForest baseline_archive_forest = incumbent;

        const bool can_run_extended_stage =
            external_time_limit_seconds > kBaselineStageLimitSeconds + kOutputReserveSeconds;

        if (can_run_extended_stage) {
            const double extended_deadline =
                external_time_limit_seconds - kOutputReserveSeconds;
            Timer extended_timer(extended_deadline, process_timer.start_time());
            SolverConfig extended_config = make_extended_config(extended_base_config);
            extended_config.enable_singleton_rescue = original_leaf_count <= 220;
            const bool use_large_final_global_repacking = original_leaf_count >= 2500;
            if (original_leaf_count >= 2500 && original_leaf_count < 5000) {
                extended_config.local_improve_max_leaves = 5000;
            }
            if (!use_large_final_global_repacking) {
                extended_config.extended_quality_recovery = true;
            }

            g_profile.event_raw(
                "extended_stage_start",
                &extended_timer,
                {
                    {"components", std::to_string(incumbent.component_count())},
                    {"deadline_seconds", std::to_string(extended_deadline)}
                }
            );

            std::vector<CoreForest> extended_global_packing_seeds;
            std::vector<CoreForest> final_global_packing_seeds;
            const std::size_t final_global_seed_cap =
                original_leaf_count >= 15000
                    ? 8
                    : (original_leaf_count >= 12000
                    ? 20
                    : (original_leaf_count >= 6000 ? 16 : 8));
            const bool use_small_exactification_tail = false;
            if (use_large_final_global_repacking || use_small_exactification_tail) {
                remember_diverse_packing_seed(
                    final_global_packing_seeds,
                    baseline_archive_forest,
                    final_global_seed_cap
                );
            }
            if (original_leaf_count >= 6000 && !extended_timer.should_stop(22.0)) {
                const bool accepted = run_incumbent_union_repacking(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    extended_timer,
                    true,
                    "main.extended_early_incumbent_union_repacking",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    AgreementPackingMode::FinalGlobal,
                    extended_config.packing_focused,
                    component_archive_ptr,
                    original_leaf_count >= 12000 ? 18000 : 24000,
                    18.0
                );
                if (accepted) {
                    remember_diverse_packing_seed(
                        final_global_packing_seeds,
                        incumbent,
                        final_global_seed_cap
                    );
                }
            }
            if (!extended_timer.should_stop(25.0)) {
                run_reduced_global_candidate_portfolio(
                    incumbent,
                    instance.trees[0],
                    instance.trees[1],
                    extended_config,
                    extended_timer,
                    true,
                    [&](CoreForest candidate) {
                        const bool accepted = consider_candidate(
                            incumbent,
                            std::move(candidate),
                            original_core_tree,
                            original_core_tree2
                        );
                        if (accepted) {
                            publish_if_global_best(incumbent);
                        }
                        return accepted;
                    },
                    "main.extended_reduced_global",
                    &extended_global_packing_seeds,
                    extended_config.packing_focused,
                    component_archive_ptr
                );
            }

            if (use_large_final_global_repacking || use_small_exactification_tail) {
                for (const CoreForest& seed : extended_global_packing_seeds) {
                    remember_diverse_packing_seed(
                        final_global_packing_seeds,
                        seed,
                        final_global_seed_cap
                    );
                }
            }

            if (!extended_global_packing_seeds.empty() && !extended_timer.should_stop(12.0)) {
                const bool crossover_accepted = run_forest_crossover_portfolio(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    extended_global_packing_seeds,
                    extended_timer,
                    "main.extended_forest_crossover",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    12,
                    2,
                    component_archive_ptr,
                    false,
                    0
                );
                if (crossover_accepted &&
                    (use_large_final_global_repacking || use_small_exactification_tail)) {
                    remember_diverse_packing_seed(
                        final_global_packing_seeds,
                        incumbent,
                        final_global_seed_cap
                    );
                }
            }

            if (!extended_global_packing_seeds.empty() && !extended_timer.should_stop(20.0)) {
                run_agreement_component_packing(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    std::move(extended_global_packing_seeds),
                    extended_timer,
                    true,
                    "main.extended_global_repacking",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    AgreementPackingMode::Normal,
                    extended_config.packing_focused,
                    false,
                    {},
                    component_archive_ptr
                );
            }

            const double final_global_reserve =
                use_large_final_global_repacking
                    ? final_global_repacking_reserve_seconds(
                        original_leaf_count,
                        extended_timer,
                        extended_config.packing_focused
                    )
                    : 0.0;
            const bool reserve_final_global =
                use_large_final_global_repacking &&
                extended_timer.remaining_seconds() >= final_global_reserve + 25.0;
            const double extended_solve_deadline = reserve_final_global
                ? std::max(
                    extended_timer.elapsed_seconds() + 1.0,
                    extended_timer.limit_seconds() - final_global_reserve
                )
                : extended_timer.limit_seconds();

            g_profile.event_raw(
                "extended_solve_budget",
                &extended_timer,
                {
                    {"n", std::to_string(original_leaf_count)},
                    {"final_global_reserve_seconds", std::to_string(final_global_reserve)},
                    {"reserve_enabled", reserve_final_global ? "true" : "false"},
                    {"solve_deadline_seconds", std::to_string(extended_solve_deadline)}
                }
            );

            auto run_extended_solve_with_config = [&](
                Timer& solve_timer,
                const SolverConfig& solve_config
            ) {
                try {
                    CoreForest extended_candidate = solve_instance(
                        instance.trees[0],
                        instance.trees[1],
                        solve_config,
                        solve_timer,
                        rng,
                        0,
                        {},
                        component_archive_ptr,
                        &cluster_profile_memo
                    );

                    remember_diverse_packing_seed(
                        final_global_packing_seeds,
                        extended_candidate,
                        final_global_seed_cap
                    );
                    if (consider_candidate(
                            incumbent,
                            std::move(extended_candidate),
                            original_core_tree,
                            original_core_tree2)) {
                        publish_if_global_best(incumbent);
                    }
                } catch (const std::exception&) {
                    // The 30-second baseline remains the valid incumbent.
                }
                sync_incumbent_from_published();
            };

            auto run_extended_solve = [&](Timer& solve_timer) {
                run_extended_solve_with_config(solve_timer, extended_config);
            };

            if (reserve_final_global) {
                Timer extended_solve_timer(extended_solve_deadline, process_timer.start_time());
                if (!extended_solve_timer.should_stop(8.0)) {
                    run_extended_solve(extended_solve_timer);
                } else {
                    g_profile.event_raw(
                        "phase_skipped",
                        &extended_timer,
                        {
                            {"phase", json_quote("main.extended_solve_instance")},
                            {"n", std::to_string(original_leaf_count)},
                            {"reason", json_quote("reserved_for_final_global_repacking")}
                        }
                    );
                }
            } else if (!extended_timer.should_stop(8.0)) {
                run_extended_solve(extended_timer);
            } else {
                g_profile.event_raw(
                    "phase_skipped",
                    &extended_timer,
                    {
                        {"phase", json_quote("main.extended_solve_instance")},
                        {"n", std::to_string(original_leaf_count)},
                        {"reason", json_quote("timer_guard")}
                    }
                );
            }

            if (!final_global_packing_seeds.empty() && !extended_timer.should_stop(8.0)) {
                const bool crossover_accepted = run_forest_crossover_portfolio(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    final_global_packing_seeds,
                    extended_timer,
                    "main.final_forest_crossover",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    12,
                    2,
                    component_archive_ptr,
                    false,
                    0
                );
                if (crossover_accepted) {
                    remember_diverse_packing_seed(
                        final_global_packing_seeds,
                        incumbent,
                        final_global_seed_cap
                    );
                }
            }

            bool late_cluster_open_retry_done = false;
            auto run_late_cluster_open_retry = [&]() {
                if (late_cluster_open_retry_done ||
                    original_leaf_count <= 150 ||
                    original_leaf_count >= 5000 ||
                    extended_timer.elapsed_seconds() <
                        extended_config.cluster_open_profile_min_elapsed_seconds ||
                    extended_timer.should_stop(20.0)) {
                    return;
                }

                late_cluster_open_retry_done = true;
                SolverConfig late_open_config = extended_config;
                late_open_config.cluster_open_profile_min_elapsed_seconds = 0.0;
                g_profile.event_raw(
                    "late_cluster_open_retry_start",
                    &extended_timer,
                    {
                        {"n", std::to_string(original_leaf_count)},
                        {"components", std::to_string(incumbent.component_count())}
                    }
                );
                run_extended_solve_with_config(extended_timer, late_open_config);
            };

            run_late_cluster_open_retry();

            std::vector<CoreForest> reserved_tail_packing_seeds;
            if (original_leaf_count >= 2500) {
                reserved_tail_packing_seeds = final_global_packing_seeds;
            }

            if (use_large_final_global_repacking && !extended_timer.should_stop(3.0)) {
                run_agreement_component_packing(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    std::move(final_global_packing_seeds),
                    extended_timer,
                    true,
                    "main.final_extended_global_repacking",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    AgreementPackingMode::FinalGlobal,
                    extended_config.packing_focused,
                    false,
                    {},
                    component_archive_ptr
                );
            } else {
                g_profile.event_raw(
                    "phase_skipped",
                    &extended_timer,
                    {
                        {"phase", json_quote("main.final_extended_global_repacking")},
                        {"n", std::to_string(original_leaf_count)},
                        {"reason", json_quote(
                            use_large_final_global_repacking
                                ? "timer_guard"
                                : "size_guard_quality_recovery"
                        )}
                    }
                );
            }

            run_late_cluster_open_retry();

            if (original_leaf_count >= 6000 && !extended_timer.should_stop(12.0)) {
                const bool accepted = run_incumbent_union_repacking(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    extended_timer,
                    true,
                    "main.extended_late_incumbent_union_repacking",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    AgreementPackingMode::FinalGlobal,
                    extended_config.packing_focused,
                    component_archive_ptr,
                    original_leaf_count >= 12000 ? 22000 : 30000,
                    10.0
                );
                if (accepted) {
                    remember_diverse_packing_seed(
                        reserved_tail_packing_seeds,
                        incumbent,
                        std::max<std::size_t>(final_global_seed_cap, 12)
                    );
                }
            }

            if (original_leaf_count < 6000 && !extended_timer.should_stop(6.0)) {
                std::vector<CoreForest> ejection_seeds = reserved_tail_packing_seeds;
                remember_diverse_packing_seed(ejection_seeds, baseline_archive_forest, 12);
                remember_diverse_packing_seed(ejection_seeds, incumbent, 12);
                const bool accepted = run_singleton_ejection_repacking(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    std::move(ejection_seeds),
                    extended_timer,
                    true,
                    "main.extended_singleton_ejection_repacking",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    AgreementPackingMode::FinalGlobal,
                    extended_config.packing_focused,
                    component_archive_ptr,
                    original_leaf_count <= 220
                        ? 18000
                        : (original_leaf_count <= 700
                            ? 18000
                            : (original_leaf_count <= 2500 ? 22000 : 18000)),
                    original_leaf_count <= 700 ? 4.0 : 6.0,
                    true
                );
                if (accepted && original_leaf_count >= 2500) {
                    remember_diverse_packing_seed(
                        reserved_tail_packing_seeds,
                        incumbent,
                        std::max<std::size_t>(final_global_seed_cap, 12)
                    );
                }
            }

            if (original_leaf_count >= 5000 && !extended_timer.should_stop(18.0)) {
                std::vector<CoreForest> balanced_seed_bank =
                    run_large_balanced_seed_bank(
                        incumbent,
                        original_core_tree,
                        original_core_tree2,
                        extended_timer,
                        [&](const CoreForest& candidate) {
                            publish_if_global_best(candidate);
                        },
                        std::max<std::size_t>(final_global_seed_cap, 10),
                        component_archive_ptr
                    );

                for (const CoreForest& seed : balanced_seed_bank) {
                    remember_diverse_packing_seed(
                        reserved_tail_packing_seeds,
                        seed,
                        std::max<std::size_t>(final_global_seed_cap, 12)
                    );
                }

                if (!balanced_seed_bank.empty() && !extended_timer.should_stop(10.0)) {
                    const bool crossover_accepted =
                        run_forest_crossover_portfolio(
                            incumbent,
                            original_core_tree,
                            original_core_tree2,
                            balanced_seed_bank,
                            extended_timer,
                            "main.large_balanced_seed_crossover",
                            [&](const CoreForest& candidate) {
                                publish_if_global_best(candidate);
                            },
                            std::max<std::size_t>(final_global_seed_cap, 10),
                            2,
                            component_archive_ptr,
                            false,
                            0
                        );
                    if (crossover_accepted) {
                        publish_if_global_best(incumbent);
                        remember_diverse_packing_seed(
                            reserved_tail_packing_seeds,
                            incumbent,
                            std::max<std::size_t>(final_global_seed_cap, 12)
                        );
                    }
                }

                if (!balanced_seed_bank.empty() && !extended_timer.should_stop(8.0)) {
                    std::vector<CoreForest> balanced_repack_seeds =
                        reserved_tail_packing_seeds;
                    for (const CoreForest& seed : balanced_seed_bank) {
                        remember_diverse_packing_seed(
                            balanced_repack_seeds,
                            seed,
                            std::max<std::size_t>(final_global_seed_cap, 12)
                        );
                    }
                    remember_diverse_packing_seed(
                        balanced_repack_seeds,
                        incumbent,
                        std::max<std::size_t>(final_global_seed_cap, 12)
                    );
                    const bool accepted = run_agreement_component_packing(
                        incumbent,
                        original_core_tree,
                        original_core_tree2,
                        std::move(balanced_repack_seeds),
                        extended_timer,
                        true,
                        "main.large_balanced_seed_repacking",
                        [&](const CoreForest& candidate) {
                            publish_if_global_best(candidate);
                        },
                        AgreementPackingMode::FinalGlobal,
                        extended_config.packing_focused,
                        false,
                        {},
                        component_archive_ptr
                    );
                    if (accepted) {
                        remember_diverse_packing_seed(
                            reserved_tail_packing_seeds,
                            incumbent,
                            std::max<std::size_t>(final_global_seed_cap, 12)
                        );
                    }
                }

                if (!balanced_seed_bank.empty() && !extended_timer.should_stop(10.0)) {
                    std::vector<CoreForest> large_ejection_seeds =
                        reserved_tail_packing_seeds;
                    for (const CoreForest& seed : balanced_seed_bank) {
                        remember_diverse_packing_seed(
                            large_ejection_seeds,
                            seed,
                            std::max<std::size_t>(final_global_seed_cap, 12)
                        );
                    }
                    remember_diverse_packing_seed(
                        large_ejection_seeds,
                        incumbent,
                        std::max<std::size_t>(final_global_seed_cap, 12)
                    );
                    const bool accepted = run_singleton_ejection_repacking(
                        incumbent,
                        original_core_tree,
                        original_core_tree2,
                        std::move(large_ejection_seeds),
                        extended_timer,
                        true,
                        "main.large_singleton_ejection_repacking",
                        [&](const CoreForest& candidate) {
                            publish_if_global_best(candidate);
                        },
                        AgreementPackingMode::FinalGlobal,
                        extended_config.packing_focused,
                        component_archive_ptr,
                        original_leaf_count >= 12000 ? 24000 : 30000,
                        10.0,
                        true
                    );
                    if (accepted) {
                        remember_diverse_packing_seed(
                            reserved_tail_packing_seeds,
                            incumbent,
                            std::max<std::size_t>(final_global_seed_cap, 12)
                        );
                    }
                }
            }

            if (use_small_exactification_tail && !extended_timer.should_stop(8.0)) {
                const std::size_t alternate_state_limit =
                    small_alternate_exactification_state_limit(original_leaf_count);
                const std::size_t alternate_archive_limit =
                    alternate_state_limit == 0
                        ? 0
                        : std::max<std::size_t>(8, alternate_state_limit + 4);
                std::vector<CoreForest> alternate_exactification_states;
                auto remember_alternate_exactification_state =
                    [&](const CoreForest& state) {
                        if (alternate_archive_limit > 0) {
                            remember_diverse_packing_seed(
                                alternate_exactification_states,
                                state,
                                alternate_archive_limit
                            );
                        }
                    };

                remember_alternate_exactification_state(baseline_archive_forest);
                for (const CoreForest& seed : final_global_packing_seeds) {
                    remember_alternate_exactification_state(seed);
                }
                remember_alternate_exactification_state(incumbent);
                enrich_small_alternate_exactification_archive(
                    alternate_exactification_states,
                    alternate_archive_limit,
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    original_leaf_count,
                    extended_timer
                );

                if (original_leaf_count <= 220) {
                    CoreForest archive_tail_candidate = baseline_archive_forest;
                    std::vector<CoreForest> archive_tail_seeds;
                    remember_diverse_packing_seed(
                        archive_tail_seeds,
                        baseline_archive_forest,
                        8
                    );
                    run_agreement_component_packing(
                        archive_tail_candidate,
                        original_core_tree,
                        original_core_tree2,
                        std::move(archive_tail_seeds),
                        extended_timer,
                        true,
                        "main.small_exactification_baseline_archive_tail",
                        [&](const CoreForest& candidate) {
                            publish_if_global_best(candidate);
                        },
                        AgreementPackingMode::FinalGlobal,
                        extended_config.packing_focused,
                        false,
                        {},
                        component_archive_ptr
                    );
                    archive_tail_candidate.normalize();
                    incumbent.normalize();
                    if (archive_tail_candidate.component_count() < incumbent.component_count() &&
                        consider_candidate(
                            incumbent,
                            std::move(archive_tail_candidate),
                            original_core_tree,
                            original_core_tree2)) {
                        publish_if_global_best(incumbent);
                    }
                    remember_alternate_exactification_state(incumbent);
                }

                if (!extended_timer.should_stop(8.0)) {
                    std::vector<CoreForest> small_tail_seeds = final_global_packing_seeds;
                    remember_diverse_packing_seed(small_tail_seeds, incumbent, 8);
                    run_agreement_component_packing(
                        incumbent,
                        original_core_tree,
                        original_core_tree2,
                        std::move(small_tail_seeds),
                        extended_timer,
                        true,
                        "main.small_exactification_packing_tail",
                        [&](const CoreForest& candidate) {
                            publish_if_global_best(candidate);
                        },
                        AgreementPackingMode::FinalGlobal,
                        extended_config.packing_focused,
                        false,
                        {},
                        component_archive_ptr
                    );
                    remember_alternate_exactification_state(incumbent);
                }

                if (alternate_state_limit > 0 && !extended_timer.should_stop(6.0)) {
                    std::vector<CoreForest> alternate_states =
                        alternate_exactification_states;
                    select_diverse_packing_archive(
                        alternate_states,
                        alternate_archive_limit
                    );

                    std::size_t alternate_attempts = 0;
                    std::size_t alternate_skipped = 0;
                    bool alternate_strict_improved = false;
                    std::size_t supplement_candidates = 0;
                    const std::size_t component_slack =
                        small_alternate_exactification_extra_slack(original_leaf_count);
                    const double per_state_budget =
                        small_alternate_exactification_local_budget(original_leaf_count);

                    for (const CoreForest& state : alternate_states) {
                        if (alternate_attempts >= alternate_state_limit ||
                            extended_timer.should_stop(4.0)) {
                            break;
                        }

                        CoreForest normalized_state = state;
                        normalized_state.normalize();
                        CoreForest normalized_incumbent = incumbent;
                        normalized_incumbent.normalize();

                        if (same_packing_seed_forest(
                                normalized_state,
                                normalized_incumbent)) {
                            ++alternate_skipped;
                            continue;
                        }

                        if (normalized_state.component_count() >
                            normalized_incumbent.component_count() + component_slack) {
                            ++alternate_skipped;
                            continue;
                        }

                        const double state_deadline = std::min(
                            extended_timer.limit_seconds(),
                            extended_timer.elapsed_seconds() + per_state_budget
                        );
                        Timer state_timer(state_deadline, process_timer.start_time());
                        if (state_timer.should_stop(1.5)) {
                            ++alternate_skipped;
                            continue;
                        }

                        CoreForest alternate_candidate = normalized_state;
                        std::vector<CoreForest> alternate_seeds =
                            alternate_exactification_states;
                        remember_diverse_packing_seed(
                            alternate_seeds,
                            normalized_incumbent,
                            8
                        );
                        remember_diverse_packing_seed(
                            alternate_seeds,
                            baseline_archive_forest,
                            8
                        );
                        remember_diverse_packing_seed(
                            alternate_seeds,
                            normalized_state,
                            8
                        );

                        ++alternate_attempts;
                        run_agreement_component_packing(
                            alternate_candidate,
                            original_core_tree,
                            original_core_tree2,
                            std::move(alternate_seeds),
                            state_timer,
                            true,
                            "main.small_exactification_alternate_state_tail",
                            std::function<void(const CoreForest&)>(),
                            AgreementPackingMode::FinalGlobal,
                            extended_config.packing_focused,
                            false,
                            {},
                            component_archive_ptr
                        );

                        alternate_candidate.normalize();
                        incumbent.normalize();
                        if (alternate_candidate.component_count() <
                                incumbent.component_count() &&
                            consider_candidate(
                                incumbent,
                                std::move(alternate_candidate),
                                original_core_tree,
                                original_core_tree2)) {
                            publish_if_global_best(incumbent);
                            remember_alternate_exactification_state(incumbent);
                            alternate_strict_improved = true;
                        }
                    }

                    const std::size_t supplement_cap =
                        small_alternate_exactification_supplement_cap(original_leaf_count);
                    if (!alternate_strict_improved &&
                        supplement_cap > 0 &&
                        !extended_timer.should_stop(5.0)) {
                        std::vector<CoreForest> supplement_source_seeds;
                        remember_diverse_packing_seed(
                            supplement_source_seeds,
                            baseline_archive_forest,
                            8
                        );
                        remember_diverse_packing_seed(
                            supplement_source_seeds,
                            incumbent,
                            8
                        );
                        std::vector<std::vector<std::uint32_t>> supplement_components =
                            generate_singleton_rescue_components(
                                incumbent,
                                supplement_source_seeds,
                                original_core_tree,
                                original_core_tree2,
                                extended_timer,
                                true,
                                AgreementPackingMode::FinalGlobal,
                                extended_config.packing_focused
                            );
                        if (supplement_components.size() > supplement_cap) {
                            supplement_components.resize(supplement_cap);
                        }
                        supplement_candidates = supplement_components.size();

                        if (!supplement_components.empty() &&
                            !extended_timer.should_stop(4.0)) {
                            CoreForest supplement_candidate = incumbent;
                            run_agreement_component_packing(
                                supplement_candidate,
                                original_core_tree,
                                original_core_tree2,
                                std::move(supplement_source_seeds),
                                extended_timer,
                                true,
                                "main.small_exactification_alternate_state_supplement_tail",
                                std::function<void(const CoreForest&)>(),
                                AgreementPackingMode::FinalGlobal,
                                extended_config.packing_focused,
                                true,
                                std::move(supplement_components),
                                component_archive_ptr
                            );

                            supplement_candidate.normalize();
                            incumbent.normalize();
                            if (supplement_candidate.component_count() <
                                    incumbent.component_count() &&
                                consider_candidate(
                                    incumbent,
                                    std::move(supplement_candidate),
                                    original_core_tree,
                                    original_core_tree2)) {
                                publish_if_global_best(incumbent);
                                remember_alternate_exactification_state(incumbent);
                            }
                        }
                    }

                    if (g_profile.enabled()) {
                        g_profile.event_raw(
                            "small_exactification_alternate_state_summary",
                            &extended_timer,
                            {
                                {"states", std::to_string(alternate_states.size())},
                                {"attempted", std::to_string(alternate_attempts)},
                                {"skipped", std::to_string(alternate_skipped)},
                                {"state_limit", std::to_string(alternate_state_limit)},
                                {"component_slack", std::to_string(component_slack)},
                                {"per_state_budget_seconds", std::to_string(per_state_budget)},
                                {"supplement_candidates", std::to_string(supplement_candidates)},
                                {"components", std::to_string(incumbent.component_count())}
                            }
                        );
                    }
                }
            } else if (use_small_exactification_tail) {
                g_profile.event_raw(
                    "phase_skipped",
                    &extended_timer,
                    {
                        {"phase", json_quote("main.small_exactification_packing_tail")},
                        {"n", std::to_string(original_leaf_count)},
                        {"reason", json_quote("timer_guard")}
                    }
                );
            }

            if (original_leaf_count >= 6000 && !extended_timer.should_stop(8.0)) {
                std::vector<CoreForest> tail_seeds = reserved_tail_packing_seeds;
                remember_diverse_packing_seed(
                    tail_seeds,
                    incumbent,
                    std::max<std::size_t>(final_global_seed_cap, 12)
                );
                const bool accepted = run_agreement_component_packing(
                    incumbent,
                    original_core_tree,
                    original_core_tree2,
                    std::move(tail_seeds),
                    extended_timer,
                    true,
                    "main.final_reserved_bucket_repacking_reserved_tail_5_cycle_0",
                    [&](const CoreForest& candidate) {
                        publish_if_global_best(candidate);
                    },
                    AgreementPackingMode::ReservedTail5,
                    extended_config.packing_focused,
                    false,
                    {},
                    component_archive_ptr
                );
                if (accepted) {
                    remember_diverse_packing_seed(
                        reserved_tail_packing_seeds,
                        incumbent,
                        std::max<std::size_t>(final_global_seed_cap, 12)
                    );
                }
            } else if (original_leaf_count >= 2500) {
                g_profile.event_raw(
                    "phase_skipped",
                    &extended_timer,
                    {
                        {"phase", json_quote("main.final_reserved_bucket_repacking")},
                        {"n", std::to_string(original_leaf_count)},
                        {"reason", json_quote(
                            original_leaf_count < 6000
                                ? "disabled_low_roi_medium_tail"
                                : "timer_guard"
                        )}
                    }
                );
            }

            if (original_leaf_count >= 1000) {
                g_profile.event_raw(
                    "phase_skipped",
                    &extended_timer,
                    {
                        {"phase", json_quote("main.extended_exact_window_repair")},
                        {"n", std::to_string(original_leaf_count)},
                        {"reason", json_quote("disabled_low_roi")}
                    }
                );
            }

            g_profile.event_raw(
                "extended_stage_complete",
                &extended_timer,
                {
                    {"components", std::to_string(incumbent.component_count())},
                    {"valid", "true"}
                }
            );
        }

        sync_incumbent_from_published();
        incumbent.normalize();
        incumbent.validate_partition_of(original_core_tree.leaf_labels);
        publish_if_global_best(incumbent);

        g_profile.event_raw(
            "main_final",
            &process_timer,
            {
                {"components", std::to_string(incumbent.component_count())},
                {"valid", "true"},
                {"used_warm_incumbent", warm_forest.has_value() ? "true" : "false"},
                {"ran_extended_stage", can_run_extended_stage ? "true" : "false"},
                {"archive_enabled", enable_component_archive ? "true" : "false"},
                {"archive_forests", std::to_string(
                    component_archive_ptr == nullptr ? 0 : component_archive_ptr->forest_count())},
                {"archive_components", std::to_string(
                    component_archive_ptr == nullptr ? 0 : component_archive_ptr->component_count())}
            }
        );
        g_profile.close(&process_timer);

        cancel_emergency_alarm();

        write_forest(original_tree_for_output, std::move(incumbent), std::cout);

        return 0;
    } catch (const std::exception& e) {
        g_profile.event_raw(
            "main_exception",
            nullptr,
            {{"message", json_quote(e.what())}}
        );
        g_profile.close(nullptr);
        cancel_emergency_alarm();

        // After a successful parse we have a valid singleton fallback.
        // Print it instead of returning blank/invalid output.
        if (!g_emergency_output.empty()) {
            std::cout << g_emergency_output;
            return 0;
        }

        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
