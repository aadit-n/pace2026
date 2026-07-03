#pragma once

#include "core/Forest.hpp"
#include "core/Timer.hpp"
#include "core/Tree.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pace26::heuristics {

class AgreementComponentPacking {
private:
    using Tree = pace26::core::Tree;
    using LabelComponent = pace26::core::LabelComponent;
    using LabelForest = pace26::core::LabelForest;
    using Timer = pace26::core::Timer;
    using Clock = std::chrono::steady_clock;

public:
    struct Stats {
        std::size_t seed_forests = 0;
        std::size_t seed_components = 0;
        std::size_t seed_groups = 0;
        std::size_t extra_candidate_components = 0;
        std::size_t extra_candidate_added = 0;
        std::size_t archived_candidate_components = 0;

        std::size_t add_attempts = 0;
        std::size_t added = 0;
        std::size_t added_from_seed = 0;
        std::size_t duplicate = 0;
        std::size_t singleton_reject = 0;
        std::size_t capacity_reject = 0;
        std::size_t size_quota_reject = 0;
        std::size_t topology_reject = 0;
        std::size_t unknown_label_reject = 0;

        std::array<std::size_t, 8> added_by_size{};
        std::array<std::size_t, 8> generated_added_by_size{};
        std::array<std::size_t, 8> capacity_reject_by_size{};
        std::array<std::size_t, 8> size_quota_reject_by_size{};
        std::array<std::size_t, 8> selected_by_size{};

        std::size_t candidates_after_seeds = 0;
        std::size_t candidates_after_pairs = 0;
        std::size_t candidates_after_extensions = 0;
        std::size_t candidates_after_structure_growth = 0;
        std::size_t structure_growth_attempts = 0;
        std::size_t structure_growth_added = 0;
        std::size_t seed_intersection_attempts = 0;
        std::size_t seed_intersection_added = 0;

        std::size_t greedy_variants = 0;
        std::size_t exchange_rounds = 0;
        std::size_t deficit_exchange_tests = 0;
        std::size_t deficit_exchange_attempts = 0;
        std::size_t deficit_exchange_accepted = 0;
        std::size_t deficit_exchange_rolled_back = 0;
        std::size_t local_mwis_runs = 0;
        std::size_t local_mwis_accepted = 0;
        std::size_t local_mwis_pool_total = 0;
        std::size_t local_mwis_largest_pool = 0;
        std::size_t local_mwis_nodes = 0;
        bool exact_ran = false;
        std::size_t exact_pool_size = 0;
        std::array<std::size_t, 8> exact_pool_by_size{};
        std::size_t exact_nodes = 0;
        std::size_t exact_conflict_edges = 0;
        std::size_t exact_components = 0;
        std::size_t exact_largest_component = 0;
        std::size_t exact_forced = 0;
        std::size_t exact_dominated = 0;

        std::size_t final_selected = 0;
        std::size_t final_uncovered_leaves = 0;
        int final_gain = 0;
        bool stopped = false;
    };

    struct Options {
        std::size_t max_candidates = 100000;
        std::size_t max_seed_candidates = std::numeric_limits<std::size_t>::max();
        std::size_t max_generated_candidates = 0;
        bool separate_seed_generated_budgets = false;
        std::size_t order_window = 10;
        std::size_t random_partners_per_leaf = 2;
        std::size_t extension_neighbors = 4;
        std::size_t all_pairs_max_leaves = 180;
        std::size_t max_generated_component_size = 4;
        std::array<std::size_t, 8> generated_bucket_capacity{};
        std::size_t structural_subtree_max_size = 0;
        std::size_t structural_subtree_node_budget = 0;
        std::size_t structure_growth_max_component_size = 0;
        std::size_t structure_growth_neighbors = 0;
        std::size_t structure_growth_branching = 0;
        std::size_t structure_growth_candidates_per_size = 0;
        std::size_t seed_intersection_candidate_limit = 0;
        std::size_t exchange_rounds = 3;
        std::size_t two_exchange_pool = 160;
        int max_refill_deficit = 0;
        std::size_t deficit_exchange_rounds = 0;
        std::size_t deficit_exchange_pool = 0;
        std::size_t deficit_refill_pool = 0;
        std::size_t local_mwis_anchors = 0;
        std::size_t local_mwis_candidate_limit = 0;
        std::size_t local_mwis_node_limit = 0;
        std::size_t exact_max_leaves = 180;
        std::size_t exact_candidate_limit = 150;
        std::size_t exact_selected_candidate_limit = std::numeric_limits<std::size_t>::max();
        std::array<std::size_t, 8> exact_pool_size_quota{};
        std::size_t exact_node_limit = std::numeric_limits<std::size_t>::max();
        double local_time_limit_seconds = 3.0;
        double guard_seconds = 1.0;
        bool enable_exact = true;
        std::size_t max_extra_candidate_components =
            std::numeric_limits<std::size_t>::max();
        std::vector<std::vector<std::uint32_t>> extra_candidate_components;
        std::function<void(const std::vector<std::uint32_t>&)>
            archive_component;
        std::size_t max_archived_candidate_components = 0;
        std::size_t min_archived_candidate_size = 3;
        Stats* stats = nullptr;
    };

    AgreementComponentPacking() : AgreementComponentPacking(Options{}) {}

    explicit AgreementComponentPacking(Options options)
        : options_(std::move(options)) {}

    LabelForest solve(
        const Tree& t1,
        const Tree& t2,
        const std::vector<LabelForest>& seed_forests,
        const Timer* timer = nullptr
    ) {
        if (t1.leaf_labels != t2.leaf_labels) {
            throw std::runtime_error("agreement packing received different leaf sets");
        }

        t1_ = &t1;
        t2_ = &t2;
        timer_ = timer;
        stats_ = options_.stats;
        if (stats_ != nullptr) {
            *stats_ = Stats{};
        }
        started_ = Clock::now();
        deadline_ = started_ + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(std::max(0.05, options_.local_time_limit_seconds))
        );

        initialize_labels();
        candidates_.clear();
        candidate_of_labels_.clear();
        seed_groups_.clear();
        seed_forests_.clear();
        generated_by_size_.fill(0);
        generated_bucket_indices_.fill({});
        generated_total_ = 0;
        seed_total_ = 0;
        archived_candidate_components_ = 0;
        initialize_signature_scratch();

        const std::size_t reserve_capacity = candidate_capacity_hint();
        candidate_of_labels_.reserve(reserve_capacity * 2 + 1);
        candidates_.reserve(std::min<std::size_t>(reserve_capacity, 700000));

        for (const LabelForest& seed : seed_forests) {
            add_seed_forest(seed);
        }
        if (!should_stop() && !options_.extra_candidate_components.empty()) {
            add_extra_candidate_components();
        }
        if (!should_stop() && options_.seed_intersection_candidate_limit > 0) {
            generate_seed_intersections();
        }
        if (stats_ != nullptr) {
            stats_->candidates_after_seeds = candidates_.size();
        }

        generate_pairs();
        if (stats_ != nullptr) {
            stats_->candidates_after_pairs = candidates_.size();
            stats_->stopped = should_stop();
        }
        if (!should_stop() && options_.max_generated_component_size >= 3) {
            generate_small_extensions();
        }
        if (stats_ != nullptr) {
            stats_->candidates_after_extensions = candidates_.size();
            stats_->stopped = stats_->stopped || should_stop();
        }
        if (!should_stop() && (
                options_.structural_subtree_max_size >= 3 ||
                options_.structure_growth_max_component_size >= 3)) {
            generate_structure_driven_growth();
        }
        if (stats_ != nullptr) {
            stats_->candidates_after_structure_growth = candidates_.size();
            stats_->stopped = stats_->stopped || should_stop();
        }

        PackingState best = best_seed_state();
        run_greedy_portfolio(best);
        run_exchange_search(best);
        run_deficit_refill_search(best);
        run_local_mwis_repair(best);

        if (options_.enable_exact &&
            labels_.size() <= options_.exact_max_leaves &&
            !should_stop()) {
            run_bounded_exact(best);
        }

        return forest_from_state(best);
    }

private:
    struct VectorHash {
        std::size_t operator()(const std::vector<std::uint32_t>& values) const noexcept {
            std::uint64_t h = 0x9e3779b97f4a7c15ULL;
            for (std::uint32_t x : values) {
                h ^= mix64(static_cast<std::uint64_t>(x) + h);
            }
            return static_cast<std::size_t>(mix64(h ^ values.size()));
        }
    };

    struct Candidate {
        std::vector<std::uint32_t> labels;
        std::vector<int> leaf_ids;
        std::vector<int> edges1;
        std::vector<int> edges2;
        int gain = 0;
        int edge_cost = 0;
        bool from_seed = false;
        std::uint64_t tie = 0;
    };

    struct PackingState {
        std::vector<char> selected;
        std::vector<int> leaf_owner;
        std::vector<int> edge_owner1;
        std::vector<int> edge_owner2;
        int gain = 0;
        std::uint64_t tie_score = 0;
    };

    Options options_;
    const Tree* t1_ = nullptr;
    const Tree* t2_ = nullptr;
    const Timer* timer_ = nullptr;
    Stats* stats_ = nullptr;
    Clock::time_point started_;
    Clock::time_point deadline_;

    std::vector<std::uint32_t> labels_;
    std::unordered_map<std::uint32_t, int> label_id_;
    std::vector<int> label_id_flat_;
    std::vector<Candidate> candidates_;
    std::unordered_map<std::vector<std::uint32_t>, int, VectorHash> candidate_of_labels_;
    std::vector<std::vector<int>> seed_groups_;
    std::vector<LabelForest> seed_forests_;
    std::array<std::size_t, 8> generated_by_size_{};
    std::array<std::vector<int>, 8> generated_bucket_indices_;
    std::size_t generated_total_ = 0;
    std::size_t seed_total_ = 0;
    std::size_t archived_candidate_components_ = 0;
    mutable std::vector<std::uint64_t> signature_hash_;
    mutable std::vector<int> signature_left_child_;
    mutable std::vector<int> signature_right_child_;
    mutable std::vector<int> signature_leaf_nodes_;
    mutable std::vector<int> signature_nodes_;
    mutable std::vector<int> signature_stack_;

    static std::uint64_t mix64(std::uint64_t x) noexcept {
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

    static std::size_t size_bucket(std::size_t size) noexcept {
        if (size <= 4) {
            return size;
        }
        if (size <= 8) {
            return 5;
        }
        if (size <= 16) {
            return 6;
        }
        return 7;
    }

    std::size_t effective_generated_capacity() const noexcept {
        return !options_.separate_seed_generated_budgets ||
               options_.max_generated_candidates == 0
            ? options_.max_candidates
            : options_.max_generated_candidates;
    }

    std::size_t candidate_capacity_hint() const noexcept {
        if (!options_.separate_seed_generated_budgets) {
            return std::max<std::size_t>(1024, options_.max_candidates);
        }

        const std::size_t seed_capacity =
            options_.max_seed_candidates == std::numeric_limits<std::size_t>::max()
                ? std::min<std::size_t>(options_.max_candidates, labels_.empty() ? 1024 : labels_.size() * 4)
                : options_.max_seed_candidates;
        return std::max<std::size_t>(
            1024,
            std::min<std::size_t>(
                800000,
                seed_capacity + effective_generated_capacity()
            )
        );
    }

    bool generated_bucket_full(std::size_t size) const noexcept {
        if (options_.separate_seed_generated_budgets &&
            generated_total_ >= effective_generated_capacity()) {
            return true;
        }
        const std::size_t bucket = size_bucket(size);
        const std::size_t quota = options_.generated_bucket_capacity[bucket];
        return quota != 0 && generated_by_size_[bucket] >= quota;
    }

    bool should_stop() const {
        if (Clock::now() >= deadline_) {
            return true;
        }
        return timer_ != nullptr && timer_->should_stop(options_.guard_seconds);
    }

    void initialize_labels() {
        labels_ = t1_->leaf_labels;
        label_id_.clear();
        label_id_.reserve(labels_.size() * 2 + 1);
        label_id_flat_.clear();

        std::uint32_t max_label = 0;
        for (std::uint32_t label : labels_) {
            max_label = std::max(max_label, label);
        }
        const std::size_t flat_limit = std::max<std::size_t>(
            1000000,
            labels_.size() * 8 + 1024
        );
        if (static_cast<std::uint64_t>(max_label) <= flat_limit) {
            label_id_flat_.assign(static_cast<std::size_t>(max_label) + 1, -1);
        }

        for (std::size_t i = 0; i < labels_.size(); ++i) {
            label_id_.emplace(labels_[i], static_cast<int>(i));
            if (!label_id_flat_.empty()) {
                label_id_flat_[static_cast<std::size_t>(labels_[i])] =
                    static_cast<int>(i);
            }
        }
    }

    void initialize_signature_scratch() {
        const std::size_t scratch_nodes = static_cast<std::size_t>(
            std::max(t1_->node_count(), t2_->node_count())
        );
        signature_hash_.assign(scratch_nodes, 0);
        signature_left_child_.assign(scratch_nodes, -1);
        signature_right_child_.assign(scratch_nodes, -1);
        signature_leaf_nodes_.reserve(labels_.size());
        signature_nodes_.reserve(labels_.size() * 2);
        signature_stack_.reserve(labels_.size());
    }

    int label_index(std::uint32_t label) const {
        if (!label_id_flat_.empty() &&
            label < label_id_flat_.size()) {
            return label_id_flat_[static_cast<std::size_t>(label)];
        }

        auto it = label_id_.find(label);
        if (it == label_id_.end()) {
            return -1;
        }

        return it->second;
    }

    static std::vector<std::uint32_t> leaf_order(const Tree& tree) {
        std::vector<int> leaf_nodes;
        leaf_nodes.reserve(tree.leaf_labels.size());
        for (std::uint32_t label : tree.leaf_labels) {
            leaf_nodes.push_back(tree.node_of_label(label));
        }

        std::sort(leaf_nodes.begin(), leaf_nodes.end(), [&](int a, int b) {
            return tree.tin[static_cast<std::size_t>(a)] <
                   tree.tin[static_cast<std::size_t>(b)];
        });

        std::vector<std::uint32_t> order;
        order.reserve(leaf_nodes.size());
        for (int node : leaf_nodes) {
            order.push_back(tree.nodes[static_cast<std::size_t>(node)].label);
        }

        return order;
    }

    static std::vector<int> induced_edges(
        const Tree& tree,
        const std::vector<std::uint32_t>& labels
    ) {
        std::vector<int> edges;
        if (labels.size() <= 1) {
            return edges;
        }

        int root = tree.node_of_label(labels.front());
        for (std::size_t i = 1; i < labels.size(); ++i) {
            root = tree.lca(root, tree.node_of_label(labels[i]));
        }

        for (std::uint32_t label : labels) {
            int u = tree.node_of_label(label);
            while (u != root) {
                edges.push_back(u);
                u = tree.parent[static_cast<std::size_t>(u)];
            }
        }

        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        return edges;
    }

    std::uint64_t restricted_signature(
        const Tree& tree,
        const std::vector<std::uint32_t>& labels
    ) const {
        if (labels.empty()) {
            return 0;
        }
        if (labels.size() == 1) {
            return mix64(0x51ed270bULL ^ labels.front());
        }

        signature_leaf_nodes_.clear();
        for (std::uint32_t label : labels) {
            signature_leaf_nodes_.push_back(tree.node_of_label(label));
        }
        std::sort(signature_leaf_nodes_.begin(), signature_leaf_nodes_.end(), [&](int a, int b) {
            return tree.tin[static_cast<std::size_t>(a)] <
                   tree.tin[static_cast<std::size_t>(b)];
        });

        signature_nodes_ = signature_leaf_nodes_;
        for (std::size_t i = 0; i + 1 < signature_leaf_nodes_.size(); ++i) {
            signature_nodes_.push_back(
                tree.lca(signature_leaf_nodes_[i], signature_leaf_nodes_[i + 1])
            );
        }

        std::sort(signature_nodes_.begin(), signature_nodes_.end(), [&](int a, int b) {
            return tree.tin[static_cast<std::size_t>(a)] <
                   tree.tin[static_cast<std::size_t>(b)];
        });
        signature_nodes_.erase(
            std::unique(signature_nodes_.begin(), signature_nodes_.end()),
            signature_nodes_.end()
        );

        signature_stack_.clear();
        for (int u : signature_nodes_) {
            while (!signature_stack_.empty() && !tree.is_ancestor(signature_stack_.back(), u)) {
                signature_stack_.pop_back();
            }
            if (!signature_stack_.empty()) {
                const int parent = signature_stack_.back();
                int& left = signature_left_child_[static_cast<std::size_t>(parent)];
                int& right = signature_right_child_[static_cast<std::size_t>(parent)];
                if (left == -1) {
                    left = u;
                } else {
                    right = u;
                }
            }
            signature_stack_.push_back(u);
        }

        for (int i = static_cast<int>(signature_nodes_.size()) - 1; i >= 0; --i) {
            const int u = signature_nodes_[static_cast<std::size_t>(i)];
            if (tree.is_leaf(u)) {
                signature_hash_[static_cast<std::size_t>(u)] =
                    mix64(0x51ed270bULL ^ tree.label(u));
                continue;
            }

            const int left = signature_left_child_[static_cast<std::size_t>(u)];
            const int right = signature_right_child_[static_cast<std::size_t>(u)];
            std::uint64_t a = left == -1
                ? 0
                : signature_hash_[static_cast<std::size_t>(left)];
            std::uint64_t b = right == -1
                ? 0
                : signature_hash_[static_cast<std::size_t>(right)];

            if (a == 0 && b == 0) {
                signature_hash_[static_cast<std::size_t>(u)] = 0;
            } else if (a == 0) {
                signature_hash_[static_cast<std::size_t>(u)] = b;
            } else if (b == 0) {
                signature_hash_[static_cast<std::size_t>(u)] = a;
            } else {
                if (b < a) {
                    std::swap(a, b);
                }
                signature_hash_[static_cast<std::size_t>(u)] =
                    mix64(0xa0761d6478bd642fULL ^ a ^ mix64(b));
            }
        }

        const std::uint64_t result =
            signature_hash_[static_cast<std::size_t>(signature_nodes_.front())];

        for (int u : signature_nodes_) {
            signature_hash_[static_cast<std::size_t>(u)] = 0;
            signature_left_child_[static_cast<std::size_t>(u)] = -1;
            signature_right_child_[static_cast<std::size_t>(u)] = -1;
        }

        return result;
    }

    bool topology_compatible(const std::vector<std::uint32_t>& labels) const {
        if (labels.size() <= 2) {
            return true;
        }
        const std::uint64_t s1 = restricted_signature(*t1_, labels);
        const std::uint64_t s2 = restricted_signature(*t2_, labels);
        return s1 != 0 && s1 == s2;
    }

    int add_candidate(
        std::vector<std::uint32_t> labels,
        bool from_seed,
        bool check_topology
    ) {
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (stats_ != nullptr) {
            ++stats_->add_attempts;
        }
        if (labels.size() <= 1) {
            if (stats_ != nullptr) {
                ++stats_->singleton_reject;
            }
            return -1;
        }

        auto existing = candidate_of_labels_.find(labels);
        if (existing != candidate_of_labels_.end()) {
            if (stats_ != nullptr) {
                ++stats_->duplicate;
            }
            if (from_seed) {
                candidates_[static_cast<std::size_t>(existing->second)].from_seed = true;
            }
            return existing->second;
        }

        if (options_.separate_seed_generated_budgets &&
            from_seed &&
            seed_total_ >= options_.max_seed_candidates) {
            if (stats_ != nullptr) {
                ++stats_->capacity_reject;
                ++stats_->capacity_reject_by_size[size_bucket(labels.size())];
            }
            return -1;
        }
        if (!from_seed) {
            const bool capacity_full = options_.separate_seed_generated_budgets
                ? generated_total_ >= effective_generated_capacity()
                : candidates_.size() >= options_.max_candidates;
            if (capacity_full) {
                if (stats_ != nullptr) {
                    ++stats_->capacity_reject;
                    ++stats_->capacity_reject_by_size[size_bucket(labels.size())];
                }
                return -1;
            }
            if (generated_bucket_full(labels.size())) {
                if (stats_ != nullptr) {
                    ++stats_->size_quota_reject;
                    ++stats_->size_quota_reject_by_size[size_bucket(labels.size())];
                }
                return -1;
            }
        }
        if (check_topology && !topology_compatible(labels)) {
            if (stats_ != nullptr) {
                ++stats_->topology_reject;
            }
            return -1;
        }

        Candidate candidate;
        candidate.labels = std::move(labels);
        candidate.from_seed = from_seed;
        candidate.gain = static_cast<int>(candidate.labels.size()) - 1;
        candidate.leaf_ids.reserve(candidate.labels.size());
        for (std::uint32_t label : candidate.labels) {
            const int id = label_index(label);
            if (id < 0) {
                if (stats_ != nullptr) {
                    ++stats_->unknown_label_reject;
                }
                return -1;
            }
            candidate.leaf_ids.push_back(id);
        }
        candidate.edges1 = induced_edges(*t1_, candidate.labels);
        candidate.edges2 = induced_edges(*t2_, candidate.labels);
        candidate.edge_cost = static_cast<int>(candidate.edges1.size() + candidate.edges2.size());

        std::uint64_t tie = 0x6a09e667f3bcc909ULL;
        for (std::uint32_t label : candidate.labels) {
            tie = mix64(tie ^ label);
        }
        candidate.tie = tie;

        const int index = static_cast<int>(candidates_.size());
        candidate_of_labels_.emplace(candidate.labels, index);
        candidates_.push_back(std::move(candidate));
        maybe_archive_candidate_component(candidates_.back().labels);
        if (stats_ != nullptr) {
            ++stats_->added;
            if (from_seed) {
                ++stats_->added_from_seed;
            }
            ++stats_->added_by_size[size_bucket(candidates_.back().labels.size())];
        }
        if (!from_seed) {
            const std::size_t bucket = size_bucket(candidates_.back().labels.size());
            ++generated_total_;
            ++generated_by_size_[bucket];
            generated_bucket_indices_[bucket].push_back(index);
            if (stats_ != nullptr) {
                ++stats_->generated_added_by_size[bucket];
            }
        } else {
            ++seed_total_;
        }
        return index;
    }

    void maybe_archive_candidate_component(
        const std::vector<std::uint32_t>& labels
    ) {
        if (!options_.archive_component ||
            options_.max_archived_candidate_components == 0 ||
            archived_candidate_components_ >=
                options_.max_archived_candidate_components ||
            labels.size() < options_.min_archived_candidate_size) {
            return;
        }

        options_.archive_component(labels);
        ++archived_candidate_components_;
        if (stats_ != nullptr) {
            ++stats_->archived_candidate_components;
        }
    }

    void add_seed_forest(const LabelForest& forest) {
        if (stats_ != nullptr) {
            ++stats_->seed_forests;
            stats_->seed_components += forest.components.size();
        }
        LabelForest normalized = forest;
        normalized.normalize();
        seed_forests_.push_back(normalized);
        std::vector<int> group;
        group.reserve(normalized.components.size());
        for (const LabelComponent& component : normalized.components) {
            const int index = add_candidate(component.labels, true, false);
            if (index >= 0) {
                group.push_back(index);
            }
        }
        if (!group.empty()) {
            seed_groups_.push_back(std::move(group));
            if (stats_ != nullptr) {
                ++stats_->seed_groups;
            }
        }
    }

    void add_extra_candidate_components() {
        const std::size_t limit = std::min(
            options_.max_extra_candidate_components,
            options_.extra_candidate_components.size()
        );
        for (std::size_t i = 0; i < limit && !should_stop(); ++i) {
            if (stats_ != nullptr) {
                ++stats_->extra_candidate_components;
            }
            const std::size_t before = candidates_.size();
            const int index = add_candidate(
                options_.extra_candidate_components[i],
                true,
                true
            );
            if (stats_ != nullptr && index >= 0 && candidates_.size() > before) {
                ++stats_->extra_candidate_added;
            }
        }
    }

    void generate_seed_intersections() {
        if (seed_forests_.size() < 2 || labels_.empty()) {
            return;
        }

        std::size_t added = 0;
        std::vector<int> component_a(labels_.size(), -1);
        std::vector<int> component_b(labels_.size(), -1);

        for (std::size_t ai = 0;
             ai + 1 < seed_forests_.size() &&
             added < options_.seed_intersection_candidate_limit &&
             !should_stop();
             ++ai) {
            std::fill(component_a.begin(), component_a.end(), -1);
            const LabelForest& a = seed_forests_[ai];
            for (std::size_t cid = 0; cid < a.components.size(); ++cid) {
                for (std::uint32_t label : a.components[cid].labels) {
                    const int id = label_index(label);
                    if (id >= 0) {
                        component_a[static_cast<std::size_t>(id)] =
                            static_cast<int>(cid);
                    }
                }
            }

            for (std::size_t bi = ai + 1;
                 bi < seed_forests_.size() &&
                 added < options_.seed_intersection_candidate_limit &&
                 !should_stop();
                 ++bi) {
                std::fill(component_b.begin(), component_b.end(), -1);
                const LabelForest& b = seed_forests_[bi];
                for (std::size_t cid = 0; cid < b.components.size(); ++cid) {
                    for (std::uint32_t label : b.components[cid].labels) {
                        const int id = label_index(label);
                        if (id >= 0) {
                            component_b[static_cast<std::size_t>(id)] =
                                static_cast<int>(cid);
                        }
                    }
                }

                std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> groups;
                groups.reserve(labels_.size() * 2 + 1);
                for (std::size_t leaf = 0; leaf < labels_.size(); ++leaf) {
                    if (component_a[leaf] < 0 || component_b[leaf] < 0) {
                        continue;
                    }
                    const std::uint64_t key =
                        (static_cast<std::uint64_t>(
                             static_cast<std::uint32_t>(component_a[leaf])) << 32) ^
                        static_cast<std::uint32_t>(component_b[leaf]);
                    groups[key].push_back(labels_[leaf]);
                }

                for (auto& entry : groups) {
                    if (added >= options_.seed_intersection_candidate_limit ||
                        should_stop()) {
                        break;
                    }
                    if (entry.second.size() < 3) {
                        continue;
                    }
                    if (stats_ != nullptr) {
                        ++stats_->seed_intersection_attempts;
                    }
                    const std::size_t before = candidates_.size();
                    const int index = add_candidate(std::move(entry.second), false, true);
                    if (index >= 0 && candidates_.size() > before) {
                        ++added;
                        if (stats_ != nullptr) {
                            ++stats_->seed_intersection_added;
                        }
                    }
                }
            }
        }
    }

    void add_order_pairs(const std::vector<std::uint32_t>& order) {
        const std::size_t n = order.size();
        for (std::size_t i = 0; i < n && !should_stop(); ++i) {
            const std::size_t end = std::min(n, i + options_.order_window + 1);
            for (std::size_t j = i + 1; j < end; ++j) {
                add_candidate({order[i], order[j]}, false, false);
            }
        }
    }

    void add_cherry_pairs(const Tree& tree) {
        for (const auto& node : tree.nodes) {
            if (node.is_leaf()) {
                continue;
            }
            const auto& left = tree.nodes[static_cast<std::size_t>(node.left)];
            const auto& right = tree.nodes[static_cast<std::size_t>(node.right)];
            if (left.is_leaf() && right.is_leaf()) {
                add_candidate({left.label, right.label}, false, false);
            }
        }
    }

    void generate_pairs() {
        const std::size_t n = labels_.size();
        if (n <= options_.all_pairs_max_leaves) {
            for (std::size_t i = 0; i < n && !should_stop(); ++i) {
                for (std::size_t j = i + 1; j < n; ++j) {
                    add_candidate({labels_[i], labels_[j]}, false, false);
                }
            }
        } else {
            add_order_pairs(leaf_order(*t1_));
            add_order_pairs(leaf_order(*t2_));
        }

        add_cherry_pairs(*t1_);
        add_cherry_pairs(*t2_);

        for (std::size_t i = 0; i < n && !should_stop(); ++i) {
            for (std::size_t k = 0; k < options_.random_partners_per_leaf; ++k) {
                const std::uint64_t key = mix64(
                    static_cast<std::uint64_t>(labels_[i]) ^
                    (0x9e3779b97f4a7c15ULL * (k + 1))
                );
                std::size_t j = static_cast<std::size_t>(key % n);
                if (j == i) {
                    j = (j + 1) % n;
                }
                add_candidate({labels_[i], labels_[j]}, false, false);
            }
        }
    }

    void insert_neighbor(
        std::vector<std::pair<int, std::uint32_t>>& neighbors,
        int cost,
        std::uint32_t label,
        std::size_t limit
    ) const {
        if (limit == 0) {
            return;
        }
        for (const auto& entry : neighbors) {
            if (entry.second == label) {
                return;
            }
        }
        neighbors.push_back({cost, label});
        std::sort(neighbors.begin(), neighbors.end());
        if (neighbors.size() > limit) {
            neighbors.pop_back();
        }
    }

    std::vector<std::vector<std::pair<int, std::uint32_t>>> pair_neighbor_lists(
        std::size_t limit
    ) const {
        std::vector<std::vector<std::pair<int, std::uint32_t>>> neighbors(labels_.size());
        if (limit == 0) {
            return neighbors;
        }

        const std::size_t count = candidates_.size();
        for (std::size_t i = 0; i < count; ++i) {
            const Candidate& candidate = candidates_[i];
            if (candidate.labels.size() != 2) {
                continue;
            }
            const int a = candidate.leaf_ids[0];
            const int b = candidate.leaf_ids[1];
            insert_neighbor(
                neighbors[static_cast<std::size_t>(a)],
                candidate.edge_cost,
                candidate.labels[1],
                limit
            );
            insert_neighbor(
                neighbors[static_cast<std::size_t>(b)],
                candidate.edge_cost,
                candidate.labels[0],
                limit
            );
        }
        return neighbors;
    }

    void generate_small_extensions() {
        std::vector<std::vector<std::pair<int, std::uint32_t>>> neighbors =
            pair_neighbor_lists(options_.extension_neighbors);

        for (std::size_t a = 0; a < labels_.size() && !should_stop(); ++a) {
            std::vector<std::uint32_t> adjacent;
            adjacent.reserve(neighbors[a].size());
            for (const auto& entry : neighbors[a]) {
                adjacent.push_back(entry.second);
            }

            for (std::size_t i = 0; i < adjacent.size(); ++i) {
                for (std::size_t j = i + 1; j < adjacent.size(); ++j) {
                    add_candidate({labels_[a], adjacent[i], adjacent[j]}, false, true);
                }
            }

            if (options_.max_generated_component_size < 4 || labels_.size() > 2500) {
                continue;
            }
            for (std::size_t i = 0; i < adjacent.size(); ++i) {
                for (std::size_t j = i + 1; j < adjacent.size(); ++j) {
                    for (std::size_t k = j + 1; k < adjacent.size(); ++k) {
                        add_candidate(
                            {labels_[a], adjacent[i], adjacent[j], adjacent[k]},
                            false,
                            true
                        );
                    }
                }
            }
        }
    }

    int add_structure_candidate(std::vector<std::uint32_t> labels) {
        const std::size_t before = generated_total_;
        if (stats_ != nullptr) {
            ++stats_->structure_growth_attempts;
        }
        const int index = add_candidate(std::move(labels), false, true);
        if (stats_ != nullptr && generated_total_ > before) {
            ++stats_->structure_growth_added;
        }
        return index;
    }

    void add_structural_subtrees_from_tree(
        const Tree& tree,
        std::size_t max_size,
        std::size_t node_budget
    ) {
        if (max_size < 3 || node_budget == 0) {
            return;
        }

        std::vector<int> nodes;
        nodes.reserve(std::min<std::size_t>(node_budget, tree.nodes.size()));
        for (int u : tree.postorder) {
            if (tree.is_leaf(u)) {
                continue;
            }
            const std::size_t leaves =
                tree.leaf_count_under[static_cast<std::size_t>(u)];
            if (leaves >= 3 && leaves <= max_size) {
                nodes.push_back(u);
            }
        }

        std::sort(nodes.begin(), nodes.end(), [&](int a, int b) {
            const std::size_t as =
                tree.leaf_count_under[static_cast<std::size_t>(a)];
            const std::size_t bs =
                tree.leaf_count_under[static_cast<std::size_t>(b)];
            if (as != bs) {
                return as > bs;
            }
            if (tree.depth[static_cast<std::size_t>(a)] !=
                tree.depth[static_cast<std::size_t>(b)]) {
                return tree.depth[static_cast<std::size_t>(a)] >
                       tree.depth[static_cast<std::size_t>(b)];
            }
            return tree.tin[static_cast<std::size_t>(a)] <
                   tree.tin[static_cast<std::size_t>(b)];
        });

        if (nodes.size() > node_budget) {
            nodes.resize(node_budget);
        }

        for (int u : nodes) {
            if (should_stop()) {
                break;
            }
            add_structure_candidate(tree.labels_under(u));
        }
    }

    void generate_neighbor_growth_candidates(
        std::size_t max_size,
        std::size_t neighbor_limit,
        std::size_t branching,
        std::size_t candidates_per_size
    ) {
        if (max_size < 3 || neighbor_limit == 0 || branching == 0 ||
            candidates_per_size == 0) {
            return;
        }

        std::vector<std::vector<std::pair<int, std::uint32_t>>> neighbors =
            pair_neighbor_lists(neighbor_limit);

        for (std::size_t current_size = 2;
             current_size < max_size && !should_stop();
             ++current_size) {
            if (generated_bucket_full(current_size + 1)) {
                continue;
            }

            const std::size_t snapshot = candidates_.size();
            std::size_t processed = 0;
            for (std::size_t i = 0;
                 i < snapshot && processed < candidates_per_size && !should_stop();
                 ++i) {
                if (candidates_[i].labels.size() != current_size) {
                    continue;
                }
                ++processed;

                // add_structure_candidate can reallocate candidates_, so retain
                // no references into it while growing this candidate.
                const std::vector<std::uint32_t> candidate_labels = candidates_[i].labels;
                const std::vector<int> candidate_leaf_ids = candidates_[i].leaf_ids;
                std::vector<std::pair<int, std::uint32_t>> choices;
                for (int leaf : candidate_leaf_ids) {
                    for (const auto& entry : neighbors[static_cast<std::size_t>(leaf)]) {
                        if (std::binary_search(
                                candidate_labels.begin(),
                                candidate_labels.end(),
                                entry.second)) {
                            continue;
                        }
                        choices.push_back(entry);
                    }
                }
                if (choices.empty()) {
                    continue;
                }

                std::sort(choices.begin(), choices.end(), [](const auto& a, const auto& b) {
                    if (a.second != b.second) {
                        return a.second < b.second;
                    }
                    return a.first < b.first;
                });

                std::vector<std::pair<int, std::uint32_t>> unique_choices;
                unique_choices.reserve(choices.size());
                for (const auto& entry : choices) {
                    if (!unique_choices.empty() &&
                        unique_choices.back().second == entry.second) {
                        unique_choices.back().first =
                            std::min(unique_choices.back().first, entry.first);
                    } else {
                        unique_choices.push_back(entry);
                    }
                }

                std::sort(
                    unique_choices.begin(),
                    unique_choices.end(),
                    [](const auto& a, const auto& b) {
                        if (a.first != b.first) {
                            return a.first < b.first;
                        }
                        return a.second < b.second;
                    }
                );

                const std::size_t take = std::min(branching, unique_choices.size());
                for (std::size_t j = 0; j < take; ++j) {
                    std::vector<std::uint32_t> grown = candidate_labels;
                    grown.push_back(unique_choices[j].second);
                    add_structure_candidate(std::move(grown));
                    if (generated_bucket_full(current_size + 1)) {
                        break;
                    }
                }
            }
        }
    }

    void generate_structure_driven_growth() {
        const std::size_t subtree_max = std::max(
            options_.structural_subtree_max_size,
            options_.max_generated_component_size
        );
        const std::size_t subtree_budget = options_.structural_subtree_node_budget;
        if (subtree_max >= 3 && subtree_budget > 0) {
            add_structural_subtrees_from_tree(*t1_, subtree_max, subtree_budget);
            if (!should_stop()) {
                add_structural_subtrees_from_tree(*t2_, subtree_max, subtree_budget);
            }
        }

        if (!should_stop()) {
            const std::size_t growth_max = std::max(
                options_.structure_growth_max_component_size,
                options_.max_generated_component_size
            );
            generate_neighbor_growth_candidates(
                growth_max,
                options_.structure_growth_neighbors,
                options_.structure_growth_branching,
                options_.structure_growth_candidates_per_size
            );
        }
    }

    PackingState empty_state() const {
        PackingState state;
        state.selected.assign(candidates_.size(), 0);
        state.leaf_owner.assign(labels_.size(), -1);
        state.edge_owner1.assign(static_cast<std::size_t>(t1_->node_count()), -1);
        state.edge_owner2.assign(static_cast<std::size_t>(t2_->node_count()), -1);
        return state;
    }

    bool can_add(const PackingState& state, int index) const {
        const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
        for (int leaf : candidate.leaf_ids) {
            if (state.leaf_owner[static_cast<std::size_t>(leaf)] != -1) return false;
        }
        for (int edge : candidate.edges1) {
            if (state.edge_owner1[static_cast<std::size_t>(edge)] != -1) return false;
        }
        for (int edge : candidate.edges2) {
            if (state.edge_owner2[static_cast<std::size_t>(edge)] != -1) return false;
        }
        return true;
    }

    bool add_to_state(PackingState& state, int index) const {
        if (state.selected[static_cast<std::size_t>(index)] || !can_add(state, index)) {
            return false;
        }
        const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
        state.selected[static_cast<std::size_t>(index)] = 1;
        for (int leaf : candidate.leaf_ids) state.leaf_owner[static_cast<std::size_t>(leaf)] = index;
        for (int edge : candidate.edges1) state.edge_owner1[static_cast<std::size_t>(edge)] = index;
        for (int edge : candidate.edges2) state.edge_owner2[static_cast<std::size_t>(edge)] = index;
        state.gain += candidate.gain;
        const std::uint64_t size = candidate.labels.size();
        state.tie_score += size * size;
        return true;
    }

    void remove_from_state(PackingState& state, int index) const {
        if (!state.selected[static_cast<std::size_t>(index)]) {
            return;
        }
        const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
        state.selected[static_cast<std::size_t>(index)] = 0;
        for (int leaf : candidate.leaf_ids) {
            if (state.leaf_owner[static_cast<std::size_t>(leaf)] == index) {
                state.leaf_owner[static_cast<std::size_t>(leaf)] = -1;
            }
        }
        for (int edge : candidate.edges1) {
            if (state.edge_owner1[static_cast<std::size_t>(edge)] == index) {
                state.edge_owner1[static_cast<std::size_t>(edge)] = -1;
            }
        }
        for (int edge : candidate.edges2) {
            if (state.edge_owner2[static_cast<std::size_t>(edge)] == index) {
                state.edge_owner2[static_cast<std::size_t>(edge)] = -1;
            }
        }
        state.gain -= candidate.gain;
        const std::uint64_t size = candidate.labels.size();
        state.tie_score -= size * size;
    }

    static bool better_state(const PackingState& a, const PackingState& b) {
        return a.gain > b.gain || (a.gain == b.gain && a.tie_score > b.tie_score);
    }

    PackingState best_seed_state() const {
        PackingState best = empty_state();
        for (const std::vector<int>& group : seed_groups_) {
            PackingState state = empty_state();
            for (int index : group) {
                add_to_state(state, index);
            }
            if (better_state(state, best)) {
                best = std::move(state);
            }
        }
        return best;
    }

    std::vector<int> candidate_order(int variant) const {
        std::vector<int> order(candidates_.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int ai, int bi) {
            const Candidate& a = candidates_[static_cast<std::size_t>(ai)];
            const Candidate& b = candidates_[static_cast<std::size_t>(bi)];
            if (variant == 0) {
                if (a.gain != b.gain) return a.gain > b.gain;
                if (a.edge_cost != b.edge_cost) return a.edge_cost < b.edge_cost;
            } else if (variant == 1) {
                const long long ad = static_cast<long long>(a.gain) * (b.edge_cost + 1);
                const long long bd = static_cast<long long>(b.gain) * (a.edge_cost + 1);
                if (ad != bd) return ad > bd;
                if (a.gain != b.gain) return a.gain > b.gain;
            } else if (variant == 2) {
                if (a.from_seed != b.from_seed) return a.from_seed;
                if (a.edge_cost != b.edge_cost) return a.edge_cost < b.edge_cost;
                if (a.gain != b.gain) return a.gain > b.gain;
            } else {
                if (a.edge_cost != b.edge_cost) return a.edge_cost < b.edge_cost;
                if (a.gain != b.gain) return a.gain > b.gain;
            }
            return a.tie < b.tie;
        });
        return order;
    }

    void run_greedy_portfolio(PackingState& best) const {
        for (int variant = 0; variant < 4 && !should_stop(); ++variant) {
            if (stats_ != nullptr) {
                ++stats_->greedy_variants;
            }
            PackingState state = empty_state();
            const std::vector<int> order = candidate_order(variant);
            for (int index : order) {
                add_to_state(state, index);
            }
            if (better_state(state, best)) {
                best = std::move(state);
            }
        }
    }

    std::vector<int> conflicts_with(const PackingState& state, int index) const {
        std::vector<int> conflicts;
        const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
        auto remember = [&](int owner) {
            if (owner < 0 || owner == index) return;
            if (std::find(conflicts.begin(), conflicts.end(), owner) == conflicts.end()) {
                conflicts.push_back(owner);
            }
        };
        for (int leaf : candidate.leaf_ids) remember(state.leaf_owner[static_cast<std::size_t>(leaf)]);
        for (int edge : candidate.edges1) remember(state.edge_owner1[static_cast<std::size_t>(edge)]);
        for (int edge : candidate.edges2) remember(state.edge_owner2[static_cast<std::size_t>(edge)]);
        return conflicts;
    }

    bool candidates_disjoint(int ai, int bi) const {
        const Candidate& a = candidates_[static_cast<std::size_t>(ai)];
        const Candidate& b = candidates_[static_cast<std::size_t>(bi)];
        auto intersects = [](const std::vector<int>& x, const std::vector<int>& y) {
            std::size_t i = 0;
            std::size_t j = 0;
            while (i < x.size() && j < y.size()) {
                if (x[i] == y[j]) return true;
                if (x[i] < y[j]) ++i; else ++j;
            }
            return false;
        };
        return !intersects(a.leaf_ids, b.leaf_ids) &&
               !intersects(a.edges1, b.edges1) &&
               !intersects(a.edges2, b.edges2);
    }

    struct PackingTransaction {
        std::vector<int> removed;
        std::vector<int> added;
        int original_gain = 0;
        std::uint64_t original_tie = 0;
    };

    void rollback_transaction(PackingState& state, const PackingTransaction& tx) const {
        for (auto it = tx.added.rbegin(); it != tx.added.rend(); ++it) {
            remove_from_state(state, *it);
        }
        for (int index : tx.removed) {
            add_to_state(state, index);
        }
        state.gain = tx.original_gain;
        state.tie_score = tx.original_tie;
    }

    void released_resources(
        const std::vector<int>& removed,
        std::vector<char>& released_leaf,
        std::vector<char>& released_edge1,
        std::vector<char>& released_edge2
    ) const {
        released_leaf.assign(labels_.size(), 0);
        released_edge1.assign(static_cast<std::size_t>(t1_->node_count()), 0);
        released_edge2.assign(static_cast<std::size_t>(t2_->node_count()), 0);
        for (int index : removed) {
            const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
            for (int leaf : candidate.leaf_ids) {
                released_leaf[static_cast<std::size_t>(leaf)] = 1;
            }
            for (int edge : candidate.edges1) {
                released_edge1[static_cast<std::size_t>(edge)] = 1;
            }
            for (int edge : candidate.edges2) {
                released_edge2[static_cast<std::size_t>(edge)] = 1;
            }
        }
    }

    bool touches_released(
        int index,
        const std::vector<char>& released_leaf,
        const std::vector<char>& released_edge1,
        const std::vector<char>& released_edge2
    ) const {
        const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
        for (int leaf : candidate.leaf_ids) {
            if (released_leaf[static_cast<std::size_t>(leaf)]) {
                return true;
            }
        }
        for (int edge : candidate.edges1) {
            if (released_edge1[static_cast<std::size_t>(edge)]) {
                return true;
            }
        }
        for (int edge : candidate.edges2) {
            if (released_edge2[static_cast<std::size_t>(edge)]) {
                return true;
            }
        }
        return false;
    }

    bool conflicts_only_with_removed_or_free(
        const PackingState& state,
        int index,
        const std::vector<char>& removed_marker
    ) const {
        const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
        auto owner_allowed = [&](int owner) {
            if (owner < 0 || owner == index) {
                return true;
            }
            return removed_marker[static_cast<std::size_t>(owner)] != 0;
        };
        for (int leaf : candidate.leaf_ids) {
            if (!owner_allowed(state.leaf_owner[static_cast<std::size_t>(leaf)])) {
                return false;
            }
        }
        for (int edge : candidate.edges1) {
            if (!owner_allowed(state.edge_owner1[static_cast<std::size_t>(edge)])) {
                return false;
            }
        }
        for (int edge : candidate.edges2) {
            if (!owner_allowed(state.edge_owner2[static_cast<std::size_t>(edge)])) {
                return false;
            }
        }
        return true;
    }

    bool try_single_exchange(PackingState& state, int index) const {
        if (state.selected[static_cast<std::size_t>(index)]) {
            return false;
        }
        std::vector<int> conflicts = conflicts_with(state, index);
        int removed_gain = 0;
        for (int conflict : conflicts) {
            removed_gain += candidates_[static_cast<std::size_t>(conflict)].gain;
        }
        if (candidates_[static_cast<std::size_t>(index)].gain <= removed_gain) {
            return false;
        }
        for (int conflict : conflicts) remove_from_state(state, conflict);
        return add_to_state(state, index);
    }

    bool try_pair_exchange(PackingState& state, int a, int b) const {
        if (a == b || state.selected[static_cast<std::size_t>(a)] ||
            state.selected[static_cast<std::size_t>(b)] || !candidates_disjoint(a, b)) {
            return false;
        }
        std::vector<int> conflicts = conflicts_with(state, a);
        for (int x : conflicts_with(state, b)) {
            if (std::find(conflicts.begin(), conflicts.end(), x) == conflicts.end()) {
                conflicts.push_back(x);
            }
        }
        int removed_gain = 0;
        for (int conflict : conflicts) removed_gain += candidates_[static_cast<std::size_t>(conflict)].gain;
        const int added_gain = candidates_[static_cast<std::size_t>(a)].gain +
                               candidates_[static_cast<std::size_t>(b)].gain;
        if (added_gain <= removed_gain) {
            return false;
        }
        for (int conflict : conflicts) remove_from_state(state, conflict);
        const bool first = add_to_state(state, a);
        const bool second = add_to_state(state, b);
        return first && second;
    }

    bool try_deficit_refill_exchange(
        PackingState& state,
        int index,
        const std::vector<int>& refill_order
    ) const {
        if (options_.max_refill_deficit <= 0 ||
            state.selected[static_cast<std::size_t>(index)]) {
            return false;
        }
        if (stats_ != nullptr) {
            ++stats_->deficit_exchange_tests;
        }

        std::vector<int> conflicts = conflicts_with(state, index);
        if (conflicts.empty()) {
            return false;
        }

        int removed_gain = 0;
        for (int conflict : conflicts) {
            removed_gain += candidates_[static_cast<std::size_t>(conflict)].gain;
        }
        const int added_gain = candidates_[static_cast<std::size_t>(index)].gain;
        const int deficit = removed_gain - added_gain;
        if (deficit < 0 || deficit > options_.max_refill_deficit) {
            return false;
        }
        if (stats_ != nullptr) {
            ++stats_->deficit_exchange_attempts;
        }

        PackingTransaction tx;
        tx.removed = std::move(conflicts);
        tx.original_gain = state.gain;
        tx.original_tie = state.tie_score;

        std::vector<char> released_leaf;
        std::vector<char> released_edge1;
        std::vector<char> released_edge2;
        released_resources(tx.removed, released_leaf, released_edge1, released_edge2);

        for (int conflict : tx.removed) {
            remove_from_state(state, conflict);
        }
        if (!add_to_state(state, index)) {
            rollback_transaction(state, tx);
            if (stats_ != nullptr) {
                ++stats_->deficit_exchange_rolled_back;
            }
            return false;
        }
        tx.added.push_back(index);

        for (int refill : refill_order) {
            if (should_stop()) {
                break;
            }
            if (state.selected[static_cast<std::size_t>(refill)] ||
                refill == index ||
                !touches_released(refill, released_leaf, released_edge1, released_edge2)) {
                continue;
            }
            if (add_to_state(state, refill)) {
                tx.added.push_back(refill);
                if (state.gain > tx.original_gain) {
                    break;
                }
            }
        }

        if (state.gain > tx.original_gain) {
            if (stats_ != nullptr) {
                ++stats_->deficit_exchange_accepted;
            }
            return true;
        }

        rollback_transaction(state, tx);
        if (stats_ != nullptr) {
            ++stats_->deficit_exchange_rolled_back;
        }
        return false;
    }

    void fill_uncontested(PackingState& state, const std::vector<int>& order) const {
        for (int index : order) {
            if (!state.selected[static_cast<std::size_t>(index)]) {
                add_to_state(state, index);
            }
        }
    }

    void run_exchange_search(PackingState& best) const {
        const std::vector<int> order = candidate_order(1);
        std::vector<int> pair_pool = order;
        if (pair_pool.size() > options_.two_exchange_pool) {
            pair_pool.resize(options_.two_exchange_pool);
        }

        for (std::size_t round = 0; round < options_.exchange_rounds && !should_stop(); ++round) {
            if (stats_ != nullptr) {
                ++stats_->exchange_rounds;
            }
            bool changed = false;
            for (int index : order) {
                if (try_single_exchange(best, index)) {
                    changed = true;
                }
            }
            fill_uncontested(best, order);

            for (std::size_t i = 0; i < pair_pool.size() && !should_stop(); ++i) {
                for (std::size_t j = i + 1; j < pair_pool.size(); ++j) {
                    if (try_pair_exchange(best, pair_pool[i], pair_pool[j])) {
                        changed = true;
                        break;
                    }
                }
            }
            fill_uncontested(best, order);
            if (!changed) {
                break;
            }
        }
    }

    void run_deficit_refill_search(PackingState& best) const {
        if (options_.max_refill_deficit <= 0 ||
            options_.deficit_exchange_rounds == 0 ||
            options_.deficit_exchange_pool == 0 ||
            options_.deficit_refill_pool == 0 ||
            should_stop()) {
            return;
        }

        std::vector<int> order = candidate_order(1);
        std::vector<int> anchor_pool = order;
        if (anchor_pool.size() > options_.deficit_exchange_pool) {
            anchor_pool.resize(options_.deficit_exchange_pool);
        }

        std::vector<int> refill_order = std::move(order);
        if (refill_order.size() > options_.deficit_refill_pool) {
            refill_order.resize(options_.deficit_refill_pool);
        }

        for (std::size_t round = 0;
             round < options_.deficit_exchange_rounds && !should_stop();
             ++round) {
            bool changed = false;
            for (int index : anchor_pool) {
                if (try_deficit_refill_exchange(best, index, refill_order)) {
                    changed = true;
                }
                if (should_stop()) {
                    break;
                }
            }
            fill_uncontested(best, refill_order);
            if (!changed) {
                break;
            }
        }
    }

    struct MwisSolution {
        int gain = 0;
        std::uint64_t tie = 0;
        std::vector<int> chosen_positions;
    };

    static bool better_mwis_solution(
        const MwisSolution& a,
        const MwisSolution& b
    ) {
        return a.gain > b.gain || (a.gain == b.gain && a.tie > b.tie);
    }

    int exact_weight(int pool_position, const std::vector<int>& pool) const {
        return candidates_[static_cast<std::size_t>(
            pool[static_cast<std::size_t>(pool_position)]
        )].gain;
    }

    std::uint64_t exact_tie(int pool_position, const std::vector<int>& pool) const {
        const Candidate& candidate = candidates_[static_cast<std::size_t>(
            pool[static_cast<std::size_t>(pool_position)]
        )];
        const std::uint64_t size = candidate.labels.size();
        return size * size;
    }

    bool exact_position_conflicts(
        int a,
        int b,
        const std::vector<std::vector<char>>& conflict
    ) const {
        return conflict[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)] != 0;
    }

    bool exact_chosen_compatible(
        int position,
        const std::vector<int>& chosen,
        const std::vector<std::vector<char>>& conflict
    ) const {
        for (int other : chosen) {
            if (exact_position_conflicts(position, other, conflict)) {
                return false;
            }
        }
        return true;
    }

    MwisSolution exact_seed_solution(
        const std::vector<int>& component,
        const std::vector<int>& pool,
        const std::vector<std::vector<char>>& conflict,
        const PackingState& incumbent
    ) const {
        MwisSolution best_seed;
        for (int position : component) {
            const int candidate_index = pool[static_cast<std::size_t>(position)];
            if (!incumbent.selected[static_cast<std::size_t>(candidate_index)] ||
                !exact_chosen_compatible(position, best_seed.chosen_positions, conflict)) {
                continue;
            }
            best_seed.chosen_positions.push_back(position);
            best_seed.gain += exact_weight(position, pool);
            best_seed.tie += exact_tie(position, pool);
        }

        MwisSolution greedy;
        std::vector<int> order = component;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const int wa = exact_weight(a, pool);
            const int wb = exact_weight(b, pool);
            if (wa != wb) return wa > wb;
            const Candidate& ca = candidates_[static_cast<std::size_t>(
                pool[static_cast<std::size_t>(a)]
            )];
            const Candidate& cb = candidates_[static_cast<std::size_t>(
                pool[static_cast<std::size_t>(b)]
            )];
            if (ca.edge_cost != cb.edge_cost) return ca.edge_cost < cb.edge_cost;
            return ca.tie < cb.tie;
        });

        for (int position : order) {
            if (!exact_chosen_compatible(position, greedy.chosen_positions, conflict)) {
                continue;
            }
            greedy.chosen_positions.push_back(position);
            greedy.gain += exact_weight(position, pool);
            greedy.tie += exact_tie(position, pool);
        }

        if (better_mwis_solution(greedy, best_seed)) {
            best_seed = std::move(greedy);
        }
        return best_seed;
    }

    int exact_clique_cover_bound(
        const std::vector<int>& active,
        const std::vector<int>& pool,
        const std::vector<std::vector<char>>& conflict
    ) const {
        std::vector<int> order = active;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            const int wa = exact_weight(a, pool);
            const int wb = exact_weight(b, pool);
            if (wa != wb) return wa > wb;
            return a < b;
        });

        std::vector<std::vector<int>> cliques;
        std::vector<int> clique_max_weight;
        for (int position : order) {
            bool placed = false;
            for (std::size_t c = 0; c < cliques.size(); ++c) {
                bool fits = true;
                for (int other : cliques[c]) {
                    if (!exact_position_conflicts(position, other, conflict)) {
                        fits = false;
                        break;
                    }
                }
                if (!fits) {
                    continue;
                }
                cliques[c].push_back(position);
                clique_max_weight[c] = std::max(
                    clique_max_weight[c],
                    exact_weight(position, pool)
                );
                placed = true;
                break;
            }
            if (!placed) {
                cliques.push_back({position});
                clique_max_weight.push_back(exact_weight(position, pool));
            }
        }

        int bound = 0;
        for (int weight : clique_max_weight) {
            bound += weight;
        }
        return bound;
    }

    int exact_branch_position(
        const std::vector<int>& active,
        const std::vector<int>& pool,
        const std::vector<std::vector<char>>& conflict
    ) const {
        int best_position = active.front();
        long long best_score = std::numeric_limits<long long>::min();

        for (int position : active) {
            int degree = 0;
            int weighted_degree = 0;
            for (int other : active) {
                if (other == position ||
                    !exact_position_conflicts(position, other, conflict)) {
                    continue;
                }
                ++degree;
                weighted_degree += exact_weight(other, pool);
            }
            const long long score =
                static_cast<long long>(exact_weight(position, pool)) * (degree + 1) +
                static_cast<long long>(weighted_degree) * 2 +
                static_cast<long long>(degree) * 3;
            if (score > best_score ||
                (score == best_score &&
                 exact_weight(position, pool) > exact_weight(best_position, pool))) {
                best_score = score;
                best_position = position;
            }
        }

        return best_position;
    }

    void exact_mwis_dfs(
        const std::vector<int>& active,
        const std::vector<int>& pool,
        const std::vector<std::vector<char>>& conflict,
        int current_gain,
        std::uint64_t current_tie,
        std::vector<int>& current_chosen,
        MwisSolution& best,
        std::size_t& nodes,
        std::size_t node_limit
    ) const {
        if (nodes >= node_limit || should_stop()) {
            return;
        }
        if (current_gain + exact_clique_cover_bound(active, pool, conflict) < best.gain) {
            return;
        }

        ++nodes;
        if (active.empty()) {
            MwisSolution candidate;
            candidate.gain = current_gain;
            candidate.tie = current_tie;
            candidate.chosen_positions = current_chosen;
            if (better_mwis_solution(candidate, best)) {
                best = std::move(candidate);
            }
            return;
        }

        const int branch = exact_branch_position(active, pool, conflict);

        std::vector<int> include_active;
        include_active.reserve(active.size());
        for (int position : active) {
            if (position != branch &&
                !exact_position_conflicts(branch, position, conflict)) {
                include_active.push_back(position);
            }
        }
        current_chosen.push_back(branch);
        exact_mwis_dfs(
            include_active,
            pool,
            conflict,
            current_gain + exact_weight(branch, pool),
            current_tie + exact_tie(branch, pool),
            current_chosen,
            best,
            nodes,
            node_limit
        );
        current_chosen.pop_back();

        std::vector<int> exclude_active;
        exclude_active.reserve(active.size() - 1);
        for (int position : active) {
            if (position != branch) {
                exclude_active.push_back(position);
            }
        }
        exact_mwis_dfs(
            exclude_active,
            pool,
            conflict,
            current_gain,
            current_tie,
            current_chosen,
            best,
            nodes,
            node_limit
        );
    }

    std::vector<std::vector<char>> build_conflict_matrix(
        const std::vector<int>& pool,
        std::size_t* conflict_edges = nullptr
    ) const {
        const std::size_t p = pool.size();
        std::vector<std::vector<char>> conflict(p, std::vector<char>(p, 0));
        std::size_t edges = 0;
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = i + 1; j < p; ++j) {
                if (!candidates_disjoint(pool[i], pool[j])) {
                    conflict[i][j] = 1;
                    conflict[j][i] = 1;
                    ++edges;
                }
            }
        }
        if (conflict_edges != nullptr) {
            *conflict_edges = edges;
        }
        return conflict;
    }

    bool try_local_mwis_anchor(
        PackingState& state,
        int anchor,
        const std::vector<int>& order
    ) const {
        if (options_.local_mwis_candidate_limit == 0 ||
            options_.local_mwis_node_limit == 0 ||
            state.selected[static_cast<std::size_t>(anchor)]) {
            return false;
        }

        std::vector<int> removed = conflicts_with(state, anchor);
        if (removed.empty()) {
            return false;
        }

        std::vector<char> removed_marker(candidates_.size(), 0);
        int removed_gain = 0;
        for (int index : removed) {
            removed_marker[static_cast<std::size_t>(index)] = 1;
            removed_gain += candidates_[static_cast<std::size_t>(index)].gain;
        }

        std::vector<char> released_leaf;
        std::vector<char> released_edge1;
        std::vector<char> released_edge2;
        released_resources(removed, released_leaf, released_edge1, released_edge2);

        std::vector<int> pool;
        pool.reserve(options_.local_mwis_candidate_limit);
        std::vector<char> included(candidates_.size(), 0);
        auto try_add_pool = [&](int index) {
            const std::size_t i = static_cast<std::size_t>(index);
            if (i >= included.size() || included[i] ||
                pool.size() >= options_.local_mwis_candidate_limit) {
                return false;
            }
            if (!conflicts_only_with_removed_or_free(state, index, removed_marker)) {
                return false;
            }
            pool.push_back(index);
            included[i] = 1;
            return true;
        };

        try_add_pool(anchor);
        for (int index : removed) {
            try_add_pool(index);
        }

        for (int index : order) {
            if (pool.size() >= options_.local_mwis_candidate_limit || should_stop()) {
                break;
            }
            if (state.selected[static_cast<std::size_t>(index)] &&
                !removed_marker[static_cast<std::size_t>(index)]) {
                continue;
            }
            if (!touches_released(index, released_leaf, released_edge1, released_edge2)) {
                continue;
            }
            try_add_pool(index);
        }

        for (std::size_t index = 0;
             index < candidates_.size() &&
             pool.size() < options_.local_mwis_candidate_limit &&
             !should_stop();
             ++index) {
            if (!candidates_[index].from_seed ||
                included[index] ||
                !touches_released(
                    static_cast<int>(index),
                    released_leaf,
                    released_edge1,
                    released_edge2)) {
                continue;
            }
            try_add_pool(static_cast<int>(index));
        }

        if (pool.size() <= removed.size()) {
            return false;
        }

        if (stats_ != nullptr) {
            ++stats_->local_mwis_runs;
            stats_->local_mwis_pool_total += pool.size();
            stats_->local_mwis_largest_pool =
                std::max(stats_->local_mwis_largest_pool, pool.size());
        }

        std::vector<std::vector<char>> conflict = build_conflict_matrix(pool);
        std::vector<int> component(pool.size());
        std::iota(component.begin(), component.end(), 0);

        MwisSolution local_best =
            exact_seed_solution(component, pool, conflict, state);
        std::vector<int> current_chosen;
        std::size_t nodes = 0;
        exact_mwis_dfs(
            component,
            pool,
            conflict,
            0,
            0,
            current_chosen,
            local_best,
            nodes,
            options_.local_mwis_node_limit
        );
        if (stats_ != nullptr) {
            stats_->local_mwis_nodes += nodes;
        }

        if (local_best.gain <= removed_gain) {
            return false;
        }

        PackingTransaction tx;
        tx.removed = std::move(removed);
        tx.original_gain = state.gain;
        tx.original_tie = state.tie_score;
        for (int index : tx.removed) {
            remove_from_state(state, index);
        }

        for (int position : local_best.chosen_positions) {
            const int index = pool[static_cast<std::size_t>(position)];
            if (!add_to_state(state, index)) {
                rollback_transaction(state, tx);
                return false;
            }
            tx.added.push_back(index);
        }

        if (state.gain > tx.original_gain) {
            if (stats_ != nullptr) {
                ++stats_->local_mwis_accepted;
            }
            return true;
        }

        rollback_transaction(state, tx);
        return false;
    }

    void run_local_mwis_repair(PackingState& best) const {
        if (options_.local_mwis_anchors == 0 ||
            options_.local_mwis_candidate_limit == 0 ||
            options_.local_mwis_node_limit == 0 ||
            should_stop()) {
            return;
        }

        const std::vector<int> order = candidate_order(1);
        std::size_t anchors_seen = 0;
        for (int anchor : order) {
            if (anchors_seen >= options_.local_mwis_anchors || should_stop()) {
                break;
            }
            if (best.selected[static_cast<std::size_t>(anchor)]) {
                continue;
            }
            std::vector<int> conflicts = conflicts_with(best, anchor);
            if (conflicts.empty()) {
                continue;
            }
            ++anchors_seen;
            if (try_local_mwis_anchor(best, anchor, order)) {
                fill_uncontested(best, order);
            }
        }
    }

    bool exact_dominates(
        int dominator,
        int dominated,
        const std::vector<int>& pool,
        const std::vector<std::vector<char>>& conflict
    ) const {
        // Essential: compatible candidates may both belong to the optimum.
        if (!exact_position_conflicts(dominator, dominated, conflict)) {
            return false;
        }

        const int wd = exact_weight(dominator, pool);
        const int wa = exact_weight(dominated, pool);

        if (wd < wa) {
            return false;
        }

        if (wd == wa &&
            exact_tie(dominator, pool) < exact_tie(dominated, pool)) {
            return false;
        }

        for (std::size_t other = 0; other < pool.size(); ++other) {
            const int o = static_cast<int>(other);

            if (o == dominator || o == dominated) {
                continue;
            }

            if (exact_position_conflicts(dominator, o, conflict) &&
                !exact_position_conflicts(dominated, o, conflict)) {
                return false;
            }
        }

        return true;
    }

    void run_bounded_exact(PackingState& best) const {
        if (stats_ != nullptr) {
            stats_->exact_ran = true;
        }
        const std::vector<int> order = candidate_order(1);
        std::vector<int> pool;
        pool.reserve(options_.exact_candidate_limit);
        std::vector<char> included(candidates_.size(), 0);

        auto try_add = [&](int index) {
            const std::size_t i = static_cast<std::size_t>(index);
            if (i >= included.size() || included[i]) {
                return false;
            }
            if (pool.size() >= options_.exact_candidate_limit) {
                return false;
            }
            pool.push_back(index);
            included[i] = 1;
            return true;
        };

        if (options_.exact_selected_candidate_limit == std::numeric_limits<std::size_t>::max()) {
            for (std::size_t i = 0; i < best.selected.size(); ++i) {
                if (best.selected[i]) {
                    pool.push_back(static_cast<int>(i));
                    included[i] = 1;
                }
            }
        } else {
            std::size_t selected_added = 0;
            for (int index : order) {
                if (selected_added >= options_.exact_selected_candidate_limit ||
                    pool.size() >= options_.exact_candidate_limit) {
                    break;
                }
                if (best.selected[static_cast<std::size_t>(index)] && try_add(index)) {
                    ++selected_added;
                }
            }
        }

        auto add_size_bucket = [&](std::size_t bucket, std::size_t quota) {
            std::size_t added = 0;
            for (int index : order) {
                if (added >= quota || pool.size() >= options_.exact_candidate_limit) {
                    break;
                }
                const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
                if (size_bucket(candidate.labels.size()) == bucket && try_add(index)) {
                    ++added;
                }
            }
        };

        for (std::size_t bucket = options_.exact_pool_size_quota.size(); bucket-- > 2;) {
            add_size_bucket(bucket, options_.exact_pool_size_quota[bucket]);
        }

        for (int index : order) {
            if (pool.size() >= options_.exact_candidate_limit) {
                break;
            }
            try_add(index);
        }
        if (stats_ != nullptr) {
            stats_->exact_pool_size = pool.size();
            stats_->exact_pool_by_size.fill(0);
            for (int index : pool) {
                const Candidate& candidate = candidates_[static_cast<std::size_t>(index)];
                ++stats_->exact_pool_by_size[size_bucket(candidate.labels.size())];
            }
        }

        const std::size_t p = pool.size();
        if (p == 0) {
            return;
        }

        std::vector<std::vector<char>> conflict(p, std::vector<char>(p, 0));
        std::size_t conflict_edges = 0;
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = i + 1; j < p; ++j) {
                if (!candidates_disjoint(pool[i], pool[j])) {
                    conflict[i][j] = 1;
                    conflict[j][i] = 1;
                    ++conflict_edges;
                }
            }
        }
        if (stats_ != nullptr) {
            stats_->exact_conflict_edges = conflict_edges;
        }

        std::vector<char> active(p, 1);
        bool changed = true;
        while (changed && !should_stop()) {
            changed = false;
            for (std::size_t a = 0; a < p && !changed; ++a) {
                if (!active[a]) {
                    continue;
                }
                for (std::size_t b = 0; b < p; ++b) {
                    if (a == b || !active[b]) {
                        continue;
                    }
                    if (exact_dominates(
                            static_cast<int>(b),
                            static_cast<int>(a),
                            pool,
                            conflict)) {
                        active[a] = 0;
                        changed = true;
                        if (stats_ != nullptr) {
                            ++stats_->exact_dominated;
                        }
                        break;
                    }
                }
            }
        }

        std::vector<int> forced_positions;
        for (std::size_t i = 0; i < p; ++i) {
            if (!active[i]) {
                continue;
            }
            bool isolated = true;
            for (std::size_t j = 0; j < p; ++j) {
                if (i != j && active[j] && conflict[i][j]) {
                    isolated = false;
                    break;
                }
            }
            if (isolated) {
                forced_positions.push_back(static_cast<int>(i));
                active[i] = 0;
            }
        }
        if (stats_ != nullptr) {
            stats_->exact_forced = forced_positions.size();
        }

        std::vector<std::vector<int>> components;
        std::vector<char> seen(p, 0);
        for (std::size_t start = 0; start < p; ++start) {
            if (!active[start] || seen[start]) {
                continue;
            }
            std::vector<int> component;
            std::vector<int> stack{static_cast<int>(start)};
            seen[start] = 1;
            while (!stack.empty()) {
                const int u = stack.back();
                stack.pop_back();
                component.push_back(u);
                for (std::size_t v = 0; v < p; ++v) {
                    if (!active[v] || seen[v] || !conflict[static_cast<std::size_t>(u)][v]) {
                        continue;
                    }
                    seen[v] = 1;
                    stack.push_back(static_cast<int>(v));
                }
            }
            components.push_back(std::move(component));
        }
        std::sort(components.begin(), components.end(), [](const auto& a, const auto& b) {
            return a.size() < b.size();
        });

        std::vector<int> selected_positions = forced_positions;
        std::size_t nodes = 0;
        for (const std::vector<int>& component : components) {
            if (stats_ != nullptr) {
                ++stats_->exact_components;
                stats_->exact_largest_component =
                    std::max(stats_->exact_largest_component, component.size());
            }
            if (nodes >= options_.exact_node_limit || should_stop()) {
                break;
            }

            MwisSolution component_best =
                exact_seed_solution(component, pool, conflict, best);
            std::vector<int> current_chosen;
            exact_mwis_dfs(
                component,
                pool,
                conflict,
                0,
                0,
                current_chosen,
                component_best,
                nodes,
                options_.exact_node_limit
            );
            selected_positions.insert(
                selected_positions.end(),
                component_best.chosen_positions.begin(),
                component_best.chosen_positions.end()
            );
        }

        PackingState exact_state = empty_state();
        for (int position : selected_positions) {
            add_to_state(exact_state, pool[static_cast<std::size_t>(position)]);
        }
        if (better_state(exact_state, best)) {
            best = std::move(exact_state);
        }
        if (stats_ != nullptr) {
            stats_->exact_nodes = nodes;
            stats_->stopped = stats_->stopped ||
                nodes >= options_.exact_node_limit ||
                should_stop();
        }
    }

    LabelForest forest_from_state(const PackingState& state) const {
        LabelForest forest;
        std::vector<char> covered(labels_.size(), 0);
        if (stats_ != nullptr) {
            stats_->selected_by_size.fill(0);
            stats_->final_selected = 0;
            stats_->final_uncovered_leaves = 0;
            stats_->final_gain = state.gain;
            stats_->stopped = stats_->stopped || should_stop();
        }
        for (std::size_t i = 0; i < state.selected.size(); ++i) {
            if (!state.selected[i]) continue;
            forest.add_component(LabelComponent(candidates_[i].labels));
            for (int leaf : candidates_[i].leaf_ids) covered[static_cast<std::size_t>(leaf)] = 1;
            if (stats_ != nullptr) {
                ++stats_->final_selected;
                ++stats_->selected_by_size[size_bucket(candidates_[i].labels.size())];
            }
        }
        for (std::size_t i = 0; i < labels_.size(); ++i) {
            if (!covered[i]) {
                forest.components.emplace_back(labels_[i]);
                if (stats_ != nullptr) {
                    ++stats_->final_uncovered_leaves;
                }
            }
        }
        forest.normalize();
        forest.validate_partition_of(labels_);
        return forest;
    }
};

}  // namespace pace26::heuristics
