#pragma once

#include "CommonSubtreeReduction.hpp"
#include "ChainReduction.hpp"
#include "ThreeTwoChainReduction.hpp"
#include "core/Forest.hpp"

using pace26::core::LabelComponent;
using pace26::core::LabelForest;

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace pace26::reductions {

class ReductionStackError : public std::runtime_error {
public:
    explicit ReductionStackError(const std::string& message)
        : std::runtime_error("Reduction stack error: " + message) {}
};

/*
 * A forest component represented purely as a set/list of labels.
 *
 * This is deliberate.
 *
 * Do not try to expand reductions directly on Newick strings at this stage.
 * Keep the forest as label blocks, then let your final ForestWriter reconstruct
 * each component's topology from the original T1 or T2.
 */
struct LabelComponent {
    std::vector<std::uint32_t> labels;
};

struct LabelForest {
    std::vector<LabelComponent> components;
};

enum class ReductionEntryKind : std::uint8_t {
    CommonSubtree,
    Chain,
    ThreeTwoChain
};

struct ReductionEntry {
    ReductionEntryKind kind;

    std::variant<
        CommonSubtreeRecord,
        ChainReductionRecord,
        ThreeTwoChainReductionRecord
    > record;
};

struct ReductionExpansionReport {
    bool certification_preserving = true;

    std::size_t common_subtree_expansions = 0;
    std::size_t chain_expansions = 0;
    std::size_t chain_suffix_labels_reattached = 0;
    std::size_t chain_suffix_singleton_fallbacks = 0;
    std::size_t chain_suffix_singleton_labels = 0;
    std::size_t three_two_chain_expansions = 0;
};

class ReductionStack {
private:
    std::vector<ReductionEntry> entries_;

    static void sort_unique(std::vector<std::uint32_t>& labels) {
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    }

    static void normalize(LabelForest& forest) {
        for (LabelComponent& component : forest.components) {
            sort_unique(component.labels);
        }

        forest.components.erase(
            std::remove_if(
                forest.components.begin(),
                forest.components.end(),
                [](const LabelComponent& c) {
                    return c.labels.empty();
                }
            ),
            forest.components.end()
        );
    }

    static bool component_contains(const LabelComponent& component, std::uint32_t label) {
        return std::find(component.labels.begin(), component.labels.end(), label) !=
               component.labels.end();
    }

    static std::optional<std::size_t> find_component_containing(
        const LabelForest& forest,
        std::uint32_t label
    ) {
        for (std::size_t i = 0; i < forest.components.size(); ++i) {
            if (component_contains(forest.components[i], label)) {
                return i;
            }
        }

        return std::nullopt;
    }

    static bool forest_contains(const LabelForest& forest, std::uint32_t label) {
        return find_component_containing(forest, label).has_value();
    }

    static void add_label_to_component(LabelComponent& component, std::uint32_t label) {
        if (!component_contains(component, label)) {
            component.labels.push_back(label);
        }
    }

    static void add_labels_to_component(
        LabelComponent& component,
        const std::vector<std::uint32_t>& labels
    ) {
        for (std::uint32_t label : labels) {
            add_label_to_component(component, label);
        }

        sort_unique(component.labels);
    }

    static void add_singleton_if_absent(LabelForest& forest, std::uint32_t label) {
        if (forest_contains(forest, label)) {
            return;
        }

        LabelComponent component;
        component.labels.push_back(label);
        forest.components.push_back(std::move(component));
    }

    static void add_singletons_if_absent(
        LabelForest& forest,
        const std::vector<std::uint32_t>& labels
    ) {
        for (std::uint32_t label : labels) {
            add_singleton_if_absent(forest, label);
        }
    }

    static void remove_label_from_component(LabelComponent& component, std::uint32_t label) {
        component.labels.erase(
            std::remove(component.labels.begin(), component.labels.end(), label),
            component.labels.end()
        );
    }

    static std::optional<std::size_t> component_containing_all(
        const LabelForest& forest,
        const std::vector<std::uint32_t>& labels
    ) {
        if (labels.empty()) {
            return std::nullopt;
        }

        const auto first_component = find_component_containing(forest, labels.front());
        if (!first_component.has_value()) {
            return std::nullopt;
        }

        const std::size_t idx = *first_component;

        for (std::uint32_t label : labels) {
            if (!component_contains(forest.components[idx], label)) {
                return std::nullopt;
            }
        }

        return idx;
    }

    static void undo_common_subtree(
        LabelForest& forest,
        const CommonSubtreeRecord& record,
        ReductionExpansionReport* report
    ) {
        if (report != nullptr) {
            ++report->common_subtree_expansions;
        }

        const auto component_index =
            find_component_containing(forest, record.placeholder_label);

        if (!component_index.has_value()) {
            throw ReductionStackError(
                "common-subtree placeholder " +
                std::to_string(record.placeholder_label) +
                " was not present in the reduced forest"
            );
        }

        LabelComponent& component = forest.components[*component_index];

        remove_label_from_component(component, record.placeholder_label);
        add_labels_to_component(component, record.leaves);

        normalize(forest);
    }

    static void undo_chain(
        LabelForest& forest,
        const ChainReductionRecord& record,
        ReductionExpansionReport* report
    ) {
        if (report != nullptr) {
            ++report->chain_expansions;
        }

        /*
         * Rule:
         *   full_chain  = x1, x2, x3, x4, ...
         *   kept_prefix = x1, x2, x3
         *   removed     = x4, ...
         *
         * If x1,x2,x3 are still in the same component, we safely add the removed
         * suffix back to that same component.
         *
         * If the solver split x1,x2,x3 across components, we restore the deleted
         * suffix as singleton components. This is always feasible. It is not a
         * proof-preserving expansion, because this stack only stores label blocks
         * and does not have the original trees needed to validate any stronger
         * enumerated placement. Exact certificates must therefore stop at this
         * fallback unless a caller proves a stronger chain expansion externally.
         */
        const auto common_component =
            component_containing_all(forest, record.kept_prefix);

        if (common_component.has_value()) {
            add_labels_to_component(
                forest.components[*common_component],
                record.removed_suffix
            );
            if (report != nullptr) {
                report->chain_suffix_labels_reattached += record.removed_suffix.size();
            }
        } else {
            add_singletons_if_absent(forest, record.removed_suffix);
            if (report != nullptr) {
                report->certification_preserving = false;
                ++report->chain_suffix_singleton_fallbacks;
                report->chain_suffix_singleton_labels += record.removed_suffix.size();
            }
        }

        normalize(forest);
    }

    static void undo_three_two_chain(
        LabelForest& forest,
        const ThreeTwoChainReductionRecord& record,
        ReductionExpansionReport* report
    ) {
        if (report != nullptr) {
            ++report->three_two_chain_expansions;
        }

        /*
         * 3-2-chain reduction deletes one taxon.
         *
         * Safe reconstruction:
         *   add it back as a singleton component.
         *
         * This is the safest possible expansion and preserves feasibility.
         */
        add_singleton_if_absent(forest, record.deleted_label);
        normalize(forest);
    }

    static void validate_no_duplicate_labels(const LabelForest& forest) {
        std::unordered_set<std::uint32_t> seen;

        for (const LabelComponent& component : forest.components) {
            for (std::uint32_t label : component.labels) {
                if (!seen.insert(label).second) {
                    throw ReductionStackError(
                        "label " + std::to_string(label) +
                        " occurs in more than one forest component"
                    );
                }
            }
        }
    }

public:
    bool empty() const noexcept {
        return entries_.empty();
    }

    std::size_t size() const noexcept {
        return entries_.size();
    }

    void clear() {
        entries_.clear();
    }

    const std::vector<ReductionEntry>& entries() const noexcept {
        return entries_;
    }

    void push_common_subtree(const CommonSubtreeRecord& record) {
        ReductionEntry entry;
        entry.kind = ReductionEntryKind::CommonSubtree;
        entry.record = record;
        entries_.push_back(std::move(entry));
    }

    void push_common_subtrees(const std::vector<CommonSubtreeRecord>& records) {
        for (const CommonSubtreeRecord& record : records) {
            push_common_subtree(record);
        }
    }

    void push_chain(const ChainReductionRecord& record) {
        ReductionEntry entry;
        entry.kind = ReductionEntryKind::Chain;
        entry.record = record;
        entries_.push_back(std::move(entry));
    }

    void push_chains(const std::vector<ChainReductionRecord>& records) {
        for (const ChainReductionRecord& record : records) {
            push_chain(record);
        }
    }

    void push_three_two_chain(const ThreeTwoChainReductionRecord& record) {
        ReductionEntry entry;
        entry.kind = ReductionEntryKind::ThreeTwoChain;
        entry.record = record;
        entries_.push_back(std::move(entry));
    }

    void push_three_two_chains(const std::vector<ThreeTwoChainReductionRecord>& records) {
        for (const ThreeTwoChainReductionRecord& record : records) {
            push_three_two_chain(record);
        }
    }

    /*
     * Expands a forest over the reduced labels back toward the original labels.
     *
     * Reductions must be undone in reverse order.
     */
    LabelForest expand(
        LabelForest forest,
        ReductionExpansionReport* report = nullptr
    ) const {
        if (report != nullptr) {
            *report = ReductionExpansionReport{};
        }

        normalize(forest);
        validate_no_duplicate_labels(forest);

        for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
            const ReductionEntry& entry = *it;

            switch (entry.kind) {
                case ReductionEntryKind::CommonSubtree:
                    undo_common_subtree(
                        forest,
                        std::get<CommonSubtreeRecord>(entry.record),
                        report
                    );
                    break;

                case ReductionEntryKind::Chain:
                    undo_chain(
                        forest,
                        std::get<ChainReductionRecord>(entry.record),
                        report
                    );
                    break;

                case ReductionEntryKind::ThreeTwoChain:
                    undo_three_two_chain(
                        forest,
                        std::get<ThreeTwoChainReductionRecord>(entry.record),
                        report
                    );
                    break;

                default:
                    throw ReductionStackError("unknown reduction entry kind");
            }

            validate_no_duplicate_labels(forest);
        }

        normalize(forest);
        validate_no_duplicate_labels(forest);

        return forest;
    }

    /*
     * Convenience helper for recursive cluster decomposition.
     *
     * If you solve bottom and top subinstances separately, their forests can be
     * unioned. Cluster decomposition itself usually does not need to be pushed
     * into this stack in the safe implementation.
     */
    static LabelForest unite(LabelForest a, const LabelForest& b) {
        for (const LabelComponent& component : b.components) {
            a.components.push_back(component);
        }

        normalize(a);
        validate_no_duplicate_labels(a);

        return a;
    }

    /*
     * Convenience helper for singleton fallback.
     */
    static LabelForest singleton_forest_from_labels(
        const std::vector<std::uint32_t>& labels
    ) {
        LabelForest forest;

        forest.components.reserve(labels.size());

        for (std::uint32_t label : labels) {
            LabelComponent component;
            component.labels.push_back(label);
            forest.components.push_back(std::move(component));
        }

        normalize(forest);
        validate_no_duplicate_labels(forest);

        return forest;
    }
};

}  // namespace pace26::reductions
