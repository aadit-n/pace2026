#pragma once

#include "Tree.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace pace26::core {

class ForestError : public std::runtime_error {
public:
    explicit ForestError(const std::string& message)
        : std::runtime_error("Forest error: " + message) {}
};

struct LabelComponent {
    std::vector<std::uint32_t> labels;

    LabelComponent() = default;

    explicit LabelComponent(std::uint32_t label) {
        labels.push_back(label);
    }

    explicit LabelComponent(std::vector<std::uint32_t> values)
        : labels(std::move(values)) {
        normalize();
    }

    void normalize() {
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    }

    bool empty() const noexcept {
        return labels.empty();
    }

    std::size_t size() const noexcept {
        return labels.size();
    }

    bool contains(std::uint32_t label) const {
        return std::find(labels.begin(), labels.end(), label) != labels.end();
    }

    void add(std::uint32_t label) {
        if (!contains(label)) {
            labels.push_back(label);
            normalize();
        }
    }

    void add_many(const std::vector<std::uint32_t>& extra) {
        for (std::uint32_t x : extra) {
            labels.push_back(x);
        }
        normalize();
    }

    void remove(std::uint32_t label) {
        labels.erase(
            std::remove(labels.begin(), labels.end(), label),
            labels.end()
        );
    }
};

struct LabelForest {
    std::vector<LabelComponent> components;

    std::size_t component_count() const noexcept {
        return components.size();
    }

    bool empty() const noexcept {
        return components.empty();
    }

    void normalize() {
        for (LabelComponent& component : components) {
            component.normalize();
        }

        components.erase(
            std::remove_if(
                components.begin(),
                components.end(),
                [](const LabelComponent& c) {
                    return c.empty();
                }
            ),
            components.end()
        );

        std::sort(
            components.begin(),
            components.end(),
            [](const LabelComponent& a, const LabelComponent& b) {
                if (a.labels.empty() || b.labels.empty()) {
                    return a.labels.size() < b.labels.size();
                }
                return a.labels.front() < b.labels.front();
            }
        );
    }

    void add_component(LabelComponent component) {
        component.normalize();
        if (!component.empty()) {
            components.push_back(std::move(component));
        }
    }

    void add_singleton(std::uint32_t label) {
        if (contains_label(label)) {
            return;
        }

        components.emplace_back(label);
    }

    bool contains_label(std::uint32_t label) const {
        return find_component_containing(label) >= 0;
    }

    int find_component_containing(std::uint32_t label) const {
        for (std::size_t i = 0; i < components.size(); ++i) {
            if (components[i].contains(label)) {
                return static_cast<int>(i);
            }
        }

        return -1;
    }

    int find_component_containing_all(const std::vector<std::uint32_t>& labels) const {
        if (labels.empty()) {
            return -1;
        }

        int idx = find_component_containing(labels.front());
        if (idx < 0) {
            return -1;
        }

        const LabelComponent& component = components[static_cast<std::size_t>(idx)];

        for (std::uint32_t label : labels) {
            if (!component.contains(label)) {
                return -1;
            }
        }

        return idx;
    }

    std::vector<std::uint32_t> all_labels() const {
        std::vector<std::uint32_t> result;
        std::size_t total_size = 0;

        for (const LabelComponent& component : components) {
            total_size += component.labels.size();
        }

        result.reserve(total_size);

        for (const LabelComponent& component : components) {
            for (std::uint32_t label : component.labels) {
                result.push_back(label);
            }
        }

        std::sort(result.begin(), result.end());
        return result;
    }

    void validate_no_duplicates() const {
        std::unordered_set<std::uint32_t> seen;
        std::size_t total_size = 0;

        for (const LabelComponent& component : components) {
            total_size += component.labels.size();
        }

        seen.reserve(total_size * 2);

        for (const LabelComponent& component : components) {
            for (std::uint32_t label : component.labels) {
                if (!seen.insert(label).second) {
                    throw ForestError(
                        "label " + std::to_string(label) +
                        " appears in more than one component"
                    );
                }
            }
        }
    }

    void validate_partition_of(const std::vector<std::uint32_t>& expected_labels) const {
        bool expected_sorted_unique = true;
        for (std::size_t i = 1; i < expected_labels.size(); ++i) {
            if (expected_labels[i - 1] >= expected_labels[i]) {
                expected_sorted_unique = false;
                break;
            }
        }

        if (expected_sorted_unique) {
            std::size_t actual_count = 0;
            for (const LabelComponent& component : components) {
                actual_count += component.labels.size();
            }
            if (actual_count != expected_labels.size()) {
                throw ForestError("forest labels do not form the expected partition");
            }

            const std::uint32_t max_label =
                expected_labels.empty() ? 0 : expected_labels.back();
            const std::size_t dense_limit =
                std::max<std::size_t>(4096, expected_labels.size() * 32 + 1024);
            if (static_cast<std::uint64_t>(max_label) <= dense_limit) {
                std::vector<unsigned char> seen(
                    static_cast<std::size_t>(max_label) + 1,
                    0
                );

                for (const LabelComponent& component : components) {
                    for (std::uint32_t label : component.labels) {
                        if (label > max_label) {
                            throw ForestError(
                                "forest labels do not form the expected partition"
                            );
                        }
                        unsigned char& marker =
                            seen[static_cast<std::size_t>(label)];
                        if (marker != 0) {
                            throw ForestError(
                                "forest labels do not form the expected partition"
                            );
                        }
                        marker = 1;
                    }
                }

                for (std::uint32_t label : expected_labels) {
                    if (seen[static_cast<std::size_t>(label)] == 0) {
                        throw ForestError(
                            "forest labels do not form the expected partition"
                        );
                    }
                }
                return;
            }

            std::vector<std::uint32_t> actual = all_labels();
            if (actual == expected_labels) {
                return;
            }
            throw ForestError("forest labels do not form the expected partition");
        }

        std::vector<std::uint32_t> actual = all_labels();
        std::vector<std::uint32_t> expected = expected_labels;
        std::sort(expected.begin(), expected.end());
        expected.erase(std::unique(expected.begin(), expected.end()), expected.end());

        if (actual != expected) {
            throw ForestError("forest labels do not form the expected partition");
        }
    }

    static LabelForest singleton_forest_from_labels(
        const std::vector<std::uint32_t>& labels
    ) {
        LabelForest forest;
        forest.components.reserve(labels.size());

        for (std::uint32_t label : labels) {
            forest.components.emplace_back(label);
        }

        bool labels_sorted_unique = true;
        for (std::size_t i = 1; i < labels.size(); ++i) {
            if (labels[i - 1] >= labels[i]) {
                labels_sorted_unique = false;
                break;
            }
        }

        if (labels_sorted_unique) {
            return forest;
        }

        forest.normalize();
        forest.validate_no_duplicates();

        return forest;
    }

    static LabelForest singleton_forest_from_tree(const Tree& tree) {
        return singleton_forest_from_labels(tree.leaf_labels);
    }

    static LabelForest unite(LabelForest a, const LabelForest& b) {
        for (const LabelComponent& component : b.components) {
            a.components.push_back(component);
        }

        a.normalize();
        a.validate_no_duplicates();

        return a;
    }
};

}  // namespace pace26::core
