#pragma once

#include "NewickParser.hpp"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <istream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pace26::io {

class PaceParseError : public std::runtime_error {
public:
    PaceParseError(std::size_t line, const std::string& message)
        : std::runtime_error("PACE parse error on line " + std::to_string(line) + ": " + message),
          line_(line) {}

    std::size_t line() const noexcept { return line_; }

private:
    std::size_t line_;
};

struct ApproximationTarget {
    double a = 0.0;
    std::size_t b = 0;
};

struct KeyValueLine {
    std::string key;
    std::string value;
};

struct PaceInstance {
    std::size_t num_trees = 0;
    std::size_t num_leaves = 0;
    std::vector<NewickTree> trees;
    std::optional<ApproximationTarget> approximation;
    std::vector<KeyValueLine> stride_metadata;
    std::vector<KeyValueLine> parameters;
    std::optional<std::string> tree_decomposition_json;
};

class PaceParser {
public:
    static PaceInstance parse(std::istream& input) {
        PaceInstance instance;
        bool saw_header = false;
        std::size_t header_line = 0;
        std::string line;

        for (std::size_t line_number = 1; std::getline(input, line); ++line_number) {
            const std::string_view content = trim(line);

            if (content.empty()) {
                continue;
            }

            if (starts_with(content, "# ")) {
                continue;
            }

            if (starts_with(content, "#p")) {
                if (saw_header) {
                    throw PaceParseError(
                        line_number,
                        "multiple #p headers; first header was on line " + std::to_string(header_line));
                }
                parse_header(content, line_number, instance);
                saw_header = true;
                header_line = line_number;
                continue;
            }

            if (starts_with(content, "#s")) {
                instance.stride_metadata.push_back(parse_key_value(content, line_number, "#s metadata"));
                continue;
            }

            if (starts_with(content, "#a")) {
                if (instance.approximation.has_value()) {
                    throw PaceParseError(line_number, "multiple #a approximation lines");
                }
                instance.approximation = parse_approximation(content, line_number);
                continue;
            }

            if (starts_with(content, "#x")) {
                KeyValueLine parameter = parse_key_value(content, line_number, "#x parameter");
                if (parameter.key == "treedecomp") {
                    instance.tree_decomposition_json = parameter.value;
                }
                instance.parameters.push_back(std::move(parameter));
                continue;
            }

            if (!content.empty() && content.front() == '#') {
                continue;
            }

            if (!saw_header) {
                throw PaceParseError(line_number, "tree appeared before #p header");
            }

            if (content.back() != ';') {
                throw PaceParseError(line_number, "expected a Newick tree ending in ';'");
            }

            const std::uint32_t root_id = compute_root_id(instance.trees.size(), instance.num_leaves, line_number);
            try {
                instance.trees.push_back(NewickParser::parse_string(content, root_id));
            } catch (const NewickParseError& error) {
                throw PaceParseError(line_number, error.what());
            }
        }

        if (!saw_header) {
            throw PaceParseError(1, "missing #p header");
        }

        if (instance.trees.size() != instance.num_trees) {
            throw PaceParseError(
                header_line,
                "#p declares " + std::to_string(instance.num_trees) + " trees, but parsed " +
                    std::to_string(instance.trees.size()));
        }

        return instance;
    }

private:
    static bool starts_with(std::string_view text, std::string_view prefix) {
        return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
    }

    static std::string_view trim(std::string_view text) {
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
            text.remove_prefix(1);
        }
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
            text.remove_suffix(1);
        }
        return text;
    }

    static std::vector<std::string_view> split_whitespace(std::string_view text) {
        std::vector<std::string_view> parts;
        while (!text.empty()) {
            text = trim_left(text);
            if (text.empty()) {
                break;
            }

            std::size_t end = 0;
            while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) {
                ++end;
            }

            parts.push_back(text.substr(0, end));
            text.remove_prefix(end);
        }
        return parts;
    }

    static std::string_view trim_left(std::string_view text) {
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
            text.remove_prefix(1);
        }
        return text;
    }

    static bool parse_unsigned(std::string_view text, std::size_t& value) {
        if (text.empty()) {
            return false;
        }

        std::size_t result = 0;
        for (char ch : text) {
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                return false;
            }

            const std::size_t digit = static_cast<std::size_t>(ch - '0');
            if (result > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
                return false;
            }
            result = result * 10 + digit;
        }

        value = result;
        return true;
    }

    static bool parse_nonnegative_double(std::string_view text, double& value) {
        std::string token(text);
        std::size_t consumed = 0;

        try {
            value = std::stod(token, &consumed);
        } catch (const std::exception&) {
            return false;
        }

        return consumed == token.size() && std::isfinite(value) && value >= 0.0;
    }

    static void parse_header(std::string_view content, std::size_t line_number, PaceInstance& instance) {
        const auto parts = split_whitespace(content);
        if (parts.size() < 3 || parts[0] != "#p") {
            throw PaceParseError(line_number, "expected '#p {num_trees} {num_leaves}'");
        }

        if (!parse_unsigned(parts[1], instance.num_trees) || !parse_unsigned(parts[2], instance.num_leaves)) {
            throw PaceParseError(line_number, "invalid numeric value in #p header");
        }

        if (instance.num_leaves == 0) {
            throw PaceParseError(line_number, "#p declares zero leaves");
        }

        instance.trees.reserve(instance.num_trees);
    }

    static ApproximationTarget parse_approximation(std::string_view content, std::size_t line_number) {
        const auto parts = split_whitespace(content);
        if (parts.size() < 3 || parts[0] != "#a") {
            throw PaceParseError(line_number, "expected '#a {a} {b}'");
        }

        ApproximationTarget target;
        if (!parse_nonnegative_double(parts[1], target.a) || !parse_unsigned(parts[2], target.b)) {
            throw PaceParseError(line_number, "invalid #a approximation values");
        }
        return target;
    }

    static KeyValueLine parse_key_value(std::string_view content, std::size_t line_number, std::string_view label) {
        if (content.size() < 3) {
            throw PaceParseError(line_number, "invalid " + std::string(label));
        }

        std::string_view rest = trim_left(content.substr(2));
        std::size_t split = 0;
        while (split < rest.size() && !std::isspace(static_cast<unsigned char>(rest[split]))) {
            ++split;
        }

        if (split == 0 || split == rest.size()) {
            throw PaceParseError(line_number, "expected key and value in " + std::string(label));
        }

        KeyValueLine result;
        result.key = std::string(rest.substr(0, split));
        result.value = std::string(trim(rest.substr(split)));
        return result;
    }

    static std::uint32_t compute_root_id(std::size_t tree_index, std::size_t num_leaves, std::size_t line_number) {
        const std::size_t internal_nodes = num_leaves - 1;
        if (internal_nodes == 0) {
            return 2;
        }

        if (tree_index == std::numeric_limits<std::size_t>::max()) {
            throw PaceParseError(line_number, "root id overflow");
        }

        const std::size_t one_based_tree_index = tree_index + 1;
        if (one_based_tree_index > (std::numeric_limits<std::size_t>::max() - 2) / internal_nodes) {
            throw PaceParseError(line_number, "root id overflow");
        }

        const std::size_t root_id = one_based_tree_index * internal_nodes + 2;
        if (root_id > std::numeric_limits<std::uint32_t>::max()) {
            throw PaceParseError(line_number, "root id does not fit in uint32_t");
        }

        return static_cast<std::uint32_t>(root_id);
    }
};

inline PaceInstance read_pace_instance(std::istream& input) {
    return PaceParser::parse(input);
}

}  // namespace pace26::io