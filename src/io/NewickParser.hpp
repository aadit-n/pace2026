#pragma once

#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pace26::io {

class NewickParseError : public std::runtime_error {
public:
    NewickParseError(std::size_t offset, const std::string& message)
        : std::runtime_error("Newick parse error at offset " + std::to_string(offset) + ": " + message),
          offset_(offset) {}

    std::size_t offset() const noexcept { return offset_; }

private:
    std::size_t offset_;
};

struct NewickNode {
    std::uint32_t id = 0;
    std::uint32_t label = 0;
    int left = -1;
    int right = -1;

    bool is_leaf() const noexcept { return left < 0 && right < 0; }
};

struct NewickTree {
    std::vector<NewickNode> nodes;
    int root = -1;

    const NewickNode& node(int index) const { return nodes.at(static_cast<std::size_t>(index)); }
};

class NewickParser {
public:
    explicit NewickParser(std::string_view text) : text_(text) {}

    NewickTree parse(std::uint32_t root_id = 0) {
        const auto parsed_root = parse_subtree(root_id);
        skip_whitespace();
        expect(';', "expected ';' after tree");
        skip_whitespace();

        if (pos_ != text_.size()) {
            fail("unexpected input after ';'");
        }

        NewickTree tree;
        tree.nodes = std::move(nodes_);
        tree.root = parsed_root.first;
        return tree;
    }

    static NewickTree parse_string(std::string_view text, std::uint32_t root_id = 0) {
        return NewickParser(text).parse(root_id);
    }

private:
    std::pair<int, std::uint32_t> parse_subtree(std::uint32_t own_id) {
        skip_whitespace();

        if (pos_ >= text_.size()) {
            fail("unexpected end of input");
        }

        if (text_[pos_] == '(') {
            ++pos_;
            const auto left = parse_subtree(increment_id(own_id));
            skip_whitespace();
            expect(',', "expected ',' between children");
            const auto right = parse_subtree(left.second);
            skip_whitespace();
            expect(')', "expected ')' after right child");

            NewickNode node;
            node.id = own_id;
            node.left = left.first;
            node.right = right.first;
            nodes_.push_back(node);
            return {static_cast<int>(nodes_.size() - 1), right.second};
        }

        if (std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            const auto label = parse_u32();
            NewickNode node;
            node.id = label;
            node.label = label;
            nodes_.push_back(node);
            return {static_cast<int>(nodes_.size() - 1), own_id};
        }

        fail("expected a leaf label or '('");
    }

    std::uint32_t parse_u32() {
        std::uint64_t value = 0;

        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            value = value * 10 + static_cast<std::uint64_t>(text_[pos_] - '0');
            if (value > std::numeric_limits<std::uint32_t>::max()) {
                fail("integer label is too large");
            }
            ++pos_;
        }

        return static_cast<std::uint32_t>(value);
    }

    std::uint32_t increment_id(std::uint32_t id) {
        if (id == std::numeric_limits<std::uint32_t>::max()) {
            fail("internal node id overflow");
        }
        return id + 1;
    }

    void expect(char expected, const char* message) {
        if (pos_ >= text_.size() || text_[pos_] != expected) {
            fail(message);
        }
        ++pos_;
    }

    void skip_whitespace() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    [[noreturn]] void fail(const std::string& message) const {
        throw NewickParseError(pos_, message);
    }

    std::string_view text_;
    std::size_t pos_ = 0;
    std::vector<NewickNode> nodes_;
};

}  // namespace pace26::io