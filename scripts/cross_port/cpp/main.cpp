// C++ half of the graph-file cross-port check (see ../graph_files.py).
//
//     cross_port_graph write    <dir>
//     cross_port_graph read     <dir> <observation.json>
//     cross_port_graph mismatch <dir>
//
// `write` saves the shared fixture graph as <dir>/graph.ison plus its
// embeddings sidecar and binary snapshot. `read` loads it with
// GraphMismatchMode::Error and writes what it found (and the load source,
// snapshot or json, with the reason a snapshot was skipped) to <observation.json>. `mismatch` loads a graph whose file was
// edited after the save and exits 0 only if the load throws
// GraphEmbeddingMismatchError.
//
// Node ids are strings in this port, so ids 1 and 2 are written as "1" and "2"
// (the other typed-id ports write integers); the orchestrator allows for that.

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "isongraph_vector.hpp"

using namespace isongraph_vector;

namespace {

constexpr size_t DIM = 384;

// A backslash, spelled so no source-level escaping can mangle it.
const std::string BS(1, static_cast<char>(92));
const std::string DOC_TEXT = "Résumé of 日本語 engineers, \"quoted\" and back" + BS + "slash";
const std::vector<std::string> QUERIES = {
    "Bob analyses ML models", "Carol plans roadmaps for über-teams", DOC_TEXT};

struct NodeSpec {
    std::string type, id;
    Properties props;
};
struct EdgeSpec {
    std::string rel;
    NodeRef source, target;
    Properties props;
};

using ison::Value;
Value str(const std::string& s) { return Value(s); }

// Keep in step with the other graph_* helpers.
std::vector<NodeSpec> nodes() {
    return {
        {"person", "1", {{"text", str("Alice is a software engineer")}, {"age", Value(30)}, {"score", Value(0.5)},
                         {"active", Value(true)}, {"nickname", Value()}}},
        {"person", "2", {{"text", str("Bob analyses ML models")}, {"age", Value(41)}, {"score", Value(2.75)},
                         {"active", Value(false)}, {"nickname", Value()}}},
        {"person", "carol", {{"text", str("Carol plans roadmaps for über-teams")}, {"age", Value(35)},
                             {"score", Value(-1.25)}, {"active", Value(true)}, {"nickname", str("Caz")}}},
        {"doc", "résumé", {{"text", str(DOC_TEXT)}}},
        {"topic", "ml", {{"text", str("machine learning models")}, {"weight", Value(0.125)}}},
    };
}

std::vector<EdgeSpec> edges() {
    return {
        {"KNOWS", NodeRef("person", "1"), NodeRef("person", "2"), {{"since", Value(2020)}, {"weight", Value(0.75)}}},
        {"WROTE", NodeRef("person", "carol"), NodeRef("doc", "résumé"), {{"note", str("first draft")}}},
        {"ABOUT", NodeRef("doc", "résumé"), NodeRef("topic", "ml"), {}},
    };
}

std::string graphPath(const std::string& dir) {
    return (std::filesystem::path(dir) / "graph.ison").string();
}

// -- JSON output ---------------------------------------------------------------

std::string jstr(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);  // UTF-8 passes through
                }
        }
    }
    return out + "\"";
}

std::string jnum(double d) {
    char buf[64];
    auto [end, ec] = std::to_chars(buf, buf + sizeof(buf), d);
    return ec == std::errc() ? std::string(buf, end) : std::string("0");
}

std::string jvalue(const Value& v) {
    if (v.is_null()) return "null";
    if (v.is_bool()) return v.as_bool() ? "true" : "false";
    if (v.is_int()) return std::to_string(v.as_int());
    if (v.is_float()) return jnum(v.as_float());
    if (v.is_string()) return jstr(v.as_string());
    return jstr(valueToText(v));
}

std::string jprops(const Properties& props) {
    std::string out = "{";
    bool first = true;
    for (const auto& [key, value] : props) {  // std::map: already sorted
        if (!first) out += ",";
        first = false;
        out += jstr(key) + ":" + jvalue(value);
    }
    return out + "}";
}

std::string jref(const NodeRef& r) { return "[" + jstr(r.type) + "," + jstr(r.id) + "]"; }

// -- commands ------------------------------------------------------------------

void write(const std::string& dir) {
    SemanticGraph graph("fixture", std::make_shared<MockEncoder>(DIM));
    for (const auto& n : nodes()) {
        graph.addNodeWithEmbed(n.type, n.id, n.props, n.props.at("text").as_string());
    }
    for (const auto& e : edges()) {
        graph.addEdge(e.rel, e.source, e.target, e.props);
    }
    std::filesystem::create_directories(dir);
    graph.save(graphPath(dir), "auto", true, true);
}

SemanticGraph load(const std::string& dir, const std::shared_ptr<MockEncoder>& encoder,
                   LoadReport* report = nullptr) {
    return SemanticGraph::load(graphPath(dir), encoder, "auto", true, true,
                               {"name", "description", "content", "text"},
                               GraphMismatchMode::Error, std::nullopt,
                               ModelMismatchMode::Warn, std::nullopt, report);
}

void read(const std::string& dir, const std::string& output) {
    auto encoder = std::make_shared<MockEncoder>(DIM);
    LoadReport report;
    SemanticGraph graph = load(dir, encoder, &report);

    std::string out = "{\"port\":\"cpp\",\"nodes\":[";
    std::vector<NodeRef> missing;
    double worst = 0;
    bool first = true;
    graph.graph().forEachNode([&](const Node& node) {
        if (!first) out += ",";
        first = false;
        out += "{\"type\":" + jstr(node.type) + ",\"id\":" + jstr(node.id) +
               ",\"props\":" + jprops(node.properties) + "}";
        const std::optional<EmbeddingRecord> record = graph.getEmbedding(NodeRef(node.type, node.id));
        if (!record) {
            missing.push_back(NodeRef(node.type, node.id));
            return;
        }
        std::vector<float> expected = encoder->encode(node.properties.at("text").as_string());
        if (expected.size() != record->vector.size()) {
            worst = 1e9;
        } else {
            for (size_t i = 0; i < expected.size(); ++i) {
                worst = std::max(worst, std::fabs(static_cast<double>(record->vector[i]) -
                                                  static_cast<double>(expected[i])));
            }
        }
    });

    out += "],\"edges\":[";
    first = true;
    for (const Edge& edge : graph.graph().getAllEdges()) {
        if (!first) out += ",";
        first = false;
        out += "{\"rel\":" + jstr(edge.relType) + ",\"source\":" + jref(edge.source) +
               ",\"target\":" + jref(edge.target) + ",\"props\":" + jprops(edge.properties) + "}";
    }

    out += "],\"embedding_count\":" + std::to_string(graph.embeddingStore().count());
    out += ",\"missing_embeddings\":[";
    for (size_t i = 0; i < missing.size(); ++i) out += (i ? "," : "") + jref(missing[i]);
    out += "],\"max_vec_diff\":" + jnum(worst) + ",\"searches\":[";
    for (size_t i = 0; i < QUERIES.size(); ++i) {
        auto hits = graph.similaritySearch(QUERIES[i], 1);
        if (i) out += ",";
        out += "{\"query\":" + jstr(QUERIES[i]) + ",\"top\":" +
               (hits.empty() ? std::string("null") : jref(hits[0].node_ref)) + ",\"score\":" +
               (hits.empty() ? std::string("null") : jnum(hits[0].score)) + "}";
    }
    out += "],\"source\":";
    out += report.source == LoadSource::Snapshot ? "\"snapshot\""
           : report.source == LoadSource::Json   ? "\"json\""
                                                 : "\"none\"";
    out += ",\"snapshot_skipped\":" +
           (report.snapshotSkipped ? jstr(*report.snapshotSkipped) : std::string("null"));
    out += "}";

    std::ofstream file(output, std::ios::binary);
    file << out;
}

int mismatch(const std::string& dir) {
    try {
        load(dir, std::make_shared<MockEncoder>(DIM));
    } catch (const GraphEmbeddingMismatchError&) {
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "expected GraphEmbeddingMismatchError, got: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "expected GraphEmbeddingMismatchError, but the load succeeded\n";
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    try {
        if (args.size() == 2 && args[0] == "write") {
            write(args[1]);
            return 0;
        }
        if (args.size() == 3 && args[0] == "read") {
            read(args[1], args[2]);
            return 0;
        }
        if (args.size() == 2 && args[0] == "mismatch") {
            return mismatch(args[1]);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    std::cerr << "usage: cross_port_graph write <dir> | read <dir> <out.json> | mismatch <dir>\n";
    return 2;
}
