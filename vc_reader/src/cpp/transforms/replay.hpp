#pragma once
#include "pipeline.hpp"

namespace vcr::transforms {
// Versioned little-endian representation for trusted replay artifacts. Like
// Python pickle, this is an internal persistence format, not an untrusted RPC.
struct Writer {
    std::string bytes = "VCA2";
    void number(std::uint64_t v) { for (int i = 0; i < 8; ++i) bytes.push_back(char(v >> (i * 8))); }
    void text(const std::string& s) { number(s.size()); bytes += s; }
    void shape(const Shape& v) { number(v.size()); for (auto x : v) number(x); }
    void floats(const std::vector<double>& v) { number(v.size()); for (auto x : v) number(std::bit_cast<std::uint64_t>(x)); }
    void field(const std::vector<float>& v) {
        number(v.size());
        for (auto x : v) { auto bits = std::bit_cast<std::uint32_t>(x); for (int i=0;i<4;++i) bytes.push_back(char(bits >> (i*8))); }
    }
};
struct Reader {
    const std::string& bytes; std::size_t pos = 4;
    explicit Reader(const std::string& b) : bytes(b) { require(b.size() >= 4 && b.substr(0, 4) == "VCA2", "unsupported replay format; regenerate plans with this version"); }
    std::uint64_t number() { require(pos <= bytes.size() && bytes.size() - pos >= 8, "truncated replay"); std::uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= std::uint64_t(static_cast<unsigned char>(bytes[pos++])) << (8 * i); return v; }
    std::size_t count(std::size_t element_size = 1) { auto n = number(); require(n <= (bytes.size() - pos) / element_size, "invalid replay length"); return std::size_t(n); }
    std::string text() { auto n = count(); auto s = bytes.substr(pos, n); pos += n; return s; }
    Shape shape() { Shape v(count(8)); for (auto& x : v) x = std::size_t(number()); return v; }
    std::vector<double> floats() { std::vector<double> v(count(8)); for (auto& x : v) { x = std::bit_cast<double>(number()); require(std::isfinite(x), "nonfinite replay parameter"); } return v; }
    std::vector<float> field() {
        std::vector<float> v(count(4));
        for (auto& x : v) { std::uint32_t bits=0; for(int i=0;i<4;++i) bits |= std::uint32_t(static_cast<unsigned char>(bytes[pos++])) << (i*8);
            x=std::bit_cast<float>(bits); require(std::isfinite(x), "nonfinite replay field"); }
        return v;
    }
};
inline std::string save_plan(const Plan& plan) {
    Writer w; w.number(plan.inputs.size());
    for (const auto& [name, input] : plan.inputs) { w.text(name); w.shape(input.shape); w.number(input.kind); w.number(input.bytes); }
    w.number(plan.steps.size());
    for (const auto& step : plan.steps) {
        w.shape(step.path); const auto& p = step.parameters; w.number(p.channels.size());
        for (const auto& c : p.channels) { w.text(c.key); w.number(c.channel); w.floats(c.values); w.field(c.field); w.number(c.seed); w.shape(c.indices); }
        w.shape(p.shape); w.floats(p.matrix); w.floats(p.offset); w.field(p.field); w.number(p.border); w.number(p.exact); w.number(p.nearest_mode);
    }
    // Detect truncation, corruption, and accidental edits before decoding arrays.
    auto checksum = split_seed(0, w.bytes); w.number(checksum); return w.bytes;
}
inline Plan load_plan(std::shared_ptr<const Program> program, const std::string& bytes) {
    require(bytes.size() >= 12, "truncated replay");
    auto payload = bytes.substr(0, bytes.size() - 8); Reader checksum_reader(bytes); checksum_reader.pos = bytes.size() - 8;
    require(checksum_reader.number() == split_seed(0, payload), "replay checksum mismatch");
    Reader r(payload); Plan plan; plan.program = std::move(program);
    auto inputs = r.count();
    for (std::size_t i = 0; i < inputs; ++i) {
        auto name = r.text(); InputSpec spec; spec.shape = r.shape(); spec.kind = char(r.number()); spec.bytes = std::size_t(r.number());
        require(plan.program->targets.count(name), "replay target is missing from pipeline"); plan.inputs.emplace(std::move(name), std::move(spec));
    }
    auto steps = r.count();
    for (std::size_t i = 0; i < steps; ++i) {
        Step step; step.path = r.shape(); resolve(*plan.program, step.path); auto count = r.count();
        for (std::size_t j = 0; j < count; ++j) { ChannelPlan c; c.key = r.text(); c.channel = std::size_t(r.number()); c.values = r.floats(); c.field = r.field(); c.seed = r.number(); c.indices = r.shape(); step.parameters.channels.push_back(std::move(c)); }
        auto& p = step.parameters; p.shape = r.shape(); p.matrix = r.floats(); p.offset = r.floats(); p.field = r.field(); p.border = r.number(); p.exact = r.number(); p.nearest_mode = int(r.number());
        plan.steps.push_back(std::move(step));
    }
    require(r.pos == payload.size(), "unexpected trailing replay data"); return plan;
}
} // namespace vcr::transforms
