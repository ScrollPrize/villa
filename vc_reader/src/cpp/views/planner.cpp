#include "common.hpp"
#include <algorithm>
#include <cmath>

namespace vcr::views {
using namespace detail;

namespace {
// Supported tokens in any box from a summed-volume table, built once.
class SummedVolume {
    std::int64_t n1_, n2_;
    std::vector<std::int64_t> table_;
    std::int64_t at(std::int64_t z, std::int64_t y, std::int64_t x) const { return table_[std::size_t((z * n1_ + y) * n2_ + x)]; }

public:
    SummedVolume(const std::uint8_t* support, const Index3& t) : n1_(t[1] + 1), n2_(t[2] + 1),
        table_(std::size_t((t[0] + 1) * n1_ * n2_), 0)
    {
        for (std::int64_t z = 0; z < t[0]; ++z)
            for (std::int64_t y = 0; y < t[1]; ++y)
                for (std::int64_t x = 0; x < t[2]; ++x) {
                    const std::int64_t v = support[std::size_t((z * t[1] + y) * t[2] + x)] != 0;
                    table_[std::size_t(((z + 1) * n1_ + y + 1) * n2_ + x + 1)] = v
                        + at(z, y + 1, x + 1) + at(z + 1, y, x + 1) + at(z + 1, y + 1, x)
                        - at(z, y, x + 1) - at(z, y + 1, x) - at(z + 1, y, x) + at(z, y, x);
                }
    }
    // Box [lo, hi), assumed nonempty and inside the grid.
    std::int64_t sum(const Index3& lo, const Index3& hi) const
    {
        return at(hi[0], hi[1], hi[2]) - at(lo[0], hi[1], hi[2]) - at(hi[0], lo[1], hi[2]) - at(hi[0], hi[1], lo[2])
             + at(lo[0], lo[1], hi[2]) + at(lo[0], hi[1], lo[2]) + at(hi[0], lo[1], lo[2]) - at(lo[0], lo[1], lo[2]);
    }
};

Index3 drawOrigin(Stream& r, const Index3& tokens, const Index3& n)
{
    Index3 o{};
    for (int a = 0; a < 3; ++a) o[a] = std::int64_t(r.below(std::uint64_t(tokens[a] - n[a] + 1)));
    return o;
}

void checkShape(const Index3& n, const Index3& tokens, const char* message)
{
    for (int a = 0; a < 3; ++a) require(n[a] > 0 && n[a] <= tokens[a], message);
}
}  // namespace

ViewSets planViewSets(const std::uint8_t* support, const Index3& tokens, const Index3& teacher,
                      const std::vector<Index3>& students, const PlanOptions& options)
{
    for (int a = 0; a < 3; ++a)
        require(tokens[a] > 0 && tokens[a] <= std::numeric_limits<std::int32_t>::max(),
                "support grid shape must be positive and fit int32");
    checkShape(teacher, tokens, "teacher shape must be positive and fit the support grid");
    for (const auto& s : students) checkShape(s, tokens, "student shapes must be positive and fit the support grid");
    require(options.maxStudentAttempts > 0 && options.maxTeacherAttempts > 0, "attempt limits must be positive");
    require(options.maxStudentAttempts <= std::numeric_limits<std::int32_t>::max() &&
            options.maxTeacherAttempts <= std::numeric_limits<std::int32_t>::max(), "attempt limits must fit int32");
    require(std::isfinite(options.minOverlapFraction) && options.minOverlapFraction >= 0,
            "min_overlap_fraction must be finite and nonnegative");
    require(options.minOverlapCount >= 0, "min_overlap_count must be nonnegative");
    require(support != nullptr, "support is required");

    const auto S = students.size(), sets = options.sets;
    ViewSets out;
    out.sets = sets;
    out.students = S;
    out.ok = true;
    out.setOk.assign(sets, 0);
    out.teacherOrigins.assign(3 * sets, 0);
    out.studentOrigins.assign(3 * sets * S, 0);
    out.studentOverlap.assign(sets * S, 0);
    out.studentAttempts.assign(sets * S, 0);
    out.teacherAttempts.assign(sets, 0);
    out.rejections.assign(sets, 0);
    if (!sets) return out;

    const SummedVolume table(support, tokens);
    std::vector<double> needed(S);
    for (std::size_t s = 0; s < S; ++s) needed[s] = options.minOverlapFraction * double(product(students[s]));
    std::vector<Index3> origins(S);
    std::vector<std::int64_t> overlap(S), attempts(S);
    Stream r(options.seed);
    for (std::size_t set = 0; set < sets; ++set) {
        std::int64_t rejected = 0, teacherAttempt = 0;
        bool placed = false;
        Index3 t{};
        while (!placed && teacherAttempt < options.maxTeacherAttempts) {
            ++teacherAttempt;
            t = drawOrigin(r, tokens, teacher);
            placed = true;
            for (std::size_t s = 0; s < S && placed; ++s) {
                bool accepted = false;
                for (std::int64_t a = 1; a <= options.maxStudentAttempts; ++a) {
                    const auto o = drawOrigin(r, tokens, students[s]);
                    Index3 lo{}, hi{};
                    bool empty = false;
                    for (int k = 0; k < 3; ++k) {
                        lo[k] = std::max(o[k], t[k]);
                        hi[k] = std::min(o[k] + students[s][k], t[k] + teacher[k]);
                        empty = empty || hi[k] <= lo[k];
                    }
                    const std::int64_t O = empty ? 0 : table.sum(lo, hi);
                    if (O >= options.minOverlapCount && double(O) >= needed[s]) {
                        origins[s] = o;
                        overlap[s] = O;
                        attempts[s] = a;
                        accepted = true;
                        break;
                    }
                    ++rejected;
                }
                placed = accepted;
            }
        }
        out.teacherAttempts[set] = std::int32_t(teacherAttempt);
        out.rejections[set] = rejected;
        if (!placed) {
            out.ok = false;
            continue;
        }
        out.setOk[set] = 1;
        for (int k = 0; k < 3; ++k) out.teacherOrigins[3 * set + k] = std::int32_t(t[k]);
        for (std::size_t s = 0; s < S; ++s) {
            for (int k = 0; k < 3; ++k) out.studentOrigins[3 * (set * S + s) + k] = std::int32_t(origins[s][k]);
            out.studentOverlap[set * S + s] = overlap[s];
            out.studentAttempts[set * S + s] = std::int32_t(attempts[s]);
        }
    }
    return out;
}
}  // namespace vcr::views
