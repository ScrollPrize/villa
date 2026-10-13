#pragma once

// Tick arithmetic for the Fiber Map's rulers, kept free of any widget so it
// can be tested on its own: the 1-2-5 step ladder, the unit a physical ruler
// labels in, and the number formatting.

#include <QString>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace vc3d::fiber_map::ruler
{

// The smallest value of the form {1, 2, 5} * 10^k that is >= minStep. A
// non-positive or non-finite minStep yields 1.
inline double niceStepAtLeast(double minStep)
{
    if (!std::isfinite(minStep) || minStep <= 0.0) {
        return 1.0;
    }
    const double magnitude = std::pow(10.0, std::floor(std::log10(minStep)));
    for (const double mantissa : {1.0, 2.0, 5.0}) {
        const double candidate = mantissa * magnitude;
        // The tolerance keeps 10^k from being skipped when minStep is exactly
        // 10^k but log10 rounded it just below.
        if (candidate >= minStep * (1.0 - 1e-12)) {
            return candidate;
        }
    }
    return 10.0 * magnitude;
}

// Winding rulers label integers only: the ladder value, never below 1.
inline int niceIntegerStepAtLeast(double minStep)
{
    // Saturated well below INT_MAX: past this no winding label would ever
    // be drawn anyway, and the narrowing stays defined.
    constexpr double kMaxStep = 1e9;
    return static_cast<int>(std::lround(std::clamp(niceStepAtLeast(minStep), 1.0, kMaxStep)));
}

enum class LengthUnit { Micrometre, Millimetre, Centimetre, Metre };

inline double lengthUnitUm(LengthUnit unit)
{
    switch (unit) {
    case LengthUnit::Micrometre:
        return 1.0;
    case LengthUnit::Millimetre:
        return 1000.0;
    case LengthUnit::Centimetre:
        return 10000.0;
    case LengthUnit::Metre:
        return 1000000.0;
    }
    return 1.0;
}

inline QString lengthUnitSuffix(LengthUnit unit)
{
    switch (unit) {
    case LengthUnit::Micrometre:
        return QStringLiteral("µm");
    case LengthUnit::Millimetre:
        return QStringLiteral("mm");
    case LengthUnit::Centimetre:
        return QStringLiteral("cm");
    case LengthUnit::Metre:
        return QStringLiteral("m");
    }
    return QString();
}

// The unit every label of one ruler shares, chosen from the tick step so the
// labels read as small whole numbers: metres from 10 cm steps up, centimetres
// from 1 cm steps, millimetres from 0.1 mm steps, micrometres below. maxUnit
// caps the climb: a ruler for a quantity that is never metres long (a
// scroll's height) stays in centimetres however coarse its ticks.
inline LengthUnit lengthUnitForStepUm(double stepUm, LengthUnit maxUnit = LengthUnit::Metre)
{
    LengthUnit unit = LengthUnit::Micrometre;
    if (stepUm >= 100000.0) {
        unit = LengthUnit::Metre;
    } else if (stepUm >= 10000.0) {
        unit = LengthUnit::Centimetre;
    } else if (stepUm >= 100.0) {
        unit = LengthUnit::Millimetre;
    }
    return lengthUnitUm(unit) > lengthUnitUm(maxUnit) ? maxUnit : unit;
}

// A length in the given unit, with only the decimals the value needs (up to
// three), so 12.5 mm and 1.25 m print as such and 20 cm prints as "20".
inline QString formatLength(double valueUm, LengthUnit unit)
{
    const double value = valueUm / lengthUnitUm(unit);
    const double rounded = std::round(value * 1000.0) / 1000.0;
    if (std::abs(rounded) < 0.0005) {
        return QStringLiteral("0");
    }
    QString text = QString::number(rounded, 'f', 3);
    while (text.endsWith(QLatin1Char('0'))) {
        text.chop(1);
    }
    if (text.endsWith(QLatin1Char('.'))) {
        text.chop(1);
    }
    return text;
}

// A voxel count for a ruler without a voxel size: whole numbers, with a "k"
// suffix from a thousand up so 20000 reads as "20k" and 2500 as "2.5k".
inline QString formatVoxels(double voxels)
{
    const double rounded = std::round(voxels);
    if (std::abs(rounded) < 0.5) {
        return QStringLiteral("0");
    }
    if (std::abs(rounded) < 1000.0) {
        return QString::number(static_cast<long long>(rounded));
    }
    QString text = QString::number(rounded / 1000.0, 'f', 2);
    while (text.endsWith(QLatin1Char('0'))) {
        text.chop(1);
    }
    if (text.endsWith(QLatin1Char('.'))) {
        text.chop(1);
    }
    return text + QLatin1Char('k');
}

// The inclusive range of tick indices k (tick at k * step) covering
// [low, high] with one spare on each side, or nullopt when the range is not
// finite or would exceed maxSteps - decided in floating point, before
// anything is narrowed to an integer.
inline std::optional<std::pair<long long, long long>> tickIndexRange(double low, double high,
                                                                     double step, int maxSteps)
{
    if (!(step > 0.0) || !std::isfinite(step) || !std::isfinite(low) || !std::isfinite(high) ||
        !(high >= low)) {
        return std::nullopt;
    }
    const double first = std::ceil(low / step) - 1.0;
    const double last = std::floor(high / step) + 1.0;
    if (!std::isfinite(first) || !std::isfinite(last) || last - first > maxSteps ||
        std::abs(first) > 1e15 || std::abs(last) > 1e15) {
        return std::nullopt;
    }
    return std::make_pair(static_cast<long long>(first), static_cast<long long>(last));
}

struct DistanceTick {
    double distance = 0.0;
    bool major = false;
};

// The ticks of a distance ruler over [low, high]: a major tick at every
// multiple of step and a minor one halfway between, with one spare step on
// each side (tickIndexRange), except that nothing lies below `floor`: the
// Fiber Map's scene continues below the sheet model's domain floor at the
// map's own scale, and there is no sheet distance there to label. Ascending.
// Empty when tickIndexRange declines the range.
inline std::vector<DistanceTick> distanceTickCandidates(double low, double high, double step,
                                                        double floor, int maxSteps)
{
    std::vector<DistanceTick> ticks;
    const auto range = tickIndexRange(low, high, step, maxSteps);
    if (!range) {
        return ticks;
    }
    for (long long k = range->first; k <= range->second; ++k) {
        for (int half = 0; half < 2; ++half) {
            const double distance = (static_cast<double>(k) + 0.5 * half) * step;
            if (distance < floor) {
                continue;
            }
            ticks.push_back(DistanceTick{distance, half == 0});
        }
    }
    return ticks;
}

// The smallest gap between neighbouring entries of an ascending sequence of
// positions, over the neighbour pairs that overlap [lo, hi]; over every pair
// when none does; `fallback` when there is no pair. The winding ruler picks
// its label step from it: the scene is scaled by sheet distance, so windings
// are not equally wide and labels must stay apart at the tightest place on
// screen.
template <class Iterator, class Position>
double narrowestNeighbourGap(Iterator first, Iterator last, Position positionOf, double lo,
                             double hi, double fallback)
{
    if (first == last) {
        return fallback;
    }
    double onScreen = std::numeric_limits<double>::infinity();
    double anywhere = std::numeric_limits<double>::infinity();
    Iterator previous = first;
    for (Iterator it = std::next(first); it != last; previous = it, ++it) {
        const double a = positionOf(*previous);
        const double b = positionOf(*it);
        const double gap = b - a;
        if (!(gap > 0.0) || !std::isfinite(gap)) {
            continue;
        }
        anywhere = std::min(anywhere, gap);
        if (a <= hi && b >= lo) {
            onScreen = std::min(onScreen, gap);
        }
    }
    if (std::isfinite(onScreen)) {
        return onScreen;
    }
    return std::isfinite(anywhere) ? anywhere : fallback;
}

} // namespace vc3d::fiber_map::ruler
