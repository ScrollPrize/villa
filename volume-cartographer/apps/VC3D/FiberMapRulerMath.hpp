#pragma once

// Tick arithmetic for the Fiber Map's rulers, kept free of any widget so it
// can be tested on its own: the 1-2-5 step ladder, the unit a physical ruler
// labels in, and the number formatting.

#include <QString>

#include <algorithm>
#include <cmath>

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

} // namespace vc3d::fiber_map::ruler
