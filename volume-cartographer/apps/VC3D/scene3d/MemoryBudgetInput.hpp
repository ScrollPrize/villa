#pragma once
#include <QDoubleSpinBox>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vc3d::scene3d
{
inline constexpr size_t defaultMemoryBudget = size_t(2048) * 1024 * 1024;

// Integer MiB stored in a double spin box avoids QSpinBox's signed-int limit.
// The only upper bound is what this process can represent as a byte count.
inline QDoubleSpinBox* makeMemoryBudgetInput(QWidget* parent, double savedMiB)
{
    auto* input = new QDoubleSpinBox(parent);
    input->setDecimals(0);
    input->setRange(64, double(std::numeric_limits<size_t>::max() / (1024 * 1024)));
    input->setSingleStep(1024);
    input->setSuffix(QStringLiteral(" MiB"));
    input->setKeyboardTracking(false);
    input->setValue(std::isfinite(savedMiB) && savedMiB > 0 ? savedMiB : 2048);
    return input;
}
inline size_t memoryBudgetBytes(const QDoubleSpinBox* input)
{
    return size_t(input->value()) * 1024 * 1024;
}
}
