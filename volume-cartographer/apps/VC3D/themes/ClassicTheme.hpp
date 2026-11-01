#pragma once

// The classic theme: VC3D's original look. The native style, a dark palette
// when the system prefers dark, and a gradient menu bar set on the main window.

#include <QString>

namespace vc3d::theme {

// Dark mode detection - works on all Qt 6.x versions
bool classicPrefersDark();
// Apply a consistent dark palette application-wide
void applyClassicDarkPalette();
// The main window's style sheet in the classic theme.
QString classicWindowStyleSheet(bool dark);

}  // namespace vc3d::theme
