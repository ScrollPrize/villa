#pragma once
#include <QColor>
#include <map>
#include <string>

// Allocate once per source ID so previews, selection and later additions do
// not recolor patches already on screen. Golden-angle hues separate neighbors.
class ConstraintGraphPatchColors {
public:
    QColor color(const std::string& id) {
        if(auto it=_colors.find(id);it!=_colors.end())return it->second;
        const auto ordinal=_colors.size();
        auto color=QColor::fromHsv(int((210+ordinal*137)%360),150+int(ordinal/360%2)*70,255);
        return _colors.emplace(id,color).first->second;
    }
private:
    std::map<std::string,QColor> _colors;
};
