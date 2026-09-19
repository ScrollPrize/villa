#pragma once
#include <QJsonArray>
#include <QJsonObject>

namespace vc3d::spiral {
inline QJsonArray mergePatchDraftRows(QJsonArray inputs, const QJsonArray& drafts)
{
    for (const auto& value : drafts) {
        QJsonObject draft = value.toObject();
        if (draft.value("local").toBool()) {
            draft["dirty"] = true;
            draft["committed"] = false;
            draft["can_restore"] = false;
        }
        bool found = false;
        for (int i = 0; i < inputs.size(); ++i) {
            QJsonObject input = inputs[i].toObject();
            if (input.value("kind") != draft.value("kind") || input.value("alias").toString(input.value("id").toString()) != draft.value("id").toString()) continue;
            for (auto field = draft.begin(); field != draft.end(); ++field)
                if (field.key() != "id") input[field.key()] = field.value();
            inputs[i] = input;
            found = true;
            break;
        }
        if (!found) inputs.append(draft);
    }
    return inputs;
}
} // namespace vc3d::spiral
