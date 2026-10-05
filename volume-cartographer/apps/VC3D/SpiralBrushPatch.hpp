#pragma once

#include <QColor>
#include <QPainterPath>
#include <QPointF>
#include <QString>
#include <memory>

class QuadSurface;
enum class SpiralGestureState { Painted, Ready, Finalizing, Finalized };

// Canonical selection stays in its original source grid. Saves snapshot it;
// completion of an older snapshot must never mark a newer edit finalized.
struct SpiralBrushPatch
{
    QString id;
    QColor color;
    std::shared_ptr<QuadSurface> source;
    QPainterPath shape;
    QPointF gridOrigin;
    QPointF columnStep;
    QPointF rowStep;
    SpiralGestureState state = SpiralGestureState::Painted;
    bool staged = false;
    bool removed = false;
    QString error;
    quint64 generation = 1;
    quint64 attempt = 0;
    quint64 submittedGeneration = 0;
    bool uploadInFlight = false;
    QPainterPath submittedShape;
    QPainterPath acceptedShape;
    QPainterPath lastNonemptyShape;
    std::shared_ptr<QuadSurface> lastPatch;
    std::shared_ptr<QuadSurface> submittedPatch;

    void changed()
    {
        state = SpiralGestureState::Painted;
        error.clear();
        ++generation;
    }
    void submitted()
    {
        state = SpiralGestureState::Finalizing;
        error.clear();
        submittedGeneration = generation;
        submittedShape = shape;
        uploadInFlight = true;
    }
    void accepted()
    {
        uploadInFlight = false;
        acceptedShape = submittedShape;
        if (!submittedShape.isEmpty()) lastNonemptyShape = submittedShape;
        if (!submittedShape.isEmpty()) lastPatch = submittedPatch;
        submittedPatch.reset();
        staged = true;
        if (generation == submittedGeneration && error.isEmpty()) state = SpiralGestureState::Finalized;
    }
    void setRemoved(bool value)
    {
        if (removed && !value && state == SpiralGestureState::Finalized && shape.isEmpty()) {
            shape = lastNonemptyShape;
            acceptedShape = shape;
        }
        removed = value;
    }
    bool visible() const { return !(removed && state == SpiralGestureState::Finalized); }
    bool removableLocally() const { return !uploadInFlight; }
    void failed(const QString& message)
    {
        uploadInFlight = false;
        submittedPatch.reset();
        if (generation != submittedGeneration) return;
        error = message;
        state = SpiralGestureState::Ready;
    }
    bool emptyLocal() const
    {
        return shape.isEmpty() && !staged && !uploadInFlight;
    }
};
