#pragma once

class SegmentationTool
{
public:
    virtual ~SegmentationTool() = default;

    virtual void cancel() = 0;
    virtual bool isActive() const = 0;
};

