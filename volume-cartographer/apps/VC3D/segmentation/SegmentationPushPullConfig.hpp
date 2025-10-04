#pragma once

struct AlphaPushPullConfig
{
    float start{-6.0f};
    float stop{30.0f};
    float step{2.0f};
    float low{0.1f};
    float high{1.0f};
    float borderOffset{1.0f};
    int blurRadius{3};
    float perVertexLimit{5.0f};
    bool perVertex{false};
};
