#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "vc/core/util/Compositing.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace {

LayerStack stack(std::vector<float> v)
{
    LayerStack s;
    s.values = std::move(v);
    s.validCount = static_cast<int>(s.values.size());
    return s;
}

LayerStack emptyStack()
{
    return LayerStack{};
}

} // namespace

TEST_CASE("CompositeMethod::mean")
{
    CHECK(CompositeMethod::mean(emptyStack()) == doctest::Approx(0.0f));
    CHECK(CompositeMethod::mean(stack({0.f, 10.f, 20.f})) == doctest::Approx(10.0f));
    CHECK(CompositeMethod::mean(stack({100.f})) == doctest::Approx(100.0f));
}

TEST_CASE("CompositeMethod::max")
{
    CHECK(CompositeMethod::max(emptyStack()) == doctest::Approx(0.0f));
    CHECK(CompositeMethod::max(stack({1.f, 5.f, 2.f})) == doctest::Approx(5.0f));
}

TEST_CASE("CompositeMethod::min")
{
    // empty → 255.0 (sentinel: nothing observed)
    CHECK(CompositeMethod::min(emptyStack()) == doctest::Approx(255.0f));
    CHECK(CompositeMethod::min(stack({5.f, 1.f, 3.f})) == doctest::Approx(1.0f));
}

TEST_CASE("CompositeMethod::alpha empty stack is 0")
{
    CompositeParams p;
    CHECK(CompositeMethod::alpha(emptyStack(), p) == doctest::Approx(0.0f));
}

TEST_CASE("CompositeMethod::alpha returns finite [0,255] value")
{
    CompositeParams p;
    p.alphaMin = 0.0f;
    p.alphaMax = 1.0f;
    p.alphaOpacity = 1.0f;
    p.alphaCutoff = 1.0f;
    float v = CompositeMethod::alpha(stack({50.f, 100.f, 200.f}), p);
    CHECK(std::isfinite(v));
    CHECK(v >= 0.0f);
    CHECK(v <= 255.0f);
}

TEST_CASE("CompositeMethod::beerLambert empty is 0")
{
    CompositeParams p;
    CHECK(CompositeMethod::beerLambert(emptyStack(), p) == doctest::Approx(0.0f));
}

TEST_CASE("CompositeMethod::beerLambert clamps to 255")
{
    CompositeParams p;
    p.blExtinction = 0.001f;
    p.blEmission = 1000.0f;
    p.blAmbient = 0.0f;
    float v = CompositeMethod::beerLambert(stack({255.f, 255.f, 255.f, 255.f}), p);
    CHECK(v <= 255.0f);
    CHECK(v > 0.0f);
}

TEST_CASE("CompositeMethod::beerLambert ambient survives when stack is all-below-threshold")
{
    CompositeParams p;
    p.blAmbient = 0.5f;
    // all values below 0.255 → loop skips them; ambient * transmittance(=1) * 255 ≈ 127.5
    float v = CompositeMethod::beerLambert(stack({0.f, 0.1f, 0.2f}), p);
    CHECK(v == doctest::Approx(127.5f).epsilon(1e-3));
}

TEST_CASE("CompositeMethod::beerLambert early break on opacity")
{
    CompositeParams p;
    p.blExtinction = 1000.0f; // huge — transmittance crashes after first layer
    p.blEmission = 0.0f;
    p.blAmbient = 0.0f;
    float v = CompositeMethod::beerLambert(stack({255.f, 255.f, 255.f}), p);
    CHECK(v == doctest::Approx(0.0f));
}

TEST_CASE("compositeLayerStack dispatches per method")
{
    LayerStack s = stack({0.f, 100.f, 200.f});

    CompositeParams p;
    p.method = "mean";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(100.0f));
    p.method = "max";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(200.0f));
    p.method = "min";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(0.0f));
}

TEST_CASE("compositeLayerStack: alpha / beerLambert dispatch returns finite")
{
    LayerStack s = stack({50.f, 100.f, 200.f});
    CompositeParams p;
    p.method = "alpha";
    CHECK(std::isfinite(compositeLayerStack(s, p)));
    p.method = "beerLambert";
    CHECK(std::isfinite(compositeLayerStack(s, p)));
}

TEST_CASE("compositeLayerStack empty stack always returns 0")
{
    CompositeParams p;
    const std::vector<std::string> methods = {
        "mean", "max", "min", "alpha", "beerLambert"
    };
    for (const auto& m : methods) {
        p.method = m;
        CHECK(compositeLayerStack(emptyStack(), p) == doctest::Approx(0.0f));
    }
}

TEST_CASE("compositeLayerStack unknown method falls back to mean")
{
    LayerStack s = stack({0.f, 100.f, 200.f});
    CompositeParams p;
    p.method = "this_does_not_exist";
    CHECK(compositeLayerStack(s, p) == doctest::Approx(100.0f));
}

TEST_CASE("CompositeParams equality")
{
    CompositeParams a;
    CompositeParams b;
    CHECK(a == b);
    b.alphaOpacity = 0.5f;
    CHECK_FALSE(a == b);
}

TEST_CASE("CompositeRenderSettings equality")
{
    CompositeRenderSettings a;
    CompositeRenderSettings b;
    CHECK(a == b);
    b.layersFront = 999;
    CHECK_FALSE(a == b);
}
