#include <utils/test.hpp>
#include <utils/compositing.hpp>
#include <vector>
#include <cmath>

using namespace utils;

// ---- composite_mean --------------------------------------------------------

TEST_CASE("composite_mean") {
    std::vector<float> layers = {1.0f, 2.0f, 3.0f};
    CHECK_NEAR(composite_mean(layers), 2.0f, 1e-6);
}

TEST_CASE("composite_mean empty") {
    std::vector<float> empty;
    CHECK_NEAR(composite_mean(empty), 0.0f, 1e-6);
}

TEST_CASE("composite_mean single") {
    std::vector<float> layers = {5.0f};
    CHECK_NEAR(composite_mean(layers), 5.0f, 1e-6);
}

// ---- composite_max / composite_min ----------------------------------------

TEST_CASE("composite_max") {
    std::vector<float> layers = {1.0f, 5.0f, 3.0f};
    CHECK_NEAR(composite_max(layers), 5.0f, 1e-6);
}

TEST_CASE("composite_min") {
    std::vector<float> layers = {1.0f, 5.0f, 3.0f};
    CHECK_NEAR(composite_min(layers), 1.0f, 1e-6);
}

TEST_CASE("composite_max empty") {
    std::vector<float> empty;
    CHECK_NEAR(composite_max(empty), 0.0f, 1e-6);
}

TEST_CASE("composite_min empty") {
    std::vector<float> empty;
    CHECK_NEAR(composite_min(empty), 0.0f, 1e-6);
}

// ---- composite_alpha -------------------------------------------------------

TEST_CASE("composite_alpha fully transparent") {
    // All values below alpha_min -> fully transparent -> 0
    std::vector<float> layers = {-1.0f, -2.0f};
    CHECK_NEAR(composite_alpha(layers, 0.0f, 1.0f), 0.0f, 1e-6);
}

TEST_CASE("composite_alpha basic") {
    // Single layer with value 127.5 (half of 255) mapped linearly from [0,1]
    // The function normalizes by dividing by 255*range, so use 255-scale values.
    std::vector<float> layers = {127.5f};
    float result = composite_alpha(layers, 0.0f, 1.0f);
    // normalized = 127.5 / (255*1) - 0 = 0.5
    // alpha = 0.5, value_acc = (1-0)*0.5 * 0.5 = 0.25
    // result = 0.25 * 255 = 63.75
    CHECK_NEAR(result, 63.75f, 1e-3);
}

TEST_CASE("composite_alpha opaque layer") {
    // Layer with value 255 at alpha_max -> fully opaque
    std::vector<float> layers = {255.0f, 127.5f};
    float result = composite_alpha(layers, 0.0f, 1.0f);
    // First layer: normalized = 255/(255*1) = 1.0, opacity=1, weight=1*1=1
    //   value_acc = 1*1 = 1, alpha = 1
    // Second layer: alpha >= cutoff (1.0), break
    // result = 1.0 * 255 = 255
    CHECK_NEAR(result, 255.0f, 1e-3);
}

// ---- composite_beer_lambert ------------------------------------------------

TEST_CASE("beer_lambert empty") {
    std::vector<float> empty;
    CHECK_NEAR(composite_beer_lambert(empty, 1.0f, 1.0f, 0.1f), 0.1f, 1e-6);
}

TEST_CASE("beer_lambert zero density") {
    std::vector<float> layers = {0.0f, 0.0f};
    // T = exp(0) = 1 for each, accumulated = 0, result = ambient * 1 = 0.1
    CHECK_NEAR(composite_beer_lambert(layers, 1.0f, 1.0f, 0.1f), 0.1f, 1e-6);
}

TEST_CASE("beer_lambert single layer") {
    std::vector<float> layers = {1.0f};
    float ext = 2.0f, em = 1.5f, amb = 0.0f;
    // accumulated = 1.5 * 1.0 * 1.0 = 1.5
    // T = exp(-2) ~ 0.1353
    // transmittance = 0.1353
    // result = 1.5 + 0.0 * 0.1353 = 1.5
    CHECK_NEAR(composite_beer_lambert(layers, ext, em, amb), 1.5f, 1e-4);
}

// ---- Lambertian lighting ---------------------------------------------------

TEST_CASE("lambertian_factor") {
    // Normal pointing up (z), light from directly above
    std::array<float, 3> normal = {0.0f, 0.0f, 1.0f};
    float factor = lambertian_factor(normal, 0.0f, static_cast<float>(M_PI / 2.0));
    CHECK_NEAR(factor, 1.0f, 1e-4);
}

TEST_CASE("lambertian_factor perpendicular") {
    // Normal pointing up, light from side
    std::array<float, 3> normal = {0.0f, 0.0f, 1.0f};
    float factor = lambertian_factor(normal, 0.0f, 0.0f);  // elevation 0
    CHECK_NEAR(factor, 0.0f, 1e-4);
}

// ---- window_level, value_stretch, saturate ---------------------------------

TEST_CASE("saturate") {
    CHECK_NEAR(saturate(-0.5f), 0.0f, 1e-6);
    CHECK_NEAR(saturate(0.5f), 0.5f, 1e-6);
    CHECK_NEAR(saturate(1.5f), 1.0f, 1e-6);
}

TEST_CASE("window_level") {
    // Window=100, Level=50: maps [0, 100] -> [0, 1]
    CHECK_NEAR(window_level(0.0f, 100.0f, 50.0f), 0.0f, 1e-6);
    CHECK_NEAR(window_level(50.0f, 100.0f, 50.0f), 0.5f, 1e-6);
    CHECK_NEAR(window_level(100.0f, 100.0f, 50.0f), 1.0f, 1e-6);
    CHECK_NEAR(window_level(-10.0f, 100.0f, 50.0f), 0.0f, 1e-6);
}

TEST_CASE("window_level zero window") {
    CHECK_NEAR(window_level(50.0f, 0.0f, 50.0f), 0.0f, 1e-6);
}

TEST_CASE("value_stretch") {
    std::vector<float> data = {2.0f, 4.0f, 6.0f, 8.0f};
    value_stretch(data);
    CHECK_NEAR(data[0], 0.0f, 1e-6);
    CHECK_NEAR(data[3], 1.0f, 1e-6);
    CHECK_NEAR(data[1], 1.0f/3.0f, 1e-5);
}

TEST_CASE("value_stretch constant") {
    std::vector<float> data = {5.0f, 5.0f, 5.0f};
    value_stretch(data);
    for (auto v : data)
        CHECK_NEAR(v, 0.0f, 1e-6);
}

TEST_CASE("value_stretch empty") {
    std::vector<float> empty;
    value_stretch(empty);  // should not crash
    CHECK(empty.empty());
}

// ---- composite_image batch -------------------------------------------------

TEST_CASE("composite_image batch mean") {
    // 3 pixels, 2 layers each
    std::vector<float> layers = {
        1.0f, 3.0f,   // pixel 0
        2.0f, 4.0f,   // pixel 1
        5.0f, 5.0f    // pixel 2
    };
    std::vector<float> output(3);
    CompositeParams params;
    params.method = CompositingMethod::mean;

    composite_image(layers, 3, 2, output, params);
    CHECK_NEAR(output[0], 2.0f, 1e-6);
    CHECK_NEAR(output[1], 3.0f, 1e-6);
    CHECK_NEAR(output[2], 5.0f, 1e-6);
}

TEST_CASE("composite_image batch max") {
    std::vector<float> layers = {
        1.0f, 3.0f,
        2.0f, 4.0f
    };
    std::vector<float> output(2);
    CompositeParams params;
    params.method = CompositingMethod::max;

    composite_image(layers, 2, 2, output, params);
    CHECK_NEAR(output[0], 3.0f, 1e-6);
    CHECK_NEAR(output[1], 4.0f, 1e-6);
}

// ---- composite_stack dispatcher --------------------------------------------

TEST_CASE("composite_stack dispatch") {
    std::vector<float> layers = {1.0f, 2.0f, 3.0f};
    CompositeParams p;
    CHECK_NEAR(composite_stack(layers, CompositingMethod::mean, p), 2.0f, 1e-6);
    CHECK_NEAR(composite_stack(layers, CompositingMethod::max, p), 3.0f, 1e-6);
    CHECK_NEAR(composite_stack(layers, CompositingMethod::min, p), 1.0f, 1e-6);
}

UTILS_TEST_MAIN()
