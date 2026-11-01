// Built only with VC_TESTING. Exercise the actual panel and shared controllers.
#include "ViewerManager.hpp"
#include "volume_viewers/VolumeViewerBase.hpp"
#include "vc/core/util/PlaneSurface.hpp"
#include "vc/core/util/QuadSurface.hpp"
#include "support/TestZarrVolume.hpp"
#include <QApplication>
#include <QElapsedTimer>
#include <QTemporaryDir>
#include <QThread>
#include <QSlider>
#include <QDoubleSpinBox>
#include <iostream>
#include "VolumePreviewTest.inc"

int runScene3DPanelTests()
{
    using namespace vc3d::scene3d;
    auto require = [](bool ok, const char* message) {
        if (!ok) throw std::runtime_error(message);
    };
    auto settle = [&](const std::function<bool()>& ready) {
        QElapsedTimer timer;
        timer.start();
        while (!ready() && timer.elapsed() < 60000) {
            QApplication::processEvents();
            QThread::msleep(5);
        }
        require(ready(), "3D panel content did not settle");
    };
    try {
        QTemporaryDir temporary;
        // Keep preferences/cache out of the user's configuration.
        qputenv("VC3D_CONFIG_DIR", temporary.path().toUtf8());
        const auto volumePath = qEnvironmentVariable("VC3D_SCENE3D_VOLUME");
        const auto surfacePath = qEnvironmentVariable("VC3D_SCENE3D_SURFACE");
        const bool render = !qEnvironmentVariableIsEmpty("VC3D_SCENE3D_SCREENSHOT");
        if (volumePath.isEmpty()) testVolumePreview(temporary.path().toStdString());
        vc::test::ZarrVolumeSpec spec;
        spec.shapeZYX = {32,32,32}; spec.chunkShapeZYX = {32,32,32};
        spec.numLevels = 4;
        auto volume = volumePath.isEmpty()
            ? vc::test::createZarrVolume(std::filesystem::path(temporary.path().toStdString()) / "ct.zarr", spec)
            : Volume::New(volumePath.toStdString());
        if (qEnvironmentVariableIsSet("VC3D_SCENE3D_BENCHMARK")) {
            require(!volumePath.isEmpty(), "benchmark requires VC3D_SCENE3D_VOLUME");
            const int mib = qEnvironmentVariableIntValue("VC3D_SCENE3D_BENCHMARK_MIB");
            const size_t budget = size_t(mib > 0 ? mib : 512)*1024*1024;
            for (int iteration=0; iteration<3; ++iteration) {
                volume->invalidateCache();
                std::atomic_bool stop{false};
                QElapsedTimer elapsed; elapsed.start();
                auto data=sampleVolume(*volume,3,budget,stop);
                require(data && !data->images.empty(), "benchmark volume is empty");
                const auto& slab=*data->images.front()->slab;
                std::cout << "iteration=" << iteration+1 << " elapsed_ms=" << elapsed.elapsed()
                          << " level=" << data->images.front()->level << " preview=" << slab.width << 'x' << slab.height << 'x' << slab.depth
                          << " texture_bytes=" << slab.pixels.size() << std::endl;
            }
            return 0;
        }
        if (volumePath.isEmpty()) {
            Array3D<uint8_t> ramp({32,32,32});
            for (int z=0; z<32; ++z)
                for (int y=0; y<32; ++y)
                    for (int x=0; x<32; ++x) ramp(z,y,x) = uint8_t(3*x + 3*y + z);
            vc::test::writeZYX(*volume, ramp, {0,0,0});
            // Distinguish the selected source level from a fine-level sample
            // at the same coordinates, rather than just inspecting metadata.
            Array3D<uint8_t> coarse({4,4,4}, uint8_t(190));
            vc::test::writeZYX(*volume, coarse, {0,0,0}, 3);
            auto oneLevelSpec = spec;
            oneLevelSpec.numLevels = 1;
            auto singleLevel = vc::test::createZarrVolume(std::filesystem::path(temporary.path().toStdString()) / "single.zarr", oneLevelSpec);
            CState fallbackState;
            fallbackState.setCurrentVolume(singleLevel);
            ViewerManager fallbackViews(&fallbackState, fallbackState.pointCollection());
            Scene3DPanel fallbackPanel(&fallbackState, &fallbackViews);
            auto* fallbackLevel = fallbackPanel.findChild<QComboBox*>("scene3DSurfaceTextureLevel");
            require(fallbackLevel && fallbackLevel->count() == 1 && fallbackLevel->currentData().toInt() == 0,
                    "single-level volumes fall back to their available level");
            fallbackState.setCurrentVolume(volume);
            require(fallbackLevel->currentData().toInt() == 3, "temporary fallback preserves the default level 3 preference");
        }
        cv::Mat_<cv::Vec3f> points(4,4);
        for (int y=0; y<4; ++y)
            for (int x=0; x<4; ++x) points(y,x) = {float(10+x),float(10+y),12};
        auto surface = surfacePath.isEmpty() ? std::make_shared<QuadSurface>(points, cv::Vec2f{1,1})
                                            : std::make_shared<QuadSurface>(surfacePath.toStdString());
        cv::Vec3f center{16,16,16};
        if (!surfacePath.isEmpty()) {
            const auto raw = surface->rawPoints();
            bool found = false;
            for (int y=0; y<raw.rows && !found; ++y)
                for (int x=0; x<raw.cols; ++x) {
                    const auto p = raw(y,x);
                    if (p[0] >= 0 && std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2])) {
                        center = p; found = true; break;
                    }
                }
            require(found, "real surface has no valid points");
        }
        CState state;
        state.setCurrentVolume(volume);
        state.setSurface("segmentation", surface);
        ViewerManager viewers(&state, state.pointCollection());
        QWidget ctHost;
        const std::array<std::string,3> names{"xy plane", "seg xz", "seg yz"};
        const std::array<cv::Vec3f,3> normals{{{0,0,1}, {0,1,0}, {1,0,0}}};
        for (int i=0; i<3; ++i) {
            state.setSurface(names[i], std::make_shared<PlaneSurface>(center, normals[i]));
            auto* view = viewers.createViewerInWidget(names[i], &ctHost);
            auto settings = view->compositeRenderSettings();
            settings.params.method = "max";
            settings.planeLayersFront = 3;
            settings.planeLayersBehind = 2;
            view->setCompositeRenderSettings(settings);
        }
        Scene3DPanel panel(&state, &viewers);
        auto* memoryBudget = panel.findChild<QDoubleSpinBox*>("scene3DMemoryBudgetMiB");
        require(memoryBudget && memoryBudget->value() == 2048, "3D memory budget defaults to 2 GiB");
        if (volumePath.isEmpty()) {
            memoryBudget->setValue(65536);
            require(memoryBudget->value() == 65536 && memoryBudgetBytes(memoryBudget) == size_t(65536)*1024*1024,
                    "memory budget accepts typed values above the old preset limits without narrowing");
            const double maximum = memoryBudget->maximum();
            memoryBudget->setValue(maximum);
            require(memoryBudgetBytes(memoryBudget)/(1024*1024) == size_t(maximum), "largest representable budget converts to bytes without overflow");
            memoryBudget->setValue(65536);
        }
        auto* canvas = panel.findChild<Scene3DCanvas*>("scene3DCanvas");
        require(canvas, "general viewer canvas exists");
        if (!render) canvas->hide();
        panel.resize(1100, 900);
        panel.show();
        settle([&] { return canvas->surfaceMesh() && !canvas->ctSlices().empty() && !canvas->ctSlices().front()->preview; });
        auto mesh = canvas->surfaceMesh();
        require(mesh->texture && !mesh->texture->pixels.empty(), "mesh has CT texture by default");
        auto* textureLevel = panel.findChild<QComboBox*>("scene3DSurfaceTextureLevel");
        require(textureLevel && textureLevel->currentData().toInt() == mesh->texture->level, "selector shows the actual sampled level");
        if (volumePath.isEmpty() && surfacePath.isEmpty()) {
            require(textureLevel->currentData().toInt() == 3 && mesh->texture->level == 3, "mesh texture defaults to OME-Zarr level 3");
            for (size_t i=0; i<mesh->texture->pixels.size(); i+=2)
                require(mesh->texture->pixels[i] == 190 && mesh->texture->pixels[i+1] == 255, "default texture reads the coarse level's data");
            textureLevel->setCurrentIndex(textureLevel->findData(2));
            textureLevel->setCurrentIndex(textureLevel->findData(0));
            settle([&] { return canvas->surfaceMesh() && canvas->surfaceMesh()->texture && canvas->surfaceMesh()->texture->level == 0; });
            mesh = canvas->surfaceMesh();
            require(QSettings(vc3d::settingsFilePath(), QSettings::IniFormat).value("scene3d/surfaceTextureLevel", -1).toInt() == 0,
                    "explicit texture level is persisted");
            Scene3DPanel restored(&state, &viewers);
            require(restored.findChild<QDoubleSpinBox*>("scene3DMemoryBudgetMiB")->value() == 65536,
                    "custom memory budgets above the old ceiling are restored");
            require(restored.findChild<QComboBox*>("scene3DSurfaceTextureLevel")->currentData().toInt() == 0,
                    "new panels restore the selected texture level");
            require(mesh->texture->width == 4 && mesh->texture->height == 4, "texture covers the grid endpoints");
            for (int y=0; y<4; ++y)
                for (int x=0; x<4; ++x) {
                    const size_t i = size_t(y*4+x)*2;
                    require(mesh->texture->pixels[i] == 72+3*x+3*y && mesh->texture->pixels[i+1] == 255,
                            "texture samples CT at the source surface coordinates");
                }
            auto masked = points.clone();
            masked(0,0) = {-1,-1,-1};
            std::atomic_bool stop{false};
            const auto maskedTexture = surface::sampleTexture(masked, {1,1}, *volume, stop, 0);
            require(maskedTexture->pixels[1] == 0 && maskedTexture->pixels.back() == 255, "texture respects invalid surface cells");
            stop.store(true);
            require(!surface::sampleTexture(points, {1,1}, *volume, stop, 0), "texture sampling can be cancelled");
        }
        auto* texture = panel.findChild<QCheckBox*>("scene3DSurfaceTexture");
        require(texture && texture->isChecked(), "CT texture is enabled by default");
        texture->setChecked(false);
        require(!textureLevel->isEnabled(), "level selection is disabled when CT texture is off");
        require(canvas->surfaceMesh() == mesh, "disabling texture reuses the mesh");
        texture->setChecked(true);
        require(textureLevel->isEnabled(), "level selection resumes with CT texture");
        require(canvas->surfaceMesh() == mesh, "enabling cached texture reuses the mesh");
        auto* zSlider = panel.findChild<QSlider*>("scene3DSliceZ");
        auto* zValue = panel.findChild<QDoubleSpinBox*>("scene3DSliceZValue");
        auto* follow = panel.findChild<QCheckBox*>("scene3DFollowMainXY");
        require(zSlider && zValue && follow && follow->isChecked(), "XY position controls center on the mesh initially");
        const auto original = canvas->ctSlices();
        require(original.size() == 1 && original.front()->plane.axis == 0, "visible mesh shows only the XY slice");
        const auto& xy = original.front()->plane;
        auto coversMesh = [&](const slices::Plane& slice, const Bounds& bounds) {
            require(slice.width <= 1024 && slice.height <= 1024, "mesh coverage preserves the texture resolution limit");
            for (int corner=0; corner<8; ++corner) {
                double u=0, v=0, right2=0, down2=0;
                for (int k=0; k<3; ++k) {
                    const double delta = ((corner & (1<<k)) ? bounds.upper[k] : bounds.lower[k]) - slice.origin[k];
                    u += delta*slice.right[k]; v += delta*slice.down[k];
                    right2 += slice.right[k]*slice.right[k]; down2 += slice.down[k]*slice.down[k];
                }
                require(u/right2 >= -1e-5 && u/right2 <= 1+1e-5 && v/down2 >= -1e-5 && v/down2 <= 1+1e-5,
                        "XY slice covers every projected mesh bounding-box corner");
            }
        };
        coversMesh(xy, mesh->bounds);
        for (int i=0; i<3; ++i)
            require(std::abs(xy.origin[i] + .5*xy.right[i] + .5*xy.down[i] -
                             .5*(mesh->bounds.lower[i]+mesh->bounds.upper[i])) < .001,
                    "XY slice is centered on the mesh bounds in all three coordinates");
        auto* xyAxis = panel.findChild<QCheckBox*>("fibers3DSliceXY");
        auto* yzAxis = panel.findChild<QCheckBox*>("fibers3DSliceYZ");
        require(xyAxis->isChecked() && !yzAxis->isChecked() && !yzAxis->isEnabled(), "mesh mode suppresses other axes");
        auto xyZ = [&]() {
            for (const auto& image : canvas->ctSlices())
                if (image->plane.axis == 0)
                    return image->plane.origin[2] + .5*image->plane.right[2] + .5*image->plane.down[2];
            return std::numeric_limits<double>::quiet_NaN();
        };
        const double originalZ = xyZ();
        const int target = std::min(zSlider->maximum(), int(originalZ)+4);
        zSlider->setValue(target-1);
        zSlider->setValue(target);
        settle([&] { return std::abs(xyZ()-target)<.001; });
        require(!follow->isChecked(), "dragging Z detaches only the XY slice from Main");
        require(canvas->surfaceMesh() == mesh, "moving slices never rebuilds or moves the mesh");
        for (const auto& before : original)
            if (before->plane.axis != 0) {
                bool same = false;
                for (const auto& after : canvas->ctSlices()) same |= after->plane == before->plane;
                require(same, "Z navigation leaves other slice planes fixed");
            }
        require(std::dynamic_pointer_cast<PlaneSurface>(state.surface("xy plane"))->origin()[2] == center[2], "Z navigation leaves Main unchanged");
        zValue->setValue(target + .25);
        settle([&] { return std::abs(xyZ()-(target+.25))<.001; });
        follow->setChecked(true);
        settle([&] { return std::abs(xyZ()-originalZ)<.001; });
        auto* showMesh = panel.findChild<QCheckBox*>("fibers3DSurfaceMesh");
        showMesh->setChecked(false);
        settle([&] { return canvas->ctSlices().size() == 2 && std::abs(xyZ()-center[2])<.001; });
        require(yzAxis->isChecked() && yzAxis->isEnabled(), "hiding mesh restores the user's slice axes");
        showMesh->setChecked(true);
        settle([&] { return canvas->ctSlices().size() == 1 && std::abs(xyZ()-originalZ)<.001; });
        auto* filter = panel.findChild<QCheckBox*>("fibers3DNearSurface");
        require(filter && !filter->isVisible(), "general viewer must not expose fiber filters");
        auto* volumeMode = panel.findChild<QCheckBox*>("fibers3DSliceVolumetric");
        volumeMode->setChecked(true);
        settle([&] { return !canvas->ctSlices().empty() && canvas->ctSlices().front()->slab && !canvas->ctSlices().front()->preview; });
        require(canvas->ctSlices().front()->plane.front == 3, "volumetric mode uses Main thickness");
        volumeMode->setChecked(false);
        settle([&] { return !canvas->ctSlices().empty() && !canvas->ctSlices().front()->slab && !canvas->ctSlices().front()->preview; });
        if (render) {
            require(canvas->isValid(), "OpenGL 3.3 context unavailable");
            canvas->fitAll();
            settle([&] { return canvas->isSurfaceMeshReady(); });
            // Allow incremental CT uploads before capturing the actual widget.
            for (int i=0; i<10; ++i) { canvas->grabFramebuffer(); QApplication::processEvents(); }
            require(panel.grab().save(qEnvironmentVariable("VC3D_SCENE3D_SCREENSHOT")), "cannot save panel screenshot");
        }
        if (volumePath.isEmpty() && surfacePath.isEmpty()) {
            // Non-square meshes larger than the old fixed 1024-voxel patch.
            // Replacing one with a wider mesh at the same center must resize
            // the slice too, even though its center has not changed.
            std::atomic_bool stop{false};
            auto largeMesh = [&](float halfWidth) {
                cv::Mat_<cv::Vec3f> grid(2,2);
                for (int y=0; y<2; ++y)
                    for (int x=0; x<2; ++x)
                        grid(y,x) = {11.5f + (x ? halfWidth : -halfWidth), 11.5f + (y ? 900.f : -900.f), 12.f + (y ? 4.f : -4.f)};
                return surface::build(grid, 1024*1024, stop);
            };
            auto wide = largeMesh(1600);
            canvas->setSurfaceMesh(wide);
            settle([&] { return canvas->ctSlices().size() == 1 &&
                distance(Point{}, canvas->ctSlices().front()->plane.right) > 3200; });
            coversMesh(canvas->ctSlices().front()->plane, wide->bounds);
            wide = largeMesh(2400);
            canvas->setSurfaceMesh(wide);
            settle([&] { return canvas->ctSlices().size() == 1 &&
                distance(Point{}, canvas->ctSlices().front()->plane.right) > 4800; });
            coversMesh(canvas->ctSlices().front()->plane, wide->bounds);
            const auto mainPlane = state.surface("xy plane");
            auto tilted = std::make_shared<PlaneSurface>(center, cv::Vec3f{.3f,.4f,1.f});
            state.setSurface("xy plane", tilted);
            settle([&] { return canvas->ctSlices().size() == 1 &&
                std::abs(canvas->ctSlices().front()->plane.right[2]) > 1; });
            coversMesh(canvas->ctSlices().front()->plane, wide->bounds);
            state.setSurface("xy plane", mainPlane);
            canvas->setSurfaceMesh(mesh);
        }
        panel.hide();
        require(canvas->ctSlices().empty() && !canvas->surfaceMesh(), "hiding panel clears its layers");
        panel.show();
        settle([&] { return canvas->surfaceMesh() && !canvas->ctSlices().empty(); });
        state.setSurface("segmentation", {});
        require(!canvas->surfaceMesh(), "surface removal clears the mesh");
        auto* displayMode = panel.findChild<QComboBox*>("scene3DDisplayMode");
        require(displayMode, "volume-only display mode exists");
        displayMode->setCurrentIndex(1);
        settle([&] { return bool(canvas->ctVolume()); });
        require(canvas->ctSlices().empty() && !canvas->surfaceMesh(), "CT volume renders without a surface or slice planes");
        require(!state.surface("segmentation"), "volume view does not create or require a segmentation");
        auto* density = panel.findChild<QDoubleSpinBox*>("scene3DVolumeDensity");
        const auto loaded = canvas->ctVolume();
        const double initialDensity = density->value();
        density->setValue(initialDensity + .25);
        require(canvas->ctVolume() == loaded, "density changes reuse resident volume data");
        density->setValue(initialDensity);
        if (render) {
            canvas->fitAll();
            settle([&] { canvas->grabFramebuffer(); return canvas->isCTVolumeReady(); });
            require(panel.grab().save(qEnvironmentVariable("VC3D_SCENE3D_SCREENSHOT") + ".volume.png"), "cannot save CT volume screenshot");
        }
        if (volumePath.isEmpty()) {
            displayMode->setCurrentIndex(0);
            require(!canvas->ctVolume(), "switching to slices removes volume data");
            displayMode->setCurrentIndex(1);
            displayMode->setCurrentIndex(0);
            settle([&] { return !canvas->ctSlices().empty(); });
            require(!canvas->ctVolume(), "rapid source changes cannot publish a stale volume");
            displayMode->setCurrentIndex(1);
            settle([&] { return bool(canvas->ctVolume()); });
            panel.hide();
            require(!canvas->ctVolume(), "hiding the panel clears the volume");
            panel.show();
            settle([&] { return bool(canvas->ctVolume()); });
        }
        state.setCurrentVolume({});
        require(!canvas->ctVolume(), "volume removal clears the resident volume");
        require(canvas->ctSlices().empty(), "volume removal clears CT");
        require(textureLevel->count() == 0 && !textureLevel->isEnabled(), "volume removal clears texture level choices");
        require(!zSlider->isEnabled() && follow->isChecked(), "volume removal resets and disables Z navigation");
        panel.hide();
        std::cout << "3D panel: CT texture, XY navigation, volumetric mode, surface, hide/show, and removal passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "3D panel test: " << error.what() << '\n';
        return 1;
    }
}
