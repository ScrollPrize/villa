// vc4d — ground-up volume viewer built on libvc.
// 4-plane orthogonal viewer (XY, XZ, YZ + future surface view).
// Plugin-based features loaded from .so files.

#include "app.hpp"
#include "codec.hpp"
#include "plugin.hpp"
#include "viewer.hpp"
#include "volume.hpp"

#include <QApplication>
#include <QDockWidget>
#include <QFileDialog>
#include <QGridLayout>
#include <QMenu>
#include <QMenuBar>
#include <QSplitter>
#include <QStatusBar>
#include <QToolBar>

#include <dlfcn.h>
#include <filesystem>
#include <format>
#include <memory>
#include <print>
#include <vector>

namespace fs = std::filesystem;

namespace vc4d {

class AppImpl : public App {
    Q_OBJECT

public:
    AppImpl() {
        setWindowTitle("vc4d");
        resize(1600, 1000);

        // 2x2 grid of viewers: XY | XZ / YZ | (reserved)
        _xy = new Viewer(Plane::XY);
        _xz = new Viewer(Plane::XZ);
        _yz = new Viewer(Plane::YZ);

        auto* top = new QSplitter(Qt::Horizontal);
        top->addWidget(_xy);
        top->addWidget(_xz);

        auto* bot = new QSplitter(Qt::Horizontal);
        bot->addWidget(_yz);

        // Placeholder for 4th panel (surface view, 3D view, etc.)
        auto* placeholder = new QWidget;
        placeholder->setStyleSheet("background: #1a1a1a;");
        bot->addWidget(placeholder);

        auto* vsplit = new QSplitter(Qt::Vertical);
        vsplit->addWidget(top);
        vsplit->addWidget(bot);
        setCentralWidget(vsplit);

        // Link focus across all viewers: when one scrolls its depth axis,
        // all viewers update their shared focus point.
        auto link = [this](vc::Vec3f f) {
            _xy->set_focus(f); _xz->set_focus(f); _yz->set_focus(f);
        };
        connect(_xy, &Viewer::focus_changed, this, link);
        connect(_xz, &Viewer::focus_changed, this, link);
        connect(_yz, &Viewer::focus_changed, this, link);

        // Forward cursor position to status bar
        auto cursor_status = [this](vc::Vec3f p) {
            statusBar()->showMessage(QString::fromStdString(
                std::format("x:{:.0f}  y:{:.0f}  z:{:.0f}", p.x, p.y, p.z)));
        };
        connect(_xy, &Viewer::cursor_world, this, cursor_status);
        connect(_xz, &Viewer::cursor_world, this, cursor_status);
        connect(_yz, &Viewer::cursor_world, this, cursor_status);

        // File menu
        auto* file = menuBar()->addMenu("&File");
        file->addAction("&Open Volume...", QKeySequence::Open, this, &AppImpl::on_open);
        file->addSeparator();
        file->addAction("&Quit", QKeySequence::Quit, this, &QWidget::close);

        statusBar()->showMessage("Ready");
    }

    // --- App interface ---
    vc::Volume* volume() override { return _vol.get(); }

    void open_volume(const fs::path& path) override {
        if (_vol) {
            _vol.reset();
            _xy->set_volume(nullptr);
            _xz->set_volume(nullptr);
            _yz->set_volume(nullptr);
            emit volume_closed();
        }

        vc::FrameCache::Config cc;
        cc.max_bytes = 2ULL << 30;
        cc.io_threads = 4;
        cc.decode = vc::h265_decode_slab;

        _vol = std::make_unique<vc::Volume>(vc::Volume::open(path, std::move(cc)));

        // All viewers share the same Volume and chunk-ready callback
        _vol->set_on_chunk_ready([this] {
            _xy->mark_dirty(); _xz->mark_dirty(); _yz->mark_dirty();
        });

        _xy->set_volume(_vol.get());
        _xz->set_volume(_vol.get());
        _yz->set_volume(_vol.get());

        auto s = _vol->shape();
        setWindowTitle(QString::fromStdString(
            std::format("vc4d — {} ({}x{}x{})", path.filename().string(), s.x, s.y, s.z)));
        statusBar()->showMessage(QString::fromStdString(
            std::format("Opened: {}", path.string())));
        emit volume_opened();
    }

    Viewer* viewer() override { return _xy; }

    QDockWidget* add_dock(const QString& title, QWidget* widget,
                           Qt::DockWidgetArea area) override {
        auto* dock = new QDockWidget(title, this);
        dock->setWidget(widget);
        addDockWidget(area, dock);
        return dock;
    }

    QMenu* add_menu(const QString& title) override {
        return menuBar()->addMenu(title);
    }

    QToolBar* add_toolbar(const QString& title) override {
        return addToolBar(title);
    }

    void set_status(const QString& msg, int timeout_ms) override {
        statusBar()->showMessage(msg, timeout_ms);
    }

    void load_plugins(const fs::path& plugin_dir) {
        if (!fs::exists(plugin_dir)) return;
        for (auto& entry : fs::directory_iterator(plugin_dir)) {
            if (entry.path().extension() != ".so") continue;
            void* handle = dlopen(entry.path().c_str(), RTLD_LAZY);
            if (!handle) {
                std::println(stderr, "plugin load failed: {} — {}", entry.path().string(), dlerror());
                continue;
            }
            auto factory = reinterpret_cast<vc4d_create_plugin_fn>(dlsym(handle, "vc4d_create_plugin"));
            if (!factory) {
                std::println(stderr, "plugin missing vc4d_create_plugin: {}", entry.path().string());
                dlclose(handle);
                continue;
            }
            auto* plugin = factory();
            plugin->init(this);
            _plugins.emplace_back(plugin);
            _handles.push_back(handle);
            std::println("loaded plugin: {} v{}", plugin->name(), plugin->version());
        }
    }

    ~AppImpl() {
        _plugins.clear();
        for (auto* h : _handles) dlclose(h);
    }

private slots:
    void on_open() {
        auto path = QFileDialog::getExistingDirectory(this, "Open Volume (zarr)");
        if (path.isEmpty()) return;
        open_volume(path.toStdString());
    }

private:
    Viewer* _xy = nullptr;
    Viewer* _xz = nullptr;
    Viewer* _yz = nullptr;
    std::unique_ptr<vc::Volume> _vol;
    std::vector<std::unique_ptr<Plugin>> _plugins;
    std::vector<void*> _handles;
};

} // namespace vc4d

#include "main.moc"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("vc4d");

    vc4d::AppImpl window;

    auto exe_dir = fs::path(argv[0]).parent_path();
    auto plugin_dir = exe_dir / "plugins";
    if (auto env = std::getenv("VC4D_PLUGINS"))
        plugin_dir = env;
    window.load_plugins(plugin_dir);

    if (argc > 1)
        window.open_volume(argv[1]);

    window.show();
    return app.exec();
}
