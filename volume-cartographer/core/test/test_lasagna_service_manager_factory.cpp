#include "LasagnaServiceManager.hpp"

#include <QCoreApplication>
#include <QObject>
#include <QPointer>

#include <cstdlib>
#include <iostream>
#include <memory>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << std::endl;
        std::exit(1);
    }
}

} // namespace

int main(int argc, char** argv)
{
    QCoreApplication app(argc, argv);

    auto parent = std::make_unique<QObject>();
    QPointer<LasagnaServiceManager> transient =
        LasagnaServiceManager::createTransient(parent.get());
    auto& shared = LasagnaServiceManager::instance();

    require(transient, "Transient Lasagna manager must be created");
    require(transient != &shared,
            "Transient Lasagna manager must be isolated from the shared singleton");
    require(transient->parent() == parent.get(),
            "Transient Lasagna manager must follow workspace QObject ownership");

    require(transient->ensureServiceRunning(),
            "Transient Lasagna manager stub must start");
    require(transient->isRunning(),
            "Transient Lasagna manager must track its own running state");
    require(!shared.isRunning(),
            "Starting a transient manager must not start the shared singleton");

    transient->stopService();
    require(!transient->isRunning(),
            "Stopping a transient manager must reset only its running state");
    require(!shared.isRunning(),
            "Stopping a transient manager must not affect the shared singleton");

    parent.reset();
    require(transient.isNull(),
            "Destroying the workspace parent must destroy its transient manager");
    return 0;
}
