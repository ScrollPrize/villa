#include "SshPrompts.hpp"

#include "vc_object_store/sftp/Sftp.hpp"

#include <QApplication>
#include <QInputDialog>
#include <QLineEdit>
#include <QMessageBox>
#include <QThread>

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace vc3d {
namespace {
std::optional<std::string> ask(const vc::sftp::Prompt& prompt)
{
    auto* parent = QApplication::activeWindow();
    const auto title = QObject::tr("SSH login: %1").arg(QString::fromStdString(prompt.destination));
    const auto text = QString::fromStdString(prompt.text);
    if (prompt.confirmation) {
        const auto choice = QMessageBox::question(parent, title, text, QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
        return choice == QMessageBox::Yes ? "yes" : "no";
    }
    bool accepted = false;
    const auto answer = QInputDialog::getText(parent, title, text, QLineEdit::Password, {}, &accepted);
    if (!accepted) return std::nullopt;
    return answer.toStdString();
}
}  // namespace

void installSshPrompts()
{
    vc::sftp::setAuthenticator([](const vc::sftp::Prompt& prompt) -> std::optional<std::string> {
        auto* app = QCoreApplication::instance();
        if (!app) return std::nullopt;
        if (QThread::currentThread() == app->thread()) return ask(prompt);
        struct Exchange {
            std::mutex mutex;
            std::condition_variable changed;
            bool started = false, done = false, abandoned = false;
            std::optional<std::string> answer;
        };
        auto exchange = std::make_shared<Exchange>();
        QMetaObject::invokeMethod(app, [exchange, prompt] {
            {
                std::lock_guard lock(exchange->mutex);
                if (exchange->abandoned) return;
                exchange->started = true;
            }
            exchange->changed.notify_all();
            auto answer = ask(prompt);
            {
                std::lock_guard lock(exchange->mutex);
                exchange->answer = std::move(answer);
                exchange->done = true;
            }
            exchange->changed.notify_all();
        }, Qt::QueuedConnection);
        std::unique_lock lock(exchange->mutex);
        // A GUI thread blocked on this very connection can never show the dialog:
        // give up instead of deadlocking, and let the operation fail and be retried.
        if (!exchange->changed.wait_for(lock, std::chrono::seconds(10), [&] { return exchange->started; })) {
            exchange->abandoned = true;
            return std::nullopt;
        }
        exchange->changed.wait(lock, [&] { return exchange->done; });
        return exchange->answer;
    });
}
}  // namespace vc3d
