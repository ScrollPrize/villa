#include "SessionRecoveryDialog.hpp"

#include "vc/ui/VCCollection.hpp"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QLabel>
#include <QLocale>
#include <QLockFile>
#include <QPushButton>
#include <QVBoxLayout>

#include <utility>

namespace {

bool sameProject(const QString& a, const QString& b)
{
    return a.isEmpty() || b.isEmpty() ? a == b : QFileInfo(a) == QFileInfo(b);
}

QString pointsText(const SessionRecovery::LeftoverSession& session)
{
    const QString points = session.pointCount == 1
        ? SessionRecoveryDialog::tr("1 point") : SessionRecoveryDialog::tr("%1 points").arg(session.pointCount);
    const QString collections = session.collectionCount == 1
        ? SessionRecoveryDialog::tr("1 collection")
        : SessionRecoveryDialog::tr("%1 collections").arg(session.collectionCount);
    return SessionRecoveryDialog::tr("%1 in %2").arg(points, collections);
}

} // namespace

SessionRecoveryDialog* SessionRecoveryDialog::offer(QWidget* parent,
                                                    SessionRecovery& recovery,
                                                    VCCollection* collection,
                                                    Actions actions)
{
    for (const auto& session : recovery.leftoverSessions()) {
        if (auto claim = SessionRecovery::claim(session)) {
            auto* dialog = new SessionRecoveryDialog(parent, &recovery, session, std::move(claim), collection,
                                                     std::move(actions));
            dialog->show();
            return dialog;
        }
    }
    return nullptr;
}

SessionRecoveryDialog::SessionRecoveryDialog(QWidget* parent,
                                             SessionRecovery* recovery,
                                             SessionRecovery::LeftoverSession session,
                                             std::shared_ptr<QLockFile> claim,
                                             VCCollection* collection,
                                             Actions actions)
    : QDialog(parent)
    , _recovery(recovery)
    , _session(std::move(session))
    , _claim(std::move(claim))
    , _collection(collection)
    , _actions(std::move(actions))
{
    setObjectName(QStringLiteral("sessionRecoveryDialog"));
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowModality(Qt::ApplicationModal);
    setWindowTitle(tr("Recover Previous Session"));
    auto* layout = new QVBoxLayout(this);

    QString text = tr("VC3D did not shut down cleanly. A copy of the %1 placed in its last session "
                      "was saved at %2.")
                       .arg(pointsText(_session),
                            QLocale().toString(_session.updated.toLocalTime(), QLocale::ShortFormat));
    if (!_session.projectPath.isEmpty()) {
        text += QLatin1Char(' ') + tr("Recover restores them into %1, opening it if needed.")
                                       .arg(QFileInfo(_session.projectPath).fileName());
    }
    auto* intro = new QLabel(text, this);
    intro->setWordWrap(true);
    intro->setToolTip(QDir::toNativeSeparators(_session.pointsFile()));
    layout->addWidget(intro);

    if (!_session.segmentId.isEmpty()) {
        _selectSegment = new QCheckBox(tr("Select segment %1 again").arg(_session.segmentId), this);
        _selectSegment->setObjectName(QStringLiteral("selectSegmentCheck"));
        _selectSegment->setChecked(true);
        layout->addWidget(_selectSegment);
    }

    auto* note = new QLabel(tr("Discard deletes the copy. Later keeps it for the next start."), this);
    note->setWordWrap(true);
    layout->addWidget(note);

    auto* buttons = new QDialogButtonBox(this);
    QPushButton* recoverButton = buttons->addButton(tr("Recover"), QDialogButtonBox::AcceptRole);
    QPushButton* discardButton = buttons->addButton(tr("Discard"), QDialogButtonBox::DestructiveRole);
    QPushButton* laterButton = buttons->addButton(tr("Later"), QDialogButtonBox::RejectRole);
    recoverButton->setObjectName(QStringLiteral("recoverSessionButton"));
    discardButton->setObjectName(QStringLiteral("discardSessionButton"));
    laterButton->setObjectName(QStringLiteral("laterSessionButton"));
    recoverButton->setDefault(true);
    layout->addWidget(buttons);

    connect(recoverButton, &QPushButton::clicked, this, &SessionRecoveryDialog::recover);
    connect(discardButton, &QPushButton::clicked, this, &SessionRecoveryDialog::discardCopy);
    connect(laterButton, &QPushButton::clicked, this, &QDialog::close);
}

bool SessionRecoveryDialog::projectIsOpen() const
{
    const QString current = _actions.currentProjectPath ? _actions.currentProjectPath() : QString();
    return sameProject(current, _session.projectPath);
}

void SessionRecoveryDialog::recover()
{
    const bool selectSegment = _selectSegment && _selectSegment->isChecked();

    // The points go back only into the project they were placed in (opening
    // a project clears the collections), and before the segment is selected,
    // which loads that segment's correction points.
    QString problem;
    if (!projectIsOpen() && !_session.projectPath.isEmpty() &&
        !(_actions.openProject && _actions.openProject(_session.projectPath, &problem))) {
        problem = tr("Could not reopen %1: %2.").arg(_session.projectPath, problem);
    } else if (!projectIsOpen()) {
        problem = tr("The points were placed in another project.");
    } else if (QString error; !SessionRecovery::restorePoints(_session, _collection, &error)) {
        problem = tr("Could not restore the points: %1.").arg(error);
    }
    if (!problem.isEmpty()) {
        finish(problem + QLatin1Char(' ') + tr("The recovery copy is kept for the next start."));
        return;
    }

    // The restored points reach disk again, in this session's own copy, before
    // the old copy goes: selecting the segment can take a while, and a crash
    // then must not lose them. The copy names the segment too, so the next
    // start would offer to select it again.
    const QString previousSegment = _recovery ? _recovery->segmentId() : QString();
    if (_recovery && selectSegment) {
        _recovery->setSegment(_session.segmentId);
    }
    const bool copied = _recovery && _recovery->writeNow();
    _claim.reset();
    QString status = tr("Recovered %1 from the previous session.").arg(pointsText(_session));
    if (copied) {
        SessionRecovery::discard(_session);
    } else {
        status += QLatin1Char(' ') + tr("The old recovery copy is kept, since a new one could not be written.");
    }
    QString error;
    if (selectSegment && !(_actions.activateSegment && _actions.activateSegment(_session.segmentId, &error))) {
        status += QLatin1Char(' ') + tr("Could not select segment %1: %2.").arg(_session.segmentId, error);
        if (_recovery) {
            _recovery->setSegment(previousSegment);
        }
    }
    finish(status);
}

void SessionRecoveryDialog::discardCopy()
{
    _claim.reset();
    SessionRecovery::discard(_session);
    finish(tr("Discarded the recovery copy of the previous session."));
}

void SessionRecoveryDialog::done(int result)
{
    _claim.reset();
    QDialog::done(result);
}

void SessionRecoveryDialog::finish(const QString& status)
{
    if (_actions.showStatus) {
        _actions.showStatus(status);
    }
    close();
}
