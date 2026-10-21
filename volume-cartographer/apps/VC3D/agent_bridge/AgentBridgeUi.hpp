#pragma once

// Pure helpers for the bridge's workspace and inspection methods. They depend
// only on QtCore so the contract test can exercise them without a running
// application.

#include <algorithm>

#include <QRect>
#include <QRegularExpression>
#include <QSize>
#include <QString>
#include <QStringList>

// Workspace ids in tab order, as stored in each tab page's "workspaceId"
// property by CWindow (the setProperty calls next to the addTab calls in
// CWindow's constructor, plus the Spiral and Fiber Map pages it creates
// later). workspace.switch's descriptor enum is built from this list, and the
// offscreen smoke test checks it against the live tab bar, so a tab added to
// CWindow must be added here.
inline QStringList agentBridgeWorkspaceIds()
{
    return {
        QStringLiteral("main"),
        QStringLiteral("lasagna"),
        QStringLiteral("atlas"),
        QStringLiteral("fiber-slice"),
        QStringLiteral("intersections"),
        QStringLiteral("spiral"),
        QStringLiteral("fiber-map"),
    };
}

// workspace.switch keeps accepting the protocol's original "fiber_slice"
// spelling; every other accepted name is already a workspace id.
inline QString canonicalWorkspaceId(const QString& name)
{
    if (name == QLatin1String("fiber_slice"))
        return QStringLiteral("fiber-slice");
    return name;
}

// True when a field's label, object name, placeholder or tooltip says it
// holds a credential: a password, passphrase, secret, token, API, access,
// private or secret key, an authorization value, or a cookie. Words are
// matched whole after camelCase and underscores are split, so "apiKeyEdit"
// and "AWSSecretKey" match while "Draws" and "Checkpoint" do not.
inline bool agentBridgeCredentialLabel(const QString& text)
{
    if (text.isEmpty())
        return false;
    static const QRegularExpression lowerUpper(QStringLiteral("([a-z0-9])([A-Z])"));
    static const QRegularExpression acronym(QStringLiteral("([A-Z]+)([A-Z][a-z])"));
    static const QRegularExpression pattern(
        QStringLiteral("\\b(?:passwords?|passphrases?|passwd|secrets?|tokens?|credentials?|"
                       "api ?keys?|access ?keys?|private ?keys?|secret ?keys?|auth|"
                       "authentication|authorization|bearer|cookies?)\\b"),
        QRegularExpression::CaseInsensitiveOption);
    QString words = text;
    words.replace(lowerUpper, QStringLiteral("\\1 \\2"));
    words.replace(acronym, QStringLiteral("\\1 \\2"));
    words.replace(QLatin1Char('_'), QLatin1Char(' '));
    words.replace(QLatin1Char('-'), QLatin1Char(' '));
    return pattern.match(words).hasMatch();
}

// True when text contains a credential itself: a URL with a user name and
// password, a URL query that carries a token, signature, key, secret or
// password (presigned S3 links among them), an AWS access key id, a bearer
// token, or a private key block.
inline bool agentBridgeCredentialContent(const QString& text)
{
    if (text.isEmpty())
        return false;
    static const QRegularExpression pattern(
        QStringLiteral(
            "[a-z][a-z0-9+.-]*://[^/\\s:@]+:[^/\\s@]+@"
            "|[?&](?:access_token|token|sig|signature|key|api_key|apikey|secret|password|"
            "x-amz-signature|x-amz-credential|x-amz-security-token)="
            "|(?-i:\\b(?:AKIA|ASIA)[0-9A-Z]{16}\\b)"
            "|\\bbearer\\s+[a-z0-9._~+/=-]{8,}"
            "|-----BEGIN [a-z ]*PRIVATE KEY-----"),
        QRegularExpression::CaseInsensitiveOption);
    return pattern.match(text).hasMatch();
}

// Splits a ui.inspect / screenshot.capture target such as "dock:name" or
// "workspace:id" into its kind and name. Returns false for a target without a
// recognized prefix or with an empty name.
inline bool splitAgentBridgeTarget(const QString& target, QString* kind, QString* name)
{
    for (const char* prefix : {"dock", "workspace"}) {
        const QString lead = QLatin1String(prefix) + QLatin1Char(':');
        if (target.startsWith(lead)) {
            *kind = QLatin1String(prefix);
            *name = target.mid(lead.size());
            return !name->isEmpty();
        }
    }
    return false;
}

// screenshot.capture's region clipped to a target of `size`, in the target's
// logical pixels. Empty when the region starts outside the target. Clipping
// happens before the rectangle is built, so any non-negative int is safe.
inline QRect agentBridgeCaptureArea(int x, int y, int width, int height, const QSize& size)
{
    if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x >= size.width() || y >= size.height())
        return QRect();
    return QRect(x, y, std::min(width, size.width() - x), std::min(height, size.height() - y));
}
