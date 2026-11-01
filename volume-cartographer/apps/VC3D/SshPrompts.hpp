#pragma once

namespace vc3d {
// Answer ssh's prompts (password, key passphrase, one-time code, new host key)
// with dialogs, for remote volumes and hosted object stores addressed as
// sftp:// or ssh://. Call once after the QApplication exists.
void installSshPrompts();
}
