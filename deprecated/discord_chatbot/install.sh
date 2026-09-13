#!/bin/bash
set -euo pipefail

if [[ "${AGENTS_AGENT_MODE:-0}" == "1" && "${AGENTS_ALLOW_INSTALL:-0}" != "1" ]]; then
  echo "INFO: discord_chatbot/install.sh is disabled by default in agent mode."
  echo "Set AGENTS_ALLOW_INSTALL=1 to run this install script."
  echo "Example: AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 ./discord_chatbot/install.sh"
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"
