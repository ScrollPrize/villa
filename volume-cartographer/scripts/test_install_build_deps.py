"""Exercise distribution/profile/agent guards using fake commands; never provision."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('install_build_deps.sh').resolve()


class Profiles(unittest.TestCase):
    def run_profile(self, distro, profile, agent='0', allow='0'):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindir = root / 'bin'
            bindir.mkdir()
            log = root / 'calls'
            release = root / 'os-release'
            release.write_text(f'ID={distro}\n')
            mock = '''#!/bin/bash
printf '%s\\n' "$(basename "$0") $*" >> "$VC_TEST_LOG"
case "$(basename "$0")" in
sudo) exec "$@" ;;
uname) if [[ "$1" == -s ]]; then echo Linux; else echo x86_64; fi ;;
unzip) mkdir -p "$4/aws"; printf '#!/bin/bash\\necho aws-install >> "$VC_TEST_LOG"\\n' > "$4/aws/install"; chmod +x "$4/aws/install" ;;
esac
'''
            for name in ['apt-get', 'add-apt-repository', 'sudo', 'uname', 'curl', 'unzip', 'ln']:
                p = bindir / name
                p.write_text(mock)
                p.chmod(0o755)
            env = dict(os.environ, PATH=f'{bindir}:/usr/bin:/bin', VC_OS_RELEASE_FILE=str(release),
                       VC_TEST_LOG=str(log), AGENTS_AGENT_MODE=agent, AGENTS_ALLOW_INSTALL=allow)
            result = subprocess.run(['bash', str(SCRIPT), profile], env=env, capture_output=True, text=True)
            return result, log.read_text() if log.exists() else ''

    def test_minimal(self):
        for distro in ['debian', 'ubuntu']:
            with self.subTest(distro=distro):
                result, log = self.run_profile(distro, 'minimal-build')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('libceres-dev', log)
                self.assertEqual('add-apt-repository -y universe' in log, distro == 'ubuntu')
                self.assertNotIn('curl ', log)
                self.assertNotIn('flang-21', log)

    def test_ci(self):
        result, log = self.run_profile('ubuntu', 'ci-tooling', '1', '1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('flang-21', log)
        self.assertIn('awscli-exe-linux-x86_64.zip', log)
        self.assertIn('aws-install', log)

    def test_guards(self):
        for args in [('debian', 'unknown'), ('fedora', 'minimal-build'),
                     ('ubuntu', 'minimal-build', '1')]:
            with self.subTest(args=args):
                result, log = self.run_profile(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('apt-get', log)
                self.assertNotIn('curl ', log)


if __name__ == '__main__':
    unittest.main()
