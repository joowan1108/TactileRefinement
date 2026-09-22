#!/usr/bin/env bash
# Reuse this computer's compiled simulator dependencies without changing them.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SHARED_SITE=/tactile/contrast/.venv-univtac/lib/python3.10/site-packages
test -d "$SHARED_SITE"
if [[ ! -x "$ROOT/.venv-benchmark/bin/python" ]]; then
    uv venv --python /usr/bin/python3.10 "$ROOT/.venv-benchmark"
fi
"$ROOT/.venv-benchmark/bin/python" - <<'PY'
import sysconfig
from pathlib import Path
site = Path(sysconfig.get_paths()['purelib'])
(site/'shared_simulator.pth').write_text("import site; site.addsitedir('/tactile/contrast/.venv-univtac/lib/python3.10/site-packages')\n")
PY
"$ROOT/.venv-benchmark/bin/python" - <<'PY'
from importlib.metadata import version
expected = {'isaacsim':'4.5.0.0', 'isaaclab':'0.41.3', 'torch':'2.5.1+cu124', 'nvidia-curobo':'0.7.7'}
for name, wanted in expected.items():
    actual = version(name)
    assert actual == wanted, (name, actual, wanted)
    print(name, actual)
PY
