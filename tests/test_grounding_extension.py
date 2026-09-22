import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from tactile_refinement import train


@pytest.mark.parametrize('completed', [True, False])
def test_extension_archives_and_resumes_only_a_completed_run(tmp_path, monkeypatch, completed):
    script = Path(__file__).resolve().parents[1]/'scripts/extend_grounding.py'
    spec = importlib.util.spec_from_file_location('extend_grounding', script)
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    monkeypatch.setattr(extension, 'ROOT', tmp_path)
    (tmp_path/'logs').mkdir()
    output = tmp_path/'run/grounding'
    output.mkdir(parents=True)
    old = {'output_dir': 'run', 'stages': {'grounding': {'updates': 3, 'learning_rate': 1e-4}}}
    new = json.loads(json.dumps(old))
    new['stages']['grounding']['updates'] = 9
    config = tmp_path/'config.json'
    config.write_text(json.dumps(new))
    for name in ('checkpoint_1.pt', 'checkpoint_2.pt', 'final.pt'):
        torch.save({'stage': 'grounding', 'step': 3 if completed else 2, 'config': old,
                    'provenance': {}, 'optimizer': {'sentinel': torch.tensor([42])}}, output/name)
    (output/'recipe.json').write_text(json.dumps(old))
    (output/'provenance.json').write_text('{}')
    before_bytes = (output/'final.pt').read_bytes()
    monkeypatch.setattr(train, 'training_provenance', lambda config: {})
    monkeypatch.setattr(sys, 'argv', ['extend_grounding.py', '--config', str(config), '--previous-updates', '3'])
    launches = []

    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            launches.append(command)
            assert command[command.index('--stage')+1] == 'grounding'
            resume = Path(command[command.index('--resume')+1])
            assert resume.read_bytes() == before_bytes
            assert kwargs['pass_fds']
        def wait(self):
            return 0

    monkeypatch.setattr(extension.subprocess, 'Popen', Child)
    if completed:
        extension.main()
        assert len(launches) == 1
        assert not list(output.glob('*.pt'))
        assert len(list((tmp_path/'run/grounding_3_before_extension').glob('*.pt'))) == 3
    else:
        with pytest.raises(ValueError, match='did not complete'):
            extension.main()
        assert not launches
        assert (output/'final.pt').read_bytes() == before_bytes
        assert not (tmp_path/'run/grounding_3_before_extension').exists()
