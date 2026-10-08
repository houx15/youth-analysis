"""模拟作业环境验证无参数提交的参数传递，不运行分析或调用真实调度器。"""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class SlurmTests(unittest.TestCase):
    def run_script(self, overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init = root / "conda.sh"
            init.write_text("conda() { :; }\n")
            fake_python = root / "python"
            fake_python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CAPTURE_ARGS"\n')
            fake_python.chmod(0o755)
            capture = root / "args"
            env = {k: v for k, v in os.environ.items() if not k.startswith("CULTURAL_")}
            env.update({"CULTURAL_CONDA_INIT": str(init), "SLURM_JOB_ID": "123", "SLURM_SUBMIT_DIR": str(root), "CAPTURE_ARGS": str(capture), "PATH": str(root) + ":" + env["PATH"]})
            config_dir = root / "cultural_participation"
            config_dir.mkdir()
            values = {"INPUT_DIR": "bangdan_data/2020", "RUN_LABEL": "pilot_2020", "MAX_LINES": "1000", **overrides}
            (config_dir / "vocabulary_job.conf").write_text("\n".join(f'{k}="{v}"' for k, v in values.items()))
            script = Path(__file__).resolve().parents[2] / "prepare_cultural_vocabulary.sh"
            result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True)
            return result, capture.read_text() if capture.exists() else ""

    def test_no_argument_pilot(self):
        result, commands = self.run_script({})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input-dir bangdan_data/2020", commands)
        self.assertIn("pilot_2020_123/topics.sqlite", commands)
        self.assertIn("--max-lines 1000", commands)

    def test_config_full_run(self):
        result, commands = self.run_script({"INPUT_DIR": "/data/hot", "RUN_LABEL": "full_v1", "MAX_LINES": "0"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--input-dir /data/hot", commands)
        self.assertIn("full_v1_123/topics.sqlite", commands)
        self.assertNotIn("--max-lines", commands)

    def test_reject_invalid_limit(self):
        result, commands = self.run_script({"MAX_LINES": "-1"})
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(commands, "")
