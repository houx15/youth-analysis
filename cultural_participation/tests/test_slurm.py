"""验证固定配置的作业入口，不执行真实分析或调用调度器。"""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cultural_participation.vocabulary_job import read_config, run_job


class SlurmTests(unittest.TestCase):
    def config(self, root, limit="1000"):
        folder = root / "cultural_participation"
        folder.mkdir()
        path = folder / "vocabulary_job.conf"
        path.write_text(f'INPUT_DIR="bangdan_data/2020"\nRUN_LABEL="pilot_2020"\nMAX_LINES={limit}\n')
        return path

    def test_pilot_and_full(self):
        for limit, expected in [("1000", 1000), ("0", None)]:
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.config(root, limit)
                with patch.dict("os.environ", {"SLURM_JOB_ID": "123"}), patch("cultural_participation.vocabulary_job.importlib.util.find_spec", return_value=True), patch("cultural_participation.vocabulary_job.collect", return_value={}) as collect, patch("cultural_participation.vocabulary_job.extract", return_value={}) as extract, patch("cultural_participation.vocabulary_job.export", return_value={}) as export, contextlib.redirect_stdout(io.StringIO()):
                    run_job(root)
                    self.assertEqual(collect.call_args.kwargs["max_lines"], expected)
                    self.assertEqual(collect.call_args.args[0], root / "bangdan_data/2020")
                    self.assertIn("pilot_2020_123", str(collect.call_args.args[2]))
                    extract.assert_called_once()
                    export.assert_called_once()

    def test_reject_invalid_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.config(Path(directory), "-1")
            with self.assertRaises(ValueError):
                read_config(path)
