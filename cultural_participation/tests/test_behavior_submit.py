"""模拟Crane/Slurm提交，检查失败后不会重复提交已有阶段。"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class SubmitTests(unittest.TestCase):
    def run_script(self, response, args=(), fail=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(Path(__file__).parents[2] / 'submit_cultural_behavior.sh', root)
            (root / 'cultural_participation').mkdir()
            (root / 'input').mkdir()
            (root / 'vocab.json').write_text('{}')
            (root / 'cultural_participation/behavior_job.conf').write_text(
                'INPUT_DIR=input\nVOCABULARY=vocab.json\nOUTPUT_ROOT=out\nRUN_LABEL=test\n'
                'BUCKET_COUNT=64\nMONTH_CONCURRENCY=3\nREDUCE_CONCURRENCY=4\n')
            mock = root / 'sbatch'
            mock.write_text('#!/bin/bash\necho "$*" >> "$MOCK_LOG"\nprintf "%s\\n" "$MOCK_RESPONSE"\nexit "$MOCK_EXIT"\n')
            mock.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+':'+os.environ['PATH'], MOCK_LOG=str(root/'calls'),
                       MOCK_RESPONSE=response, MOCK_EXIT='1' if fail else '0')
            result = subprocess.run(['bash', str(root/'submit_cultural_behavior.sh'), *args], env=env, capture_output=True, text=True)
            calls = (root/'calls').read_text().splitlines() if (root/'calls').exists() else []
            return result, calls

    def test_formats(self):
        for text in ['Job id allocated: 17339.', 'Submitted batch job 17339']:
            result, calls = self.run_script(text)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all('--parsable' not in c for c in calls))
            self.assertIn('--dependency=afterok:17339', calls[1])
            self.assertIn('--dependency=afterok:17339', calls[2])

    def test_continue(self):
        result, calls = self.run_script('Job id allocated: 18000.', ['--month-job','17339'])
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(calls),2)
        self.assertIn('--dependency=afterok:17339',calls[0])
        self.assertIn('--dependency=afterok:18000',calls[1])
        result,calls = self.run_script('Submitted batch job 18001',['--reduce-job','18000'])
        self.assertEqual(result.returncode,0)
        self.assertEqual(len(calls),1)
        self.assertIn('--dependency=afterok:18000',calls[0])

    def test_bad_responses_stop(self):
        for text,fail in [('unknown 123',False),('Job id allocated: 123.',True),('Submitted batch job 123\nSubmitted batch job 456',False)]:
            result,calls=self.run_script(text,fail=fail)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(len(calls),1)
        result,calls=self.run_script('Submitted batch job 123',['--month-job','bad'])
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(calls,[])
