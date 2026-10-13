"""验证最终筛选不能因模型一致而恢复已排除词或需要语境词。"""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from cultural_participation.finalize_vocabulary import export
from cultural_participation.vocabulary import Vocabulary


class FinalizeVocabularyTest(unittest.TestCase):
    def test_review_overrides_consensus_and_context_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = ["term", "freq", "ds_domains", "qw_domains", "ds_decision", "qw_decision"]
            examples = [
                ["双域", 10, "business|technology", "technology|business", "standalone", "standalone"],
                ["需语境", 9, "arts", "arts", "context_required", "standalone"],
                ["已排除", 8, "arts", "arts", "standalone", "standalone"],
                ["新人物", 7, "arts", "none", "context_required", "exclude"],
                ["都不是", 6, "none", "none", "standalone", "standalone"],
                ["未裁决分歧", 5, "arts", "sports", "standalone", "standalone"],
            ]
            with (root / "paired.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(examples)
            reviews = [
                {"term": "已排除", "scope_id": "A1", "final_domains": ["none"], "final_decision": "exclude", "adjudication_reason": "通用词"},
                {"term": "新人物", "scope_id": "B1", "final_domains": ["arts"], "final_decision": "standalone", "adjudication_reason": "明确人名", "term_type": "person"},
            ]
            (root / "reviews.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reviews), encoding="utf-8")
            result = export(root / "paired.csv", root / "reviews.jsonl", root / "out", "test")
            self.assertEqual(result["unique_terms"], 2)
            self.assertEqual(result["multi_domain_terms"], 1)
            vocab = Vocabulary(root / "out/vocabulary.json")
            self.assertEqual(set(vocab.rules), {"双域", "新人物"})
            self.assertEqual(vocab.rules["双域"]["domains"], ["business", "technology"])


if __name__ == "__main__":
    unittest.main()
