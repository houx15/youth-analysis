"""联结验证：词表版本、每词系数与缺失处理。"""
import unittest

import pandas as pd

from cultural_participation.behavior_semantic_link import METRICS, association_table, validate_join


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.sem = pd.DataFrame([dict(term=t, domains="arts", covered=True, known_person=False,
            gender_marked=False, gender=i, prestige=i*.2, status=i*.3,
            cultivation=i*.4, potency=-i, morality=i*.5, affluence=i*.6) for i,t in enumerate(["甲", "乙", "丙"])])
        self.gaps = pd.DataFrame([dict(term=t, metric=metric, model=model,
            status="ok", estimate_f_minus_m=i*.01) for metric in METRICS
            for model in ["unadjusted_OLS", "activity_adjusted_OLS"]
            for i,t in enumerate(self.sem.term)])
        self.coverage = pd.DataFrame([dict(term=t, gender=g, retweets=10, expressions=20,
            retweet_users=5, expression_users=8) for t in self.sem.term for g in ["m", "f"]])
        self.sm = {"input": {"signature": {"vocabulary_sha256": "same"}}}
        self.bm = {"status": "complete", "source_signature": {"vocabulary_sha256": "same"}}

    def join(self):
        return validate_join(self.sem, self.gaps, self.coverage, self.sm, self.bm)

    def test_exact_term_link_and_status_axis(self):
        result = self.join()
        self.assertEqual(len(result), 36)
        self.assertTrue((result[result.term == "乙"].estimate_f_minus_m == .01).all())
        self.assertTrue((result[result.term == "乙"].status == .3).all())
        table = association_table(result)
        row = table[(table.variant == "all_covered_estimable") & (table.scope == "all") & (table.axis == "gender")].iloc[0]
        self.assertAlmostEqual(row.pearson, 1)
        self.assertEqual(row.n_terms, 3)
        self.assertEqual(table[table.variant == "at_least_20_participants"].n_terms.max(), 0)

    def test_invalid_version_and_incomplete_keys_rejected(self):
        self.bm["source_signature"]["vocabulary_sha256"] = "other"
        with self.assertRaises(ValueError): self.join()
        self.bm["source_signature"]["vocabulary_sha256"] = "same"
        self.gaps = self.gaps.iloc[1:]
        with self.assertRaises(ValueError): self.join()

    def test_duplicate_and_missing_vectors(self):
        self.sem.loc[0, "covered"] = False
        self.gaps.loc[self.gaps.term == "乙", "status"] = "insufficient_or_singular"
        table = association_table(self.join())
        self.assertEqual(table[(table.variant == "all_covered_estimable") & (table.scope == "all")].n_terms.max(), 1)
        self.gaps = pd.concat([self.gaps, self.gaps.iloc[:1]])
        with self.assertRaises(ValueError): self.join()


if __name__ == "__main__": unittest.main()
