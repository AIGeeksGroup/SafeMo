import unittest

from safemo_mmu.reporting import build_summary, summary_text


class ReportingTest(unittest.TestCase):
    def test_public_table_contains_only_paper_metrics(self):
        cells = [("static_forget", "SafeMo-Static", "forget", 1.0)]
        raw = {
            "repetitions": 1,
            "cells": {
                "static_forget": [
                    {
                        "eligible_samples": 35,
                        "effective_samples": 32,
                        "metrics": {
                            "FID": 1.0,
                            "Diversity": 2.0,
                            "R@1": 0.1,
                            "R@2": 0.2,
                            "R@3": 0.3,
                        },
                    }
                ]
            },
        }
        summary = build_summary(raw, cells, "identity")
        protocol = {
            "identity": {
                "dataset_scope": "test",
                "seed": 10,
                "repetitions": 1,
                "unsafe_ids_sha256": "abc",
            }
        }
        rendered = summary_text(summary, protocol)
        self.assertIn("Method\tSubset\tAlpha\tFID\tDiversity\tR@1\tR@2\tR@3", rendered)
        self.assertNotIn("Eligible", rendered)
        self.assertNotIn("Effective", rendered)
        self.assertNotIn("eligible_samples", summary["rows"][0])
        self.assertNotIn("effective_samples", summary["rows"][0])


if __name__ == "__main__":
    unittest.main()
