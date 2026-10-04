from django.test import SimpleTestCase

from scraper.evaluation import metrics as M


def rec(**kw):
    base = {"has_gt": True, "gt_available": True, "success": True, "exact_match": True,
            "cardinality": "single", "list_tp": None, "list_fp": None, "list_fn": None,
            "n_values": 0, "token_f1": 1.0}
    base.update(kw)
    return base


class MetricTests(SimpleTestCase):
    def test_prf(self):
        self.assertAlmostEqual(M.prf(8, 2, 2)["f1"], 0.8)
        self.assertEqual(M.prf(0, 3, 3)["f1"], 0.0)
        self.assertIsNone(M.prf(0, 0, 0)["f1"])

    def test_single_field_counting(self):
        self.assertEqual(M.record_counts(rec()), (1, 0, 0))                       # correct
        self.assertEqual(M.record_counts(rec(exact_match=False)), (0, 1, 1))      # wrong value
        self.assertEqual(M.record_counts(rec(success=False, exact_match=False)), (0, 0, 1))  # missing
        self.assertEqual(M.record_counts(rec(gt_available=False)), (0, 1, 0))     # value for absent field
        self.assertEqual(M.record_counts(rec(gt_available=False, success=False)), (0, 0, 0))  # TN
        self.assertIsNone(M.record_counts(rec(has_gt=False)))                     # unlabelled

    def test_list_field_counting_and_correctness(self):
        r = rec(cardinality="many", list_tp=9, list_fp=1, list_fn=1)
        self.assertEqual(M.record_counts(r), (9, 1, 1))
        self.assertTrue(M.is_correct(r, 0.9))
        self.assertFalse(M.is_correct(rec(cardinality="many", list_tp=5, list_fp=0, list_fn=5), 0.9))

    def test_aggregate(self):
        agg = M.aggregate([rec(), rec(exact_match=False), rec(success=False, exact_match=False),
                           rec(has_gt=False)])
        self.assertEqual((agg["tp"], agg["fp"], agg["fn"], agg["fields"]), (1, 1, 2, 3))
        self.assertAlmostEqual(agg["precision"], 0.5)
        self.assertAlmostEqual(agg["recall"], 1 / 3)
        self.assertAlmostEqual(agg["field_accuracy"], 1 / 3)

    def test_binary_eval(self):
        ev = M.binary_eval([True, True, False, False], [True, False, True, False])
        self.assertEqual((ev["tp"], ev["fp"], ev["fn"], ev["tn"]), (1, 1, 1, 1))
        self.assertAlmostEqual(ev["fpr"], 0.5)
        self.assertAlmostEqual(ev["false_alarm_share"], 0.25)

    def test_spearman(self):
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(M.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertIsNone(M.spearman([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(M.spearman([1, 2], [1, 2]))

    def test_sign_test(self):
        self.assertAlmostEqual(M.sign_test(10, 0), 2 / 1024)
        self.assertEqual(M.sign_test(5, 5), 1.0)
        self.assertIsNone(M.sign_test(0, 0))

    def test_describe(self):
        d = M.describe([1, 2, 3, 4, None])
        self.assertEqual((d["n"], d["median"], d["min"], d["max"]), (4, 2.5, 1, 4))
        self.assertIsNone(M.describe([])["mean"])
