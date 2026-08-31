from django.test import SimpleTestCase

from scraper.services.change_detector import detect_change, structural_distance


class ChangeDetectorTests(SimpleTestCase):
    def test_identical_is_zero(self):
        self.assertEqual(structural_distance("a>b>c", "a>b>c"), 0.0)

    def test_no_previous(self):
        report = detect_change("", "a>b>c")
        self.assertFalse(report.previous_available)
        self.assertFalse(report.changed)

    def test_detects_change(self):
        report = detect_change("div>h1>span>span", "section>div>h2>p>p>a")
        self.assertTrue(report.previous_available)
        self.assertGreater(report.structural_distance, 0.0)
        self.assertTrue(report.changed)
