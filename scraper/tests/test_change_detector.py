from django.test import SimpleTestCase

from scraper.services.change_detector import detect_change, dom_profiles, weighted_jaccard_distance
from scraper.services.htmlutil import parse_html


def profiles(html):
    return dom_profiles(parse_html(html))


BASE = "<html><body><main><div class='a'><span class='p'>1</span><span class='q'>2</span></div></main></body></html>"


class ChangeDetectorTests(SimpleTestCase):
    def test_identical_is_zero(self):
        p = profiles(BASE)
        report = detect_change(p, p)
        self.assertEqual(report.structural_distance, 0.0)
        self.assertEqual(report.markup_distance, 0.0)
        self.assertFalse(report.changed)

    def test_no_previous(self):
        report = detect_change(None, profiles(BASE))
        self.assertFalse(report.previous_available)
        self.assertFalse(report.changed)

    def test_class_rename_is_markup_only(self):
        renamed = BASE.replace("class='p'", "class='x9f3'")
        report = detect_change(profiles(BASE), profiles(renamed))
        self.assertEqual(report.structural_distance, 0.0)
        self.assertGreater(report.markup_distance, 0.0)
        self.assertTrue(report.changed)

    def test_wrapping_changes_structure(self):
        wrapped = BASE.replace("<div class='a'>", "<section><div class='a'>").replace("</div>", "</div></section>")
        report = detect_change(profiles(BASE), profiles(wrapped))
        self.assertGreater(report.structural_distance, 0.0)

    def test_distance_bounds(self):
        self.assertEqual(weighted_jaccard_distance({}, {}), 0.0)
        self.assertEqual(weighted_jaccard_distance({"a": 1}, {"b": 1}), 1.0)
        self.assertAlmostEqual(weighted_jaccard_distance({"a": 2}, {"a": 1}), 0.5)
