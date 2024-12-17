import unittest
from recipe_package_template.utils import Timer


class TestUtils(unittest.TestCase):

    def test_timer(self):
        with Timer() as t:
            print("Captured with Timer.")
        self.assertIsInstance(t.elapsed, str)
