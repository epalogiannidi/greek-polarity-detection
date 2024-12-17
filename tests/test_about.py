
import unittest

from recipe_package_template.about import __version__, __author__, __summary__, __title__, __copyright__
from recipe_package_template.about import __MAJOR__, __MINOR__, __PATCH__


class TestBrain(unittest.TestCase):
    """ src.brain unit tests """

    def test_about(self):
        """ Test recipe_package_template.about """
        print(__title__)
        self.assertIsInstance(__version__, str)
        self.assertIsInstance(__author__, str)
        self.assertIsInstance(__summary__, str)
        self.assertIsInstance(__copyright__, str)
        self.assertEqual(__title__, "recipe_package_template")
        self.assertIsInstance(__MAJOR__, int)
        self.assertIsInstance(__MINOR__, int)
        self.assertIsInstance(__PATCH__, int)
        self.assertEqual(len(__version__.split(".")), 3)