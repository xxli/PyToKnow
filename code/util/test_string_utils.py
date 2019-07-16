import unittest

from string_utils import *

class TestStringUtils(unittest.TestCase):

    def test_is_digit(self):
        self.assertTrue(is_digit('10'))
        self.assertTrue(is_digit('-10.10'))
        self.assertTrue(is_digit('0.1'))
        self.assertTrue(is_digit('a'))

if __name__ == '__main__':
    unittest.main()
    
