import random
import tempfile
import unittest

import src.utility as util


class TestSaving(unittest.TestCase):
    def test_1(self):
        big_object = [random.random() for _ in range(1000)]

        with tempfile.TemporaryDirectory() as tmpdirname:
            util.pickle_into_several_parts(tmpdirname, 'test', big_object, 8)
            result = util.pickle_from_several_parts(tmpdirname, 'test')
            self.assertEqual(big_object, result)

    def test_2(self):
        big_object = [random.random() for _ in range(1000)]
        big_object2 = [random.random() for _ in range(1000)]

        with tempfile.TemporaryDirectory() as tmpdirname:
            util.pickle_into_several_parts(tmpdirname, 'test', big_object, 8)
            util.pickle_into_several_parts(tmpdirname, 'test2', big_object2, 6)
            result = util.pickle_from_several_parts(tmpdirname, 'test')
            result2 = util.pickle_from_several_parts(tmpdirname, 'test2')
            self.assertEqual(big_object, result)
            self.assertEqual(big_object2, result2)

