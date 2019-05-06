from unittest import TestCase
from main.common.configuration import Configuration


class TestConfiguration(TestCase):

    def test(self):
        conf = Configuration('main/conf/config.yml', 'main/conf/logging.yml')
        print(conf.get('vocab-file'))



