from unittest import TestCase

from main.common.common import *
from main.common.util.file_util import FileUtil
from main.data.giga import GigaDataLoader


class TestDataLoader(TestCase):

    def test(self):
        dataloader = GigaDataLoader(
            FileUtil.get_file_path(conf.get('train:article-file')),
            FileUtil.get_file_path(conf.get('train:summary-file')), 15)

        batch = dataloader.next()

        print(batch)








