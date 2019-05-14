from unittest import TestCase

from test.common import *

from main.common.util.file_util import FileUtil
from main.data.giga_world import GigaWorldDataLoader


class TestDataLoader(TestCase):

    def test(self):
        dataloader = GigaWorldDataLoader(
            FileUtil.get_file_path(conf('train:article-file')),
            FileUtil.get_file_path(conf('train:summary-file')), 15)

        counter = 0
        while True:
            batch = dataloader.next_batch()
            if batch is None:
                break

            counter += len(batch)

            print(counter)

            print(batch)








