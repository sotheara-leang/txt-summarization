from unittest import TestCase

from main.data.giga_world import *
from main.common.util.file_util import FileUtil
from main.common.common import *


class TestMemoryDataLoader(TestCase):

    def test(self):
        dataloader = GigaWorldMemoryDataLoader(
            FileUtil.get_file_path(conf.get('train:article-file')),
            FileUtil.get_file_path(conf.get('train:summary-file')), 15)

        for i in range(dataloader.get_num_batch()):

            batch = dataloader.get_batch(i)

            print(batch)

