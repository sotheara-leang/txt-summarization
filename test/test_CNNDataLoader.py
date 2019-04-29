from unittest import TestCase

from main.data.cnn_mail import *


class TestCNNDataLoader(TestCase):

    def testCNNProcessedDataLoader(self):
        dataloader = CNNProcessedDataLoader('/home/vivien/Downloads/finished_files/chunked/train_*', 15)

        while True:
            batch = dataloader.next_batch()

            if batch is None:
                break

            print(batch)
