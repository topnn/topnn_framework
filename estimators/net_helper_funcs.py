import json
import os
import sys
import tarfile
import urllib
from io import BytesIO
from typing import List

import cv2
import numpy as np
import tensorflow as tf


def num_of_corrects(y_true, y_pred):
    actual = tf.argmax(y_true, axis=1)
    pred = tf.argmax(y_pred, axis=1)
    equality = tf.equal(pred, actual)
    return tf.reduce_sum(tf.cast(equality, tf.int64))





def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """Downloads the `tarball_url` and uncompresses it locally.
    Args:
      tarball_url: The URL of a tarball file.
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def download_and_extract_model(url, ckpt_path):
    checkpoint_dir = os.path.dirname(ckpt_path)
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)
    if not tf.gfile.Exists(ckpt_path):
        download_and_uncompress_tarball(url, checkpoint_dir)
