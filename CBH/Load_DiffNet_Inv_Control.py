#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

# from six.moves import xrange  # pylint: disable=redefined-builtin

import h5py


def extract_images(filename, imageName):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    fData = h5py.File(filename, 'r')
    inData = fData.get(imageName)

    if len(inData.shape) > 3:
        num_images = inData.shape[0]
        rows = inData.shape[1]
        cols = inData.shape[2]
        chan = inData.shape[3]
        data = numpy.array(inData)
        data = data.reshape(num_images, rows, cols, chan)
        print(num_images, rows, cols, chan)
    else:
        num_images = inData.shape[0]
        rows = inData.shape[1]
        cols = inData.shape[2]

        data = numpy.array(inData)
        data = data.reshape(num_images, rows, cols, 1)
        print(num_images, rows, cols)

    return data


# def dense_to_one_hot(labels_dense, num_classes=10):
#  """Convert class labels from scalars to one-hot vectors."""
#  num_labels = labels_dense.shape[0]
#  index_offset = numpy.arange(num_labels) * num_classes
#  labels_one_hot = numpy.zeros((num_labels, num_classes))
#  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#  return labels_one_hot


class DataSet(object):

    def __init__(self, images, true, controls):
        """Construct a DataSet"""

        assert images.shape[0] == true.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape,
                                                       true.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        #    assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1], images.shape[2], images.shape[3])
        controls = controls.reshape(controls.shape[0],
                                controls.shape[1], controls.shape[2], controls.shape[3])

        true = true.reshape(true.shape[0],
                            true.shape[1], true.shape[2], true.shape[3])

        #    Maybe -1 to zero mean
        #    images = numpy.multiply(10.0,images)
        #      # Convert from [0, 255] -> [0.0, 1.0].
        #      images = images.astype(numpy.float32)
        #      images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._true = true
        self._controls = controls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def controls(self):
        return self._controls

    @property
    def true(self):
        return self._true

    @property
    def grad(self):
        return self._grad

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._true = self._true[perm]
            self._controls = self._controls[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._true[start:end],self._controls[start:end]


def read_data_sets(FileNameTrain, FileNameTest):
    class DataSets(object):
        pass

    data_sets = DataSets()

    TRAIN_SET = FileNameTrain
    TEST_SET = FileNameTest
    IMAGE_NAME = 'imagesDiff'
    CONTROL_NAME = 'imagesControl'
    TRUE_NAME = 'imagesInput'

    print('Start loading data')
    train_images = extract_images(TRAIN_SET, IMAGE_NAME)
    train_true = extract_images(TRAIN_SET, TRUE_NAME)
    train_control = extract_images(TRAIN_SET, CONTROL_NAME)

    test_images = extract_images(TEST_SET, IMAGE_NAME)
    test_true = extract_images(TEST_SET, TRUE_NAME)
    test_control = extract_images(TEST_SET, CONTROL_NAME)

    data_sets.train = DataSet(train_images, train_true, train_control)
    data_sets.test = DataSet(test_images, test_true,test_control)

    return data_sets