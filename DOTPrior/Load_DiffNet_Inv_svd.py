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

    dims = numpy.array(list(range(len(inData.shape))));
    data = numpy.array(inData)
    data = data.transpose(numpy.flip(dims))
    print(data.shape)
    return data


# def dense_to_one_hot(labels_dense, num_classes=10):
#  """Convert class labels from scalars to one-hot vectors."""
#  num_labels = labels_dense.shape[0]
#  index_offset = numpy.arange(num_labels) * num_classes
#  labels_one_hot = numpy.zeros((num_labels, num_classes))
#  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#  return labels_one_hot


class DataSet(object):

    def __init__(self, indata, list_of_arrays):
        """Construct a DataSet"""
        for i in range(0, len(list_of_arrays)):
            cmd_dict2var = "%s = indata['%s']" % (list_of_arrays[i], list_of_arrays[i])
            exec(cmd_dict2var)

        for i in range(0, len(list_of_arrays) - 1):
            cmd_assert = "assert %s.shape[0] == %s.shape[0]" % (list_of_arrays[i], list_of_arrays[i + 1])
            exec(cmd_assert)

        cmd_num_examples = "self._num_examples = %s.shape[0]" % (list_of_arrays[0]);
        exec(cmd_num_examples)

        for i in range(0, len(list_of_arrays)):
            cmd_define = "self._%s = %s" % (list_of_arrays[i], list_of_arrays[i])
            exec(cmd_define)
        self._listnames = list_of_arrays;
        self._numListnames = len(list_of_arrays);
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    def data_array(self, num):
        cmd = "self._%s " % (self._listnames[num])

        return eval(cmd)

    @property
    def truea(self):
        return self._truea

    @property
    def trues(self):
        return self._trues

    @property
    def recona(self):
        return self._recona

    @property
    def recons(self):
        return self._recons

    @property
    def data(self):
        return self._data

    @property
    def prior(self):
        return self._prior

    @property
    def priorslice(self):
        return self._priorslice

    @property
    def priorslice(self):
        return self._U

    @property
    def priorslice(self):
        return self._s

    @property
    def priorslice(self):
        return self._Vt

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
            for i in range(len(self._listnames)):
                exec("self._%s = self.data_array(i)[perm]" % (self._listnames[i]))

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples, "batch size excedes maximum"
        end = self._index_in_epoch
        out = dict();  # [None]*self._numListnames;
        for i in range(len(self._listnames)):
            exec("out['%s'] = numpy.array(self._%s[start:end])" % (self._listnames[i], self._listnames[i]))
        return out


def read_data_sets(FileNameTrain, FileNameTest, string_array):
    class DataSets(object):
        pass

    data_sets = DataSets()
    string_array = ['recona', 'recons', 'priorslice', 'prior', 'truea', 'trues', 'data', 'U', 's', 'Vt'];

    TRAIN_SET = FileNameTrain
    TEST_SET = FileNameTest

    IMAGE_NAME = 'imagesDiff'
    TRUE_NAME = 'imagesInput'
    print('Start loading data')
    train = dict()  # [None]* len(string_array);
    test = dict()  # [None] * len(string_array);

    for i in range(0, len(string_array)):
        cmd_train = "train_%s = extract_images(TRAIN_SET, '%s');" % (string_array[i], string_array[i]);
        cmd_test = "test_%s = extract_images(TEST_SET, '%s');" % (string_array[i], string_array[i]);

        exec(cmd_train);
        cmd_list = "train['%s'] = train_%s " % (string_array[i], string_array[i])
        exec(cmd_list)

        exec(cmd_test)
        cmd_list = "test['%s'] = test_%s " % (string_array[i], string_array[i])
        exec(cmd_list)

    data_sets.train = DataSet(train,
                              string_array)  # [ ('train_').__add__(string_array[i]) for i in range(len(string_array))] )
    data_sets.test = DataSet(test,
                             string_array)  # [ ('test_').__add__(string_array[i]) for i in range(len(string_array)) ] )

    return data_sets