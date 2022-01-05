import os
import shutil
from os.path import join, expanduser, exists
import tensorflow as tf
import sys

# import demandimport
# with demandimport.enabled():
#   import tensorflow as tf

__all__ = ('get_base_dir',
           'default_checkpoint_path', 'default_tensorboard_dir',
           'summary_writers')


def get_base_dir():
    """Get the data directory."""
    base_odl_dir = os.environ.get('ADLER_HOME',
                                  expanduser(join('~', '.adler')))
    data_home = join(base_odl_dir, 'tensorflow')
    if not exists(data_home):
        os.makedirs(data_home)
    return data_home


def default_checkpoint_path(name):
    checkpoint_dir = join(get_base_dir(), 'checkpoints')
    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = join(checkpoint_dir,
                           '{}.ckpt'.format(name))

    return checkpoint_path


def default_tensorboard_dir(name):
    #tensorboard_dir = '/cs/academic/phd3/gdisciac/DOCM_diffNet/tensorboard/priorLearn/MinimisePriorDiff_MULTC/'
    #tensorboard_dir = '/cs/research/medim/gdisciac/tensorboard_prior/GaussBlur/s1anisox/'+ sys.argv[1] + '/';#sys.argv[4]
    #tensorboard_dir = '/cs/research/medim/gdisciac/tensorboard_prior/Optical/'+sys.argv[4]+sys.argv[6]+'/'+ sys.argv[1] + '/';#sys.argv[4]
    tensorboard_dir = '/home/gdisciac/DOTPrior/tensorboard_prior/' + sys.argv[4] +sys.argv[6]+ '/' + sys.argv[1] + '/'
    #tensorboard_dir = '/scratch0/gdisciac/tensorboard_prior/' + sys.argv[4] +sys.argv[6]+ '/' + sys.argv[1] + '/'
    if not exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    return tensorboard_dir


def summary_writers(name, expName, cleanup=False, session=None):
    #    print('In summary writers')
    if session is None:
        session = tf.get_default_session()

    dname = default_tensorboard_dir(name)
    #    print(dname)
    if cleanup and os.path.exists(dname):
        shutil.rmtree(dname)

    test_summary_writer = [None]*len(expName);
    for i in range(0, len(expName)):
        test_summary_writer[i] = tf.summary.FileWriter(dname + '/test_' + expName[i], session.graph)
    #    print('passed test')

    train_summary_writer = tf.summary.FileWriter(dname + '/train_' + expName[0])
    #    print('passed train')
    return test_summary_writer, train_summary_writer
