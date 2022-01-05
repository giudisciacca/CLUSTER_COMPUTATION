import tensorflow as tf
from tensorflow.python.framework import ops
import jax.numpy as np
import scipy.sparse as ssp
from jax import grad

def numpy_square(x):
    x = [x,x,x,x,0,0]
    #x = csr_matrix(x);
    return np.sum(np.square(x))    #.toarray()

# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_function(func, inp, Tout,name=name)


# Def custom square function using np.square instead of tf.square:
def mysquare(x, name=None):
    with ops.op_scope([x], name, "Mysquare") as name:
        sqr_x = py_func(numpy_square,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_MySquareGrad)  # <-- here's the call to the gradient
        return sqr_x[0]

def ggrad():
    return grad(mysquare)

# Actual gradient:
def _MySquareGrad(op, grad):
    x = op.inputs[0]
    #el = grad(mysquare)
    return [np.float32(el)]# [tf.py_func(el, inp=[op.inputs], Tout = tf.float32)]# add a "small" error just to see the difference:

el = grad(mysquare)

with tf.Session() as sess:
    x = tf.constant([1., 2.])
    y = mysquare(x)
    tf.global_variables_initializer().run()

    print(x.eval(), y.eval(), tf.gradients(y, x)[0].eval())
