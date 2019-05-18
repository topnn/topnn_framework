import tensorflow as tf

def print_tensor(sess, tensor_name):
    """prints value of a given tensor"""

    var = [v for v in tf.trainable_variables() if v.name == "fully_connected/weights:0"][0]
    print(sess.run(var))

def print_tensors(sess):
    """ prints values of all tensors."""

    vars = [(v.name,  sess.run(v)) for v in tf.trainable_variables()]
    print(vars)