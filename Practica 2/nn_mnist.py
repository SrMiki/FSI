import gzip
import cPickle

import tensorflow as tf
import numpy as np

#Imports para nuevas funciones y grafica
import matplotlib
import matplotlib.pyplot as plot

matplotlib.use('Agg')

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

#Train
x_train_data, y_train_data = train_set
y_train_data = one_hot(y_train_data, 10)

#Validaciones
x_valid_data, y_valid_data = valid_set
y_valid_data = one_hot(y_valid_data, 10)

#Test
x_test_data, y_test_data = test_set
y_test_data = one_hot(y_test_data, 10)


#Practica!
# TODO: the neural net!!
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

# 28*28 pixels
""" Neuronas, entradas // pesos """
W1 = tf.Variable(np.float32(np.random.rand(784, 10)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

Wh = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
bh = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(10, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
h2 = tf.matmul(h, Wh) + bh
y = tf.nn.softmax(tf.matmul(h2, W2) + b2)


loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

#muestra
batch_size = 50
#Empezamos de un valor alto para que se realicen mas iteraciones
last_valid_data_error = 1000000000
valid_data_error = 0
#etapa
epoch = 0
#Comparcion de errores
error_difference = 1
#Lista para las graficas
train_data_list = []
valid_data_list = []

#El training acaba cuando el error llegue a un umbral, en esta caso 0.001
while error_difference >= 0.001:
    epoch = epoch + 1
    for jj in xrange(len(x_train_data) / batch_size):
        # Train
        batch_xs = x_train_data[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_train_data[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Train error
    train_data = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    train_data_list.append(train_data)

    if epoch > 1:
        last_valid_data_error = valid_data_error

    # Valid error
    valid_data_error = sess.run(loss, feed_dict={x: x_valid_data, y_: y_valid_data})
    valid_data_list.append(valid_data_error)

    error_difference = last_valid_data_error - valid_data_error
    print "Epoch:", epoch, "Train error: ", train_data#, "||    Data error: ",\
        #valid_data_error, "    Data error prev: ", last_valid_data_error,\
        #"||    Diferencia: ", error_difference

print "----------------------"
print "   Test result...     "
print "----------------------"

total = 0.0
error = 0.0
test_data = sess.run(y, feed_dict={x: x_test_data})
for b, r in zip(y_test_data, test_data):
        if np.argmax(b) != np.argmax(r):
            error += 1

        total += 1
    #Tasa de error final
fail = error / total * 100.0
print "Porcentaje de error: ", fail,"% y portenjate de exito", (100.0 - fail), "%"

"""Grafica"""
plot.ylabel('Errores')
plot.xlabel('Epocas')
tr_handle, = plot.plot(train_data_list)
vl_handle, = plot.plot(valid_data_list)
plot.legend(handles=[tr_handle, vl_handle],
            labels=['Error entrenamiento', 'Error validacion'])
plot.savefig('Grafica_mnist.png')
