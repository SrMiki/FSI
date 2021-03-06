import tensorflow as tf
import numpy as np

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

data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

# Seccionamos los datos

train_range_x = len(x_data) * 0.7
train_range_y = len(y_data) * 0.7

# 70% para entrenamiento
x_data_train = x_data[:int(np.floor(train_range_x))]
y_data_train = y_data[:int(np.floor(train_range_y))]

valid_range_x = train_range_x + len(x_data) * 0.15
valid_range_y = train_range_y + len(y_data) * 0.15

# 15% para validacion
x_data_valid = x_data[int(np.round(train_range_x)):int(np.floor(valid_range_x))]
y_data_valid = y_data[int(np.round(train_range_y)):int(np.floor(valid_range_y))]

# 15% para test
x_data_test = x_data[int(np.round(valid_range_x)):]
y_data_test = y_data[int(np.round(valid_range_y)):]


print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# entrada
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)


W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

# Listas para cargar en la grafica
train_list = []
validation_list = []

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
epoch = 0
diferencia = 100.0
#for epoch in xrange(100): # hay que sustituirlo por un while
while diferencia > 0.001:
    epoch += 1
    for jj in xrange(len(x_data_train) / batch_size):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    data_train = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    train_list.append(data_train)
    print "Epoch Train#:", epoch, "Error: ", data_train
    result = sess.run(y, feed_dict={x: batch_xs})
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print ""

    # Errores de validacion 15%
    data_valid = sess.run(loss, feed_dict={x: x_data_valid, y_: y_data_valid})
    validation_list.append(data_valid)
    print "Epoch Validation#:", epoch, "Error: ", data_valid
    if epoch > 1:
        diferencia = validation_list[-2] - data_valid
    print "Diferencia ", diferencia
    print "----------------------------------------------------------------------------------"

# Errores de validacion 15%
print "----------------------"
print "   Test result...     "
print "----------------------"

total = 0.0
error = 0.0
test_data = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, test_data):
    if np.argmax(b) != np.argmax(r):
        error += 1
    total += 1
fail = error / total * 100.0
print "Porcentaje de error: ", fail,"% y portenjate de exito", (100.0 - fail), "%"

"""Grafica"""
plot.ylabel('Errores')
plot.xlabel('Epocas')
tr_handle, = plot.plot(train_list)
vl_handle, = plot.plot(validation_list)
plot.legend(handles=[tr_handle, vl_handle],
            labels=['Error entrenamiento', 'Error validacion'])
plot.savefig('Grafica_entrenamiento_validacion_while.png')