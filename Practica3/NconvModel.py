# -*- coding: utf-8 -*-

# Sample code to use string producer.
"""Necesario para las graficas"""
import matplotlib.pyplot as plot
"""Convmodel entrena la red"""
import tensorflow as tf
import numpy as np
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)
    o_h[x] = 1
    return o_h

#Cambio a 3!
num_classes = 3
#imagenes
batch_size = 4

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image),  one_hot(int(i), num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.image.rgb_to_grayscale(image)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)

        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
# se ha cambiao a 3 y sigmoide a softmax! (sigmoide sirve para 2 clases pero estamos usando 3)
        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y
"""Imagenes"""
example_batch_train, label_batch_train = dataSource(["Train/0/*.jpg", "Train/1/*.jpg" , "Train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["Valid/0/*.jpg", "Valid/1/*.jpg", "Valid/2/*.jpg"], batch_size=batch_size)
#Se añade TEST
example_batch_test, label_batch_test = dataSource(["Test/0/*.jpg", "Test/1/*.jpg", "Test/2/*.jpg"], batch_size=batch_size)
"""exe"""
example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
#Se añade TEST
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(tf.cast(example_batch_train_predicted,tf.float64) - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(tf.cast(example_batch_valid_predicted,tf.float64) - label_batch_valid))
#Se añade TEST
cost_test = tf.reduce_sum(tf.square(tf.cast(example_batch_test_predicted,tf.float64) - label_batch_test))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

#Objeto saver para guardar pesos
saver = tf.train.Saver()

#Lista para las graficas
train_data_list= []
valid_data_list = []


with tf.Session() as sess:

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    """Bucles de entrenamiento"""
    for _ in range(200):
        #Llamamos al optimizador
        sess.run(optimizer)
        #agrupamos en 20
        if _ % 20 == 0:
            print "Iteration:", _, "======================================================="

            #print sess.run(label_batch_valid)
            #print sess.run(example_batch_valid_predicted)
            print"Valid error: ", sess.run(cost_valid), "        Train error: ", sess.run(cost)
            #Se insertan valores a las listas
            train_data_list.append(sess.run(cost))
            valid_data_list.append(sess.run(cost_valid))
            print ""

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print"Model saved in file: %s" % save_path
            
    coord.request_stop()
    coord.join(threads)


"""Grafica"""
plot.ylabel('Errores')
plot.xlabel('Epocas')
tr_handle, = plot.plot(train_data_list)
vl_handle, = plot.plot(valid_data_list)
plot.legend(handles=[tr_handle, vl_handle],
       labels=['Error entrenamiento', 'Error validacion'])
plot.savefig('Grafica_3.png')
