import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from alexNet import AlexNet
from caffe_classes import class_names

%matplotlib inline

if __name__ == "__main__":

    imagenet_mean = np.array([104., 117., 124.], dtype = np.float32)

    current_dir = os.getcwd()
    image_dir = os.path.join(current_dir, 'images')

    img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                        if f.endswith('jpeg')]

    # Laod all images
    imgs = []
    for f in img_files:
        imgs.append(cv2.imread(f))

    # plot images
    fig = plt.figure(figsize=(15,6))
    for i, img in enumerate(imgs):
        fig.add_subplot(1, 3, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')

    # placeholder for input and output dropout rate
    x = tf.placeholder(tf.float32, [1, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    # create model with default config
    model = AlexNet(x, keep_prob, 1000, [])

    # activation of the last layer
    score = model.fc8

    # softmax the output
    softmax = tf.nn.softmax(score)

    # run tensorflow session
    with tf.session as sess:
        # initialize all variables
        sess.run(tf.global_variables_initializer())

        # laod pretrained weights
        model.load_initial_weights(sess)

        # figure handle
        fig2 = plt.figure(figsize = (15, 6))

        # for all images
        for i, image in enomerate(imgs):

            # convert images to float32 and resize to (227x227)
            img = cv2.resize(image.astype(np.float32), (227, 227))

            # subtract the imagenet mean
            img -= imagenet_mean

            # reshape
            img = img.reshape((1, 227, 227, 3))

            # calculate class probability
            prob = sess.run(softmax, feed_dict = {x:img, keep_prob:1})

            # get class name of highest probability class
            class_name = class_names[np.argmax(prob)]

            #plot images
            fig2.add_subplot(1, 3, i+1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Class: " + clacc_name + ", probability: %.4f" %prob[0, np.argmax(prob)])
            plt.axis('off')
