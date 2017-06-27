from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import tensorflow as tf
import numpy as np

emotions = {
    0:'anger',
    1:'disgust',
    2:'fear',
    3:'happy',
    4:'sad',
    5:'surprise',
    6:'neutral'
}

def to_gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#img = mpimg.imread('example-monalisa.jpg')
#img = mpimg.imread('example-anger.jpg')
#img = mpimg.imread('example-happy.jpg')
img = mpimg.imread('example-fear.jpg')



img = to_gray(img)
img = misc.imresize(img, (48,48), interp='bilinear', mode=None)
#img = np.resize(img,(48,48))
plt.imshow(img, cmap = plt.get_cmap('gray'))
plt.show()

sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('emotion_classification.meta')
new_saver.restore(sess, 'emotion_classification')
tf.get_default_graph().as_graph_def()

x = sess.graph.get_tensor_by_name("x:0")
y_conv = sess.graph.get_tensor_by_name("logits:0")
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")

image_0 = np.resize(img,(1,48,48,1))

result = sess.run(y_conv, feed_dict={x:image_0, keep_prob:1.0})
label = sess.run(tf.argmax(result, 1))
print(emotions[label[0]])