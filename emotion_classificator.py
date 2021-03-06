import pandas as pd
import numpy as np
import tensorflow as tf
#import scipy.misc as misc
from sklearn.preprocessing import OneHotEncoder
import pickle
from matplotlib import pyplot as plt
#import matplotlib.image as mpimg


# ====================================================================================================================
# Funtions
# ====================================================================================================================

def make_onehot(x,num_labels=7):
    """
    Creates dummy variables for a given column

    :param x: original column
    :param num_labels: number of classes in the column
    :return: dummy-coded columns
    """
    enc = OneHotEncoder(n_values=num_labels)
    return enc.fit_transform(np.array(x).reshape(-1, 1)).toarray()

# def load_custom_image(path='my_pic.jpg', show=False):
#     img = mpimg.imread(path)
#     gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
#     if show:
#         plt.imshow(gray, cmap=plt.get_cmap('gray'))
#         plt.show()
#     return np.resize(gray, (1, 48, 48, 1))

def next_batch(images, labels, step, batch_size):
    """
    Takes a batch from a dataset step by step

    :param images: original dataset (features)
    :param labels: original dataset (labels)
    :param step: number of step
    :param batch_size: size of every batch
    :return: batch
    """
    offset = (step * batch_size) % (images.shape[0] - batch_size)
    batch_images = images[offset: offset + batch_size]
    batch_labels = labels[offset:offset + batch_size]
    return batch_images, batch_labels

def input_tesors(image_shape, n_classes):
    """
    Returns 3 input tensors

    :param image_shape: Shape of the images
    :param n_classes: Number of classes
    :return: Tensors for image inputs, for label inputs and for keep_probability
    """
    tensor_x = tf.placeholder(
        tf.float32,
        [None, image_shape[0], image_shape[1], image_shape[2]],
        name='x'
    )
    tensor_y = tf.placeholder(
        tf.float32,
        [None, n_classes],
        name='y'
    )
    tensor_keep_prob = tf.placeholder(
        tf.float32,
        name='keep_prob'
    )
    return tensor_x, tensor_y, tensor_keep_prob

def layer_conv2d_maxpool(
        x_tensor,
        conv_num_outputs,
        conv_ksize,
        conv_strides,
        pool_ksize,
        pool_strides
):
    """
    Applies convolution with relu activation and max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    weights = tf.Variable(
        tf.random_normal(
            [
                conv_ksize[0],
                conv_ksize[1],
                x_tensor.get_shape().as_list()[-1],
                conv_num_outputs
            ],
            stddev=0.1
        )
    )
    bias = tf.Variable(
        tf.zeros(conv_num_outputs, dtype=tf.float32)
    )
    conv = tf.nn.conv2d(
        x_tensor,
        weights,
        strides=[1, conv_strides[0], conv_strides[1], 1],
        padding='SAME'

    )
    conv = tf.nn.relu(
        tf.nn.bias_add(conv, bias)
    )
    return tf.nn.max_pool(
        conv,
        ksize=[1, pool_ksize[0], pool_ksize[1], 1],
        strides=[1, pool_strides[0], pool_strides[1], 1],
        padding='SAME'
    )

def layer_flatten(x_tensor):
    """
    Flattens x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    return tf.reshape(
        x_tensor,
        [-1, (x_tensor.shape[1] * x_tensor.shape[2] * x_tensor.shape[3]).value]
    )

def layer_fully_connected(x_tensor, num_outputs):
    """
    Applies a fully connected layer with relu activation to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    weights = tf.Variable(
        tf.random_normal(
            [x_tensor.shape[1].value, num_outputs],
            stddev=0.1
        )
    )
    bias = tf.Variable(tf.zeros([num_outputs]))
    return tf.nn.relu(
        tf.add(tf.matmul(x_tensor, weights), bias)
    )

def layer_output(x_tensor, num_outputs):
    """
    Applies an output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    weights = tf.Variable(
        tf.random_normal(
            [x_tensor.shape[1].value, num_outputs],
            stddev=0.1
        )
    )
    bias = tf.Variable(tf.zeros([num_outputs]))
    return tf.add(tf.matmul(x_tensor, weights), bias)

def neural_network(x, keep_prob):
    """
    Creates a neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """

    conv1 = layer_conv2d_maxpool(
        x,
        conv_num_outputs=16,
        conv_ksize=[5,5],
        conv_strides=[1,1],
        pool_ksize=[2, 2],
        pool_strides=[2, 2]
    )

    conv2 = layer_conv2d_maxpool(
        conv1,
        conv_num_outputs=32,
        conv_ksize=[5, 5],
        conv_strides=[1,1],
        pool_ksize=[2, 2],
        pool_strides=[2, 2]
    )

    conv3 = layer_conv2d_maxpool(
        conv2,
        conv_num_outputs=64,
        conv_ksize=[5,5],
        conv_strides=[1,1],
        pool_ksize=[2, 2],
        pool_strides=[2, 2]
    )


    conv_f = layer_flatten(conv3)

    conv_fc1 = tf.nn.dropout(layer_fully_connected(conv_f, 512), keep_prob)
    conv_fc2 = tf.nn.dropout(layer_fully_connected(conv_fc1, 256), keep_prob)
    conv_fc3 = tf.nn.dropout(layer_fully_connected(conv_fc2, 128), keep_prob)

    return layer_output(conv_fc3, 7)

def train_step(session, optimizer, keep_probability, batch_image, batch_label):
    """
    Optimizes the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(
        optimizer,
        feed_dict={
            x: batch_image,
            y: batch_label,
            keep_prob: keep_probability
        }
    )

def print_statistics(session, batch_image, batch_label, cost, accuracy, type="VALIDATION"):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    : return: validation loss, validation accuracy
    """
    loss = session.run(
        cost,
        feed_dict={
            x: batch_image,
            y: batch_label,
            keep_prob: 1.0
        }
    )
    accuracy = session.run(
        accuracy,
        feed_dict={
            x: batch_image,
            y: batch_label,
            keep_prob: 1.0
        }
    )
    print("{} :: Loss = {} ; Accuracy = {} ".format(type, loss, accuracy))
    return loss, accuracy

def plot_stat(values, label, color='g'):
    """
    Creates a plot

    :param values: dataset to visualize
    :param label: plot label
    :param color: line color
    :return: plot
    """
    plt.plot(range(len(values)), values, '-', color=color, label=label)
    # plt.plot(range(len(validation_accuracy)), validation_loss, '-', color='b', label='Recall')
    plt.title("Training")
    plt.xlabel("iteration")
    plt.ylabel('Score')
    plt.legend(loc='upper left')
    #plt.ylim([0, 1])
    plt.show()

# ====================================================================================================================
# Hyperparameters and options
# ====================================================================================================================

# Original Kaggle Dataset
filename_orig="train.csv"
# Ratios to split into training, testing and validation datasets
perc_validation=0.1
perc_test=0.1
# Size of the original images
image_size=48
# Save the split datasets
save_pickle=True
filename_pickle="datasets_train_valid_test.pickle"
# Number of iterations durint the training phase
epochs = 55
# Batch size
batch_size = 512
# Dropout keep probability
keep_probability = 0.9
# Path to save tensorflow model
save_model_path='./emotion_classification'

# ====================================================================================================================
# Data wrangling and preprocessing
# ====================================================================================================================

# Reading the original Kaggle dataset
data_frame = pd.read_csv(filename_orig)

# Balancing the dataset
dataset_bal = data_frame.loc[(data_frame["Emotion"] != 6) & (data_frame["Emotion"] != 3),:]
dataset_imb1 = data_frame.loc[data_frame["Emotion"] == 6,:]
dataset_imb1 = dataset_imb1.sample(n=450)
dataset_imb2 = data_frame.loc[data_frame["Emotion"] == 3,:]
dataset_imb2 = dataset_imb2.sample(n=450)
dataset = dataset_bal.append(dataset_imb1)
dataset = dataset.append(dataset_imb2)

# Normalizing the dataset
data_frame['Pixels'] = (
    data_frame['Pixels']
        .apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        .dropna())

# Reshaping to a proper shape for tensorflow
df_images = np.vstack(data_frame['Pixels']).reshape(-1, image_size, image_size, 1)
#print(df_images)

# Creating dummy variables
df_labels = make_onehot(data_frame['Emotion'])
#print(df_labels)

# Shuffling the dataset
shuffle = np.random.permutation(df_images.shape[0])
df_images = df_images[shuffle]
df_labels = df_labels[shuffle]

# Splitting into validation, testing and training datasets
marker_validation = int(df_images.shape[0] * perc_validation)
marker_test = marker_validation + int(df_images.shape[0] * perc_test)
validation_labels = df_labels[:marker_validation]
test_labels = df_labels[marker_validation:marker_test]
train_labels = df_labels[marker_test:]
validation_images = df_images[:marker_validation]
test_images = df_images[marker_validation:marker_test]
train_images = df_images[marker_test:]

# Pickling the datasets
if save_pickle:
    with open(filename_pickle, "wb") as file:
        save = {
            "validation_labels": validation_labels,
            "test_labels": test_labels,
            "train_labels": train_labels,
            "validation_images": validation_images,
            "test_images": test_images,
            "train_images": train_images
        }
        pickle.dump(save, file)


# ====================================================================================================================
# Training the model
# ====================================================================================================================


tf.reset_default_graph()

# defining input tensors
x, y, keep_prob = input_tesors((48, 48, 1), 7)

# Defining the model
logits = neural_network(x, keep_prob)
logits = tf.identity(logits, name='logits')

# Variable to minimize during the training step
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
)

# Training algorithm
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Defining the accuracy metric
correct_prediction = tf.equal(
    tf.argmax(logits, 1),
    tf.argmax(y, 1)
)
accuracy = tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32),
    name='accuracy'
)

# Store variables to plot them later
validation_accuracy = []
validation_loss = []

print('>> TRAINING')

# Starting a session
sess =  tf.Session()

# Initializing the variables
sess.run(tf.global_variables_initializer())

# Training cycle
for epoch in range(epochs):
    print('Epoch {:>2}:  '.format(epoch + 1))
    # Loop over all batches
    n_batches = int(train_images.shape[0] / batch_size)
    for batch_i in range(1, n_batches + 1):
        batch_features, batch_labels = next_batch(
                train_images,
                train_labels,
                batch_i,
                batch_size=batch_size)
        train_step(
            session=sess,
            optimizer=optimizer,
            keep_probability=keep_probability,
            batch_image=batch_features,
            batch_label=batch_labels
        )
    # print validation statistics for this epoch
    v_loss, v_acc = print_statistics(sess, validation_images, validation_labels, cost, accuracy, type="VALIDATION")
    validation_loss.append(v_loss)
    validation_accuracy.append(v_acc)

# Print final test statistics
print_statistics(sess, test_images, test_labels, cost, accuracy, type="TEST")

# Plot validation metrics
plot_stat(validation_accuracy, "Validation Accuracy")
plot_stat(validation_loss, "Validation Loss")


# Save Model
saver = tf.train.Saver()
save_path = saver.save(sess, save_model_path)

