import os
import gzip
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "./data/"

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 10000  # Size of the validation set.

def maybe_download(filename):
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        size = os.stat(filepath).st_size
        print('Successfully downloaded', filename, size, 'bytes.')

# Extract the images
def extract_data(filename):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    if filename=='train-images-idx3-ubyte.gz':
        num_images = 60_000
    elif filename=='t10k-images-idx3-ubyte.gz':
        num_images = 10_000
    
    filepath = os.path.join(DATA_DIRECTORY, filename)
    with gzip.open(filepath) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        data = data/255 - 0.5 # Normalize
    return data

# Extract the labels
def extract_labels(filename):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    if filename=='train-labels-idx1-ubyte.gz':
        num_images = 60_000
    elif filename=='t10k-labels-idx1-ubyte.gz':
        num_images = 10_000
    
    filepath = os.path.join(DATA_DIRECTORY, filename)
    with gzip.open(filepath) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

# Filter dataset helper function to include only eights and threes, 
# then restricting to first 500 samples
def filter_dataset(data, labels):
    req_idx = (labels == 3) | (labels == 8)
    return data[req_idx][:500], labels[req_idx][:500]

# Preprocess dataset
def preprocess_dataset():
    train_data = extract_data('train-images-idx3-ubyte.gz')
    train_labels = extract_labels('train-labels-idx1-ubyte.gz')
    test_data = extract_data('t10k-images-idx3-ubyte.gz')
    test_labels = extract_labels('t10k-labels-idx1-ubyte.gz')
    
    train_data, train_labels = filter_dataset(train_data, train_labels)
    test_data, test_labels = filter_dataset(test_data, test_labels)
    
    return train_data, train_labels, test_data, test_labels

# Plot first 25 images from the data
def plot_samples(data):
    _, axes = plt.subplots(5, 5)
    axes = axes.flatten()
    for ax, img in zip(axes, data[:25]):
        ax.imshow(img.reshape((IMAGE_SIZE, IMAGE_SIZE)), cmap=plt.get_cmap('gray'), vmin=-0.5, vmax=0.5)
        ax.axis('off')
        
    plt.show()
    
# Flatten image matrices
def flatten_data(data):
    return data.reshape((500, -1))

# Plot Histogram of label statistics
def plot_histogram(labels):
    _ = plt.figure(figsize=(5,6))
    # Gridlines
    plt.grid(visible=True, color ='grey', linestyle ='-.', linewidth = 0.5,
             alpha = 0.4)
    label_types = [3, 8]
    label_counts = [np.sum(labels==3), np.sum(labels==8)]
    ax = plt.bar(label_types, label_counts, width=0.4)
    plt.xlabel("Label")
    plt.ylabel("No. of samples")
    # Annotations
    for count, p in zip(label_counts, ax.patches):
        plt.text(p.get_x() + p.get_width()/2, p.get_height()+3,
                 count, ha='center', va='bottom')
    plt.show()

# Relabel (3, 8) to (-1, +1)
def relabel_data(labels):
    labels[labels==3] = -1
    labels[labels==8] = 1
    return labels

# Plot the trajectories of train and test accuracy convergence
def plot_trajectories(trajectories, rolling_avg=True, window_sz=None):
        iter_idx = list(range(1,len(trajectories['train'])+1))
        train_acc_values = trajectories['train']
        test_acc_values = trajectories['test']
        if rolling_avg and window_sz is None:
            train_acc_values = np.cumsum(trajectories['train']) / iter_idx
            test_acc_values = np.cumsum(trajectories['test']) / iter_idx
        elif rolling_avg:
            window_ones = np.ones((window_sz,))
            full_ones = np.ones_like(trajectories['train'])
            dr_factor = np.convolve(full_ones, window_ones)[:len(trajectories['train'])]
            train_acc_values = np.convolve(trajectories['train'], window_ones)[:len(trajectories['train'])] / dr_factor
            test_acc_values = np.convolve(trajectories['test'], window_ones)[:len(trajectories['test'])] / dr_factor
            
        plt.plot(iter_idx, train_acc_values, label='Train Accuracy')
        plt.plot(iter_idx, test_acc_values, color='green', linestyle='dashed', label='Test Accuracy')
        # Gridlines
        plt.grid(visible=True, color ='grey', linestyle ='-.', linewidth = 0.5,
                 alpha = 0.4)
        plt.ylabel(f'Accuracy{"" if not rolling_avg else " (Rolling Average)" if window_sz is None else " (Rolling Average with window = "+str(window_sz)+")"}')
        plt.xlabel('Iterations')
        plt.legend()
        plt.show()

# Preprocess dataset
train_data, train_labels, test_data, test_labels = preprocess_dataset()

# Plot first 25 samples from training data
# plot_samples(train_data)

# Flatten the data feature vectors
train_data, test_data = flatten_data(train_data), flatten_data(test_data)

# Plot histogram of the label counts in training dataset
# plot_histogram(train_labels)

# Relabel the data categories for suiting the learning algorithm
train_labels, test_labels = relabel_data(train_labels), relabel_data(test_labels)

# Visualize trajectories during training of the learning algorithm
learning_algo = Perceptron(train_data.shape[-1])
learning_algo.train(train_data, train_labels, test_data, test_labels, 2000)
plot_trajectories(learning_algo.trajectories, True, 20)

# Final train and test accuracy values at the end of training
print('Final Train Accuracy:', learning_algo.trajectories['train'][-1])
print('Final Test Accuracy:', learning_algo.trajectories['test'][-1])
