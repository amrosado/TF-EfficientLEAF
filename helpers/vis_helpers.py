import tensorflow as tf

from matplotlib import pyplot as plt

# TODO: finish and test visual representation

def graph_spectrogram(batch_spectogram):
    for i in range(batch_spectogram.shape[0]):
        # show only first 500 given long file hard to appreciate.
        comp_spec = batch_spectogram[i,:500,:,0]
        comp_spec = tf.transpose(comp_spec, (1, 0))
        plt.imshow(comp_spec.numpy())
        plt.xlabel('Time')
        plt.ylabel('Frequency Filter')
        plt.title('Efficient LEAF Example #{}'.format(i))
        plt.savefig('eleaf_example_{}.png'.format(i))