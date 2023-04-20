import os

import tensorflow as tf

from sequences import HuggingFaceAudioSeq

from datasets import load_dataset, Audio

def set_gpu_server_env_var(target_gpus='6'):
    """
    Set environmental variables that are specific to running code on GPU server
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = target_gpus
    print("Set target devices for CUDA {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

    os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
    print("Set XLA_FLAGS for CUDA = {}".format(os.environ["XLA_FLAGS"]))

    os.environ['TF_XLA_FLAGS'] = "--tf_xla_enable_xla_devices"
    print("Set TF_XLA_FLAGS for TF = {}".format(os.environ["TF_XLA_FLAGS"]))

def create_keras_seq(dataset_name, split, batch_size, sampling_rate, max_audio_len_s, max_target_len, cache_dir=None):
    # Create dataset
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    # Cast column for audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # Create sequence as datat for keras fit
    seq = HuggingFaceAudioSeq(dataset, batch_size=batch_size, sr=sampling_rate,
                                max_audio_len_s=max_audio_len_s, max_target_len=max_target_len)

    return seq


def load_saved_model(model, model_load_path, train_seq):
    # Load previously saved model

    with tf.device('/GPU:0'):
        first_input = (train_seq[0]["source"], train_seq[0]["target"])

        model(first_input)
        model.load_weights(model_load_path)