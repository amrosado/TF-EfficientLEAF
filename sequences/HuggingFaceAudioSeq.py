import math

import tensorflow as tf
from tensorflow import keras

class HuggingFaceAudioSeq(keras.utils.Sequence):
    def __init__(self, huggingface_dataset, batch_size):
        self.batch_size = batch_size
        self.hugging_face_dataset = huggingface_dataset
        # 16000 sample rate times 30s
        self.max_len = 16000 * 30

    def __len__(self):
        return math.ceil(len(self.hugging_face_dataset) / self.batch_size)

    def __getitem__(self, idx):
        low = idx*self.batch_size
        high = (idx+1)*self.batch_size
        # max len = 475760 w/ sr 16000 which is ~30 seconds
        padded_audio = []
        items = self.hugging_face_dataset[low:high]["audio"]
        for i in items:
            padded_audio.append(tf.pad(i["array"], [[0, self.max_len - i["array"].shape[0]]]))
            if i["array"].shape[0] > self.max_len:
                pass
        text = self.hugging_face_dataset[low:high]["text"]

        return (tf.convert_to_tensor(padded_audio), text), []


