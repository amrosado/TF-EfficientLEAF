import math
import random

import tensorflow as tf
import numpy as np
from tensorflow import keras

"""
Vectorizer
"""

class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
                ["-", "#", "<", ">"]
                + [chr(i + 96) for i in range(1, 27)]
                + [" ", ".", ",", "?", "'"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx[ch] for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab

class HuggingFaceAudioSeq(keras.utils.Sequence):
    def __init__(self, huggingface_dataset, batch_size, sr=16000, max_audio_len_s=35, max_target_len=600):
        self.batch_size = batch_size
        self.hugging_face_dataset = huggingface_dataset
        # 16000 sample rate times 30s
        self.max_len = sr * max_audio_len_s
        self.max_target_len = max_target_len  # all transcripts in out data are < 550 characters
        self.vectorizer = VectorizeChar(self.max_target_len)
        self.indexes = [i for i in range(huggingface_dataset.num_rows)]
        random.shuffle(self.indexes)
        print("Vocab size for vectorizer", len(self.vectorizer.get_vocabulary()))

    def __len__(self):
        return math.floor(len(self.indexes) / self.batch_size)

    def on_epoch_end(self):
        random.shuffle(self.indexes)
    def __getitem__(self, idx):
        idxs = self.indexes[idx:(idx+self.batch_size)]

        audios = self.hugging_face_dataset[idxs]["audio"]
        texts = self.hugging_face_dataset[idxs]["text"]

        source = []
        target = []

        for i in range(len(audios)):
            audio_array = tf.convert_to_tensor(audios[i]["array"], dtype=tf.float32)
            text = self.vectorizer(texts[i])

            try:
                source.append(tf.pad(audio_array, [[0, self.max_len - audio_array.shape[0]]]))
            except:
                print("Shape of audio array {}".format(audio_array.shape))
                print("Audio array {}".format(audio_array))
                exit(9999)
            target.append(text)

        return {"source": tf.stack(source), "target": tf.stack(target)}





