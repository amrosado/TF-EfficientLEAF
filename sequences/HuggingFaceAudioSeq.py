import math
import random

import tensorflow as tf
from tensorflow import keras

"""
Vectorizer
"""

class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
                ["-", "#", "<", ">"]
                + [chr(i + 96) for i in range(1, 27)]
                + [" ", ".", ",", "?"]
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
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

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
        self.indexes = [i for i in range(len(huggingface_dataset)//batch_size)]
        random.shuffle(self.indexes)
        print("Vocab size for vectorizer", len(self.vectorizer.get_vocabulary()))

    def __len__(self):
        return len(self.indexes)

    def on_epoch_end(self):
        random.shuffle(self.indexes)
    def __getitem__(self, idx):
        i = self.indexes[idx]

        low = i*self.batch_size
        high = (i+1)*self.batch_size
        # max len = 475760 w/ sr 16000 which is ~35 seconds
        padded_audios = []
        vec_texts = []

        items = self.hugging_face_dataset[low:high]["audio"]
        texts = self.hugging_face_dataset[low:high]["text"]

        for i in items:
            padded_audios.append(tf.pad(i["array"], [[0, self.max_len - i["array"].shape[0]]]))

        for i in texts:
            text = i.numpy().decode("utf-8")
            text = self.vectorizer(text)
            vec_texts.append(text)

        return {"source": tf.convert_to_tensor(padded_audios), "target": tf.convert_to_tensor(vec_texts)}


