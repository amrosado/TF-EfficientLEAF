"""
Title: Automatic Speech Recognition with Transformer
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2021/01/13
Last modified: 2021/01/13
Description: Training a sequence-to-sequence Transformer for automatic speech recognition.
Accelerator: GPU
"""
"""
## Introduction

Automatic speech recognition (ASR) consists of transcribing audio speech segments into text.
ASR can be treated as a sequence-to-sequence problem, where the
audio can be represented as a sequence of feature vectors
and the text as a sequence of characters, words, or subword tokens.

For this demonstration, we will use the LJSpeech dataset from the
[LibriVox](https://librivox.org/) project. It consists of short
audio clips of a single speaker reading passages from 7 non-fiction books.
Our model will be similar to the original Transformer (both encoder and decoder)
as proposed in the paper, "Attention is All You Need".


**References:**

- [Attention is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- [Very Deep Self-Attention Networks for End-to-End Speech Recognition](https://arxiv.org/pdf/1904.13377.pdf)
- [Speech Transformers](https://ieeexplore.ieee.org/document/8462506)
- [LJSpeech Dataset](https://keithito.com/LJ-Speech-Dataset/)
"""


import os
import random
from glob import glob
import tensorflow as tf
from tensorflow import keras

from models.EfficientLEAF import EfficientLeaf

import matplotlib.pyplot as plt


"""
## Define the Transformer Input Layer

When processing past target tokens for the decoder, we compute the sum of
position embeddings and token embeddings.

When processing audio features, we apply convolutional layers to downsample
them (via convolution stides) and process local relationships.
"""


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


"""
## Transformer Encoder Layer
"""


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""
## Transformer Decoder Layer
"""


class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.self_att = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = tf.keras.layers.Dropout(0.5)
        self.enc_dropout = tf.keras.layers.Dropout(0.1)
        self.ffn_dropout = tf.keras.layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                tf.keras.layers.Dense(feed_forward_dim, activation="relu"),
                tf.keras.layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.

        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


"""
## Complete the Transformer model

Our model takes audio spectrograms as inputs and predicts a sequence of characters.
During training, we give the decoder the target character sequence shifted to the left
as input. During inference, the decoder uses its own past predictions to predict the
next token.
"""


class Transformer(keras.Model):
    def __init__(
            self,
            num_hid=64,
            num_head=2,
            num_feed_forward=128,
            source_maxlen=100,
            target_maxlen=100,
            num_layers_enc=4,
            num_layers_dec=1,
            num_classes=10,
    ):
        super().__init__()
        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        """
        E leaf frontend to generate mel spectrogram like input into encoder
        """
        self.frontend = EfficientLeaf()

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = tf.keras.layers.Dense(num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source, target = inputs
        source = self.frontend(source)[:, :, :, 0]

        # for i in range(source.shape[0]):
        #     fig, ax = plt.subplots()
        #     spec = tf.transpose(source[i,:100,:,0])
        #     ax.imshow(spec)
        #     plt.show()

        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        if isinstance(batch, tuple):
            batch = batch[0]
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self((source, dec_input))
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        try:
            if len(trainable_vars) != len(gradients):
                for i in range(len(trainable_vars)):
                    print("Trainable vars {}".format(trainable_vars[i]))

                for i in range(len(gradients)):
                    print("Gradients {}".format(gradients[i]))

            self.optimizer.apply_gradients(zip(gradients, trainable_vars, strict=False))
        except:
            print("Length of preds {}".format(preds.shape))
            print("Length of mask {}".format(mask.shape))
            print("Length of loss {}".format(loss.shape))
            print("Length of trainable vars {}".format(len(trainable_vars)))
            print("Length of gradients {}".format(len(gradients)))


        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        if isinstance(batch, tuple):
            batch = batch[0]
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        source = self.frontend(source)[:, :, :, 0]
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input


# """
# ## Download the dataset
#
# Note: This requires ~3.6 GB of disk space and
# takes ~5 minutes for the extraction of files.
# """
#
# keras.utils.get_file(
#     os.path.join(os.getcwd(), "data.tar.gz"),
#     "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
#     extract=True,
#     archive_format="tar",
#     cache_dir=".",
# )


# saveto = "./datasets/LJSpeech-1.1"
# wavs = glob("{}/**/*.wav".format(saveto), recursive=True)

# id_to_text = {}
# with open(os.path.join(saveto, "metadata.csv"), encoding="utf-8") as f:
#     for line in f:
#         id = line.strip().split("|")[0]
#         text = line.strip().split("|")[2]
#         id_to_text[id] = text


# def get_data(wavs, id_to_text, maxlen=50):
#     """returns mapping of audio paths and transcription texts"""
#     data = []
#     for w in wavs:
#         id = w.split("/")[-1].split(".")[0]
#         if len(id_to_text[id]) < maxlen:
#             data.append({"audio": w, "text": id_to_text[id]})
#     return data



