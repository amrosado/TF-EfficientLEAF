import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print("Set target devices for CUDA {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

from models.TransformerASR import Transformer

from sequences import HuggingFaceAudioSeq

from datasets import load_dataset, Audio

def main():

    # Set environmental variables for computer the code will run in


    # os.environ["HF_HOME"] = '/opt/localdata/Data/laryn/hugging_face/'
    #
    # print(os.environ["HF_HOME"])

    # os.environ["HF_DATASETS_DOWNLOADED_DATASETS_PATH"] = '/opt/localdata/Data/laryn/hugging_face/downloads'

    # Specify shared dataset configuration values that will be used for train, test, and validation

    batch_size = 20
    max_audio_len_s = 35
    max_target_len = 600
    sampling_rate = 16000

    # all_dataset = load_dataset("librispeech_asr")

    hugging_face_cache_dir = os.path.join('/opt', 'localdata', 'Data', 'laryn', 'hugging_face', 'cache')

    train_dataset = load_dataset("librispeech_asr", split='train.clean.360', cache_dir=hugging_face_cache_dir)
    test_dataset = load_dataset("librispeech_asr", split='test.clean', cache_dir=hugging_face_cache_dir)
    val_dataset = load_dataset("librispeech_asr", split='validation.clean', cache_dir=hugging_face_cache_dir)

    # train_dataset = load_dataset("librispeech_asr", split='train.clean.360')
    # test_dataset = load_dataset("librispeech_asr", split='test.clean')
    # val_dataset = load_dataset("librispeech_asr", split='validation.clean')
    #
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate)).with_format("tf")
    val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate)).with_format("tf")
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=sampling_rate)).with_format("tf")


    train_seq = HuggingFaceAudioSeq(train_dataset, batch_size=batch_size, sr=sampling_rate,
                                    max_audio_len_s=max_audio_len_s, max_target_len=max_target_len)
    test_seq = HuggingFaceAudioSeq(test_dataset, batch_size=batch_size, sr=sampling_rate,
                                   max_audio_len_s=max_audio_len_s, max_target_len=max_target_len)
    val_seq = HuggingFaceAudioSeq(val_dataset, batch_size=batch_size, sr=sampling_rate,
                                  max_audio_len_s=max_audio_len_s, max_target_len=max_target_len)

    # train max_len_audio = 475760, max_len_text=524
    # test max_len_audio = 559280, max_len_text = 576
    # val max_len_audio = 522320, max_len_text = 516

    # max len = 559280 w/ sr 16000 which is ~35 seconds
    #
    # for i in val_dataset:
    #     audio = i["audio"]
    #     text = i["text"]
    #     array = audio["array"]
    #
    #     if array.shape[0] > max_len_audio:
    #         max_len_audio = array.shape[0]
    #
    #     if len(text) > max_len_txt:
    #         max_len_txt = len(text)



    """
    ## Preprocess the dataset
    """



    # def create_text_ds(data):
    #     texts = [_["text"] for _ in data]
    #     text_ds = [vectorizer(t) for t in texts]
    #     text_ds = tf.data.Dataset.from_tensor_slices(text_ds)
    #     return text_ds


    def path_to_audio(audio):
        # spectrogram using stft
        # audio = tf.io.read_file(path)
        # audio, _ = tf.audio.decode_wav(audio, 1)
        # audio = tf.squeeze(audio, axis=-1)
        stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
        x = tf.math.pow(tf.abs(stfts), 0.5)
        # normalisation
        means = tf.math.reduce_mean(x, 1, keepdims=True)
        stddevs = tf.math.reduce_std(x, 1, keepdims=True)
        x = (x - means) / stddevs
        audio_len = tf.shape(x)[0]
        # padding to 10 seconds
        pad_len = 2754
        paddings = tf.constant([[0, pad_len], [0, 0]])
        x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
        return x


    def create_audio_ds(data):
        flist = [_["audio"] for _ in data]
        audio_ds = tf.data.Dataset.from_tensor_slices(flist)
        audio_ds = audio_ds.map(path_to_audio, num_parallel_calls=tf.data.AUTOTUNE)
        return audio_ds


    # def create_tf_dataset(data, bs=4):
    #     audio_ds = create_audio_ds(data)
    #     text_ds = create_text_ds(data)
    #     ds = tf.data.Dataset.zip((audio_ds, text_ds))
    #     ds = ds.map(lambda x, y: {"source": x, "target": y})
    #     ds = ds.batch(bs)
    #     ds = ds.prefetch(tf.data.AUTOTUNE)
    #     return ds

    """
    ## Callbacks to display predictions
    """


    class DisplayOutputs(keras.callbacks.Callback):
        def __init__(
                self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
        ):
            """Displays a batch of outputs after every epoch

            Args:
                batch: A test batch containing the keys "source" and "target"
                idx_to_token: A List containing the vocabulary tokens corresponding to their indices
                target_start_token_idx: A start token index in the target vocabulary
                target_end_token_idx: An end token index in the target vocabulary
            """
            self.batch = batch
            self.target_start_token_idx = target_start_token_idx
            self.target_end_token_idx = target_end_token_idx
            self.idx_to_char = idx_to_token

        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 != 0:
                return
            source = self.batch["source"]
            target = self.batch["target"]
            bs = tf.shape(source)[0]
            preds = self.model.generate(source, self.target_start_token_idx)
            preds = preds.numpy()
            for i in range(bs):
                target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
                prediction = ""
                for idx in preds[i, :]:
                    prediction += self.idx_to_char[idx]
                    if idx == self.target_end_token_idx:
                        break
                print(f"target:     {target_text.replace('-','')}")
                print(f"prediction: {prediction}\n")


    """
    ## Learning rate schedule
    """


    class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(
                self,
                init_lr=0.00001,
                lr_after_warmup=0.001,
                final_lr=0.00001,
                warmup_epochs=15,
                decay_epochs=85,
                steps_per_epoch=203,
        ):
            super().__init__()
            self.init_lr = init_lr
            self.lr_after_warmup = lr_after_warmup
            self.final_lr = final_lr
            self.warmup_epochs = warmup_epochs
            self.decay_epochs = decay_epochs
            self.steps_per_epoch = steps_per_epoch

        def calculate_lr(self, epoch):
            """linear warm up - linear decay"""
            warmup_lr = (
                    self.init_lr
                    + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * float(epoch)
            )
            decay_lr = tf.math.maximum(
                self.final_lr,
                self.lr_after_warmup
                - (float(epoch) - self.warmup_epochs)
                * (self.lr_after_warmup - self.final_lr)
                / (float(self.decay_epochs)),
                )
            return tf.math.minimum(warmup_lr, decay_lr)

        def __call__(self, step):
            epoch = step // self.steps_per_epoch
            return self.calculate_lr(epoch)

    """
    ## Create & train the end-to-end model
    """

    # for i in train_dataset:
    #     # path = i["file"].numpy().decode('utf-8')
    #     path_to_audio(i["audio"]["array"])

    batch = val_seq[0]

    # The vocabulary to convert predicted indices into characters
    idx_to_char = train_seq.vectorizer.get_vocabulary()
    display_cb = DisplayOutputs(
        batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
    )  # set the arguments as per vocabulary index for '<' and '>'

    model = Transformer(
        num_hid=200,
        num_head=2,
        num_feed_forward=400,
        target_maxlen=max_target_len,
        num_layers_enc=4,
        num_layers_dec=1,
        num_classes=34,
    )

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        label_smoothing=0.1,
    )

    learning_rate = CustomSchedule(
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=len(train_dataset),
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn)

    current_time = datetime.now()
    output_dir = os.path.join('saved_models', '{}'.format(current_time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(output_dir, exist_ok=True)
    model_output_path = os.path.join(output_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5')
    model_load_path = os.path.join('saved_models', '20230407_104034', 'model.18-0.20.h5')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_output_path, save_weights_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    model_callbacks = [
        display_cb,
        model_checkpoint_callback,
        tensorboard_callback
    ]

    first_input = [train_seq[0]["source"], train_seq[0]["target"]]

    model(first_input)
    model.load_weights(model_load_path)

    history = model.fit(x=train_seq, validation_data=val_seq, callbacks=model_callbacks, epochs=30, initial_epoch=18)

    pass

    """
    In practice, you should train for around 100 epochs or more.
    
    Some of the predicted text at or around epoch 35 may look as follows:
    ```
    target:     <as they sat in the car, frazier asked oswald where his lunch was>
    prediction: <as they sat in the car frazier his lunch ware mis lunch was>
    
    target:     <under the entry for may one, nineteen sixty,>
    prediction: <under the introus for may monee, nin the sixty,>
    ```
    """

if __name__ == "__main__":
    main()