import os
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from models.TransformerASR import Transformer
from helpers.train_helpers import create_keras_seq, set_gpu_server_env_var, load_saved_model

"""
EfficientLEAF implementation example in tensorflow using Keras TransfomerASR example.

Tested on both windows and linux using tensorflow 2.10.x and 2.12.x.

"""

def main():
    """
    Set environmental variables specific for the code to run on.
    Uncomment if desired to run on specific GPU.
    set_gpu_server_env_var('6')
    """

    """
    Setup datasets in a keras sequence.  Can change dataset name to other huggingface ASR dataset.
    """

    batch_size = 40
    max_audio_len_s = 35
    max_target_len = 600
    sampling_rate = 16000

    cache_dir = os.path.join('/opt', 'localdata', 'Data', 'laryn', 'hugging_face', 'cache')

    train_seq = create_keras_seq("librispeech_asr", "train.clean.360",
                                 batch_size, sampling_rate, max_audio_len_s, max_target_len, cache_dir)
    test_seq = create_keras_seq("librispeech_asr", "test.clean",
                                batch_size, sampling_rate, max_audio_len_s, max_target_len, cache_dir)
    val_seq = create_keras_seq("librispeech_asr", "test.clean",
                                batch_size, sampling_rate, max_audio_len_s, max_target_len, cache_dir)

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
        steps_per_epoch=len(train_seq) // batch_size,
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Save models every epoch

    current_time = datetime.now()
    output_dir = os.path.join('saved_models', '{}'.format(current_time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(output_dir, exist_ok=True)
    model_output_path = os.path.join(output_dir, 'model.{epoch:02d}-{val_loss:.2f}.h5')



    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_output_path, save_weights_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    model_callbacks = [
        display_cb,
        model_checkpoint_callback,
        tensorboard_callback
    ]

    # Load saved model

    model_load_path = os.path.join('saved_models', 'latest_model.h5')
    load_saved_model(model, model_load_path, train_seq)

    history = model.fit(x=train_seq, validation_data=val_seq, callbacks=model_callbacks, epochs=100, initial_epoch=0)

    model.evaluate(x=test_seq)

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