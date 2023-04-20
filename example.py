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

set_gpu_server_env_var()

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

            # if model is None:
            #     preds = self.model.generate(source, self.target_start_token_idx)
            # else:
            #     preds = model.generate(source, self.target_start_token_idx)
            preds = model.generate(source, self.target_start_token_idx)
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

    history = model.fit(x=train_seq, validation_data=val_seq, callbacks=model_callbacks, epochs=100, initial_epoch=15)

    model.evaluate(x=test_seq)

    """
    In practice, you should train for around 100 epochs or more.
    
    Some of the predicted text at or around epoch 35 may look as follows:
    ```
    target:     <forgive me i hardly know what i am saying a thousand times forgive me madame was right quite right this brutal exile has completely turned my brain>
    prediction: <for get me i hardly no on what i am saying the thousand time me madame was right quite right as brittle excile as complictly turned my brain>
    
    target:     <there cannot be a doubt he received you kindly for in fact you returned without his permission>
    prediction: <their cannot be a doubt he received you kindly for in fact you returned without his permission>
    
    target:     <oh mademoiselle why have i not a devoted sister or a true friend such as yourself>
    prediction: <oh met must elie have i not a divoted sister or a true or a true friend the such as yourself>
    
    target:     <what already here they said to her>
    prediction: <what already here may said to her>
    
    target:     <i have been here this quarter of an hour replied la valliere>
    prediction: <i have been here this corder of an hour replied loviet lovieta loviety>
    
    target:     <did not the dancing amuse you no>
    prediction: <did not the dancing amuse you no>
    
    target:     <no more than the dancing>
    prediction: <no more then the dancing>
    
    target:     <la valliere is quite a poetess said tonnay charente>
    prediction: <the value is quite a poetice a tumish shonal up>
    
    target:     <i am a woman and there are few like me whoever loves me flatters me whoever flatters me pleases me and whoever pleases well said montalais you do not finish>
    prediction: <i am a woman and are few like me who ever loves me flatters me who have his me pleases me and who ever pleases well said not to lay you do not lay you do not finish>
    
    target:     <it is too difficult replied mademoiselle de tonnay charente laughing loudly>
    prediction: <it is ta difficult replied mudde muddenish all detenish all hall thing ladly>
    
    target:     <look yonder do you not see the moon slowly rising silvering the topmost branches of the chestnuts and the oaks>
    prediction: <look younder do you not saw rising slowly rising so hearing the top most branches of the chest and the oaks>
    
    target:     <exquisite soft turf of the woods the happiness which your friendship confers upon me>
    prediction: <exquisit soft turfable the woods the happiness which our friendship compars upon me>
    
    target:     <well said mademoiselle de tonnay charente i also think a good deal but i take care>
    prediction: <well said my must my should i all so sand i also think a good dial but i take care>
    
    target:     <to say nothing said montalais so that when mademoiselle de tonnay charente thinks athenais is the only one who knows it>
    prediction: <to say nothing said montally so there when mud mud was all deternatior obtanks at the nay is the only one who nose it>
    
    target:     <quick quick then among the high reed grass said montalais stoop athenais you are so tall>
    prediction: <quick then among the high read grassed montal a stoop at the nay you are so tall>
    
    target:     <the young girls had indeed made themselves small indeed invisible>
    prediction: <the hungerls had indeed made themselve small indeed and visible>
    
    target:     <she was here just now said the count>
    prediction: <she was her just now said the count>
    
    target:     <you are positive then>
    prediction: <you are positive then>
    
    target:     <yes but perhaps i frightened her in what way>
    prediction: <yes the perhaps are frightened what way>
    
    target:     <how is it la valliere said mademoiselle de tonnay charente that the vicomte de bragelonne spoke of you as louise>
    prediction: <how is it lowelly a said muddemose of the that the recompt a becompt a bright alone spoke a you as leas>
    
    target:     <it seems the king will not consent to it>
    prediction: <it seems the kingle not consemed to it>
    
    target:     <good gracious has the king any right to interfere in matters of that kind>
    prediction: <good gracians as the can any right to inner matters of that time>
    
    target:     <i give my consent>
    prediction: <i give my conset>
    
    target:     <oh i am speaking seriously replied montalais and my opinion in this case is quite as good as the king's i suppose is it not louise>
    prediction: <all i am speaking seriously replied montally and my opinion and this case his quite of the kings i suppose is a not louise is a not louise>
    
    target:     <let us run then said all three and gracefully lifting up the long skirts of their silk dresses they lightly ran across the open space between the lake and the thickest covert of the park>
    prediction: <what as run then sat all three and gracefully lifting up the long spirts of their so gresses they lightly ran crosseile conspace between the lake and the lake and the park>
    
    target:     <in fact the sound of madame's and the queen's carriages could be heard in the distance upon the hard dry ground of the roads followed by the mounted cavaliers>
    prediction: <and fact the sound of the dams and a queens carriages could be heard in the distance upon the hard dright round of the roads followed by the mountain cavaliers>
    
    target:     <in this way the fete of the whole court was a fete also for the mysterious inhabitants of the forest for certainly the deer in the brake the pheasant on the branch the fox in its hole were all listening>
    prediction: <in this way the fed of the whole core was a fet also for the misterious and habitants of the forest persertainly the dear in the break the fas intone the fox in its hole were all this and it holl listening>
    
    target:     <at the conclusion of the banquet which was served at five o'clock the king entered his cabinet where his tailors were awaiting him for the purpose of trying on the celebrated costume representing spring which was the result of so much imagination and had cost so many efforts of thought to the designers and ornament workers of the court>
    prediction: <at the conclusion of the bankly which was served it flive a court the cart the came in or his cavin or his tailors or awaiting him for the purpose of trying on the selebrated castoom representing spring which was the result of somewhat imagination and had cause some many of her suffers of thought to thes ts ous fin ofin th th the t cathathay omateschereley andgrd ocheraffocttoromugrmasle imaratinopatobinowledgatovide>
    
    target:     <ah very well>
    prediction: <how very will>
    
    target:     <let him come in then said the king and as if colbert had been listening at the door for the purpose of keeping himself au courant with the conversation he entered as soon as the king had pronounced his name to the two courtiers>
    prediction: <lien come and them said the king and as if cold bare had the most any the door for the purpose of keeping himself alcorance with the compersation he entered a soon as the king had pronounced his main to the shout corty years>
    
    target:     <gentlemen to your posts whereupon saint aignan and villeroy took their leave>
    prediction: <judgment dear posts were upon sand and and beautor thirly to thirly>
    
    target:     <certainly sire but i must have money to do that what>
    prediction: <certainly sire but i must have money to do that>
    
    target:     <what do you mean inquired louis>
    prediction: <what do mean in quartleries>
    
    target:     <he has given them with too much grace not to have others still to give if they are required which is the case at the present moment>
    prediction: <he has given them with two much grace not to have utterstolt to give if they are required which as the case of the present moment>
    
    target:     <it is necessary therefore that he should comply the king frowned>
    prediction: <it is messary their for that he should comply the king frowned>
    
    target:     <does your majesty then no longer believe the disloyal attempt>
    prediction: <those your magisty then no longer believe the disoilettent>
    
    target:     <not at all you are on the contrary most agreeable to me>
    prediction: <not it all you are on the conferry most agreeable to me>
    
    target:     <your majesty's plan then in this affair is>
    prediction: <your majesty plan then and this affair is>
    
    target:     <you will take them from my private treasure>
    prediction: <you will take them from my private treasure>
    
    target:     <the news circulated with the rapidity of lightning during its progress it kindled every variety of coquetry desire and wild ambition>
    prediction: <the new his circulated with the repetty elightening during of progress it candled every varied of coachery desire and while and while and bision>
    ```
    """

if __name__ == "__main__":
    main()