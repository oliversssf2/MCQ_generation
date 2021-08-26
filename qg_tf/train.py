from datasets import load_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import transformers
import datasets
from transformers import AutoTokenizer, TFT5ForConditionalGeneration
import datetime
import os
from tf_t5_model import TFT5

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=1e4):
    super().__init__()

    self.warmup_steps = tf.cast(warmup_steps, tf.float32)
    
  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    m = tf.maximum(self.warmup_steps, step)
    m = tf.cast(m, tf.float32)
    lr = tf.math.rsqrt(m)
    
    return lr 

if __name__ == "__main__":


    tf_train_ds = tf.data.experimental.load("../demos_data/train_data_qg_hl_t5_tf")
    tf_valid_ds = tf.data.experimental.load("../demos_data/valid_data_qg_hl_t5_tf")
    
    tf_train_ds = tf_train_ds.shuffle(1000)
    tf_train_ds = tf_train_ds.batch(4)
    tf_train_ds = tf_train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    tf_valid_ds = tf_valid_ds.batch(4)
    tf_valid_ds = tf_valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

    #================================================#
    warmup_steps = 1e4
    batch_size = 4
    encoder_max_len = 250
    decoder_max_len = 54
    buffer_size = 1000
    ntrain = len(tf_train_ds)
    nvalid = len(tf_valid_ds)
    steps = int(np.ceil(ntrain/batch_size))
    valid_steps = int(np.ceil(nvalid/batch_size))
    print("Total Steps: ", steps)
    print("Total Validation Steps: ", valid_steps)
    data_dir = "../output_models"
    log_dir = f"{data_dir}/experiments/t5_qg_hl/logs"
    save_path = f"{data_dir}/experiments/t5_qg_hl/models"
    #================================================#

    start_profile_batch = steps+10
    stop_profile_batch = start_profile_batch + 100
    profile_range = f"{start_profile_batch},{stop_profile_batch}"

    log_path = log_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
                                                        update_freq=20,profile_batch=profile_range)

    checkpoint_filepath = save_path + "/" + "T5-{epoch:04d}-{val_loss:.4f}.ckpt"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    callbacks = [tensorboard_callback, model_checkpoint_callback] 
    metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy') ]

    learning_rate = CustomSchedule()
    # learning_rate = 0.001  # Instead set a static learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model = TFT5.from_pretrained("t5-small")
    model.compile(optimizer=optimizer, metrics=metrics)

    epochs_done = 0
    model.fit(tf_train_ds, epochs=1, steps_per_epoch=steps, callbacks=callbacks, 
          validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)
    model.save_pretrained(save_path)