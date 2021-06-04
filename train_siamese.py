import argparse
import os
from datetime import datetime

import tensorflow as tf
import tensorflow_addons as tfa
from absl import app

from data.data_generator import DataGenerator
# from model.siamese.config import cfg
from model.siamese.model_generator import create_model, base_models

TRAINABLE = True

base_model = list(base_models.keys())[2]  # MobileNetV2


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--weights-path',default="model/siamese/weight",help="weight path")
    parser.add_argument('--dirTrain', default=None, help="Data to train model, Ex: '/datasets/trains' ")
    parser.add_argument('--dirValid', default=None, help="Data to Valid in train step, Ex: '/datasets/validation'")
    args = parser.parse_args()

    model = create_model(trainable=TRAINABLE, base_model=base_model)
    prefix = "block3c_add"
    try:
        tf.keras.utils.plot_model(
                model,
                to_file=f"assets/{base_model}_model_fig.png",
                show_shapes=True,
                expand_nested=True,
                )
    except ImportError as e:
        print(f"Failed to plot keras model: {e}")

    ds_generator = DataGenerator(
            file_ext=["png", "jpg"],
            folder_path=args.dirTrain,
            exclude_aug=True,
            step_size=4,
            )

    # train_ds = ds_generator.get_dataset()

    learning_rate = args.lr

    # optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_fun = tfa.losses.TripletSemiHardLoss()
    model.compile(loss=loss_fun, optimizer=optimizer, metrics=[])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            args.weights_path + "/" + base_model + "/siam-{epoch}-"+str(learning_rate)+"-"+str(prefix)+"_{loss:.4f}.h5",
            monitor="loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            )
    # stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=cfg.TRAIN.PATIENCE, mode="min")
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.6, patience=5, min_lr=1e-6, verbose=1,
    #                                                  mode="min")

    # Define the Keras TensorBoard callback.
    logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


    model.fit(
            ds_generator,
            epochs=args.epochs,
            callbacks=[tensorboard_callback, checkpoint],
            verbose=1
            )
    # model.save('./siamese.h5')


if __name__ == "__main__":
   main()

