import argparse
from siamese import target_shape,SiameseModel,DistanceLayer,embeddingNet,preprocess_image,preprocess_triplets
import os
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import resnet



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout-ratio', type=float,default=0.5, help='dropout ratio')
    parser.add_argument('--epochs', type=int, default=20, help='Maximum number of epochs')
    parser.add_argument('--mode', type=int, default=0, help="0: train from scratch, 1: load checkpoint")
    parser.add_argument('--checkpoint-path',default="checkpoint_path/network.ckpt",help="Checkpoint path")
    parser.add_argument('--dir-save-model',default='saved_model',help='directory will save model after training')
    parser.add_argument('--dirTrain', default=None, help="Data to train model, Ex: '/datasets/trains' ")
    parser.add_argument('--dirValid', default=None, help="Data to Valid in train step, Ex: '/datasets/validation'")
    args = parser.parse_args()

    path = 'checkpoint_path'
    checkpoint_path = 'checkpoint_path/vgg16.ckpt'
    if os.path.exists(path) == False:
        os.mkdir(path)    

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255., horizontal_flip=True, validation_split=0.2)
    train_dataset = datagen.flow_from_directory(args.dirTrain,target_size=target_shape,batch_size=32,subset="training")
    val_dataset = datagen.flow_from_directory(args.dirValid,target_size=target_shape,batch_size=32,subset="validation")

    anchor_input = layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = layers.Input(name="negative", shape=target_shape + (3,))

    base_cnn = resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
        )

    flatten = layers.Flatten()(base_cnn.output)
    dense1 = layers.Dense(512, activation="relu")(flatten)
    dense1 = layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(256, activation="relu")(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    output = layers.Dense(256)(dense2)

    embedding = Model(base_cnn.input, output, name="Embedding")

    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    distances = DistanceLayer()(
        embedding(resnet.preprocess_input(anchor_input)),
        embedding(resnet.preprocess_input(positive_input)),
        embedding(resnet.preprocess_input(negative_input)),
    )

    siamese_network = Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )
    
    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=optimizers.Adam(args.lr))

    callbacks_list= tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
    siamese_model.fit(train_dataset, epochs=args.epochs,batch_size=args.batch_size ,validation_data=val_dataset,callbacks=[callbacks_list])
    siamese_model.save(args.dir_save_model)


if __name__=="__main__":
    main()
