#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Exercise 1

# Usage:

# $ CUDA_VISIBLE_DEVICES=2 python practico_1.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100

# To know which GPU to use, you can check it with the command

# $ nvidia-smi


# In[ ]:


import argparse

import os
import mlflow
import numpy
import pandas
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Activation


# In[ ]:


TARGET_COL = 'AdoptionSpeed'


# In[ ]:


def read_args():
    parser = argparse.ArgumentParser(
        description='Training a MLP on the petfinder dataset')
    # Here you have some examples of classifier parameters. You can add
    # more arguments or change these if you need to.
    parser.add_argument('--dataset_dir', default='../petfinder_dataset/', type=str,
                        help='Directory with the training and test files.')
    parser.add_argument('--hidden_layer_sizes', nargs='+', default=[100], type=int,
                        help='Number of hidden units of each hidden layer.')
    parser.add_argument('--epochs', default=10, type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dropout', nargs='+', default=[0.5], type=float,
                        help='Dropout ratio for every layer.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of instances in each batch.')
    parser.add_argument('--experiment_name', type=str, default='Base model',
                        help='Name of the experiment, used in mlflow.')
    args = parser.parse_args()

    assert len(args.hidden_layer_sizes) == len(args.dropout)
    return args


# In[ ]:


def process_features(df, one_hot_columns, numerical_columns, embedded_columns, test=False):
    direct_features = []

    # one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    # numeric
    direct_features.append(tf.keras.utils.normalize(df[numerical_columns]))
    # TODO Create and append numeric columns
    # Don't forget to normalize!
    # ....

    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': numpy.hstack(direct_features)}

    # Create embedding columns - nothing to do here. We will use the zero embedding for OOV
    for embedded_col in embedded_columns.keys():
        features[embedded_col] = df[embedded_col].values

    if not test:
        nlabels = df[TARGET_COL].unique().shape[0]
        # Convert labels to one-hot encodings
        targets = tf.keras.utils.to_categorical(df[TARGET_COL], nlabels)
    else:
        targets = None

    return features, targets


# In[ ]:


def load_dataset(dataset_dir, batch_size):

    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join('petfinder_dataset/train.csv')), test_size=0.2)

    test_dataset = pandas.read_csv(os.path.join('petfinder_dataset/test.csv'))

    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))

    return dataset, dev_dataset, test_dataset


# In[ ]:


def main():
    args = read_args()
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]

    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in ['Gender', 'Color1']
    }
    embedded_columns = {
        embedded_col: dataset[embedded_col].max() + 1
        for embedded_col in ['Breed1']
    }
    numerical_columns =  ['Age', 'Fee']

    # TODO (optional) put these three types of columns in the same dictionary with "column types"
    X_train, y_train = process_features(dataset, one_hot_columns, numerical_columns, embedded_columns)
    direct_features_input_shape = (X_train['direct_features'].shape[1],)
    X_dev, y_dev = process_features(dev_dataset, one_hot_columns, numerical_columns, embedded_columns)

    # Create the tensorflow Dataset
    batch_size = 32
    # TODO shuffle the train dataset!
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
    #x_batch, y_batch = next(iter(train_ds))
    dev_ds = tf.data.Dataset.from_tensor_slices((X_dev, y_dev)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(process_features(
        test_dataset, one_hot_columns, numerical_columns, embedded_columns, test=True)[0]).batch(batch_size)

#     model = Sequential([
#     Dense(64, input_shape=(784,), activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#     model.fit(train_ds, y_train, 
#           batch_size=batch_size, epochs=args.epochs, 
#           validation_data=(test_ds, y_test), verbose=1);
    tf.keras.backend.clear_session()

    hidden_layer_size = 64

    # Add one input and one embedding for each embedded column
    embedding_layers = []
    inputs = []
    for embedded_col, max_value in embedded_columns.items():
        input_layer = layers.Input(shape=(1,), name=embedded_col)
        inputs.append(input_layer)
        # Define the embedding layer
        embedding_size = int(max_value / 4)
        embedding_layers.append(
            tf.squeeze(layers.Embedding(input_dim=max_value, output_dim=embedding_size)(input_layer), axis=-2))
        print('Adding embedding of size {} for layer {}'.format(embedding_size, embedded_col))

    # Add the direct features already calculated
    direct_features_input = layers.Input(shape=direct_features_input_shape, name='direct_features')
    inputs.append(direct_features_input)

    # Concatenate everything together
    features = layers.concatenate(embedding_layers + [direct_features_input])

    dense1 = layers.Dense(hidden_layer_size, activation='relu')(features)
    output_layer = layers.Dense(nlabels, activation='softmax')(dense1)

    model = models.Model(inputs=inputs, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

    # TODO: Fit the model
#     mlflow.set_experiment(args.experiment_name)

#     with mlflow.start_run(nested=True):
#         # Log model hiperparameters first
#         mlflow.log_param('hidden_layer_size', args.hidden_layer_sizes)
#         mlflow.log_param('embedded_columns', embedded_columns)
#         mlflow.log_param('one_hot_columns', one_hot_columns)
#         mlflow.log_param('numerical_columns', numerical_columns)  # Not using these yet
#         #mlflow.log_param('epochs', args.epochs)
        
#         epochs = 30
        
#         history = model.fit(train_ds, epochs=epochs)
#         loss, accuracy = model.evaluate(test_ds)
#         print("*** Test loss: {} - accuracy: {}".format(loss, accuracy))
#         mlflow.log_metric('epochs', epochs)
#         mlflow.log_metric('loss', loss)
#         mlflow.log_metric('accuracy', accuracy)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_size', hidden_layer_size)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numerical_columns', numerical_columns)  # Not using these yet

        # Train
        epochs = 30
        history = model.fit(train_ds, epochs=epochs)

        # Evaluate
        loss, accuracy = model.evaluate(dev_ds)
        print("*** Test loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('epochs', epochs)
        mlflow.log_metric('loss', loss)
        mlflow.log_metric('accuracy', accuracy)

        #Train
        #history = model.fit(train_ds, epochs=args.epochs)
        # TODO: analyze history to see if model converges/overfits

        # TODO: Evaluate the model, calculating the metrics.
        # Option 1: Use the model.evaluate() method. For this, the model must be
        # already compiled with the metrics.
        # performance = model.evaluate(X_test, y_test)

#         loss, accuracy = 0, 0
#         # loss, accuracy = model.evaluate(dev_ds)
#         print("*** Dev loss: {} - accuracy: {}".format(loss, accuracy))
#         mlflow.log_metric('loss', loss)
#         mlflow.log_metric('accuracy', accuracy)

        # Option 2: Use the model.predict() method and calculate the metrics using
        # sklearn. We recommend this, because you can store the predictions if
        # you need more analysis later. Also, if you calculate the metrics on a
        # notebook, then you can compare multiple classifiers.

#         predictions = model.predict(test_ds)

#         # TODO: Convert predictions to classes
#         # TODO: Save the results for submission
#         # ...
#         print(predictions)

#     predictions = numpy.argmax(model.predict(test_ds), axis=1)
#     seaborn.countplot(predictions)


print('All operations completed')

if __name__ == '__main__':
    main()
