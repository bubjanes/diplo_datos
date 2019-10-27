# $ CUDA_VISIBLE_DEVICES=2 python deep_practico2.py --dataset_dir ../ --epochs 30 --dropout 0.1 0.1 --hidden_layer_sizes 200 100
# To know which GPU to use, you can check it with the command
# $ nvidia-smi

import argparse

import os
import mlflow
import numpy
import pandas
import tensorflow as tf
import seaborn

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import regularizers
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
TARGET_COL = 'AdoptionSpeed'

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

def process_features(df,one_hot_columns, numerical_columns, embedded_columns,test=False):
    direct_features = []

    # one hot encodings
    for one_hot_col, max_value in one_hot_columns.items():
        direct_features.append(tf.keras.utils.to_categorical(df[one_hot_col] - 1, max_value))

    # numeric
    scaler.fit(df[numerical_columns])
    numeric_col = scaler.transform(df[numerical_columns])
    #direct_features.append(tf.keras.utils.normalize(df[numerical_columns]))

    # Concatenate all features that don't need further embedding into a single matrix.
    features = {'direct_features': numpy.hstack(direct_features), 'numerical_columns': numeric_col}

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

def load_dataset(dataset_dir, batch_size):

    # Read train dataset (and maybe dev, if you need to...)
    dataset, dev_dataset = train_test_split(
        pandas.read_csv(os.path.join('petfinder_dataset/train.csv')), test_size=0.2)

    test_dataset = pandas.read_csv(os.path.join('petfinder_dataset/test.csv'))

    print('Training samples {}, test_samples {}'.format(
        dataset.shape[0], test_dataset.shape[0]))

    return dataset, dev_dataset, test_dataset

def main():
    args = read_args()
    dataset, dev_dataset, test_dataset = load_dataset(args.dataset_dir, args.batch_size)
    nlabels = dataset[TARGET_COL].unique().shape[0]

    # It's important to always use the same one-hot length
    one_hot_columns = {
        one_hot_col: dataset[one_hot_col].max()
        for one_hot_col in ['Gender','Color1','Breed2','Vaccinated','Health','MaturitySize','FurLength','Dewormed','Sterilized']
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
    test_ds = tf.data.Dataset.from_tensor_slices(process_features(test_dataset, one_hot_columns, numerical_columns, embedded_columns, test=True)[0]).batch(batch_size)
    
    #MODEL CREATION
    
    tf.keras.backend.clear_session()

    hidden_layer_size = 64
    hidden_layer_size2 = 20

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

    dense1 = layers.Dense(hidden_layer_size, activation='relu', kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l2(0.001))(features)
    dense2 = layers.Dense(hidden_layer_size2, activation='relu', kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(0.001),
                activity_regularizer=regularizers.l2(0.001))(dense1)
    output_layer = layers.Dense(nlabels, activation='softmax')(dense2)
    

    model = models.Model(inputs=inputs, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

  
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(nested=True):
        # Log model hiperparameters first
        mlflow.log_param('hidden_layer_size', hidden_layer_size)
        mlflow.log_param('embedded_columns', embedded_columns)
        mlflow.log_param('one_hot_columns', one_hot_columns)
        mlflow.log_param('numerical_columns', numerical_columns)  
        
        # Train
        epochs = 30
        history = model.fit(train_ds, epochs=epochs)

        # Evaluate
        loss, accuracy = model.evaluate(dev_ds)
        print("*** Test loss: {} - accuracy: {}".format(loss, accuracy))
        mlflow.log_metric('epochs', epochs)
        mlflow.log_metric('loss', loss)
        mlflow.log_metric('accuracy', accuracy)

    model.save("models/mode10.h5")
    pred = model.predict(test_ds)
    ypred = numpy.argmax(pred, axis=1)
    submission = pandas.DataFrame(data= list(zip(test_dataset.PID, ypred)), columns=["PID", "AdoptionSpeed"])
    submission.to_csv("./submission.csv", header=True, index=False)
    
print('All operations completed')

if __name__ == '__main__':
    main()

