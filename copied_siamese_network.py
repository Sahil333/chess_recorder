import os

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input, Dropout
from keras.optimizers import SGD
from keras.regularizers import l2
import tensorflow as tf
import keras.layers as nn

import numpy as np

from chess_similarity_loader import ChessLoader


class ChessPieceClassifier:
    """Class that constructs a Chess Piece Classification Network

    This Class was constructed to create a classifier for chess piece types.

    Attributes:
        input_shape: image size
        model: current classification model
        learning_rate: SGD learning rate
        chess_loader: instance of ChessLoader
        summary_writer: tensorflow writer to store the logs
    """

    def __init__(self, dataset_path, learning_rate, batch_size, use_augmentation,
                 learning_rate_multipliers, l2_regularization_penalization, tensorboard_log_path):
        """Inits ChessPieceClassifier with the provided values for the attributes.

        It also constructs the classification network architecture, creates a dataset 
        loader and opens the log file.

        Arguments:
            dataset_path: path of Chess piece dataset    
            learning_rate: SGD learning rate
            batch_size: size of the batch to be used in training
            use_augmentation: boolean that allows us to select if data augmentation 
                is used or not
            learning_rate_multipliers: learning-rate multipliers (relative to the learning_rate
                chosen) that will be applied to each fo the conv and dense layers
            l2_regularization_penalization: l2 penalization for each layer.
            tensorboard_log_path: path to store the logs                
        """
        self.input_shape = (105, 105, 1)  # Size of images (RGB)
        self.model = []
        self.learning_rate = learning_rate
        self.chess_loader = ChessLoader(dataset_path=dataset_path, use_augmentation=use_augmentation, batch_size=batch_size)
        self.summary_writer = tf.summary.create_file_writer(tensorboard_log_path)
        self._construct_classification_architecture(learning_rate_multipliers,
                                              l2_regularization_penalization)

    def _construct_classification_architecture(self, learning_rate_multipliers,
                                         l2_regularization_penalization):
        """ Constructs the classification architecture and stores it in the class

        Arguments:
            learning_rate_multipliers
            l2_regularization_penalization
        """

        # Define the cnn architecture - simplified version
        model = Sequential()
        # First conv block - smaller kernel with padding
        model.add(Conv2D(filters=32, kernel_size=(3, 3),
                         activation='relu',
                         padding='same',
                         input_shape=self.input_shape,
                         kernel_regularizer=l2(
                             l2_regularization_penalization.get('Conv1', 1e-4)),
                         name='Conv1'))
        model.add(MaxPool2D())

        # Second conv block
        model.add(Conv2D(filters=64, kernel_size=(3, 3),
                         activation='relu',
                         padding='same',
                         kernel_regularizer=l2(
                             l2_regularization_penalization.get('Conv2', 1e-4)),
                         name='Conv2'))
        model.add(MaxPool2D())
        
        # Flatten and dense layers
        model.add(Flatten())
        model.add(
            Dense(units=128, activation='relu',
                  kernel_regularizer=l2(
                      l2_regularization_penalization.get('Dense1', 1e-4)),
                  name='Dense1'))
        
        # Add dropout for regularization
        model.add(Dropout(0.25))
        
        # Add final classification layer with 4 outputs (for the chess piece types)
        model.add(
            Dense(units=4, activation='softmax',
                  kernel_regularizer=l2(
                      l2_regularization_penalization.get('Dense2', 1e-4)),
                  name='Dense2'))

        self.model = model

        # Define the optimizer and compile the model
        optimizer = SGD(
            learning_rate=self.learning_rate,
            momentum=0.5)

        self.model.compile(loss='categorical_crossentropy', 
                           metrics=['accuracy'],
                           optimizer=optimizer)

    def _write_logs_to_tensorboard(self, current_iteration, train_losses,
                                    train_accuracies, validation_accuracy,
                                    evaluate_each):
        """ Writes the logs to a tensorflow log file

        This allows us to see the loss curves and the metrics in tensorboard.
        If we wrote every iteration, the training process would be slow, so 
        instead we write the logs every evaluate_each iteration.

        Arguments:
            current_iteration: iteration to be written in the log file
            train_losses: contains the train losses from the last evaluate_each
                iterations.
            train_accuracies: the same as train_losses but with the accuracies
                in the training set.
            validation_accuracy: accuracy in the validation set
            evaluate each: number of iterations defined to evaluate the model
        """
        with self.summary_writer.as_default():
            for index in range(0, evaluate_each):
                tf.summary.scalar('Train Loss', train_losses[index], 
                                 step=current_iteration - evaluate_each + index + 1)
                tf.summary.scalar('Train Accuracy', train_accuracies[index], 
                                 step=current_iteration - evaluate_each + index + 1)
                
                if index == (evaluate_each - 1):
                    tf.summary.scalar('Validation Accuracy', validation_accuracy, 
                                     step=current_iteration)
            
            self.summary_writer.flush()

    def train_classification_network(self, number_of_iterations,
                              final_momentum, momentum_slope, evaluate_each,
                              model_name):
        """ Train the Classification network

        This is the main function for training the classification network. 
        Every evaluate_each train iterations we evaluate on the validation set.

        Arguments:
            number_of_iterations: maximum number of iterations to train.
            final_momentum: momentum value to reach. Each layer starts at 0.5 momentum
                but evolves linearly to final_momentum
            momentum_slope: slope of the momentum evolution.
            evaluate each: number of iterations defined to evaluate on validation set.
            model_name: save_name of the model

        Returns: 
            Best validation accuracy
        """

        # Variables that will store iterations losses and accuracies
        train_losses = np.zeros(shape=(evaluate_each))
        train_accuracies = np.zeros(shape=(evaluate_each))
        count = 0
        
        # Stop criteria variables
        best_validation_accuracy = 0.0
        best_accuracy_iteration = 0
        validation_accuracy = 0.0

        # Train loop
        for iteration in range(number_of_iterations):

            # Get batch from train set
            images, labels = self.chess_loader.get_train_batch()
            train_loss, train_accuracy = self.model.train_on_batch(images, labels)

            # Decay learning rate 1% per 500 iterations and update momentum linearly
            if (iteration + 1) % 500 == 0:
                K.set_value(self.model.optimizer.learning_rate, K.get_value(
                        self.model.optimizer.learning_rate) * 0.99)
            if K.get_value(self.model.optimizer.momentum) < final_momentum:
                K.set_value(self.model.optimizer.momentum, K.get_value(
                    self.model.optimizer.momentum) + momentum_slope)

            train_losses[count] = train_loss
            train_accuracies[count] = train_accuracy

            count += 1
            print('Iteration %d/%d: Train loss: %f, Train Accuracy: %f, lr = %f' %
                  (iteration + 1, number_of_iterations, train_loss, train_accuracy, K.get_value(
                      self.model.optimizer.learning_rate)))

            # Each evaluate_each iterations evaluate on validation set
            if (iteration + 1) % evaluate_each == 0:
                # Evaluate on validation set
                validation_accuracy = self.chess_loader.evaluate(self.model)

                self._write_logs_to_tensorboard(
                    iteration + 1, train_losses, train_accuracies,
                    validation_accuracy, evaluate_each)
                count = 0

                # Save the model if it improves
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_accuracy_iteration = iteration
                    
                    model_json = self.model.to_json()

                    if not os.path.exists('./models'):
                        os.makedirs('./models')
                    with open('models/' + model_name + '.json', "w") as json_file:
                        json_file.write(model_json)
                    self.model.save_weights('models/' + model_name + '.h5')
                    print(f'New best validation accuracy: {best_validation_accuracy}')

            # Early stopping if accuracy doesn't improve for 10000 iterations
            if iteration - best_accuracy_iteration > 10000:
                print('Early Stopping: validation accuracy did not increase for 10000 iterations')
                print('Best Validation Accuracy = ' + str(best_validation_accuracy))
                break

        print('Training Ended!')
        return best_validation_accuracy