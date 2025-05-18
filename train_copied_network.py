from copied_siamese_network import ChessPieceClassifier


def main():
    dataset_path = 'dataset/chess_piece_similarity'
    use_augmentation = True
    learning_rate = 10e-4
    batch_size = 32
    
    # Learning Rate multipliers for each layer
    learning_rate_multipliers = {}
    learning_rate_multipliers['Conv1'] = 1
    learning_rate_multipliers['Conv2'] = 1
    learning_rate_multipliers['Conv3'] = 1
    learning_rate_multipliers['Conv4'] = 1
    learning_rate_multipliers['Dense1'] = 1
    learning_rate_multipliers['Dense2'] = 1
    
    # l2-regularization penalization for each layer
    l2_penalization = {}
    l2_penalization['Conv1'] = 1e-2
    l2_penalization['Conv2'] = 1e-2
    l2_penalization['Conv3'] = 1e-2
    l2_penalization['Conv4'] = 1e-2
    l2_penalization['Dense1'] = 1e-4
    l2_penalization['Dense2'] = 1e-4
    
    # Path where the logs will be saved
    tensorboard_log_path = './logs/chess_classifier'
    
    # Create the classifier
    chess_classifier = ChessPieceClassifier(
        dataset_path=dataset_path,
        learning_rate=learning_rate,
        batch_size=batch_size, 
        use_augmentation=use_augmentation,
        learning_rate_multipliers=learning_rate_multipliers,
        l2_regularization_penalization=l2_penalization,
        tensorboard_log_path=tensorboard_log_path
    )
    
    # Training parameters
    momentum = 0.9
    momentum_slope = 0.01
    evaluate_each = 100
    number_of_train_iterations = 20000

    # Train the model
    best_validation_accuracy = chess_classifier.train_classification_network(
        number_of_iterations=number_of_train_iterations,
        final_momentum=momentum,
        momentum_slope=momentum_slope,
        evaluate_each=evaluate_each, 
        model_name='chess_piece_classifier'
    )
    
    print('Final Best Validation Accuracy = ' + str(best_validation_accuracy))


if __name__ == "__main__":
    main()