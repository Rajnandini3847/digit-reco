import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import matplotlib.pyplot as plt

def create_model():
    """Create the CNN model with improved architecture"""
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

def plot_training_history(history):
    """Plot training & validation accuracy and loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('model/plots'):
        os.makedirs('model/plots')
    plt.savefig('model/plots/training_history.png')
    plt.close()

def main():
    # Create model directory if it doesn't exist
    if not os.path.exists('model'):
        os.makedirs('model')

    # Load and preprocess the MNIST dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize and reshape the data
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

    # Create data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    # Create and compile the model
    print("Creating and compiling model...")
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary
    model.summary()

    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'model/digit_model_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train the model
    print("Training model...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=20,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save the final model
    print("Saving model...")
    model.save('model/digit_model.keras')
    print("Training completed successfully!")

    # Test prediction on a few samples
    print("\nTesting predictions on sample images...")
    test_samples = x_test[:5]
    predictions = model.predict(test_samples)
    for i, prediction in enumerate(predictions):
        predicted_digit = np.argmax(prediction)
        actual_digit = y_test[i]
        confidence = np.max(prediction)
        print(f"Sample {i+1}: Predicted={predicted_digit}, Actual={actual_digit}, Confidence={confidence:.4f}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")