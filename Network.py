import tensorflow as tf
from keras import layers, models
from keras import backend as K


class NN():
    def __init__(self, marker_value):
        self.marker = marker_value

    def create_model(self, input_shape=(256, 384, 64), n_max=10):
        inputs = layers.Input(shape=input_shape)
        
        # Example convolutional layers
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        
        # Flatten and fully connected layers
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(n_max * 2, activation='linear')(x)
        
        # Reshape to N_max x 2
        outputs = layers.Reshape((n_max, 2))(x)
        
        model = models.Model(inputs, outputs)
        
        # Compile the model with a custom loss function
        model.compile(optimizer='adam', loss=lambda y_true, y_pred: self.custom_loss_function(y_true, y_pred, self.marker))
        
        return model

    def custom_loss_function(self, y_true, y_pred, marker_value):
        # Create a mask to ignore the special marker values
        mask = tf.reduce_all(tf.not_equal(y_true, marker_value), axis=-1)
        
        # Apply the mask to the true and predicted values
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        
        # Calculate mean squared error for the valid points
        mse_loss = K.mean(K.square(y_true_masked - y_pred_masked))
        
        return mse_loss
