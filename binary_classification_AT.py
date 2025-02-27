"""
    TensorFlow training code for
    "Data-Integrated Semi-Supervised Attention Enhances Performance 
    and Interpretability of Biological Classification Tasks"
    
    This file includes:
     * binary classification training code which reproduces
       sparse attention transfer

    2025 Jun Kim
"""
import h5py
import random
import numpy as np
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Binary Classification')
parser.add_argument('--k', default=0.01)

def convolutional_model(input_shape):
    # Convolutional model architectur
    input_img = tf.keras.Input(shape=input_shape, name='tf_input')
    Z1 = tf.keras.layers.Conv2D(8, 4, strides=(1, 1), padding='same', input_shape=input_shape)(input_img)
    A1 = tf.keras.layers.ReLU()(Z1)
    P1 = tf.keras.layers.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same')(A1)
    Z2 = tf.keras.layers.Conv2D(16, 2, strides=(1, 1), padding='same', input_shape=input_shape)(P1)
    A2 = tf.keras.layers.ReLU()(Z2)
    P2 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(A2)
    F = tf.keras.layers.Flatten()(P2)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

def tiny_convolutional_model(input_shape):
    # Define the convolutional model architectur
    input_img = tf.keras.Input(shape=input_shape, name='tf_input')
    Z1 = tf.keras.layers.Conv2D(4, 4, strides=(1, 1), padding='same', input_shape=input_shape)(input_img)
    A1 = tf.keras.layers.ReLU()(Z1)
    P1 = tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(2, 2), padding='same')(A1)
    F = tf.keras.layers.Flatten()(P1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model

#@tf.function
def attention_transfer_loss_topk_euclidean_distance_both(teacher_saliency_map, student_saliency_map, k):
    teacher_saliency_map = tf.cast(teacher_saliency_map, tf.float32)
    student_saliency_map = tf.cast(student_saliency_map, tf.float32)
    batch_size = tf.shape(teacher_saliency_map)[0]
    teacher_flat = tf.reshape(teacher_saliency_map, [batch_size, -1])
    student_flat = tf.reshape(student_saliency_map, [batch_size, -1])
    num_pixels = tf.shape(teacher_flat)[1]
    num_top_values = tf.cast(k * tf.cast(num_pixels, tf.float32), tf.int32)
    num_bottom_values = tf.cast(0.01 * tf.cast(num_pixels, tf.float32), tf.int32)
    # Use top_k for top values
    top_values, _ = tf.math.top_k(teacher_flat, k=num_top_values, sorted=True)
    # For bottom values, get the smallest k values by negating
    bottom_values, _ = tf.math.top_k(-teacher_flat, k=num_bottom_values, sorted=True)
    bottom_values = -bottom_values
    # Gather corresponding student values
    # Find indices
    sorted_indices = tf.argsort(teacher_flat, axis=1, direction='ASCENDING')
    top_indices = sorted_indices[:, -num_top_values:]
    bottom_indices = sorted_indices[:, :num_bottom_values]
    # Use batch gather
    batch_indices = tf.reshape(tf.range(batch_size), [-1, 1])
    batch_indices = tf.tile(batch_indices, [1, num_top_values])
    top_gather_indices = tf.stack([batch_indices, top_indices], axis=-1)
    student_top_values = tf.gather_nd(student_flat, top_gather_indices)
    batch_indices_bottom = tf.reshape(tf.range(batch_size), [-1, 1])
    batch_indices_bottom = tf.tile(batch_indices_bottom, [1, num_bottom_values])
    bottom_gather_indices = tf.stack([batch_indices_bottom, bottom_indices], axis=-1)
    student_bottom_values = tf.gather_nd(student_flat, bottom_gather_indices)
    # Calculate Euclidean distance for top and bottom values
    euclidean_top = tf.sqrt(tf.reduce_sum(tf.square(top_values - student_top_values), axis=1))
    euclidean_bottom = tf.sqrt(tf.reduce_sum(tf.square(bottom_values - student_bottom_values), axis=1))
    # Total loss
    total_euclidean_distance = tf.reduce_mean(euclidean_top + euclidean_bottom)
    return total_euclidean_distance

#@tf.function
def train_step(conv_model, batch_inputs, batch_labels, teacher_saliency, alpha, k, optimizer):
    with tf.GradientTape() as tape:
        # Forward pass for student
        student_outputs = conv_model(batch_inputs, training=True)
        student_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(batch_labels, student_outputs)
        )

        # Compute student saliency maps
        with tf.GradientTape() as tape_student:
            tape_student.watch(batch_inputs)
            student_outputs = conv_model(batch_inputs, training=True)
            student_loss_inner = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(batch_labels, student_outputs)
            )
        student_grads = tape_student.gradient(student_loss_inner, batch_inputs)
        
        att_loss = attention_transfer_loss_topk_euclidean_distance_both(teacher_saliency, student_grads, k)
        total_loss = student_loss + alpha*att_loss

    # Compute gradients and apply
    gradients = tape.gradient(total_loss, conv_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, conv_model.trainable_variables))
    return total_loss, student_loss, att_loss

class Saliency(object):
    def __init__(self, model, output_index=0):
        self.model = model  # Store the model
        self.output_index = output_index  # Store the output index
    def get_grad(self, input_image):
        input_tensor = tf.Variable(tf.cast(input_image, tf.float64))  # Ensure input tensor is float64
        with tf.GradientTape() as tape:
            output = self.model(input_tensor)  # Forward pass through the model
            loss = output[0, self.output_index]  # Select the output to compute gradients for
        gradients = tape.gradient(loss, input_tensor)  # Compute the gradients
        return gradients[0].numpy()  # Return the gradients as a NumPy array

def main():
    seeds = [40, 41, 42]
    k=parser.parse_args().k
    results =[]
    for seed in seeds:
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        
        # Load dataset from h5 files
        train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
        X_train_orig = train_dataset["train_set_x"][:]
        Y_train_orig = train_dataset["train_set_y"][:]
        
        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        X_test_orig = test_dataset["test_set_x"][:]
        Y_test_orig = test_dataset["test_set_y"][:]
        
        Y_test_orig = 1 - Y_test_orig
        Y_train_orig = 1 - Y_train_orig
        
        # Display sample images
        images_iter = iter(X_train_orig)
        labels_iter = iter(Y_train_orig)
        plt.figure(figsize=(15, 15))
        '''
        for i in range(20):
            plt.subplot(5, 10, i + 1)
            current_image = next(images_iter)
            current_label = next(labels_iter)
            plt.imshow(current_image)
            plt.title(current_label)
            plt.axis("off")
        plt.show()'''
        
        X_train = X_train_orig / 255.0
        X_test = X_test_orig / 255.0
        Y_train = Y_train_orig
        Y_test = Y_test_orig
        
        Y_train = Y_train.reshape(-1,1)
        Y_test = Y_test.reshape(-1,1)
        
        conv_model = convolutional_model((64, 64, 3))
        conv_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        #conv_model.summary()
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
        conv_model.fit(train_dataset, epochs=120, validation_data=test_dataset, verbose=0)
        
        # Precompute teacher saliency maps
        teacher_saliency_map = []
        for batch_inputs, batch_labels in train_dataset:
            with tf.GradientTape() as tape:
                tape.watch(batch_inputs)
                teacher_outputs = conv_model(batch_inputs, training=True)
                teacher_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(batch_labels, teacher_outputs))
            teacher_grads = tape.gradient(teacher_loss, batch_inputs)
            teacher_saliency_map.append(teacher_grads)
        '''
        single = Saliency(conv_model)
        grad_control = single.get_grad(tf.expand_dims(current_image/255.0, axis = 0))
        plt.imshow(current_image.reshape((64, 64, 3)), cmap = 'gray')
        plt.show()
        plt.imshow(np.sum(grad_control.reshape((64, 64, 3)), axis=2), cmap='inferno', interpolation='nearest')
        plt.colorbar() 
        plt.show()'''
        
        alphas = [0, 300]
        #k=0.003
        for alpha in alphas:
            # Define and compile the student model
            conv_model2 = tiny_convolutional_model((64, 64, 3))
            conv_model2.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            optimizer = tf.keras.optimizers.Adam()
            
            EPOCHS = 120
            best_validation_accuracy = 0
            for epoch in range(EPOCHS):
                batch_index = 0
                for batch_inputs, batch_labels in train_dataset:
                    # Fetch precomputed teacher saliency map
                    teacher_saliency = teacher_saliency_map[batch_index]
                    
                    # Perform a training step
                    total_loss, student_loss, att_loss = train_step(conv_model2, batch_inputs, batch_labels, teacher_saliency, alpha, k, optimizer)
                    
                    batch_index += 1
                # Evaluation on test dataset
                test_loss = 0.0
                test_accuracy = 0.0
                num_batches = 0
                for test_inputs, test_labels in test_dataset:
                    test_outputs = conv_model2(test_inputs, training=False)
                    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(test_labels, test_outputs))
                    accuracy = tf.reduce_mean(
                        tf.keras.metrics.binary_accuracy(test_labels, test_outputs)
                    )
                    test_loss += student_loss
                    test_accuracy += accuracy
                    num_batches += 1
                test_loss /= num_batches
                test_accuracy /= num_batches
                if test_accuracy > best_validation_accuracy:
                    best_validation_accuracy = test_accuracy
            print(f'Random Seed: {seed}, Alpha: {alpha}, Best Accuracy: {best_validation_accuracy}')
            results.append([seed, alpha, best_validation_accuracy])
    for seed, alpha, accuracy in results:
        print(f'Random Seed: {seed}, Alpha: {alpha}, Best Accuracy: {accuracy}')
if __name__ == '__main__':
    main()
