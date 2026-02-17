!pip uninstall -y flwr
!pip install -U "flwr[simulation]"

!pip install -U "flwr[simulation]"

import flwr as fl
import tensorflow as tf
import numpy as np

# Load MNIST dataset
# This dataset has images of handwritten numbers (0 to 9)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Normalize the images
# Divide by 255 to make pixel values between 0 and 1
# This helps the model learn better
x_train, x_test = x_train / 255.0, x_test / 255.0


# Reshape the images for CNN
# CNN needs 4D input: (samples, height, width, channels)
# Since images are black & white, channels = 1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

def create_model():
    # Create a simple CNN model
    model = tf.keras.Sequential([
        
        # Convolution layer
        # Looks for patterns in the image (like edges)
        # 32 filters of size 3x3
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        
        # Max pooling layer
        # Reduces image size to make training faster
        tf.keras.layers.MaxPooling2D(2,2),
        
        # Flatten layer
        # Converts 2D data into 1D so it can go into Dense layer
        tf.keras.layers.Flatten(),
        
        # Dense layer with 128 neurons
        # Learns important features
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Output layer
        # 10 neurons for digits 0–9
        # Softmax gives probability for each digit
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    # Adam → optimizer (helps model learn)
    # sparse_categorical_crossentropy → used for digit labels (0–9)
    # accuracy → shows how correct the model is
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Number of clients in federated learning
NUM_CLIENTS = 3

# Divide training data equally among clients
client_data_size = len(x_train) // NUM_CLIENTS


def get_client_data(client_id):
    # Find starting index for this client
    start = client_id * client_data_size
    
    # Find ending index
    end = start + client_data_size
    
    # Return that part of the dataset
    return x_train[start:end], y_train[start:end]

# Define a Flower client
# Each client trains the model on its own data
class FlowerClient(fl.client.NumPyClient):
    
    def __init__(self, model, x_train, y_train):
        # Store model and local dataset
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def get_parameters(self, config):
        # Send model weights to the server
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Receive global weights from server
        self.model.set_weights(parameters)
        
        # Train the model on local data
        self.model.fit(self.x_train, self.y_train, 
                       epochs=1, batch_size=32, verbose=0)
        
        # Send updated weights back to server
        # Also send number of training samples
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # Set global weights before evaluation
        self.model.set_weights(parameters)
        
        # Test model on test dataset
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        # Send loss and accuracy to server
        return loss, len(x_test), {"accuracy": accuracy}

# This function creates a new client
# It is called by the Flower server
def client_fn(cid):
    
    # Create a new model for the client
    model = create_model()
    
    # Get this client's part of the training data
    x_c, y_c = get_client_data(int(cid))
    
    # Return a FlowerClient object
    # Each client has its own model and local data
    return FlowerClient(model, x_c, y_c)

import numpy as np

# Function to calculate average accuracy from all clients
def weighted_average(metrics):
    accuracies = []
    
    # Loop through each client's metrics
    for _, m in metrics:
        accuracies.append(float(m["accuracy"]))  # Get accuracy
    
    # Return average accuracy
    return {"accuracy": np.mean(accuracies)}


# Define Federated Averaging strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,  # Use our average function
)


# Start federated learning simulation
history = fl.simulation.start_simulation(
    client_fn=client_fn,          # Function that creates clients
    num_clients=NUM_CLIENTS,      # Total number of clients
    config=fl.server.ServerConfig(num_rounds=3),  # Train for 3 rounds
    strategy=strategy,            # Use FedAvg strategy
)


# -------- CLEAN PRINT --------
# Print accuracy after each round
print("\nClean Accuracy Output:")
for round_num, acc in history.metrics_distributed["accuracy"]:
    print(f"Round {round_num}: {float(acc)*100:.2f}%")

# FedProx Client (Improved version of FedAvg)
class FedProxClient(fl.client.NumPyClient):

    def __init__(self, model, x_train, y_train, mu=0.01):
        # Store model and local data
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        
        # mu controls how strongly we keep model close to global weights
        self.mu = mu

    def get_parameters(self, config):
        # Send model weights to server
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set global weights received from server
        self.model.set_weights(parameters)

        # Convert global weights to tensors
        global_weights = [tf.convert_to_tensor(w) for w in parameters]

        optimizer = tf.keras.optimizers.Adam()
        batch_size = 32

        # Create batches of local data
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        ).batch(batch_size)

        # Training loop (1 epoch)
        for epoch in range(1):
            for x_batch, y_batch in dataset:
                with tf.GradientTape() as tape:

                    # Forward pass
                    predictions = self.model(x_batch, training=True)

                    # Normal classification loss
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        y_batch, predictions
                    )
                    loss = tf.reduce_mean(loss)

                    # Proximal term (extra penalty in FedProx)
                    # Keeps local model close to global model
                    prox_term = 0.0
                    for w, w_global in zip(self.model.trainable_variables, global_weights):
                        prox_term += tf.reduce_sum(tf.square(w - w_global))

                    # Add proximal penalty to loss
                    loss += (self.mu / 2) * prox_term

                # Compute gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)

                # Update model weights
                optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

        # Send updated weights back to server
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # Set global weights before testing
        self.model.set_weights(parameters)

        # Evaluate on test data
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)

        # Send results to server
        return loss, len(x_test), {"accuracy": accuracy}

# This function creates a FedProx client
def client_fn_prox(cid):
    
    # Create a new model for this client
    model = create_model()
    
    # Get this client's local data
    x_c, y_c = get_client_data(int(cid))
    
    # Return FedProxClient with mu value
    # mu controls how strongly the model stays close to global weights
    return FedProxClient(model, x_c, y_c, mu=0.01)

# Define strategy (still using FedAvg aggregation on server)
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,  # Average accuracy from clients
)

# Start FedProx simulation
history_prox = fl.simulation.start_simulation(
    client_fn=client_fn_prox,          # Use FedProx clients
    num_clients=NUM_CLIENTS,           # Total number of clients
    config=fl.server.ServerConfig(num_rounds=3),  # Train for 3 rounds
    strategy=strategy,                 # Server uses FedAvg aggregation
)

# Print accuracy for each round
print("\nFedProx Accuracy:")
for round_num, acc in history_prox.metrics_distributed["accuracy"]:
    print(f"Round {round_num}: {float(acc)*100:.2f}%")

# FedPer Client (Personalized Federated Learning)
class FedPerClient(fl.client.NumPyClient):

    def __init__(self, model, x_train, y_train):
        # Store model and local data
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

        # Freeze the last layer (classifier layer)
        # This layer becomes personal for each client
        # It will NOT be updated during global training
        self.model.layers[-1].trainable = False

    def get_parameters(self, config):
        # Send model weights to server
        return self.model.get_weights()

    def fit(self, parameters, config):
        # Set global weights received from server
        self.model.set_weights(parameters)

        # Train model on local data (except frozen layer)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )

        # Send updated weights back to server
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        # Set global weights before testing
        self.model.set_weights(parameters)

        # Evaluate on test dataset
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=0)

        # Send results to server
        return loss, len(x_test), {"accuracy": accuracy}

# This function creates a FedPer client
def client_fn_per(cid):
    
    # Create FedPer model (model designed for personalization)
    model = create_fedper_model()
    
    # Get this client's local data
    x_c, y_c = get_client_data(int(cid))
    
    # Return FedPer client with model and local data
    return FedPerClient(model, x_c, y_c)

# Define server strategy (using FedAvg aggregation)
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,  # Average accuracy from clients
)

# Start FedPer simulation
history_per = fl.simulation.start_simulation(
    client_fn=client_fn_per,          # Use FedPer clients
    num_clients=NUM_CLIENTS,          # Total number of clients
    config=fl.server.ServerConfig(num_rounds=3),  # Train for 3 rounds
    strategy=strategy,                # Server aggregates using FedAvg
)

# Print accuracy for each round
print("\nFedPer Accuracy:")
for round_num, acc in history_per.metrics_distributed["accuracy"]:
    print(f"Round {round_num}: {float(acc)*100:.2f}%")

# Define server strategy (using FedAvg aggregation)
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,  # Average accuracy from clients
)

# Start FedPer simulation
history_per = fl.simulation.start_simulation(
    client_fn=client_fn_per,          # Use FedPer clients
    num_clients=NUM_CLIENTS,          # Total number of clients
    config=fl.server.ServerConfig(num_rounds=3),  # Train for 3 rounds
    strategy=strategy,                # Server aggregates using FedAvg
)

# Print accuracy for each round
print("\nFedPer Accuracy:")
for round_num, acc in history_per.metrics_distributed["accuracy"]:
    print(f"Round {round_num}: {float(acc)*100:.2f}%")

