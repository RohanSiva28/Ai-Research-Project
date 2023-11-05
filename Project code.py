import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10

# Environment Setup
class CustomEnvironment:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
        self.num_classes = 10
        self.state_dim = self.x_train.shape[1:]
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.x_train[self.current_step]

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.x_train)
        next_state = self.x_train[self.current_step]
        reward = calculate_reward(action)
        return next_state, reward, done, None

env = CustomEnvironment()

# Search Space for Architectures
search_space = [
    # Architecture 1: Convolutional Neural Network
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=env.state_dim),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(env.num_classes, activation='softmax')
    ],

    # Architecture 2: A simpler architecture
    [
        tf.keras.layers.Flatten(input_shape=env.state_dim),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(env.num_classes, activation='softmax')
    ]
]

# Reward Function
def calculate_reward(action):
    # Ensure the action is within a valid range
    action = max(0, min(action, len(search_space) - 1))

    architecture = search_space[action]

    model = tf.keras.models.Sequential(architecture)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(env.x_train, env.y_train, epochs=5, verbose=0)
    _, accuracy = model.evaluate(env.x_test, env.y_test, verbose=0)

    # Reward based on accuracy (higher accuracy is rewarded)
    reward = accuracy

    return reward

# Hyperparameters for RL
learning_rate = 0.001
num_episodes = 100

# Policy Network
def build_policy_network():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=env.state_dim),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(search_space), activation='softmax')
    ])
    return model

# RL Algorithm (Proximal Policy Optimization)
def train_rl_policy():
    policy_network = build_policy_network()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action_probs = policy_network.predict(np.expand_dims(state, axis=0))[0]
            selected_architecture = np.random.choice(range(len(search_space)), p=action_probs)

            next_state, reward, done, _ = env.step(selected_architecture)

            with tf.GradientTape() as tape:
                action_mask = tf.one_hot(selected_architecture, len(search_space))
                selected_action_prob = tf.reduce_sum(action_probs * action_mask)
                loss = -tf.math.log(selected_action_prob) * reward
            gradients = tape.gradient(loss, policy_network.trainable_variables)

            state = next_state

    return policy_network

# Main
if __name__ == "__main__":
    policy_network = train_rl_policy()

    # Evaluate the final selected architecture
    final_architecture_idx = np.argmax(policy_network.predict(np.expand_dims(env.x_test[0], axis=0)))
    model = tf.keras.models.Sequential(search_space[final_architecture_idx])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    _, accuracy = model.evaluate(env.x_test, env.y_test)
    print("Final architecture accuracy:", accuracy)
