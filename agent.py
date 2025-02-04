import tensorflow as tf
import keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.0015, env=None,
                 gamma=0.95, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=256, noise=0.8):
        self.gamma = gamma
        self.tau = tau
        breakpoint()
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.policy_delay = 1


        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic1 = CriticNetwork(name='critic1')
        self.critic2 = CriticNetwork(name='critic2')

        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor')
        self.target_critic1 = CriticNetwork(name='target_critic1')
        self.target_critic2 = CriticNetwork(name='target_critic2')


        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic1.compile(optimizer=Adam(learning_rate=beta))
        self.critic2.compile(optimizer=Adam(learning_rate=beta))

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic2.compile(optimizer=Adam(learning_rate=beta))
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic1.weights
        for i, weight in enumerate(self.critic1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic1.set_weights(weights)

        weights = []
        targets = self.target_critic2.weights
        for i, weight in enumerate(self.critic2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic2.set_weights(weights)
    def remember(self, state, action, reward, new_state, done, optimal_action):
        self.memory.store_transition(state, action, reward, new_state, done, optimal_action)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic1.save_weights(self.critic1.checkpoint_file)
        self.critic2.save_weights(self.critic2.checkpoint_file)
        self.target_critic1.save_weights(self.target_critic1.checkpoint_file)
        self.target_critic2.save_weights(self.target_critic2.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        path_actor = './saved/actor_ddpg.weights.h5'
        path_target_actor = './saved/target_actor_ddpg.weights.h5'
        path_critic = './saved/critic_ddpg.weights.h5'
        path_target_critic = './saved/target_critic_ddpg.weights.h5'
        self.actor.load_weights(path_actor)
        self.target_actor.load_weights(path_target_actor)
        self.critic.load_weights(path_critic)
        self.target_critic.load_weights(path_target_critic)

        # self.actor.load_weights(self.target_actor.checkpoint_file)
        # self.target_actor.load_weights(self.target_actor.checkpoint_file)
        # self.critic.load_weights(self.critic.checkpoint_file)
        # self.target_critic.load_weights(self.target_critic.checkpoint_file)
    
    def choose_action(self, observation, evaluate):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        state = tf.reshape(state, (1, -1))  # Reshape to (1, 2)
        actions = self.actor(state)
        actions = tf.clip_by_value(actions , self.min_action, self.max_action)  # Reduce voltage effect

        self.noise = max(0.1, self.noise * 0.99)
        
        if not evaluate:
            noise = tf.random.normal(shape=[self.n_actions], mean=0.0, stddev = self.noise)
            actions += noise
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]

    def learn(self, total_steps):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done, optimal_action= \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        target_actions = self.target_actor(states_)


        target_noise = tf.random.normal(shape=target_actions.shape, mean=0.0, stddev=0.2)
        target_noise = tf.clip_by_value(target_noise, -0.5, 0.5)
        target_actions = tf.clip_by_value(target_actions + target_noise, self.min_action, self.max_action)
        target_q1 = self.target_critic1(states_, target_actions)
        target_q2 = self.target_critic2(states_, target_actions)
        target_q = 0.1 * target_q1 + 0.9
        target = rewards + self.gamma * target_q * (1 - done)


        with tf.GradientTape() as tape:
            q1 = self.critic1(states, actions)
            critic_loss1 = tf.keras.losses.MSE(target, q1)
        critic1_grads = tape.gradient(critic_loss1, self.critic1.trainable_variables)
        self.critic1.optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        if tf.reduce_mean(tf.abs(critic_loss1)) < 1e-3:
            print("Warning: Critic 1 loss quá nhỏ, có thể đã bị stuck.")

        # Update Critic 2
        with tf.GradientTape() as tape:
            q2 = self.critic2(states, actions)
            critic_loss2 = tf.keras.losses.MSE(target, q2)
        critic2_grads = tape.gradient(critic_loss2, self.critic2.trainable_variables)
        self.critic2.optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        # print(f"Critic Loss1: {critic_loss1.numpy():.4f} | Critic Loss2: {critic_loss2.numpy():.4f}")
        # print(f"Critic Loss1: {float(critic_loss1.numpy()):.4f} | Critic Loss2: {float(critic_loss2.numpy()):.4f}")


        # Delayed Actor and Target Update
        if total_steps % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                new_policy_actions = self.actor(states)
                # actor_loss = -tf.math.reduce_mean(self.critic1(states, new_policy_actions))  # Only use Critic 1
                actor_loss = -self.critic1(states, new_policy_actions)
                supervised_loss = tf.keras.losses.MSE(optimal_action, new_policy_actions)
                total_loss = tf.math.reduce_mean(actor_loss) + 0.5 * tf.math.reduce_mean(supervised_loss)
            actor_grads = tape.gradient(total_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            self.update_network_parameters() 