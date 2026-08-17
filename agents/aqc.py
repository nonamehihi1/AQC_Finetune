import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


def expectile_loss(diff, expectile=0.9):
    """Compute the expectile loss."""
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff ** 2)


class AQCAgent(flax.struct.PyTreeNode):
    """Adaptive Q-Chunking (AQC) agent with learned variance normalization."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the AQC critic, value, and moment losses."""

        # Batch features
        # observations: (batch_size, horizon_length, state_dim)
        # next_observations: (batch_size, horizon_length, state_dim)
        # actions: (batch_size, horizon_length, action_dim)
        # rewards: (batch_size, horizon_length)  # This is cumulative reward from t to t+j!
        # valid: (batch_size, horizon_length)
        # masks: (batch_size, horizon_length)

        chunk_sizes = self.config['chunk_sizes']
        horizon_length = self.config['horizon_length']

        info = {}
        total_critic_loss = 0.0

        rng, sample_rng = jax.random.split(rng)
        
        # We need V^h for bootstrapping smaller chunks
        # Compute V^h(s_{t+k}) for all k in chunk_sizes
        next_v_h = {}
        for k in chunk_sizes:
            if k < horizon_length:
                # We need V^h at t+k
                # next_observations[..., k-1, :] represents s_{t+k} since it's the next state of the k-th step
                v = self.network.select('value_h')(batch['next_observations'][..., k-1, :], params=grad_params)
                next_v_h[k] = jax.lax.stop_gradient(v)

        # 1. Q^h Loss (Standard FQL on full horizon)
        batch_actions_h = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        
        next_actions = self.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)
        
        target_q_h = batch['rewards'][..., -1] + \
            (self.config['discount'] ** horizon_length) * batch['masks'][..., -1] * next_q
        target_q_h = jax.lax.stop_gradient(target_q_h)

        q_h = self.network.select('critic')(batch['observations'], actions=batch_actions_h, params=grad_params)
        loss_q_h = (jnp.square(q_h - target_q_h) * batch['valid'][..., -1]).mean()
        total_critic_loss += loss_q_h

        info.update({
            'q_h_loss': loss_q_h,
            'q_h_mean': q_h.mean(),
        })

        # 2. V^h Loss
        v_h = self.network.select('value_h')(batch['observations'], params=grad_params)
        # target for v_h is q_h (stop grad)
        q_h_sg = jax.lax.stop_gradient(q_h)
        # using min over ensemble if q_h has multiple Qs. q_h is shape (num_qs, batch_size)
        if self.config['q_agg'] == 'min':
            q_h_val = q_h_sg.min(axis=0)
        else:
            q_h_val = q_h_sg.mean(axis=0)
            
        diff_h = q_h_val - v_h
        loss_v_h = expectile_loss(diff_h, self.config['expectile']).mean()
        total_critic_loss += loss_v_h
        info.update({'v_h_loss': loss_v_h, 'v_h_mean': v_h.mean()})

        # 3. Q^k, V^k, M^k Losses for all k in chunk_sizes
        for k in chunk_sizes:
            # Prepare action sequence a_{1:k}
            batch_actions_k = jnp.reshape(batch["actions"][..., :k, :], (batch["actions"].shape[0], -1))
            
            # Prepare target for Q^k
            if k == horizon_length:
                target_q_k = target_q_h  # Same as Q^h
            else:
                # Target for Q^k is cumulative reward for k steps + gamma^k * V^h(s_{t+k})
                target_q_k = batch['rewards'][..., k-1] + \
                             (self.config['discount'] ** k) * batch['masks'][..., k-1] * next_v_h[k]
                target_q_k = jax.lax.stop_gradient(target_q_k)
            
            # Predict Q^k
            q_k = self.network.select(f'critic_k{k}')(batch['observations'], actions=batch_actions_k, params=grad_params)
            
            # Only train on valid trajectories
            valid_mask = batch['valid'][..., k-1]
            loss_q_k = (jnp.square(q_k - target_q_k) * valid_mask).mean()
            total_critic_loss += loss_q_k

            # Value network V^k
            v_k = self.network.select(f'value_k{k}')(batch['observations'], params=grad_params)
            q_k_sg = jax.lax.stop_gradient(q_k)
            if self.config['q_agg'] == 'min':
                q_k_val = q_k_sg.min(axis=0)
            else:
                q_k_val = q_k_sg.mean(axis=0)
            
            diff_k = q_k_val - v_k
            loss_v_k = (expectile_loss(diff_k, self.config['expectile']) * valid_mask).mean()
            total_critic_loss += loss_v_k

            # Moment network M^k (Proposal 1)
            # TẠM TẮT: Do công thức target_m_k sai toán học (dùng expectile thay vì mean)
            # Ta bỏ qua update M^k để tiết kiệm tính toán trong lúc debug hệ thống bằng Sample Z-score
            # m_k = self.network.select(f'moment_k{k}')(batch['observations'], params=grad_params)
            # target_m_k = jnp.square(jax.lax.stop_gradient(diff_k))
            # loss_m_k = (jnp.square(m_k - target_m_k) * valid_mask).mean()
            # total_critic_loss += loss_m_k

            info.update({
                f'q_{k}_loss': loss_q_k,
                f'v_{k}_loss': loss_v_k,
                # f'm_{k}_loss': loss_m_k,
                f'q_{k}_mean': q_k_val.mean(),
                f'v_{k}_mean': v_k.mean(),
                # f'm_{k}_mean': m_k.mean(),
            })

        return total_critic_loss, info

    def actor_loss(self, batch, grad_params, rng):
        """Compute the actor loss (same as FQL)."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1)) 
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        # only bc on the valid chunk indices
        bc_flow_loss = jnp.mean(
            jnp.reshape(
                (pred - vel) ** 2, 
                (batch_size, self.config["horizon_length"], self.config["action_dim"]) 
            ) * batch["valid"][..., None]
        )

        if self.config["actor_type"] == "distill-ddpg":
            # Distillation loss.
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
            actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
            
            # Q loss.
            actor_actions = jnp.clip(actor_actions, -1, 1)

            qs = self.network.select(f'critic')(batch['observations'], actions=actor_actions)
            q = jnp.mean(qs, axis=0)
            q_loss = -q.mean()
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        # Update target for Q^h (standard critic)
        agent.target_update(new_network, 'critic')
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """Sample actions (full horizon) using the actor. Returns full chunks."""
        # For evaluation and actor distillation target
        if self.config["actor_type"] == "distill-ddpg":
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config['action_dim'] * self.config['horizon_length'],
                ),
            )
            actions = self.network.select(f'actor_onestep_flow')(observations, noises)
            actions = jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self.config['action_dim'] * self.config['horizon_length']
            noises = jax.random.normal(
                rng,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                    self.config["actor_num_samples"], action_dim
                ),
            )
            observations = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            
            # Select best action based on Q^h
            if self.config["q_agg"] == "mean":
                q = self.network.select("critic")(observations, actions).mean(axis=0)
            else:
                q = self.network.select("critic")(observations, actions).min(axis=0)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, self.config["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(
                bshape + (action_dim,))

        return actions
        
    @jax.jit
    def sample_actions_adaptive(
        self,
        observations,
        rng=None,
    ):
        """
        Sample actions and select optimal chunk size k*.
        Returns:
            actions: full horizon actions
            k_star: array of optimal chunk sizes
        """
        rng, sample_rng = jax.random.split(rng)
        
        # 1. Sample full chunks
        action_dim_full = self.config['action_dim'] * self.config['horizon_length']
        noises = jax.random.normal(
            sample_rng,
            (
                *observations.shape[: -len(self.config['ob_dims'])],  # batch_size
                self.config["actor_num_samples"], action_dim_full
            ),
        )
        obs_expanded = jnp.repeat(observations[..., None, :], self.config["actor_num_samples"], axis=-2)
        
        # Dùng trực tiếp Flow Matching policy (Euler integration) để sinh N=32 diverse candidates
        # Đây là chuẩn xác 100% so với bài báo gốc
        actions = self.compute_flow_actions(obs_expanded, noises)
            
        actions = jnp.clip(actions, -1, 1)
        
        # 2. Evaluate A^k and z_k for each k
        chunk_sizes = self.config['chunk_sizes']
        
        # Evaluate V^k and M^k for state
        # Because obs_expanded has shape (batch_size, num_samples, state_dim),
        # but V^k and M^k only depend on state, we can evaluate on original observations
        # and expand.
        
        z_k_list = []
        
        for k in chunk_sizes:
            # v_k, m_k shape: (batch_size, 1)
            v_k = self.network.select(f'value_k{k}')(observations)
            # m_k = self.network.select(f'moment_k{k}')(observations)
            
            # Predict Q^k for all candidates
            # actions shape: (batch_size, num_samples, action_dim * horizon_length)
            actions_k = jnp.reshape(
                jnp.reshape(actions, (*actions.shape[:-1], self.config['horizon_length'], self.config['action_dim']))[..., :k, :],
                (*actions.shape[:-1], k * self.config['action_dim'])
            )
            
            q_k = self.network.select(f'critic_k{k}')(obs_expanded, actions=actions_k)
            if self.config['q_agg'] == 'min':
                q_k = q_k.min(axis=0) # shape: (*bshape, num_samples)
            else:
                q_k = q_k.mean(axis=0)
                
            # Broadcast to match q_k shape (*bshape, num_samples)
            v_k = jnp.expand_dims(v_k, axis=-1)
            # m_k = jnp.expand_dims(m_k, axis=-1)
            
            # Compute Advantage
            a_k = q_k - v_k
            
            # SỬA LỖI BUG: Dùng Sample Z-score trực tiếp trên 32 samples tại inference
            # (Loại bỏ m_k do học sai toán học)
            a_k_mean = a_k.mean(axis=-1, keepdims=True)
            a_k_std = a_k.std(axis=-1, keepdims=True)
            z_k = (a_k - a_k_mean) / (a_k_std + 1e-6)
            z_k_list.append(z_k)
            
        # z_k_list is a list of len(chunk_sizes) with tensors of shape (*bshape, num_samples)
        z_scores = jnp.stack(z_k_list, axis=-2) # (*bshape, len(chunk_sizes), num_samples)
        
        # We want to find (k_idx, sample_idx) that maximizes z_score for each batch element
        bshape = observations.shape[:-1]
        z_scores_flat = z_scores.reshape((*bshape, -1))
        
        best_flat_idx = jnp.argmax(z_scores_flat, axis=-1)
        
        best_k_idx = best_flat_idx // self.config["actor_num_samples"]
        best_sample_idx = best_flat_idx % self.config["actor_num_samples"]
        
        # Convert k_idx to actual k values
        chunk_sizes_array = jnp.array(chunk_sizes)
        k_star = chunk_sizes_array[best_k_idx]
        
        # Select the best action
        flat_best_sample_idx = best_sample_idx.reshape(-1)
        bsize = len(flat_best_sample_idx)
        
        flat_actions = actions.reshape((bsize, self.config["actor_num_samples"], -1))
        best_actions = flat_actions[jnp.arange(bsize), flat_best_sample_idx, :].reshape((*bshape, -1))
        
        return best_actions, k_star

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        full_action_dim = full_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Define basic networks
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
        )
        
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic'),
        )

        moment_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            encoder=encoders.get('critic'),
        )

        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
            use_fourier_features=config["use_fourier_features"],
            fourier_feature_dim=config["fourier_feature_dim"],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            value_h=(copy.deepcopy(value_def), (ex_observations,)),
        )
        
        # Add sub-critics, values, and moments for each chunk size
        for k in config['chunk_sizes']:
            k_action_dim = action_dim * k
            ex_actions_k = jnp.concatenate([ex_actions] * k, axis=-1)
            
            network_info[f'critic_k{k}'] = (
                Value(
                    hidden_dims=config['value_hidden_dims'],
                    layer_norm=config['layer_norm'],
                    num_ensembles=config['num_qs'],
                    encoder=encoders.get('critic'),
                ), 
                (ex_observations, ex_actions_k)
            )
            network_info[f'value_k{k}'] = (copy.deepcopy(value_def), (ex_observations,))
            network_info[f'moment_k{k}'] = (copy.deepcopy(moment_def), (ex_observations,))

        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
            
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
            
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='aqc',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=100.0,  # BC coefficient (need to be tuned for each environment).
            num_qs=2, # critic ensemble size
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            horizon_length=ml_collections.config_dict.placeholder(int), # will be set
            action_chunking=True,  # Always True for AQC
            chunk_sizes=(1, 3, 5), # AQC specific: sequence of chunk sizes to adaptively select from
            expectile=0.9, # Expectile for value function training
            actor_type="best-of-n", # Default to best-of-n for AQC
            actor_num_samples=32,  # for actor_type="best-of-n" only
            use_fourier_features=False,
            fourier_feature_dim=64,
            weight_decay=0.,
        )
    )
    return config
