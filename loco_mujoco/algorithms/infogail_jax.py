import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import flax
import optax

from loco_mujoco.algorithms import ActorCritic, FullyConnectedNet, Transition, TrainState, TrainStateBuffer, \
    MetricHandlerTransition
from loco_mujoco.algorithms.gail_jax import GAILJax, GAILAgentConf, GAILAgentState, GailSummaryMetrics
from loco_mujoco.core.wrappers import LogEnvState
from loco_mujoco.utils import MetricsHandler, ValidationSummary
from dataclasses import replace
from omegaconf import OmegaConf


@struct.dataclass
class InfoGailSummaryMetrics(GailSummaryMetrics):
    posterior_loss: float = 0.0
    posterior_accuracy: float = 0.0
    mean_log_posterior: float = 0.0


@dataclass(frozen=True)
class InfoGAILAgentConf(GAILAgentConf):
    posterior: FullyConnectedNet = None
    post_tx: Any = None

    def serialize(self):
        _all = super().serialize()
        _all["posterior"] = flax.serialization.to_state_dict(self.posterior)
        return _all

    @classmethod
    def from_dict(cls, d):
        config = OmegaConf.create(d["config"])
        tx, disc_tx = InfoGAILJax._get_optimizer(config)
        post_tx = InfoGAILJax._get_posterior_optimizer(config)
        return cls(config=config,
                   network=flax.serialization.from_state_dict(ActorCritic, d["network"]),
                   discriminator=flax.serialization.from_state_dict(FullyConnectedNet, d["discriminator"]),
                   tx=tx, disc_tx=disc_tx,
                   expert_dataset=None,
                   posterior=flax.serialization.from_state_dict(FullyConnectedNet, d["posterior"]),
                   post_tx=post_tx)


@struct.dataclass
class InfoGAILAgentState(GAILAgentState):
    post_train_state: TrainState

    def serialize(self):
        d = super().serialize()
        d["posterior"] = flax.serialization.to_state_dict(self.post_train_state)
        return d

    @classmethod
    def from_dict(cls, d, agent_conf):
        base = GAILAgentState.from_dict(d, agent_conf)
        post_train_state = TrainState.create(
            apply_fn=agent_conf.posterior.apply,
            tx=agent_conf.post_tx,
            params=d["posterior"]["params"],
            run_stats=d["posterior"]["run_stats"],
        )
        post_opt_state = flax.serialization.from_state_dict(
            post_train_state.opt_state, d["posterior"]["opt_state"]
        )
        post_train_state = post_train_state.replace(opt_state=post_opt_state)
        return cls(base.train_state, base.disc_train_state, post_train_state)


class InfoGAILJax(GAILJax):
    """
    InfoGAIL (Li et al., 2017): GAIL with a latent code c that is sampled per episode, fed to the policy, and
    made recoverable from the visited states via a learned posterior Q(c|s). Uses a WGAN critic
    (weight clipping, RMSprop) as the discriminator, as in the original implementation.

    Reward: (1 - p_env) * (disc_reward_scale * D(s) + reward_bias) + p_env * r_env + info_coef * log Q(c|s)

    """

    _agent_conf = InfoGAILAgentConf
    _agent_state = InfoGAILAgentState

    @classmethod
    def init_agent_conf(cls, env, config):

        if hasattr(config.experiment, "len_obs_history"):
            assert config.experiment.len_obs_history == 1, "InfoGAIL does not support len_obs_history > 1."

        agent_conf = super().init_agent_conf(env, config)

        # append the one-hot latent code to the actor and critic inputs
        obs_dim = env.info.observation_space.shape[0]
        n_codes = config.experiment.n_codes
        code_ind = obs_dim + jnp.arange(n_codes)
        network = agent_conf.network.clone(
            actor_obs_ind=jnp.concatenate([agent_conf.network.actor_obs_ind, code_ind]),
            critic_obs_ind=jnp.concatenate([agent_conf.network.critic_obs_ind, code_ind]))

        posterior = FullyConnectedNet(activation=config.experiment.activation,
                                      hidden_layer_dims=agent_conf.network.hidden_layer_dims,
                                      output_dim=n_codes, output_activation=None,
                                      use_running_mean_stand=True,
                                      squeeze_output=False)

        return replace(agent_conf, network=network, posterior=posterior,
                       post_tx=cls._get_posterior_optimizer(config))

    @classmethod
    def _get_optimizer(cls, config):
        tx, _ = super()._get_optimizer(config)
        # WGAN critics are trained with RMSprop (momentum-based optimizers are unstable with weight clipping)
        disc_tx = optax.rmsprop(config.experiment.disc_lr)
        return tx, disc_tx

    @classmethod
    def _get_posterior_optimizer(cls, config):
        return optax.chain(
            optax.clip_by_global_norm(config.experiment.max_grad_norm),
            optax.adam(config.experiment.posterior_lr, eps=1e-5),
        )

    @staticmethod
    def _augment_obs(obs, codes, n_codes):
        return jnp.concatenate([obs, jax.nn.one_hot(codes, n_codes)], axis=-1)

    @classmethod
    def _critic_scores(cls, obs, discriminator, disc_train_state):
        scores, _ = discriminator.apply({'params': disc_train_state.params,
                                         'run_stats': disc_train_state.run_stats},
                                        obs, mutable=["run_stats"])
        return scores

    @classmethod
    def _log_posterior(cls, obs, codes, posterior, post_train_state):
        logits, _ = posterior.apply({'params': post_train_state.params,
                                     'run_stats': post_train_state.run_stats},
                                    obs, mutable=["run_stats"])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(log_probs, codes[:, None], axis=-1)[:, 0]

    @classmethod
    def _train_fn(cls, rng, env,
                  agent_conf: InfoGAILAgentConf,
                  agent_state: InfoGAILAgentState = None,
                  mh: MetricsHandler = None):

        config, network, discriminator, posterior, tx, disc_tx, post_tx, expert_dataset = \
            (agent_conf.config.experiment, agent_conf.network, agent_conf.discriminator, agent_conf.posterior,
             agent_conf.tx, agent_conf.disc_tx, agent_conf.post_tx, agent_conf.expert_dataset)

        obs_dim = env.info.observation_space.shape[0]
        n_codes = config.n_codes

        if agent_state is not None:
            train_state, disc_train_state, post_train_state = \
                agent_state.train_state, agent_state.disc_train_state, agent_state.post_train_state
        else:
            train_state, disc_train_state, post_train_state = None, None, None

        if train_state is None:
            rng, _rng1, _rng2, _rng3 = jax.random.split(rng, 4)
            network_params = network.init(_rng1, jnp.zeros(obs_dim + n_codes))
            discrim_params = discriminator.init(_rng2, jnp.zeros(obs_dim))
            post_params = posterior.init(_rng3, jnp.zeros(obs_dim))

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params["params"] if train_state is None else train_state.params,
            run_stats=network_params["run_stats"] if train_state is None else train_state.run_stats,
            tx=tx,
        )
        disc_train_state = TrainState.create(
            apply_fn=discriminator.apply,
            params=discrim_params["params"] if disc_train_state is None else disc_train_state.params,
            run_stats=discrim_params["run_stats"] if disc_train_state is None else disc_train_state.run_stats,
            tx=disc_tx,
        )
        post_train_state = TrainState.create(
            apply_fn=posterior.apply,
            params=post_params["params"] if post_train_state is None else post_train_state.params,
            run_stats=post_params["run_stats"] if post_train_state is None else post_train_state.run_stats,
            tx=post_tx,
        )

        env = cls._wrap_env(env, config)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = env.reset(reset_rng)

        # latent codes are sampled uniformly per env; a new code is drawn one step after an episode ends, so that
        # the (final observation, old code) pair used for value bootstrapping stays consistent
        rng, _rng = jax.random.split(rng)
        codes = jax.random.randint(_rng, (config.num_envs,), 0, n_codes)
        pending_reset = jnp.zeros(config.num_envs, dtype=bool)

        train_state_buffer = TrainStateBuffer.create(train_state, config.validation.num)

        def _update_step(runner_state, unused):

            def _env_step(runner_state, unused):
                (train_state, disc_train_state, post_train_state, env_state, last_obs, codes, pending_reset,
                 train_state_buffer, rng) = runner_state

                rng, _rng = jax.random.split(rng)
                obs_aug = cls._augment_obs(last_obs, codes, n_codes)
                y, updates = network.apply({'params': train_state.params,
                                            'run_stats': train_state.run_stats},
                                           obs_aug, mutable=["run_stats"])
                pi, value = y
                train_state = train_state.replace(run_stats=updates['run_stats'])
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                logged_metrics = env_state.find(LogEnvState).metrics

                transition = Transition(
                    done, absorbing, action, value, reward, log_prob, obs_aug, info,
                    env_state.additional_carry.traj_state, logged_metrics
                )

                rng, _rng = jax.random.split(rng)
                new_codes = jax.random.randint(_rng, codes.shape, 0, n_codes)
                codes = jnp.where(pending_reset, new_codes, codes)
                pending_reset = done.astype(bool)

                runner_state = (train_state, disc_train_state, post_train_state, env_state, obsv, codes,
                                pending_reset, train_state_buffer, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            (train_state, disc_train_state, post_train_state, env_state, last_obs, codes, pending_reset,
             train_state_buffer, rng) = runner_state
            y, _ = network.apply({'params': train_state.params,
                                  'run_stats': train_state.run_stats},
                                 cls._augment_obs(last_obs, codes, n_codes), mutable=["run_stats"])
            pi, last_val = y

            # REWARDS: WGAN critic score + log posterior of the code that generated the visited state
            obs_raw = traj_batch.obs[..., :obs_dim]
            traj_codes = jnp.argmax(traj_batch.obs[..., obs_dim:], axis=-1)
            flat_obs = obs_raw.reshape((-1, obs_dim))
            scores = cls._critic_scores(flat_obs, discriminator, disc_train_state).reshape(traj_batch.reward.shape)
            log_q = cls._log_posterior(flat_obs, traj_codes.reshape(-1), posterior,
                                       post_train_state).reshape(traj_batch.reward.shape)
            disc_reward = config.disc_reward_scale * scores + config.reward_bias
            rewards = (config.proportion_env_reward * traj_batch.reward +
                       (1 - config.proportion_env_reward) * disc_reward +
                       config.info_coef * log_q)

            def _calculate_gae(traj_batch, last_val, rewards):
                def _get_advantages(gae_and_next_value, x):
                    gae, next_value = gae_and_next_value
                    done, absorbing, value, reward = x
                    delta = reward + config.gamma * next_value * (1 - absorbing) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    (traj_batch.done, traj_batch.absorbing, traj_batch.value, rewards),
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, rewards)

            # UPDATE ACTOR & CRITIC NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        y, _ = network.apply({'params': params, 'run_stats': train_state.run_stats},
                                             traj_batch.obs, mutable=["run_stats"])
                        pi, value = y
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.clip_eps, config.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.minibatch_size * config.num_minibatches
                assert (
                    batch_size == config.num_steps * config.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config.num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
            train_state = update_state[0]
            rng = update_state[-1]

            def _get_one_batch(data, batch_size, rng):
                idx = jax.random.randint(rng, shape=(batch_size,), minval=0, maxval=data.shape[0])
                return data[idx]

            # UPDATE WGAN CRITIC
            def _update_discriminator(carry, unused):
                disc_train_state, rng = carry

                def _critic_loss(params, disc_train_state, inputs):
                    scores, updates = discriminator.apply({'params': params,
                                                           'run_stats': disc_train_state.run_stats},
                                                          inputs, mutable=["run_stats"])
                    disc_train_state = disc_train_state.replace(run_stats=updates["run_stats"])
                    plcy_scores = scores[:config.disc_minibatch_size]
                    exp_scores = scores[config.disc_minibatch_size:]
                    loss = jnp.mean(plcy_scores) - jnp.mean(exp_scores)
                    return loss, (disc_train_state, plcy_scores, exp_scores)

                rng, _rng1, _rng2 = jax.random.split(rng, 3)
                plcy_input = _get_one_batch(flat_obs, config.disc_minibatch_size, _rng1)
                demo_input = _get_one_batch(expert_dataset.observations, config.disc_minibatch_size, _rng2)
                inputs = jnp.concatenate([plcy_input, demo_input], axis=0)

                grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)
                (_, (disc_train_state, plcy_scores, exp_scores)), grads = grad_fn(
                    disc_train_state.params, disc_train_state, inputs)
                disc_train_state = disc_train_state.apply_gradients(grads=grads)

                # Lipschitz constraint via weight clipping
                clipped = jax.tree.map(lambda p: jnp.clip(p, -config.wgan_clip, config.wgan_clip),
                                       disc_train_state.params)
                disc_train_state = disc_train_state.replace(params=clipped)

                return (disc_train_state, rng), (jnp.mean(plcy_scores), jnp.mean(exp_scores))

            (disc_train_state, rng), (plcy_scores, exp_scores) = jax.lax.scan(
                _update_discriminator, (disc_train_state, rng), xs=None, length=config.n_disc_epochs)

            # UPDATE POSTERIOR Q(c|s): maximize the variational lower bound on I(c; s)
            def _update_posterior(carry, unused):
                post_train_state, rng = carry

                def _posterior_loss(params, post_train_state, inputs, labels):
                    logits, updates = posterior.apply({'params': params,
                                                       'run_stats': post_train_state.run_stats},
                                                      inputs, mutable=["run_stats"])
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    loss = -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))
                    acc = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
                    return loss, (updates["run_stats"], acc)

                rng, _rng = jax.random.split(rng)
                idx = jax.random.randint(_rng, (config.post_minibatch_size,), 0, flat_obs.shape[0])
                inputs, labels = flat_obs[idx], traj_codes.reshape(-1)[idx]

                grad_fn = jax.value_and_grad(_posterior_loss, has_aux=True)
                (loss, (run_stats, acc)), grads = grad_fn(post_train_state.params, post_train_state, inputs, labels)
                post_train_state = post_train_state.apply_gradients(grads=grads).replace(run_stats=run_stats)
                return (post_train_state, rng), (loss, acc)

            (post_train_state, rng), (post_loss, post_acc) = jax.lax.scan(
                _update_posterior, (post_train_state, rng), xs=None, length=config.n_post_epochs)

            counter = ((train_state.step + 1) // config.num_minibatches) // config.update_epochs

            logged_metrics = traj_batch.metrics
            metric = InfoGailSummaryMetrics(
                discriminator_output_policy=jnp.mean(plcy_scores),
                discriminator_output_expert=jnp.mean(exp_scores),
                posterior_loss=jnp.mean(post_loss),
                posterior_accuracy=jnp.mean(post_acc),
                mean_log_posterior=jnp.mean(log_q),
                mean_episode_return=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_returns, 0.0)) / jnp.sum(logged_metrics.done),
                mean_episode_length=jnp.sum(jnp.where(logged_metrics.done, logged_metrics.returned_episode_lengths, 0.0)) / jnp.sum(logged_metrics.done),
                max_timestep=jnp.max(logged_metrics.timestep * config.num_envs),
            )

            def _evaluation_step():
                # evaluation envs cycle through the latent codes
                eval_codes = jnp.arange(config.validation.num_envs) % n_codes

                def _eval_env(carry, unused):
                    train_state, env_state, last_obs, rng = carry
                    rng, _rng = jax.random.split(rng)
                    y, updates = train_state.apply_fn({'params': train_state.params,
                                                       'run_stats': train_state.run_stats},
                                                      cls._augment_obs(last_obs, eval_codes, n_codes),
                                                      mutable=["run_stats"])
                    pi, _ = y
                    train_state = train_state.replace(run_stats=updates['run_stats'])
                    action = pi.sample(seed=_rng)

                    obsv, reward, absorbing, done, info, env_state = env.step(env_state, action)

                    logged_metrics = env_state.find(LogEnvState).metrics
                    transition = MetricHandlerTransition(env_state, logged_metrics)
                    return (train_state, env_state, obsv, rng), transition

                eval_rng = jax.random.fold_in(rng, 1)
                reset_rng = jax.random.split(eval_rng, config.validation.num_envs)
                obsv, env_state = env.reset(reset_rng)

                _, traj_batch_eval = jax.lax.scan(
                    _eval_env, (train_state, env_state, obsv, eval_rng), None, config.validation.num_steps
                )
                return mh(traj_batch_eval.env_state)

            if mh is None:
                validation_metrics = ValidationSummary()
            else:
                validation_metrics = jax.lax.cond(counter % config.validation_interval == 0, _evaluation_step,
                                                  mh.get_zero_container)

            if config.debug:
                def callback(metrics):
                    return_values = metrics.returned_episode_returns[metrics.done]
                    timesteps = metrics.timestep[metrics.done] * config.num_envs

                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")

                jax.debug.callback(callback, env_state.metrics)

            train_state_buffer = jax.lax.cond(counter % config.validation_interval == 0,
                                              lambda x, y: TrainStateBuffer.add(x, y),
                                              lambda x, y: x, train_state_buffer, train_state)

            runner_state = (train_state, disc_train_state, post_train_state, env_state, last_obs, codes,
                            pending_reset, train_state_buffer, rng)
            return runner_state, (metric, validation_metrics)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, disc_train_state, post_train_state, env_state, obsv, codes, pending_reset,
                        train_state_buffer, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config.num_updates)

        agent_state = cls._agent_state(train_state=runner_state[0], disc_train_state=runner_state[1],
                                       post_train_state=runner_state[2])

        return {"agent_state": agent_state,
                "training_metrics": metrics[0],
                "validation_metrics": metrics[1]}

    @classmethod
    def play_policy(cls, env,
                    agent_conf: InfoGAILAgentConf,
                    agent_state: InfoGAILAgentState,
                    n_envs: int, n_steps=None, render=True,
                    record=False, rng=None, deterministic=False,
                    use_mujoco=False, wrap_env=True,
                    train_state_seed=None, code=None):
        """
        Play the policy. If code is None, env i uses latent code i % n_codes, otherwise all envs use the given code.
        """

        config = agent_conf.config.experiment
        n_codes = config.n_codes

        if use_mujoco:
            assert n_envs == 1, "Only one mujoco env can be run at a time."
        if code is not None:
            assert 0 <= code < n_codes, f"code must be in [0, {n_codes})."

        def sample_actions(ts, obs, codes, _rng):
            y, updates = agent_conf.network.apply({'params': ts.params,
                                                   'run_stats': ts.run_stats},
                                                  cls._augment_obs(obs, codes, n_codes), mutable=["run_stats"])
            ts = ts.replace(run_stats=updates['run_stats'])
            pi, _ = y
            a = pi.sample(seed=_rng)
            return a, ts

        train_state = agent_state.train_state

        if deterministic:
            train_state.params["log_std"] = np.ones_like(train_state.params["log_std"]) * -np.inf

        if config.n_seeds > 1:
            assert train_state_seed is not None, ("Loaded train state has multiple seeds. Please specify "
                                                  "train_state_seed for replay.")
            train_state = jax.tree.map(lambda x: x[train_state_seed], train_state)

        if not render and n_steps is None and not record:
            warnings.warn("No rendering, no record, no n_steps specified. This will run forever with no effect.")

        if wrap_env and not use_mujoco:
            env = cls._wrap_env(env, config)

        if rng is None:
            rng = jax.random.key(0)

        keys = jax.random.split(rng, n_envs + 1)
        rng, env_keys = keys[0], keys[1:]

        if use_mujoco:
            codes = jnp.asarray(0 if code is None else code)
        elif code is not None:
            codes = jnp.full((n_envs,), code)
        else:
            codes = jnp.arange(n_envs) % n_codes

        plcy_call = jax.jit(sample_actions)

        if use_mujoco:
            obs = env.reset()
            env_state = None
        else:
            obs, env_state = env.reset(env_keys)

        if n_steps is None:
            n_steps = np.iinfo(np.int32).max

        for i in range(n_steps):

            rng, _rng = jax.random.split(rng)
            action, train_state = plcy_call(train_state, obs, codes, _rng)
            action = jnp.atleast_2d(action)

            if use_mujoco:
                obs, reward, absorbing, done, info = env.step(action)
                env.render(record=True)
                if done:
                    obs = env.reset()
            else:
                obs, reward, absorbing, done, info, env_state = env.step(env_state, action)
                env.mjx_render(env_state, record=record)

        env.stop()

    @classmethod
    def play_policy_mujoco(cls, env,
                           agent_conf: InfoGAILAgentConf,
                           agent_state: InfoGAILAgentState,
                           n_steps=None, render=True,
                           record=False, rng=None, deterministic=False,
                           train_state_seed=None, code=None):

        cls.play_policy(env, agent_conf, agent_state, 1, n_steps, render, record, rng, deterministic,
                        True, False, train_state_seed, code)
