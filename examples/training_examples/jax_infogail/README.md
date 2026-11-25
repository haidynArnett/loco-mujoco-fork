## InfoGAIL Imitation Learning Example

This example demonstrates training an InfoGAIL ([Li et al., 2017](https://arxiv.org/abs/1703.08840)) agent on the 
Unitree H1 robot to mimic a dataset of human running. InfoGAIL extends GAIL with a categorical latent code `c` that is 
sampled per episode and fed to the policy. A posterior network `Q(c|s)` is trained to recover the code from the 
visited states, and `log Q(c|s)` is added to the reward, so different codes are encouraged to produce distinguishable 
behaviors. It serves as a minimal showcase of the algorithm and is **not** intended to produce state-of-the-art results.

Following the original implementation, the discriminator is a WGAN critic (weight clipping, RMSprop). The reward is

```
p_env * r_env + (1 - p_env) * (disc_reward_scale * D(s) + reward_bias) + info_coef * log Q(c|s)
```

This example uses `NoGoal`, i.e., no trajectory goal in the observation. With a goal such as `GoalTrajMimic`, 
the policy would only track the reference trajectory and the latent code would have nothing left to control. 
The defined reward function in the configuration `MimicReward` will be only used for evaluation since the 
`proportion_env_reward` is set to 0.0.

> ⚠️ **Note:** The InfoGAIL hyperparameters in `conf.yaml` (`info_coef`, `disc_reward_scale`, `reward_bias`, 
> `wgan_clip`) follow the original implementation and have not been tuned for this setup.

---

### 🚀 Training

To train the agent, run:

```bash
python experiment.py
```

This command will:

- Train the InfoGAIL agent on the Unitree H1 robot for 75 million steps.
- Save the trained agent (as `InfoGAILJax_saved.pkl` in the `outputs` folder).
- Perform a final rendering of the trained policy, where the rendered environments cycle through the latent codes.
- Save a video of the rendering to the `LocoMuJoCo_recordings/` directory.
- Upload the video to Weights & Biases (WandB) for further analysis (check the command line logs for details).

In addition to the metrics logged by the GAIL example, the posterior loss and accuracy (`Posterior/*`) are logged. 
The accuracy should rise above chance level (`1 / n_codes`) if the latent code influences the behavior.

#### Validation Loop During Training

Throughout training, the agent will be evaluated using various trajectory-based metrics, including 
Euclidean distance, Dynamic Time Warping (DTW), and discrete Fréchet distance. These metrics will be 
computed on different entities such as joint positions, joint velocities, and site positions and orientations. 
The validation environments cycle through the latent codes. All results will be logged to Weights & Biases (WandB).

---

### 📈 Evaluation

To evaluate the trained agent, run:

```bash
python eval.py --path path/to/agent_file
```

To condition the policy on a specific latent code, use:

```bash
python eval.py --path path/to/agent_file --code 1
```

If you'd like to evaluate the agent using MuJoCo (instead of Mjx), run:

```bash
python eval.py --path path/to/agent_file --use_mujoco
```

> ⚠️ **Note:** Evaluating with MuJoCo may not yield results as robust as with Mjx due to simulator differences. For reliable policy transfer between the two, consider applying domain randomization techniques.
