import optuna
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from baloo_gym.utils.helpers import make_parallel_env


def optimize_ppo(trial):
    """
    Suggests hyperparameters for PPO and evaluates the policy performance.
    """
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps",
                                        [128, 256, 512, 1024, 2048, 4096])

    batch_size = trial.suggest_categorical("batch_size",
                                           [32, 64, 128, 256, 512])

    if n_steps % batch_size != 0:
        raise optuna.exceptions.TrialPruned(
            "n_steps must be divisible by batch_size")

    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True)

    config = {
        "total_timesteps": 25000,
        "ctrl_timestep": .1,
        "env_name": "baloo_v4",
        "time_limit_sec": 30,
        "time_aware_obs": True,
        "curriculum_selection": ["manipuland_initial_position"],
    }

    env = make_parallel_env(config,
                            '.',
                            baseline=False,
                            monitor=False,
                            record_video=False,
                            num_envs=9,
                            wandb=False)

    rl_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
    )

    rl_model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
    )

    # Evaluate the model
    mean_reward, _ = evaluate_policy(rl_model, env, n_eval_episodes=5)
    print(f"Mean reward: {mean_reward}")
    env.close()

    # Negative because Optuna minimizes objective
    return -mean_reward


# Optuna study for optimization
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize",
                                storage="sqlite:///ppo.db")
    study.optimize(optimize_ppo, n_trials=50)

    print("Best hyperparameters:", study.best_params)
