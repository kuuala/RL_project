import matplotlib.pyplot as plt
import numpy as np

from dqn import DQN, DQN_MODE, DQN_ENV


def run_and_plot(envs, modes, filename_prefix):
    plt.figure(figsize=(12, 7))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, env in enumerate(envs):
        plt.clf()
        for j, mode in enumerate(modes):
            print(f"Обучение в {env.value}, режим {mode.name}")
            dqn = DQN(env_name=env, mode_name=mode)
            episode_rewards = dqn.train_dqn()

            color = color_cycle[j % len(color_cycle)]
            plt.plot(range(len(episode_rewards)), episode_rewards, linestyle='--', alpha=0.3,
                     color=color, label=f"{mode.name} (raw)")

            smoothed = np.convolve(episode_rewards, np.ones(10) / 10, mode='valid')
            plt.plot(range(len(smoothed)), smoothed, linestyle='-', color=color,
                     label=f"{mode.name} (smoothed)")

        plt.title(f"DQN: сравнение режимов в {env.value}")
        plt.xlabel("Эпизод")
        plt.ylabel("Награда")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_{env.value}.png")


def run_all_envs():
    envs = list(DQN_ENV)
    modes = list(DQN_MODE)
    run_and_plot(envs, modes, "dqn_comparison")


def main(seed=42):
    np.random.seed(seed)
    run_all_envs()


if __name__ == "__main__":
    main()
