import wandb
from tqdm import tqdm

guided_reward_run_names = [
    "curtiscjohnson/ppo_baloo/k1b5sgl8", "curtiscjohnson/ppo_baloo/clb77zh6",
    "curtiscjohnson/ppo_baloo/h4jhragr", "curtiscjohnson/ppo_baloo/14wid7ha",
    "curtiscjohnson/ppo_baloo/yrqgtt10", "curtiscjohnson/ppo_baloo/zffvdqe5",
    "curtiscjohnson/ppo_baloo/ucw6yr8d", "curtiscjohnson/ppo_baloo/5ms4h5te",
    "curtiscjohnson/ppo_baloo/0hk9i5w8", "curtiscjohnson/ppo_baloo/ip8q2cme"
]

shaped_reward_run_names = [
    "curtiscjohnson/ppo_baloo/yemzaj5w", "curtiscjohnson/ppo_baloo/8nahvcli",
    "curtiscjohnson/ppo_baloo/no0fcm1r", "curtiscjohnson/ppo_baloo/u4ki660w",
    "curtiscjohnson/ppo_baloo/211eax8i", "curtiscjohnson/ppo_baloo/kbk4uaon",
    "curtiscjohnson/ppo_baloo/5pbqvx9l", "curtiscjohnson/ppo_baloo/zpohzs46",
    "curtiscjohnson/ppo_baloo/vp6wmank", "curtiscjohnson/ppo_baloo/n2wjujs4"
]

api = wandb.Api()

for name in tqdm(guided_reward_run_names):
    run = api.run(name)
    print(run.name)
    for file in run.files():
        file.download(root=f"./wandb_download/guided_reward_runs/{run.name}/")

for name in tqdm(shaped_reward_run_names):
    run = api.run(name)
    print(run.name)
    for file in run.files():
        file.download(root=f"./wandb_download/shaped_reward_runs/{run.name}/")
