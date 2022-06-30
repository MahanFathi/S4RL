import random # TODO: need to do some bookkeeping for seeding
import pickle
import numpy as np
from torch.utils import data
from ml_collections import FrozenConfigDict
from dataset.util import NumpyLoader


class D4RLTrajectoryDataset(data.Dataset):
    def __init__(
            self,
            cfg: FrozenConfigDict,
            dataset_path: str,
    ):

        self.seq_len = cfg.MODEL.SEQ_LEN

        # load dataset
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        states = []
        actions = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            states.append(traj['observations'])
            states.append(traj['actions'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        actions = np.concatenate(actions, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        self.action_mean, self.action_std = np.mean(actions, axis=0), np.std(actions, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std
            traj['actions'] = (traj['actions'] - self.action_mean) / self.action_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]

        if traj_len >= self.seq_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.seq_len)

            states = traj['observations'][si : si + self.seq_len]
            actions = traj['actions'][si : si + self.seq_len]

            # all ones since no padding
            traj_mask = np.ones(self.seq_len)

        else:
            padding_len = self.seq_len - traj_len

            # padding with zeros
            states = traj['observations']
            states = np.concatenate([
                states,
                np.zeros(([padding_len] + list(states.shape[1:])), dtype=states.dtype)
            ])

            actions = traj['actions']
            actions = np.concatenate([
                actions,
                np.zeros(([padding_len] + list(actions.shape[1:])), dtype=actions.dtype)
            ])

            traj_mask = np.concatenate([
                np.ones(traj_len, dtype=np.float),
                np.zeros(padding_len, dtype=np.float)
            ])

        return states, actions, traj_mask # pure imitation task


def get_d4lr_dataset_dataloader(cfg: FrozenConfigDict):

    env_name = cfg.ENV.ENV_NAME
    ds_name = cfg.DATA.DS_NAME.split("/")[-1]
    d4rl_env_name = f'{env_name}-{ds_name}-v2' # TODO: v2 only?
    dataset_dir = './data' # lazy hardcoding, sue me
    dataset_path = f'{dataset_dir}/{d4rl_env_name}.pkl'

    batch_size = cfg.TRAIN.BATCH_SIZE

    dataset = D4RLTrajectoryDataset(cfg, dataset_path)
    dataloader = NumpyLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    return dataset, dataloader
