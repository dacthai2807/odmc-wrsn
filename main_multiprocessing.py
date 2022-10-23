import random
import argparse
import os
import torch
import time
import numpy as np
from datetime import datetime

from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from multiprocessing import set_start_method

from model import MCActor, Critic
from utils import Config, DrlParameters as dp, WrsnParameters as wp
from utils import logger, device, writer, make_logger, device_str

from worker import train, validate

def main(log_dir='', num_sensors=20, num_targets=10, config=None,
        checkpoint=None, save_dir='checkpoints', seed=123, 
        mode='train', epoch_start=0, render=False, verbose=False):

    logger.info("Running problem with %d sensors %d targets: " + 
                "(checkpoint: %s, seed : %d, config: %s)", 
                num_sensors, num_targets, checkpoint, seed, config or 'default')

    if config is not None:
        wp.from_file(config)
        dp.from_file(config)

    if config is not None:
        basefile = os.path.splitext(os.path.basename(config))[0]
    else:
        basefile = 'default'

    save_dir = os.path.join(save_dir, basefile)

    actor = MCActor(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE, 
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size,
                    dp.dropout).to(device)

    critic = Critic(dp.MC_INPUT_SIZE,
                    dp.DEPOT_INPUT_SIZE,
                    dp.SN_INPUT_SIZE,
                    dp.hidden_size).to(device)

    if checkpoint is not None:
        path = os.path.join(checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if mode == 'train':
        actor.share_memory()
        critic.share_memory()

        processes = []
        num_processes = 7

        counter = mp.Value('i', 0)
        lock = mp.Lock()

        for rank in range(0, num_processes):
            p = mp.Process(target=train, args=(rank + 1, counter, log_dir, lock, seed, num_sensors, num_targets, actor, critic, save_dir, config))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

    else:
        test_data = WRSNDataset(num_sensors, num_targets, dp.test_size, seed + 2)
        test_loader = DataLoader(test_data, 1, False, num_workers=0)

        ret = validate(test_loader, decision_maker, (actor,) , wp, render, verbose, max_step=dp.max_step)
        lifetime, travel_dist = ret['lifetime_mean'], ret['travel_dist_mean']

        logger.info("Test metrics: Mean network lifetime %2.4f, mean travel distance: %2.4f",
                    lifetime, travel_dist)

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    parser = argparse.ArgumentParser(description="Mobile Charger Trainer")
    parser.add_argument('--num_sensors', '-ns', default=20, type=int)
    parser.add_argument('--num_targets', '-nt', default=10, type=int)
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'eval'])
    parser.add_argument('--config', '-cf', default=None, type=str)
    parser.add_argument('--checkpoint', '-cp', default=None, type=str)
    parser.add_argument('--save_dir', '-sd', default='checkpoints', type=str)
    parser.add_argument('--epoch_start', default=0, type=int)
    parser.add_argument('--render', '-r', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--seed', '-s', default=123, type=int)

    args = parser.parse_args()

    if args.config is not None:
        basefile = os.path.splitext(os.path.basename(args.config))[0]
    else:
        basefile = 'default'

    now = datetime.now()
    dt_str = now.strftime("%d_%m_%Y_%H_%M_%S")
    log_dir = "logs/{}_{}".format(basefile, dt_str)
    logger, writer = make_logger(log_dir)

    logger.info("Running on device: %s", device_str)
    logger.info("Log dir: %s", log_dir)

    torch.set_printoptions(sci_mode=False)

    torch.manual_seed(args.seed)
    
    np.random.seed(args.seed + 1)
    np.set_printoptions(suppress=True)

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    main(log_dir, **vars(args))