import random
import argparse
import os
import torch
import time
import numpy as np
from datetime import datetime

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import MCActor, Critic
from environment import WRSNEnv
from utils import NetworkInput, WRSNDataset, Point
from utils import Config, DrlParameters as dp, WrsnParameters as wp
from utils import logger, gen_cgrg, device, writer, make_logger, device_str

def decision_maker(mc_state, depot_state, sn_state, adj, mask, actor):
    actor.eval()
    mc_state = mc_state.unsqueeze(0)
    depot_state = depot_state.unsqueeze(0)
    sn_state = sn_state.unsqueeze(0)

    with torch.no_grad():
        logit = actor(mc_state, depot_state, sn_state, adj)

    logit = logit + mask.log()
    prob = F.softmax(logit, dim=-1)

    prob, action = torch.max(prob, 1)  # Greedy selection
    actor.train()
    return action.squeeze().item(), prob

def validate(w1, w2, data_loader, decision_maker, args=None, wp=wp,
             render=False, verbose=False, max_step=None, normalize=True,
             on_validation_begin=None, on_validation_end=None, 
             on_episode_begin=None, on_episode_end=None):
    if on_validation_begin is not None:
        on_validation_begin(*args)

    rewards = []
    mean_policy_losses = []
    mean_entropies = []
    times = [0]
    net_lifetimes = []
    mc_travel_dists = []
    mean_aggregated_ecrs = []
    mean_node_failures = []
    inf_lifetimes = []
    steps = []

    for idx, data in enumerate(data_loader):
        if verbose: print("Test %d" % idx)

        sensors, targets = data
        
        if on_episode_begin is not None:
            on_episode_begin(*args)

        env = WRSNEnv(sensors=sensors.squeeze(), 
                      targets=targets.squeeze(), 
                      wp=wp,
                      normalize=normalize)

        mc_state, depot_state, sn_state = env.reset()
        
        mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
        depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
        sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

        adj = env.get_sn_adjacency()

        aggregated_ecrs = []
        node_failures = []
        total_reward = 0

        mask = torch.ones(env.action_space.n).to(device)

        max_step = max_step or dp.max_step

        for step in range(max_step):
            if render:
                env.render()

            if args is not None:
                action, prob = decision_maker(mc_state, depot_state, sn_state, adj, mask, *args)
            else:
                action, prob = decision_maker(mc_state, depot_state, sn_state, adj, mask)
            
            mask[env.last_action] = 1.0
            (mc_state, depot_state, sn_state), reward, done, _ = env.step(action)
            # mask[env.last_action] = 0.0
            # mask[0] = 1.0
                
            mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
            depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
            sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

            if verbose: 
                print("Step %d: Go to %d (prob: %2.4f) => reward (%2.4f, %2.4f)\n" % 
                      (step, action, prob, reward[0], reward[1]))
                print("Aggregated ecr %2.4f, node failures %2.4f\n" % 
                       (env.net.aggregated_ecr, env.net.node_failures))
                print("Current network lifetime: %2.4f, mc_battery: %2.4f \n\n" % 
                       (env.net.network_lifetime, env.mc.cur_energy))

            total_reward += w1 * reward[0] - w2 * reward[1]
            aggregated_ecrs.append(env.net.aggregated_ecr)
            node_failures.append(env.net.node_failures)

            if done:
                if verbose: print("End episode! Press any button to continue...")
                if render: env.render()
                if render or verbose: input()
                env.close()
                break

            if render:
                time.sleep(0.5)
                # pass

        if on_episode_end is not None:
            on_episode_end(*args)
            
        rewards.append(total_reward)
        steps.append(step)
        net_lifetimes.append(env.get_network_lifetime())
        mc_travel_dists.append(env.get_travel_distance())
        mean_aggregated_ecrs.append(np.mean(aggregated_ecrs))
        mean_node_failures.append(np.mean(node_failures))
        inf_lifetimes.append(env.get_network_lifetime() 
                             if done else np.inf)

    ret = {}
    ret['inf_lifetimes'] = inf_lifetimes
    ret['lifetime_mean'] = np.mean(net_lifetimes)
    ret['lifetime_std'] = np.std(net_lifetimes)
    ret['travel_dist_mean'] = np.mean(mc_travel_dists)
    ret['travel_dist_std'] = np.std(mc_travel_dists)
    ret['aggregated_ecr_mean'] = np.mean(mean_aggregated_ecrs)
    ret['aggregated_ecr_std'] = np.std(mean_aggregated_ecrs)
    ret['node_failures_mean'] = np.mean(mean_node_failures)
    ret['node_failures_std'] = np.std(mean_node_failures)
    ret['step_mean'] = np.mean(steps)
    ret['reward_mean'] = np.mean(rewards)
    ret['reward_std'] = np.std(rewards)
    ret['k_bit'] = wp.k_bit
    ret['E_s'] = wp.E_s
    ret['E_mc'] = wp.E_mc

    if on_validation_end is not None:
        on_validation_end(*args)
    return ret

def train(w1, w2, actor, critic, train_data, valid_data, save_dir, 
          epoch_start_idx=0, wp=wp, dp=dp):
    logger.info("Begin training phase")
    train_loader = DataLoader(train_data, 1, True, num_workers=0)
    valid_loader = DataLoader(valid_data, 1, False, num_workers=0)

    actor_optim = optim.Adam(actor.parameters(), dp.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), dp.critic_lr)

    best_params = None
    best_reward = np.inf
    sample_inp = None

    for epoch in range(epoch_start_idx, dp.num_epoch):
        logger.info("Start epoch %d" % epoch)
        actor.train()
        critic.train()

        epoch_start = time.time()
        start = epoch_start

        mean_policy_losses = []
        mean_entropies = []
        mean_aggregated_ecrs = []
        times = [0]
        net_lifetimes = []
        mc_travel_dists = []
        steps = []
        mean_rewards = []
        total_rewards = []

        for idx, data in enumerate(train_loader):
            sensors, targets = data

            env = WRSNEnv(sensors=sensors.squeeze(), 
                          targets=targets.squeeze(),
                          wp=wp, 
                          normalize=True)
            num_sensors = env.net.num_sensors

            mc_state, depot_state, sn_state = env.reset()
            mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
            depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
            sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

            adj = env.get_sn_adjacency()

            values = []
            log_probs = []
            rewards = []
            entropies = []
            aggregated_ecrs = []
            actions = []

            mask = torch.ones(env.action_space.n).to(device)

            for step in range(dp.max_step):
                mc_state = mc_state.unsqueeze(0)
                depot_state = depot_state.unsqueeze(0)
                sn_state = sn_state.unsqueeze(0)
                
                if sample_inp is None:
                    sample_inp = (mc_state, depot_state, sn_state)

                logit = actor(mc_state, depot_state, sn_state, adj)
                logit = logit + mask.log()

                prob = F.softmax(logit, dim=-1)

                value = critic(mc_state, depot_state, sn_state, adj)

                m = torch.distributions.Categorical(prob)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                action = m.sample()
                logp = m.log_prob(action)
                entropy = m.entropy()

                mask[env.last_action] = 1.0
                (mc_state, depot_state, sn_state), reward, done, info = env.step(action.squeeze().item())
                # mask[env.last_action] = 0.0
                # mask[0] = 1 # always allow MC staying at depot

                mc_state = torch.from_numpy(mc_state).to(dtype=torch.float32, device=device)
                depot_state = torch.from_numpy(depot_state).to(dtype=torch.float32, device=device)
                sn_state = torch.from_numpy(sn_state).to(dtype=torch.float32, device=device)

                values.append(value) 
                rewards.append(reward)
                log_probs.append(logp)
                entropies.append(entropy)
                aggregated_ecrs.append(env.net.aggregated_ecr)
                actions.append(action.squeeze().item())

                if done:
                    env.close()
                    break
            steps.append(step)

            R = torch.zeros(1, 1).to(device)
            if not done:
                value = critic(mc_state.unsqueeze(0), 
                               depot_state.unsqueeze(0), 
                               sn_state.unsqueeze(0), adj)
                R = value.detach() if value is not None else value

            values.append(R)
            
            net_lifetimes.append(env.get_network_lifetime())
            mc_travel_dists.append(env.get_travel_distance())
            mean_aggregated_ecrs.append(np.mean(aggregated_ecrs))
            
            gae = torch.zeros(1, 1).to(device)
            policy_losses = torch.zeros(len(rewards))
            value_losses = torch.zeros(len(rewards))
            total_reward = 0

            R = values[-1]
            
            for i in reversed(range(len(rewards))):
                reward = w1 * rewards[i][0] - w2 * rewards[i][1] 
                total_reward += reward
                R = dp.gamma * R + reward
                advantage = R - values[i]

                value_losses[i] = 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta_t = reward + dp.gamma * \
                    values[i + 1] - values[i]
                gae = gae * dp.gamma * dp.gae_lambda + delta_t

                policy_losses[i] = -log_probs[i] * gae.detach() - \
                                     dp.entropy_coef * entropies[i]

            actor_optim.zero_grad()
            policy_losses.sum().backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), dp.max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            value_losses.sum().backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), dp.max_grad_norm)
            critic_optim.step()

            with torch.no_grad():
                pl = torch.mean(policy_losses).item()
                mean_policy_losses.append(pl)
                vl = torch.mean(value_losses).item()

                e = torch.mean(torch.Tensor(entropies)).item()
                mean_entropies.append(e)

                r = np.mean([(w1 * reward[0] - w2 * reward[1]) for reward in rewards])
                mean_rewards.append(r)
                total_rewards.append(total_reward)

            if (idx + 1) % dp.log_size == 0:
                end = time.time()
                times.append(end-start)
                start = end

                mm_policy_loss = np.mean(mean_policy_losses[-dp.log_size:])
                mm_entropies = np.mean(mean_entropies[-dp.log_size:])
                m_net_lifetime = np.mean(net_lifetimes[-dp.log_size:])
                m_mc_travel_dist = np.mean(mc_travel_dists[-dp.log_size:])
                mm_rewards = np.mean(mean_rewards[-dp.log_size:])
                m_steps = np.mean(steps[-dp.log_size:])
                m_total_rewards = np.mean(total_rewards[-dp.log_size:])
                mm_aggregated_ecr = np.mean(mean_aggregated_ecrs[-dp.log_size:])
                bins = np.array(list(range(num_sensors+2)))
                values = np.array(actions).astype(float).reshape(-1)
                counts, limits = np.histogram(values, bins=bins)
                sum_sq = values.dot(values)

                global_step = (idx + epoch * len(train_loader)) / dp.log_size
                writer.add_scalar('batch/policy_loss', mm_policy_loss, global_step)
                writer.add_scalar('batch/entropy', mm_entropies, global_step)
                writer.add_scalar('batch/net_lifetime', m_net_lifetime, global_step)
                writer.add_scalar('batch/mc_travel_dist', m_mc_travel_dist, global_step)
                writer.add_scalar('batch/reward_per_step', mm_rewards, global_step)
                writer.add_scalar('batch/num_steps', m_steps, global_step)
                writer.add_scalar('batch/total_rewards', m_total_rewards, global_step)
                writer.add_histogram_raw('batch/action_hist', 
                                         min=values.min(),
                                         max=values.max(),
                                         num=len(values),
                                         sum=values.sum(),
                                         sum_squares=sum_sq,
                                         bucket_limits=limits[1:].tolist(),
                                         bucket_counts=counts.tolist(),
                                         global_step=global_step)

                msg = '\tBatch %d/%d, mean_policy_losses: %2.3f, ' + \
                    'mean_net_lifetime: %2.4f, mean_mc_travel_dist: %2.4f, ' + \
                    'mean_rewards: %2.4f, total_rewards: %2.4f, mean_steps: %2.4f, mean_ecr: %2.4f ' + \
                    'mean_entropies: %2.4f, took: %2.4fs'
                logger.info(msg % (idx, len(train_loader), mm_policy_loss, 
                                   m_net_lifetime, m_mc_travel_dist,
                                   mm_rewards, m_total_rewards, m_steps, mm_aggregated_ecr,
                                   mm_entropies, times[-1]))

        mm_policy_loss = np.mean(mean_policy_losses)
        mm_entropies = np.mean(mean_entropies)
        m_net_lifetime = np.mean(net_lifetimes)
        m_mc_travel_dist = np.mean(mc_travel_dists)

        # Save the weights
        epoch_dir = os.path.join(save_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        res = validate(w1, w2, valid_loader, decision_maker, (actor,), wp, max_step=dp.max_step)
        m_net_lifetime_valid = res['lifetime_mean'] 
        m_mc_travel_dist_valid = res['travel_dist_mean']

        writer.add_scalar('epoch/policy_loss', mm_policy_loss, epoch)
        writer.add_scalar('epoch/entropy', e, epoch)
        writer.add_scalars('epoch/net_lifetime', 
                           {'train': m_net_lifetime,
                            'valid': m_net_lifetime_valid},
                           epoch)
        writer.add_scalars('epoch/mc_travel_dist', 
                           {'train': m_mc_travel_dist,
                            'valid': m_mc_travel_dist_valid},
                           epoch)

        # Save best model parameters
        if m_net_lifetime_valid < best_reward:

            best_reward = m_net_lifetime_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        msg = 'Epoch %d: mean_policy_losses: %2.3f, ' + \
            'mean_net_lifetime: %2.4f, mean_mc_travel_dist: %2.4f, ' + \
            'mean_entropies: %2.4f, m_net_lifetime_valid: %2.4f, ' + \
            'took: %2.4fs, (%2.4f / 100 batches)\n'
        logger.info(msg % (epoch, mm_policy_loss, m_net_lifetime, 
                           m_mc_travel_dist, mm_entropies, m_net_lifetime_valid,
                           time.time() - epoch_start, np.mean(times)))

    #writer.add_graph(actor, sample_inp)

def main(w1=1, w2=0, num_sensors=20, num_targets=10, config=None,
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
    save_dir = os.path.join(save_dir, 'w_%2.2f_%2.2f' % (w1, w2))

    last_checkpoint_dir = save_dir

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
        logger.info("Generating training dataset")
        train_data = WRSNDataset(num_sensors, num_targets, dp.train_size, seed)
        logger.info("Generating validation dataset")
        valid_data = WRSNDataset(num_sensors, num_targets, dp.valid_size, seed + 1)
        train(w1, w2, actor, critic, train_data, valid_data, save_dir, epoch_start, wp, dp)

    test_data = WRSNDataset(num_sensors, num_targets, dp.test_size, seed + 2)
    test_loader = DataLoader(test_data, 1, False, num_workers=0)

    ret = validate(w1, w2, test_loader, decision_maker, (actor,) , wp, render, verbose, max_step=dp.max_step)
    lifetime, travel_dist = ret['lifetime_mean'], ret['travel_dist_mean']

    logger.info("Test metrics: Mean network lifetime %2.4f, mean travel distance: %2.4f",
                lifetime, travel_dist)

    return last_checkpoint_dir

if __name__ == '__main__':
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
    torch.manual_seed(args.seed + 11)
    np.random.seed(args.seed + 12)
    np.set_printoptions(suppress=True)

    # main(0.3, 0.7, **vars(args))
    T = 100
    w2_list = np.arange(T + 1)/T
    last_checkpoint_dir = None
    for i in range(0, T + 1):
        if i == 0:
            # The first subproblem is trained from scratch or loaded from checkpoint.
            last_checkpoint_dir = main(1, 0, **vars(args))
        else:
            # Parameter transfer, train based on the parameters of the previous subproblem
            args.checkpoint = last_checkpoint_dir
            last_checkpoint_dir = main(1 - w2_list[i], w2_list[i], **vars(args))

