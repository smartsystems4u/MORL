import itertools
# import torch.multiprocessing as mp
import queue
import datetime
import multiprocessing as mp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from deap.tools.indicator import hv
from torch.utils.tensorboard import SummaryWriter
from deep_sea_treasure_env.deep_sea_treasure_env import DeepSeaTreasureEnv

# Hyperparameters
n_train_processes = 5
learning_rate = 0.00002
update_interval = 5
gamma = 0.98
max_train_ep = 2000
max_test_ep = 2000
goal_size = 10


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(3, 256).float()
        self.fc_pi = nn.Linear(256, 4).float()
        self.fc_v = nn.Linear(256, 1).float()

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x)).float()
        x = self.fc_pi(x).float()
        prob = F.softmax(x, dim=softmax_dim).float()
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x)).float()
        v = self.fc_v(x).float()
        return v


def train(rank, weights, data_pool ):
    print(f'agent_{rank} starting...')
    local_model = ActorCritic()
    #local_model.load_state_dict(global_model.state_dict())

    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

    env = DeepSeaTreasureEnv()

    for n_epi in range(max_train_ep):
        epoch_reward1 = []
        epoch_reward2 = []
        epoch_loss = []
        epoch_pi = []
        epoch_v = []
        epoch_advantage = []
        done = False
        s = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            s_prime = None
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)

                s = s_prime
                if done:
                    break

            s_final = torch.tensor(s_prime, dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + weights[0] * reward[0] + weights[1] * reward[1]
                td_target_lst.append([R])
            td_target_lst.reverse()

            s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                torch.tensor(td_target_lst)
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())

            optimizer.zero_grad()
            loss.mean().backward()
            # for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            #     global_param._grad = local_param.grad
            optimizer.step()
            # local_model.load_state_dict(global_model.state_dict())

            # share scalarized average score
            #scalarized_reward = weights[0] * (sum([reward[0] for reward in r_lst]) / len(r_lst)) + \
            #    weights[1] * (sum([reward[1] for reward in r_lst]) / len(r_lst))
            avg_reward_1 = sum([reward[0] for reward in r_lst]) / len(r_lst)
            avg_reward_2 = sum([reward[1] for reward in r_lst]) / len(r_lst)
            epoch_reward1.append(avg_reward_1)
            epoch_reward2.append(avg_reward_2)
            epoch_v.append(local_model.v(s_batch).detach().mean())
            epoch_pi.append(pi.argmax(dim=1).median())
            epoch_advantage.append(advantage.detach().mean())
            epoch_loss.append(loss.detach().mean())

        sent = False
        while not sent:
            try:
                data_pool.put_nowait((n_epi, rank, sum(epoch_loss) / len(epoch_loss),
                                    sum(epoch_pi) / len(epoch_pi),
                                    sum(epoch_advantage) / len(epoch_advantage),
                                    sum(epoch_reward1) / len(epoch_reward1),
                                    sum(epoch_reward2) / len(epoch_reward2)))

                sent = True
            except queue.Full:
                print('queue in the queue, waiting....')
                time.sleep(0.1)

        time.sleep(0.1)  # Yield remaining time
        if n_epi % 100 == 0:
            time.sleep(1)

    env.close()

    # while not data_pool.empty():
    #     print(f'not all data consumed, waiting...')
    #     time.sleep(1)

    print("Training process {} reached maximum episode.".format(rank))

def data_complete(loss_list, epoch):
    for i in range(1, n_train_processes+1):
        if loss_list[i, epoch] == 0:
            return False

    return True

def test(weights, data_pool):
    summary_writer = SummaryWriter(filename_suffix=datetime.datetime.now().ctime().replace(" ", "_"))
    time.sleep(10)
    # while data_pool.empty():
    #     print('Start Logging: Waiting for data...')
    #     time.sleep(1)

    reward_list = np.empty((100, max_test_ep), dtype=tuple)
    loss_list = np.empty((100, max_test_ep), dtype=float)
    pi_list = np.empty((100, max_test_ep), dtype=float)
    advantage_list = np.empty((100, max_test_ep), dtype=float)

    for i in range(0, max_test_ep-1):
        # receive rewards
        print(f'checking for new data for epoch {i}')
        while not data_complete(loss_list, i):
            try:
                data = data_pool.get_nowait()

                print(f'got data: {data}')
                n_epi = data[0]
                rank = data[1]
                loss = data[2]
                pi = data[3]
                advantage = data[4]
                avg_reward_1 = data[5]
                avg_reward_2 = data[6]

                reward_list[rank][n_epi] = (avg_reward_1, avg_reward_2)
                loss_list[rank][n_epi] = loss
                pi_list[rank][n_epi] = pi
                advantage_list[rank][n_epi] = advantage
            except queue.Empty:
                print('Log Epoch: Waiting for data...')
                time.sleep(1)


        # calculate hypervolume
        print('calculating hypervolume')
        reward_set = reward_list[:, i]
        reward_set = list(filter(None, reward_set))
        if reward_set:
            hypervolume = hv.hypervolume(reward_set, [100, 100])
            print(f'Hypervolume indicator for episode {i}: {hypervolume} for {len(reward_set)} points')
            summary_writer.add_scalar("hypervolume_indicator", hypervolume, i)
        else:
            print('reward_set is empty')

        for agent_rank in range(1, n_train_processes+1):
            summary_writer.add_scalar(f'agent_{agent_rank}_weight_1', weights[agent_rank][0], i)
            summary_writer.add_scalar(f'agent_{agent_rank}_weight_2', weights[agent_rank][1], i)

            if loss_list[agent_rank][i]:
                summary_writer.add_scalar(f'agent_{agent_rank}_loss',loss_list[agent_rank][i], i)
                # summary_writer.add_scalar(
                #     f'agent_{agent_rank}_weights_{weights[agent_rank][0]}_{weights[agent_rank][1]}_loss',
                #     loss_list[agent_rank][i], i)

            if pi_list[agent_rank][i]:
                summary_writer.add_scalar(f'agent_{agent_rank}_pi', pi_list[agent_rank][i], i)
                # summary_writer.add_scalar(
                #     f'agent_{agent_rank}_weights_{weights[agent_rank][0]}_{weights[agent_rank][1]}_pi',
                #     pi_list[agent_rank][i], i)

            if advantage_list[agent_rank][i]:
                summary_writer.add_scalar(f'agent_{agent_rank}_advantage', advantage_list[agent_rank][i], i)
                # summary_writer.add_scalar(
                #     f'agent_{agent_rank}_weights_{weights[agent_rank][0]}_{weights[agent_rank][1]}_advantage',
                #     advantage_list[agent_rank][i], i)

            if reward_list[agent_rank][i]:
                summary_writer.add_scalar(f'agent_{agent_rank}_reward_1', reward_list[agent_rank][i][0], i)
                # summary_writer.add_scalar(
                #     f'agent_{agent_rank}_weights_{weights[agent_rank][0]}_{weights[agent_rank][1]}_reward_1',
                #     reward_list[agent_rank][i][0], i)

            if reward_list[agent_rank][i]:
                summary_writer.add_scalar(f'agent_{agent_rank}_reward_2', reward_list[agent_rank][i][1], i)
                # summary_writer.add_scalar(
                #     f'agent_{agent_rank}_weights_{weights[agent_rank][0]}_{weights[agent_rank][1]}_reward_2',
                #     reward_list[agent_rank][i][1], i)


if __name__ == '__main__':
    # mp.set_start_method('spawn')  # Deal with fork issues
    global_model = ActorCritic()
    global_model.share_memory()
    data_pool = mp.Queue()

    weights = np.array(list(itertools.product(range(0, goal_size, int(goal_size / n_train_processes)),
                                     range(0, goal_size, int(goal_size / n_train_processes)))))

    #randomly sample from weightspace
    selected_weights = np.random.choice(len(weights), n_train_processes+1, replace=False)

    processes = []
    for rank in range(0, n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(weights[selected_weights], data_pool))
        else:
            p = mp.Process(target=train, args=(rank, weights[selected_weights][rank-1], data_pool))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
