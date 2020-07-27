import itertools
import os
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
n_train_processes = 5  # Number of workers
learning_rate = 0.00002  # LR of 0.00002 works well with deep sea treasure env.
update_interval = 5  # nr of steps before actor critic network update
gamma = 0.98  # discount factor
max_train_ep = 1000  # maximum episodes for training
max_test_ep = 1000  # maximum episodes for testing/logging
goal_size = 10  # weight range for agents
goal_partition = 5  # step size for weight range
log_interval = 1000  # interval for log messages
queue_read_interval = 3000  # max items read from queue
run_timestamp = datetime.datetime.now().ctime().replace(" ", "_")


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc_pi = nn.Linear(256, 4)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


def train(rank, weights, data_pool ):
    print(f'agent_{rank} starting...', flush=True)
    local_model = ActorCritic()

    optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)

    env = DeepSeaTreasureEnv()

    for n_epi in range(max_train_ep):
        if n_epi % log_interval == 0:
            print(f'agent {rank} starting epoch {n_epi}', flush=True)
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
                td_target_lst.append([np.double(R)])
            td_target_lst.reverse()

            s_batch, a_batch, td_target = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                torch.tensor(td_target_lst)
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch, softmax_dim=1)
            pi_a = pi.gather(1, a_batch)
            loss = -torch.log(pi_a) * advantage.float().detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.float().detach())

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

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
                print('queue in the queue, waiting....', flush=True)
                time.sleep(0.1)

        time.sleep(0)  # Yield remaining time
        # if n_epi % 100 == 0:
        #     time.sleep(1)

    env.close()

    while not data_pool.empty():
        print(f'Agent {rank}: not all data consumed, waiting...', flush=True)
        time.sleep(10)

    print("Training process {} reached maximum episode.".format(rank))

    torch.save(local_model.state_dict(), f'./agents/{run_timestamp}_agent_{rank}.ai')

def data_complete(epi_list, epoch):
    for i in range(n_train_processes):
        if epi_list[i, epoch] != epoch:
            return False

    return True

def first_missing(epi_list, epoch):
    if epoch >= max_test_ep:
        return -1

    for i in range(n_train_processes):
        if epi_list[i, epoch] != epoch:
            return i

    return -1

def test(weights, data_pools):
    summary_writer = SummaryWriter(filename_suffix=run_timestamp)

    epi_list = np.empty((n_train_processes+1, max_test_ep), dtype=int)
    reward_list = np.empty((n_train_processes+1, max_test_ep), dtype=tuple)
    loss_list = np.empty((n_train_processes+1, max_test_ep), dtype=float)
    pi_list = np.empty((n_train_processes+1, max_test_ep), dtype=float)
    advantage_list = np.empty((n_train_processes+1, max_test_ep), dtype=float)

    i_epi = 0
    while i_epi < (max_test_ep):
        test_iteration = 0
        while data_complete(epi_list, i_epi) and test_iteration < queue_read_interval:
            # receive rewards
            if i_epi % log_interval == 0:
                print(f'processing data for epoch {i_epi}', flush=True)

            reward_set = reward_list[:, i_epi]
            reward_set = list(filter(None, reward_set))
            if reward_set:
                hypervolume = hv.hypervolume(reward_set, [100, 100])
                # if i_epi % log_interval == 0:
                print(f'Hypervolume indicator for episode {i_epi}: {hypervolume} for {len(reward_set)} points', flush=True)
                summary_writer.add_scalar("hypervolume_indicator", hypervolume, i_epi)
            else:
                print('reward_set is empty')

            for agent_rank in range(1, n_train_processes + 1):
                summary_writer.add_scalar(f'agent_{agent_rank}_weight_1', weights[agent_rank][0], i_epi)
                summary_writer.add_scalar(f'agent_{agent_rank}_weight_2', weights[agent_rank][1], i_epi)

                if loss_list[agent_rank][i_epi]:
                    summary_writer.add_scalar(f'agent_{agent_rank}_loss', loss_list[agent_rank][i_epi], i_epi)

                if pi_list[agent_rank][i_epi]:
                    summary_writer.add_scalar(f'agent_{agent_rank}_pi', pi_list[agent_rank][i_epi], i_epi)

                if advantage_list[agent_rank][i_epi]:
                    summary_writer.add_scalar(f'agent_{agent_rank}_advantage', advantage_list[agent_rank][i_epi], i_epi)

                if reward_list[agent_rank][i_epi]:
                    summary_writer.add_scalar(f'agent_{agent_rank}_reward_1', reward_list[agent_rank][i_epi][0], i_epi)

                if reward_list[agent_rank][i_epi]:
                    summary_writer.add_scalar(f'agent_{agent_rank}_reward_2', reward_list[agent_rank][i_epi][1], i_epi)

            test_iteration += 1
            i_epi += 1
            if not i_epi < max_test_ep:
                break

        print(f'Waiting for epoch {i_epi} to be completed by all workers', flush=True)
        print(f'Waiting for worker: {first_missing(loss_list, i_epi)}', flush=True)

        for i in range(n_train_processes):  # iterate over worker queues
            queue_not_empty = True

            read_counter = 0
            if epi_list[i, max_test_ep-1] != (max_test_ep-1):  # only read from unfinished workers
                while queue_not_empty and read_counter < queue_read_interval:
                    try:
                        data = data_pools[i].get_nowait()

                        n_epi = data[0]
                        rank = data[1]
                        loss = data[2]
                        pi = data[3]
                        advantage = data[4]
                        avg_reward_1 = data[5]
                        avg_reward_2 = data[6]

                        epi_list[rank][n_epi] = n_epi
                        reward_list[rank][n_epi] = (avg_reward_1, avg_reward_2)
                        loss_list[rank][n_epi] = loss
                        pi_list[rank][n_epi] = pi
                        advantage_list[rank][n_epi] = advantage

                        read_counter += 1
                    except queue.Empty:
                        queue_not_empty = False
                        if read_counter > 0:
                            print(f'read_queue for agent {i}, got {read_counter} datapoints', flush=True)
                            print(f'last datapoint {data}', flush=True)
                if read_counter == queue_read_interval:
                    print(f'read queue for agent {i}, got {read_counter} datapoints', flush=True)
                    print(f'last datapoint {data}', flush=True)

if __name__ == '__main__':
    mp.set_start_method('spawn')  # Deal with fork issues
    try:
        os.mkdir('./agents',)
    except FileExistsError:
        pass
    global_model = ActorCritic()
    global_model.share_memory()
    data_pools = []

    for i in range(n_train_processes):
        data_pools.append(mp.Queue())

    weights = np.array(list(itertools.product(range(0, goal_size, int(goal_size / goal_partition)),
                                     range(0, goal_size, int(goal_size / goal_partition)))))

    #randomly sample from weightspace
    selected_weights = np.random.choice(len(weights), n_train_processes+1, replace=False)

    processes = []
    for rank in range(0, n_train_processes + 1):  # + 1 for test process
        if rank == 0:
            p = mp.Process(target=test, args=(weights[selected_weights], data_pools))
        else:
            p = mp.Process(target=train, args=(rank-1, weights[selected_weights][rank-1], data_pools[rank-1]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
