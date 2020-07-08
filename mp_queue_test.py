import torch.multiprocessing as mp
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
import time

max_step = 10


def train(rank, model, Q):
    print('Do something')
    for step in range(0,max_step):
        Q.put(rank, True, 0.1)
        time.sleep(0.5)


def log(model, Q):
    print('Start logging')
    writer = SummaryWriter(filename_suffix=datetime.datetime.now().ctime().replace(" ", "_"))

    for step in range(0, max_step):
        while not Q.empty():
            data = Q.get()
            print(f'received data: {data}')
            writer.add_scalar('data', data, step)
        time.sleep(0.5)


if __name__ == '__main__':
    num_processes = 4
    model = torch.nn.Module()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    Q = mp.Queue()
    #summary_writer = SummaryWriter(filename_suffix=datetime.datetime.now().ctime().replace(" ", "_"))
    processes = []
    for rank in range(num_processes):
        if rank == 0:
        # p = mp.Process(target=log, args=(model, Q, summary_writer))
            p = mp.Process(target=log, args=(model, Q, ))
        else:
            p = mp.Process(target=train, args=(rank, model, Q))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()