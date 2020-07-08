import torch.multiprocessing as mp
import torch

def train(model):


if __name__ == '__main__':
    num_processes = 4
    model = torch.nn.Module()
    # NOTE: this is required for the ``fork`` method to work
    model.share_memory()
    mp.SimpleQueue()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()