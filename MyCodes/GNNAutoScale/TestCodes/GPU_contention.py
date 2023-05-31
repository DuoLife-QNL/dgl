import itertools
import torch
import multiprocessing
from multiprocessing import Process

from torch import Tensor
from torch_autoscale import Metric

numbers = [0, 1, 2, 3]


def run(metric_name: str, dev_id: int, data_push: Tensor, data_pull: Tensor):
    device = f'cuda:{dev_id}'
    metric = Metric(name="{}, GPU = {}".format(metric_name, dev_id))
    metric.start("transfer to gpu")
    # first move data_push to GPU
    data_push = data_push.to(device)
    metric.stop("transfer to gpu")
    data_push += 1

    metric.start("push")
    # from GPU to CPU
    data_push = data_push.to('cpu')
    metric.stop("push")
    
    metric.start("pull")
    # from CPU to GPU
    data_pull = data_pull.to(device)
    data_pull += 1
    metric.stop("pull")

    metric.print_metrics()
    return data_pull


def main():
    runs = 3
    metric = Metric()
    combinations = []
    for i in range(1, 5):
        for combination in itertools.combinations(numbers, i):
            combinations.append(combination)
    
    data_push = torch.randn(153026, 256)
    # data_push = torch.randn(58241, 256)
    data_pull = torch.randn(153026, 256)
    # 用fork比spawn快很多
    multiprocessing.set_start_method('fork')
    for _ in range(runs):
        for combination in combinations:
            workers = []
            metric_name = "GPUs = {}".format(combination)
            metric.start(metric_name + ', total_time')

            # 这个阶段几乎不花时间
            for i in range(len(combination)):
                wp = Process(target=run, name=f'worker {i + 1}', args=(metric_name, combination[i], data_push, data_pull))
                workers.append(wp)

            for wp in workers:
                wp.start()

            for wp in workers:
                wp.join()

            metric.stop(metric_name + ', total_time')
    metric.print_metrics()
    
if __name__ == '__main__':
    main()