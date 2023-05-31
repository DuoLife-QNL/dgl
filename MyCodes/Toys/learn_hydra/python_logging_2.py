# You'll need these imports in your own code
import logging
import logging.handlers
import hydra
import multiprocessing
from omegaconf import DictConfig

import logging
import logging.config
import logging.handlers
from torch.multiprocessing import Queue
import random
import threading
import time

import torch.multiprocessing as mp

def logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(__name__)
        logger.handle(record)


def worker_process(q):
    qh = logging.handlers.QueueHandler(q)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(qh)
    levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR,
              logging.CRITICAL]
    loggers = ['foo', 'foo.bar', 'foo.bar.baz',
               'spam', 'spam.ham', 'spam.ham.eggs']
    for i in range(100):
        lvl = random.choice(levels)
        logger = logging.getLogger(random.choice(loggers))
        logger.log(lvl, 'Message no. %d', i)

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(_cfg: DictConfig):
    mp.set_start_method('forkserver', force=True)
    q = mp.Queue()
    workers = []
    lp = threading.Thread(target=logger_thread, args=(q,))
    lp.start()
    # mp.set_start_method('spawn', force=True)
    for i in range(4):
        wp = mp.Process(target=worker_process, name='worker %d' % (i + 1), args=(q,))
        workers.append(wp)
        wp.start()
    # At this point, the main process could do some useful work of its own
    # Once it's done that, it can wait for the workers to terminate...
    for wp in workers:
        wp.join()
    # And now tell the logging thread to finish up, too
    q.put(None)
    lp.join()

def run():
    pass

@hydra.main(version_base=None, config_path='conf', config_name='config')
def warm_up(_cfg: DictConfig):
    # q = mp.Queue()
    # # workers = []
    # lp = threading.Thread(target=logger_thread, args=(q,))
    # lp.start()
    mp.set_start_method('spawn', force=True)
    # for i in range(4):
    #     wp = mp.Process(target=run, name='worker %d' % (i + 1), args=())
    #     workers.append(wp)
    #     wp.start()
    # # At this point, the main process could do some useful work of its own
    # # Once it's done that, it can wait for the workers to terminate...
    # for wp in workers:
    #     wp.join()
    # And now tell the logging thread to finish up, too
    # q.put(None)
    # lp.join()


if __name__ == '__main__':
    # warm_up()
    main()

# if __name__ == '__main__':
#     main()


# @hydra.main()
# def main(_cfg: DictConfig):
#     # multiprocessing.set_start_method('spawn')
#     q = Queue()
#     workers = []
#     lp = threading.Thread(target=logger_thread, args=(q,))
#     lp.start()
#     mp.spawn(worker_process, args=(q,), nprocs=4)
#     # for i in range(5):
#         # wp = Process(target=worker_process, name='worker %d' % (i + 1), args=(q,))
#         # workers.append(wp)
#         # wp.start()
#     # At this point, the main process could do some useful work of its own
#     # Once it's done that, it can wait for the workers to terminate...
#     # for wp in workers:
#     #     wp.join()
#     # And now tell the logging thread to finish up, too
#     q.put(None)
#     lp.join()

# if __name__ == '__main__':
#     main()