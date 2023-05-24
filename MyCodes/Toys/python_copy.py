import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
import time
class B(object):
    def __init__(self, x: int) -> None:
        self.x = x
    
    def add_one(self) -> None:
        self.x += 1
    
    def get_x(self) -> int:
        return self.x

class A(object):
    def __init__(self, b: B) -> None:
        self.b = b
    
    def add_one(self) -> None:
        self.b.add_one()
    
    def get_b(self) -> B:
        return self.b

def run(p_id, shared_a):
    if p_id == 0:
        shared_a.get_b().add_one()
    if p_id == 1:
        time.sleep(1)
    print("Process {} has x = {}".format(p_id, shared_a.get_b().get_x()))

def main():
    # Create a Manager object to create the shared namespace
    BaseManager.register('A', A)
    BaseManager.register('B', B)
    with BaseManager() as manager:
        # Create a shared B object
        
        # b = B(0)
        shared_a = manager.A(manager.B(0))
        # Spawn the processes
        mp.spawn(run, args=(shared_a, ), nprocs=2)

if __name__ == '__main__':
    main()