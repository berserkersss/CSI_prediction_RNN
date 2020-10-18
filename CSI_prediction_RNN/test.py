import random
import numpy as np
from models.model import GRU
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import random
from queue import Queue

bs_queue = Queue(maxsize=0)

bs_queue.put(1)
print(bs_queue.qsize())
x = bs_queue.get()
print(bs_queue.qsize())
print(bs_queue.empty())


xx = []
xx.append(3)
print(xx)
xx.append(2)
print(xx)
xx.append(1)
print(min(xx))
xx.pop(-2)
print(len(xx))
xx.insert(0, 333)
print(xx)
