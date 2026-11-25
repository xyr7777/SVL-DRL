import random
import numpy as np
import pytest
torch = pytest.importorskip("torch")
from utils.reproducibility import set_global_seed, seed_worker


def test_seed_reproduces_random_sequences():
    set_global_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_global_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second

def test_seed_worker_is_repeatable():
    torch.manual_seed(99)
    seed_worker(0)
    a = (random.random(), np.random.rand())
    torch.manual_seed(99)
    seed_worker(0)
    b = (random.random(), np.random.rand())
    assert a == b
