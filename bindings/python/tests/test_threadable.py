import unittest
from concurrent import futures
import threading
import torch
from safetensors.torch import save_file, load_file
import time
import os
import copy
import hashlib


class TestCase(unittest.TestCase):
    def test_threadable_available(self):
        rand_huge_tensor = {
            "tensor_a": torch.randn(2000, 20000),
            "tensor_b": torch.randint(0, 128, (20000, 2000), dtype=torch.int8),
        }
        lock = threading.Lock()
        executor = futures.ThreadPoolExecutor()
        # create thread now
        executor.submit(lambda: None)
        executor.submit(lambda: None)

        def counting_thread():
            lock.acquire()  # ensure saving thread starting before counting thread
            st = time.monotonic()
            # emulating computing bound thread
            time.sleep(0.5)
            x = "0"
            for i in range(20_000_000):
                x += f"{i} counted..."
            ed = time.monotonic()
            return ed - st

        def saving_thread(save_func):
            lock.release()
            file_name = "./out_rand_huge_tensor_for_threadable_test.safetensors"
            # st = time.monotonic()
            save_func(rand_huge_tensor, file_name)
            # print(f"{save_func} save cost {time.monotonic() - st}")
            os.remove(file_name)

        lock.acquire()
        f1 = executor.submit(saving_thread, save_file)
        f2 = executor.submit(counting_thread)
        f1.result()
        cost1 = f2.result()

        f1 = executor.submit(saving_thread, save_file)
        f2 = executor.submit(counting_thread)
        f1.result()
        cost2 = f2.result()
        # print(cost1, cost2)
        self.assertLess(cost2, cost1)

    def test_consistancy(self):
        rand_huge_tensor = {
            "tensor_a": torch.randn(1000, 20000),
            "tensor_b": torch.randint(0, 128, (10000, 2000), dtype=torch.int8),
        }
        backup_tensor = copy.deepcopy(rand_huge_tensor)
        fn1 = "./out_rand_huge_tensor_for_consistancy_test1.safetensors"
        fn2 = "./out_rand_huge_tensor_for_consistancy_test2.safetensors"
        save_file(rand_huge_tensor, fn1)
        with open(fn1, "rb") as f:
            hsh1 = hashlib.md5(f.read()).hexdigest()
        with open(fn2, "rb") as f:
            hsh2 = hashlib.md5(f.read()).hexdigest()
        self.assertEqual(hsh1, hsh2)
        saved_tensor = load_file(fn2)
        for k in backup_tensor:
            self.assertTrue(torch.equal(saved_tensor[k], backup_tensor[k]))


if __name__ == "__main__":
    unittest.main()
