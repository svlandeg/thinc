from pathlib import Path

from thinc.neural.ops import CupyOps, NumpyOps
from spacy.compat import cupy

if __name__ == "__main__":
    ops = CupyOps()

    PWD = Path(__file__).parent
    print("PWD", PWD)
    SRC = (PWD / "neural" / "toy.cu").open("r", encoding="utf8").read()

    hash_data_kernel = cupy.RawKernel(SRC, "hash_data") #, options=('-G'))

    num_blocks = 128
    threads_per_block = 128

    shape = (5,)
    ids = ops.allocate(shape, dtype="uint64")
    out = cupy.zeros((ids.shape[0], 4), dtype="uint32")

    out_size = 16
    in_size = 8

    seed = 0

    hash_data_kernel((num_blocks,), (threads_per_block,),
                     (out, ids, out_size, in_size, ids.shape[0], seed))

    print()
    print("IN", ids)
    print()
    print("OUT", out)


