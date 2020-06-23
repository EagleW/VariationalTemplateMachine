from benchmark_reader import Benchmark
from benchmark_reader import select_files

b = Benchmark()
files = select_files('./webnlg_challenge_2017/train')
b.fill_benchmark(files)