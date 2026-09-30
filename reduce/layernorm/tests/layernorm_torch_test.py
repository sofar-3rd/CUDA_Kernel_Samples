"""Compare the CUB or Welford kernel with PyTorch on the same CUDA tensors.

Run with the project's CUDA-enabled Python environment:
    python reduce/layernorm/tests/layernorm_torch_test.py
    python reduce/layernorm/tests/layernorm_torch_test.py --benchmark
    python reduce/layernorm/tests/layernorm_torch_test.py --benchmark --shape 1024 2048
    python reduce/layernorm/tests/layernorm_torch_test.py --implementation welford --benchmark

Requires PyTorch, Ninja, a C++ compiler, and a CUDA Toolkit compatible with
PyTorch. Set CUDA_HOME if nvcc is not on PATH. The extension is built in
PyTorch's cache; the existing CMake build and CUDA source are unchanged.
"""

import argparse
from pathlib import Path
import statistics
import sys
import unittest

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


EPSILON = 1e-5
ATOL = RTOL = 2e-4
SHAPES = ((1024, 2048), (4096, 2048), (256, 8192), (4096, 1025),
          (4096, 1026), (64, 4096), (8, 1024))


def load_extension(implementation="cub"):
    root = Path(__file__).resolve().parents[1]
    defines = ["-DLAYERNORM_USE_WELFORD"] if implementation == "welford" else []
    return load(
        name=f"layernorm_{implementation}_torch_test",
        sources=[str(root / "tests/layernorm_torch_binding.cpp"),
                 str(root / f"src/layernorm_{implementation}.cu")],
        extra_include_paths=[str(root / "include")],
        extra_cflags=["-O3", *defines],
        extra_cuda_cflags=["-O3", "-lineinfo"],
    )


def make_inputs(shape):
    generator = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator)
    columns = torch.arange(shape[1], device="cuda", dtype=torch.float32)
    gamma = 0.5 + columns.remainder(17) / 16
    beta = -0.25 + columns.remainder(13) / 24
    return x, gamma, beta


class LayerNormTorchTest(unittest.TestCase):
    extension = None

    def compare(self, x, gamma, beta, epsilon=EPSILON):
        output = torch.full_like(x, float("nan"))
        expected = F.layer_norm(x, (x.shape[1],), gamma, beta, eps=epsilon)
        reference = F.layer_norm(x.double(), (x.shape[1],), gamma.double(),
                                 beta.double(), eps=epsilon).float()
        originals = [tensor.clone() for tensor in (x, gamma, beta)]
        self.extension.out(x, gamma, beta, output, epsilon)
        torch.testing.assert_close(output, expected, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(output, reference, atol=ATOL, rtol=RTOL)
        for tensor, original in zip((x, gamma, beta), originals):
            torch.testing.assert_close(tensor, original, atol=0, rtol=0)
        error = (output - expected).abs().max().item()
        print(f"[MATCH] shape={tuple(x.shape)} eps={epsilon:g} max_abs={error:.8g}",
              flush=True)

    def test_shapes_match_torch(self):
        for shape in ((1024, 2048), (33, 1026), (33, 1025), (33, 255),
                      (257, 256), (3, 1), (3, 2), (3, 4), (8, 16384),
                      (3, 31), (3, 32), (3, 33), (3, 127), (3, 128), (3, 129)):
            with self.subTest(shape=shape):
                self.compare(*make_inputs(shape))

    def test_constant_rows(self):
        x, gamma, beta = make_inputs((17, 2048))
        x.fill_(3.25)
        self.compare(x, gamma, beta)

    def test_large_offset_small_variance(self):
        x, gamma, beta = make_inputs((17, 2048))
        x[:, 0::2] = 9999
        x[:, 1::2] = 10001
        self.compare(x, gamma, beta)

    def test_distinct_row_statistics(self):
        x, gamma, beta = make_inputs((5, 2048))
        scales = torch.tensor([0.01, 0.1, 1, 10, 100], device="cuda")
        shifts = torch.tensor([-2, -1, 0, 1, 2], device="cuda")
        x.mul_(scales[:, None]).add_(shifts[:, None])
        self.compare(x, gamma, beta)

    def test_non_default_stream_and_epsilon(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x, gamma, beta = make_inputs((33, 1026))
            x.mul_(0.01)
            self.compare(x, gamma, beta, epsilon=0.125)
        stream.synchronize()

    def test_cuda_graph_recomputes_changed_input(self):
        x, gamma, beta = make_inputs((33, 1026))
        output = torch.empty_like(x)

        def call():
            self.extension.out(x, gamma, beta, output, EPSILON)
            return output

        graph, result = capture_calls(call, 2)
        x[:, 0].add_(10)
        output.fill_(float("nan"))
        graph.replay()
        expected = F.layer_norm(x, (x.shape[1],), gamma, beta, eps=EPSILON)
        torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)

    def test_rejects_unsupported_inputs(self):
        x, gamma, beta = make_inputs((2, 4))
        output = torch.empty_like(x)
        offset_x = torch.empty(9, device="cuda")[1:].view(2, 4)
        cases = (
            ("dtype", x.double(), gamma, beta, output),
            ("device", x.cpu(), gamma, beta, output),
            ("shape", x, gamma[:3], beta, output),
            ("strides", x.t(), gamma, beta, output),
            ("alignment", offset_x, gamma, beta, output),
            ("alias", x, gamma, beta, x),
            ("empty", x[:0], gamma, beta, output[:0]),
            ("output shape", x, gamma, beta, output[:1]),
            ("beta shape", x, gamma, beta[:3], output),
        )
        for name, *tensors in cases:
            with self.subTest(case=name), self.assertRaises(RuntimeError):
                self.extension.out(*tensors, EPSILON)
        for epsilon in (0, -1, float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaises(RuntimeError):
                self.extension.out(x, gamma, beta, output, epsilon)


def capture_calls(call, iterations):
    # Keep the final output alive while the graph owns intermediate allocations.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(100):
            result = call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            result = call()
    return graph, result


def benchmark(extension, shape, iterations, implementation="cub"):
    x, gamma, beta = make_inputs(shape)
    output = torch.empty_like(x)

    def custom():
        extension.out(x, gamma, beta, output, EPSILON)
        return output

    def native():
        return F.layer_norm(x, (shape[1],), gamma, beta, eps=EPSILON)

    torch.testing.assert_close(custom(), native(), atol=ATOL, rtol=RTOL)
    captures = [capture_calls(call, iterations) for call in (custom, native)]
    for _, result in captures:
        result.fill_(float("nan"))
    for graph, _ in captures:
        for _ in range(10):
            graph.replay()
    torch.cuda.synchronize()
    samples = [[], []]
    # Alternate order to reduce clock/temperature drift between implementations.
    for repeat in range(7):
        for index in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            captures[index][0].replay()
            stop.record()
            stop.synchronize()
            samples[index].append(start.elapsed_time(stop) * 1000 / iterations)
    for _, result in captures:
        torch.testing.assert_close(result, native(), atol=ATOL, rtol=RTOL)
    custom_us, torch_us = (statistics.median(sample) for sample in samples)
    print(f"[BENCH] M={shape[0]} N={shape[1]} {implementation}={custom_us:.3f} us "
          f"torch={torch_us:.3f} us speedup(torch/{implementation})={torch_us / custom_us:.3f}x",
          flush=True)


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--implementation", choices=("cub", "welford"), default="cub")
    parser.add_argument("--shape", nargs=2, type=positive_int, metavar=("M", "N"))
    parser.add_argument("--iterations", type=positive_int, default=100,
                        help="calls per CUDA graph (default: 100)")
    args = parser.parse_args()
    if args.shape and not args.benchmark:
        parser.error("--shape requires --benchmark")
    if not torch.cuda.is_available():
        parser.error("a CUDA-enabled PyTorch installation and NVIDIA GPU are required")
    print(f"torch={torch.__version__} cuda={torch.version.cuda} "
          f"gpu={torch.cuda.get_device_name()} dtype=float32 eps={EPSILON}", flush=True)
    with torch.inference_mode():
        extension = load_extension(args.implementation)
        LayerNormTorchTest.extension = extension
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(LayerNormTorchTest)
        result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
        if not result.wasSuccessful():
            return 1
        if args.benchmark:
            print(f"Timing: CUDA Graph, {args.iterations} calls/replay, median of 7; "
                  "excludes compilation, input setup and CPU dispatch.", flush=True)
            for shape in (tuple(args.shape),) if args.shape else SHAPES:
                benchmark(extension, shape, args.iterations, args.implementation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
