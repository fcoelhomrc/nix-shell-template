import time

import torch


def main():
    print(f"PyTorch Version: {torch.__version__}")

    # 1. Hardware Check
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        print(f"✅ CUDA is available!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")

    # 2. Stress Test Parameters
    # Creating two large matrices (approx 400MB each)
    size = 10000
    iterations = 1000

    print(f"\n🚀 Initializing {size}x{size} matrices on {device}...")
    try:
        # Generate random tensors directly on the device
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        print(f"🔥 Starting stress test ({iterations} iterations of matmul)...")
        start_time = time.time()

        for i in range(iterations):
            # Matrix multiplication: c = a * b
            c = torch.matmul(a, b)

            # Periodically print progress
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}/{iterations} complete.")

        # Ensure all kernels are finished before stopping the clock
        # if device.type == "cuda":
        #     torch.cuda.synchronize()

        end_time = time.time()
        print(f"\n✨ Test Complete!")
        print(f"Total time: {end_time - start_time:.2f} seconds")

    except RuntimeError as e:
        print(f"\n💥 Stress test failed: {e}")


if __name__ == "__main__":
    main()
