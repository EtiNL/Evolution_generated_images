import pycuda.driver as cuda
import pycuda.autoinit

# Function to get the maximum number of threads per block
def get_max_threads_per_block():
    try:
        cuda.init()  # Initialize the CUDA driver
        device = cuda.Device(0)  # Get the first CUDA device
        max_threads_per_block = device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        return max_threads_per_block
    except cuda.Error as e:
        print(f"CUDA Error: {e}")
        return None

if __name__ == '__main__':
    max_threads = get_max_threads_per_block()
    if max_threads is not None:
        print(f"Maximum threads per block: {max_threads}")
    else:
        print("Failed to get the maximum threads per block.")