import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device() -> torch.device:
    """
    Determines and returns the appropriate device for PyTorch operations.
    
    Checks for CUDA availability and MPS (for Apple Silicon) as a fallback,
    otherwise defaults to CPU.
    
    Returns:
        torch.device: The selected device object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    # Fallback for Apple Silicon
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("MPS is available. Using Apple Silicon GPU.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA and MPS not available. Using CPU.")
        
    return device

# You can call this function once at the start of your script
# and pass the device object around.
DEVICE = get_device()

if __name__ == '__main__':
    # Example of how to use the device
    print(f"Selected device: {DEVICE}")

    # 1. Move a model to the selected device
    #    (assuming a simple model is defined)
    try:
        model = torch.nn.Sequential(torch.nn.Linear(10, 1))
        model.to(DEVICE)
        print("Model successfully moved to the device.")
    except Exception as e:
        print(f"Could not move model to device: {e}")

    # 2. Move a tensor to the selected device
    try:
        tensor = torch.randn(4, 10)
        tensor_on_device = tensor.to(DEVICE)
        print(f"Tensor successfully moved to the device. Current device: {tensor_on_device.device}")
    except Exception as e:
        print(f"Could not move tensor to device: {e}")
