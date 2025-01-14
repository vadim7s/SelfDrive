#this is to see if pytorch can see GPU as a program
# in jupyter there is something between conda and pip which see different packages in same env
import torch
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    print("No GPU available. Training will run on CPU.")