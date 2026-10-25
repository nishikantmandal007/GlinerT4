import sys,subprocess,torch
print(sys.version,torch.__version__,torch.cuda.get_device_name(),flush=True)
subprocess.run(['nvcc','--version'])
subprocess.run(['nvidia-smi'])
subprocess.run(['ls','/usr/local'])
print('ENV_DONE',flush=True)
