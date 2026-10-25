import subprocess
for cmd in (['nvcc','--version'],['nvidia-smi'],['ls','/usr/local'],['which','uv']):
 r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(r.stdout,flush=True)
print('RUNTIME_DONE',flush=True)
