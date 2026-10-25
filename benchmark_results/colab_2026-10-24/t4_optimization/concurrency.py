"""Same 290 chunks and batch boundaries, distributed over independent model/stream pairs."""
import concurrent.futures,inspect,os
select('base')
models=[fused];streams=[torch.cuda.Stream()]
for slot in range(1,4):
 model=AutoExtractor.from_pretrained(name,map_location='cuda',quantize=True,attention_backend='disentangled_flash')
 model.eval();model.eval=types.MethodType(lambda self:self,model)
 assert install_bulk_token_id_lookup(model.processor.tokenizer)
 models.append(model);streams.append(torch.cuda.Stream())
 emit('MODEL_READY',{'slot':slot,'allocated_mb':torch.cuda.memory_allocated()/2**20})
# Keep original 16-chunk boundaries; joining assigned batches does not change the shapes seen by a model.
batches=[list(range(i,min(i+16,len(chunks)))) for i in range(0,len(chunks),16)]
@torch.inference_mode()
def worker(slot,indices):
 with torch.cuda.stream(streams[slot]):
  out=[models[slot].batch_extract_entities([chunks[i] for i in indices],p,batch_size=16,threshold=.65,
       include_confidence=True,include_spans=True) for p in passes]
  streams[slot].synchronize()
 return indices,out
pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
def concurrent_extract(count):
 split_cached.cache_clear();torch.cuda.synchronize();start=time.perf_counter()
 jobs=[pool.submit(worker,slot,[i for batch in batches[slot::count] for i in batch]) for slot in range(count)]
 outputs=[[None]*len(chunks) for _ in passes]
 for job in jobs:
  indices,result=job.result()
  for j,part in enumerate(result):
   for i,value in zip(indices,part):outputs[j][i]=value
 torch.cuda.synchronize()
 return time.perf_counter()-start,outputs
concurrency={'cpu_count':os.cpu_count(),'torch_threads':torch.get_num_threads(),'paths':{},'scope':'One T4; independent model and CUDA stream per worker; same 290 unique chunks; batch 16; no result cache or fast-count. Word cache cleared before each sweep.'}
paths=['base',*sorted(strict_ok-{'base'})]
best=min(paths,key=lambda k:statistics.median(experiment['extraction'][k]['seconds']))
configs=[('base',1),('base',2),('base',4)]
if best!='base':configs += [(best,1),(best,2),(best,4)]
for path,count in configs:
 select(path);seconds,out=concurrent_extract(count)
 equal=out==reference_outputs
 emit('CONCURRENCY_WARMUP',{'path':path,'workers':count,'full_json_equal':equal,'chunks_per_s':len(chunks)/seconds})
 concurrency['paths'][f'{path}_{count}']={'samples':[],'full_json_equal':equal,'difference_count':sum(a!=b for aa,bb in zip(out,reference_outputs) for a,b in zip(aa,bb))}
for trial in range(3):
 for path,count in configs[trial%len(configs):]+configs[:trial%len(configs)]:
  select(path);seconds,out=concurrent_extract(count)
  row=concurrency['paths'][f'{path}_{count}'];row['samples'].append(seconds);row['full_json_equal'] &= out==reference_outputs
  emit('CONCURRENCY_SAMPLE',{'trial':trial,'path':path,'workers':count,'chunks_per_s':len(chunks)/seconds,'full_json_equal':out==reference_outputs})
for row in concurrency['paths'].values():
 row['chunks_per_s']=len(chunks)/statistics.median(row['samples'])
 row['input_tokens_per_s']=report['input_tokens']/statistics.median(row['samples'])
concurrency['allocated_mb']=torch.cuda.memory_allocated()/2**20
concurrency['peak_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
pool.shutdown()
(root/'t4-concurrency.json').write_text(json.dumps(concurrency,indent=2))
# Save installed preparation code to inspect whether shape churn or forced packing can be removed safely.
import gliner2.models.base as base_module
(root/'installed-base.py').write_text(inspect.getsource(base_module))
print('PACKED_ATTRS',[(k,str(v)) for k,v in vars(fused.encoder).items() if 'pack' in k or 'length' in k],flush=True)
emit('CONCURRENCY_DONE',concurrency)
