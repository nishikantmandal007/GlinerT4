"""Paired GLiNER2 encoder and extraction benchmarks; no serving or result cache."""
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
from gliner2 import AutoExtractor

ROOT = Path('/content/gliner-ab')
torch.set_num_threads(4)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
cap = torch.cuda.get_device_capability()
report = {
    'gpu': torch.cuda.get_device_name(), 'capability': cap,
    'versions': {p: importlib.metadata.version(p) for p in ('torch', 'triton', 'gliner2', 'disentangled-flash', 'transformers')},
    'cuda': torch.version.cuda, 'cpu_threads': torch.get_num_threads(),
    'scope': 'Single GPU, single serving thread, batch 16, FP16, grouped three passes. Same model instance for Triton/custom. Compilation and warmup excluded.',
    'kernel': 'Original T4 v4 source recompiled for actual GPU; no tile or algorithm tuning.',
}
def emit(label, value):
    print(label + ' ' + json.dumps(value), flush=True)
def save():
    (ROOT / 'result.json').write_text(json.dumps(report, indent=2))
    emit('RESULT', report)
emit('ENV', report)
subprocess.run(['nvidia-smi'], check=True)
subprocess.run(['cmake', '-S', str(ROOT), '-B', str(ROOT/'build'),
                '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES='+str(cap[0]*10+cap[1]),
                '-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path], check=True)
subprocess.run(['cmake', '--build', str(ROOT/'build'), '--parallel', '2'], check=True)
sys.path.insert(0, str(ROOT/'torch-ext'))
import gliner_t4
from disentangled_flash import kernel
cls = kernel.InferenceDisentangledSelfAttention
originals = cls.forward_prepared, cls.forward_packed
gliner_t4.install()
custom = cls.forward_prepared, cls.forward_packed
def select(path):
    cls.forward_prepared, cls.forward_packed = custom if path == 'custom' else originals

docs = json.loads((ROOT/'corpus.json').read_text())
chunks = sorted([s for d in docs for s in d['chunks']], key=len)
report['unique_corpus_chunks'] = len(chunks)
report['corpus_sha256'] = hashlib.sha256((ROOT/'corpus.json').read_bytes()).hexdigest()
name = 'fastino/gliner2-multi-v1'
fused = AutoExtractor.from_pretrained(name, map_location='cuda', quantize=True,
                                     attention_backend='disentangled_flash')
fused.eval()
modules = [m for m in fused.encoder.modules() if isinstance(m, cls)]
assert len(modules) == 12, len(modules)
assert all(m.backend != 'torch' for m in modules), 'Triton backend did not activate'
report['attention_backends'] = [m.backend for m in modules]
report['model_commit'] = getattr(fused.encoder.config, '_commit_hash', None)
tokenizer = fused.processor.tokenizer
tokens = tokenizer(' '.join(chunks), add_special_tokens=False)['input_ids']
report['input_tokens'] = sum(len(tokenizer(s, add_special_tokens=False)['input_ids']) for s in chunks)
plain = AutoExtractor.from_pretrained(name, map_location='cuda')
plain.eval()
encoder_rows = []

@torch.inference_mode()
def forward(model, ids, mask):
    return model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state

@torch.inference_mode()
def timed_encoder(ids, mask, path, repetitions=5):
    select(path)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        forward(fused, ids, mask)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions

for length in (256, 384, 512, 1024):
    for padded in (False, True):
        ids = torch.tensor([tokens[i*length:(i+1)*length] for i in range(16)], device='cuda')
        mask = torch.ones_like(ids)
        if padded:
            for row in range(16):
                mask[row, max(1, length-row*length//20):] = 0
        keep = mask.bool()[..., None]
        reference = forward(plain, ids, mask).float()
        select('triton')
        native = forward(fused, ids, mask).float()
        select('custom')
        calls = [0]
        original_attention = gliner_t4.attention
        def counted(*args, **kwargs):
            calls[0] += 1
            return original_attention(*args, **kwargs)
        gliner_t4.attention = counted
        try:
            ours = forward(fused, ids, mask).float()
        finally:
            gliner_t4.attention = original_attention
        assert calls[0] == 12, ('custom kernel fallback', calls[0])
        assert torch.isfinite(ours).all(), 'Non-finite custom encoder result'
        def distance(output):
            delta = (output-reference).abs().masked_select(keep)
            return {'mean_abs': delta.mean().item(), 'max_abs': delta.max().item()}
        timings = {'triton': [], 'custom': []}
        for path in timings:
            select(path)
            for _ in range(3): forward(fused, ids, mask)
        for trial in range(7):
            order = ('triton', 'custom') if trial % 2 == 0 else ('custom', 'triton')
            for path in order: timings[path].append(timed_encoder(ids, mask, path))
        row = {'length': length, 'padded': padded, 'batch':16, 'custom_calls':calls[0],
               'triton_ms':statistics.median(timings['triton']),
               'custom_ms':statistics.median(timings['custom']),
               'timing_samples':timings, 'triton_vs_fp32':distance(native), 'custom_vs_fp32':distance(ours)}
        row['speedup'] = row['triton_ms']/row['custom_ms']
        encoder_rows.append(row)
        emit('ENCODER', row)
report['encoder'] = encoder_rows
del plain, reference, native, ours
torch.cuda.empty_cache()

labels = {
 'Person':'First and last names of people. NOT company names, street addresses, cities, or identifier codes.',
 'Organization':'Company, corporation, institution, agency, or legal entity name. NOT person names, street addresses, cities, or identifier codes.',
 'Street':'Street, road, avenue, boulevard, or lane name. NOT city names, building names, or postal codes.',
 'City':'City'}
passes = [{'Organization':labels['Organization']},
          {k:labels[k] for k in ('Person','Street','City')},
          {'City':'Name of a city, town or village.'}]

@torch.inference_mode()
def extract(path, data):
    select(path)
    torch.cuda.synchronize()
    start = time.perf_counter()
    outputs = [fused.batch_extract_entities(data, p, batch_size=16, threshold=.65,
                                          include_confidence=True, include_spans=True) for p in passes]
    torch.cuda.synchronize()
    return time.perf_counter()-start, outputs

for path in ('triton','custom'):
    extract(path, chunks)
pipeline_times = {'triton':[], 'custom':[]}
for trial in range(3):
    order = ('triton','custom') if trial % 2 == 0 else ('custom','triton')
    for path in order:
        elapsed, outputs = extract(path, chunks)
        pipeline_times[path].append(elapsed)
        (ROOT/(path+'_outputs.json')).write_text(json.dumps(outputs))
        emit('EXTRACTION', {'trial':trial, 'path':path, 'seconds':elapsed, 'chunks_per_s':len(chunks)/elapsed})
report['extraction'] = {path:{'seconds':samples, 'median_seconds':statistics.median(samples),
    'chunks_per_s':len(chunks)/statistics.median(samples),
    'input_tokens_per_s':report['input_tokens']/statistics.median(samples)} for path,samples in pipeline_times.items()}
report['extraction']['speedup'] = statistics.median(pipeline_times['triton'])/statistics.median(pipeline_times['custom'])

def normalize(outputs):
    merged = [dict() for _ in chunks]
    for part in outputs:
        assert len(part) == len(chunks)
        for i, item in enumerate(part):
            for label, values in item.get('entities',{}).items():
                threshold = .85 if label == 'Organization' else .65
                for value in values:
                    if isinstance(value,str):text=value
                    else:
                        if value.get('confidence', value.get('score',1)) < threshold:continue
                        text=value.get('text', value.get('value',''))
                    if text:merged[i].setdefault(label,set()).add(text)
    return merged
native_values = normalize(json.loads((ROOT/'triton_outputs.json').read_text()))
custom_values = normalize(json.loads((ROOT/'custom_outputs.json').read_text()))
differences = [{'chunk':i,'triton':{k:sorted(v) for k,v in a.items()},'custom':{k:sorted(v) for k,v in b.items()}}
               for i,(a,b) in enumerate(zip(native_values,custom_values)) if a != b]
report['parity'] = {'identical_chunks':len(chunks)-len(differences),'total_chunks':len(chunks),'differences':differences}
report['peak_allocated_mb'] = torch.cuda.max_memory_allocated()/2**20
save()
