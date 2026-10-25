"""Generate isolated memory-movement variants; leave original sources untouched."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
payload=json.loads((root.parent/'payload.json').read_text())
original=payload['csrc/disentangled_attention_v4.cu']
helper=r'''
// Copy the same 16 bytes as fetch_rows/store_rows without retaining them in registers.
template <int CHUNKS, int THREADS>
__device__ __forceinline__ void copy_rows_async(__half* tile, const __half* base, int64_t stride, int row0, int limit) {
#pragma unroll
  for (int m = 0; m < CHUNKS; ++m) {
    const int index = threadIdx.x + m * THREADS, row = row0 + (index >> 3);
    const bool valid = row >= 0 && row < limit;
    const __half* src = base + (valid ? row : 0) * stride + (index & 7) * 8;
    const uint32_t dst = smem_u32(tile + swz(index >> 3, index & 7));
    const int size = valid ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(size) : "memory");
  }
}
__device__ __forceinline__ void copy_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void copy_wait() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

'''
insert=original.index('template <int NT>\n__device__ __forceinline__ void warp_gemm_abt')
common=original[:insert]+helper+original[insert:]
common=common.replace('8 / WARPS','12 / WARPS')
common=common.replace('    configured = true;',
 '    cudaFuncSetAttribute(disentangled_attention_v4_kernel<WARPS>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);\n    configured = true;')
a=common.index('  uint4 k_regs[')
b=common.index('  float o[8][4];',a)
prefix=common[:a]
suffix=common[b:]
start=suffix.index('    __syncthreads();  // last tile')
end=suffix.index('    float s[4][4];',start)
pq_start=suffix.index('    __syncthreads();  // pos_tile free')
pq_end=suffix.index('    {\n      // p2c block',pq_start)

simple_start=r'''    __syncthreads();
    copy_rows_async<S::KV_CHUNKS / 2, S::THREADS>(k_tile, k_base, k_sl, j0, real);
    copy_rows_async<S::KV_CHUNKS / 2, S::THREADS>(v_tile, v_base, v_sl, j0, real);
    copy_rows_async<S::POS_CHUNKS, S::THREADS>(pos_tile, pk_base, D, slot_lo, pos_rows);
    copy_commit();
    copy_wait();
    __syncthreads();

'''
simple_pq=r'''    __syncthreads();
    copy_rows_async<S::POS_CHUNKS, S::THREADS>(pos_tile, pq_base, D, slot_lo, pos_rows);
    copy_commit();
    copy_wait();
    __syncthreads();
'''
simple_suffix=suffix[:start]+simple_start+suffix[end:pq_start]+simple_pq+suffix[pq_end:]
simple_suffix=simple_suffix.replace('const int slot_lo = slot_lo_next;','const int slot_lo = window_start(j0);')
simple=prefix+simple_suffix

pipeline_init=r'''  int slot_lo_next = window_start(0);
  __syncthreads();  // all warps have finished reading Q from pos_tile
  copy_rows_async<S::KV_CHUNKS / 2, S::THREADS>(k_buffers, k_base, k_sl, 0, real);
  copy_rows_async<S::KV_CHUNKS / 2, S::THREADS>(v_buffers, v_base, v_sl, 0, real);
  copy_rows_async<S::POS_CHUNKS, S::THREADS>(pos_tile, pk_base, D, slot_lo_next, pos_rows);
  copy_rows_async<S::POS_CHUNKS, S::THREADS>(pos_query_tile, pq_base, D, slot_lo_next, pos_rows);
  copy_commit();
  copy_wait();
  __syncthreads();

'''
pipeline_pq=r'''    __syncthreads();  // C2P is stored; the position-key buffer is free
    if (has_next) {
      slot_lo_next = window_start(j0 + BN);
      const int next_buffer = buffer ^ 1;
      copy_rows_async<S::KV_CHUNKS / 2, S::THREADS>(k_buffers + next_buffer * BN * D, k_base, k_sl, j0 + BN, real);
      copy_rows_async<S::KV_CHUNKS / 2, S::THREADS>(v_buffers + next_buffer * BN * D, v_base, v_sl, j0 + BN, real);
      copy_rows_async<S::POS_CHUNKS, S::THREADS>(pos_tile, pk_base, D, slot_lo_next, pos_rows);
      copy_commit();
    }
'''
pipeline_suffix=suffix[:start]+suffix[end:pq_start]+pipeline_pq+suffix[pq_end:]
pipeline_suffix=pipeline_suffix.replace('    const bool has_next = j0 + BN < real;',
 '    const bool has_next = j0 + BN < real;\n    const int buffer = (j0 / BN) & 1;\n    k_tile = k_buffers + buffer * BN * D;\n    v_tile = v_buffers + buffer * BN * D;')
pipeline_suffix=pipeline_suffix.replace('warp_gemm_abt<S::PK_COLS / 8>(c, k_frag, pos_tile,',
 'warp_gemm_abt<S::PK_COLS / 8>(c, k_frag, pos_query_tile,')
pipeline_suffix=pipeline_suffix.replace('    float tile_max_a = -INFINITY, tile_max_b = -INFINITY;',
 '''    if (has_next) {
      copy_rows_async<S::POS_CHUNKS, S::THREADS>(pos_query_tile, pq_base, D, slot_lo_next, pos_rows);
      copy_commit();
    }
    float tile_max_a = -INFINITY, tile_max_b = -INFINITY;''')
tail='      }\n    }\n  }\n\n  sum_a +='
assert pipeline_suffix.count(tail)==1
pipeline_suffix=pipeline_suffix.replace(tail,'      }\n    }\n    if (has_next) copy_wait();\n    __syncthreads();\n  }\n\n  sum_a +=')
pipeline=prefix+pipeline_init+pipeline_suffix
pipeline=pipeline.replace('(2 * BN * D + CTA_WINDOW * D + WARPS * 16 * WARP_WINDOW + BN * CTA_WINDOW)',
 '(4 * BN * D + 2 * CTA_WINDOW * D + WARPS * 16 * WARP_WINDOW + BN * CTA_WINDOW)')
pipeline=pipeline.replace('  __half* k_tile = smem;\n  __half* v_tile = k_tile + BN * D;\n  __half* pos_tile = v_tile + BN * D;\n  __half* cq = pos_tile + S::CTA_WINDOW * D;',
 '''  __half* k_buffers = smem;
  __half* v_buffers = k_buffers + 2 * BN * D;
  __half* pos_tile = v_buffers + 2 * BN * D;
  __half* pos_query_tile = pos_tile + S::CTA_WINDOW * D;
  __half* k_tile = k_buffers;
  __half* v_tile = v_buffers;
  __half* cq = pos_query_tile + S::CTA_WINDOW * D;''')
variants={'async_simple':simple,'async_pipeline':pipeline}
for label,cu in variants.items():
 assert 'k_regs' not in cu and 'pq_regs' not in cu
 assert cu.count('copy_wait();')>=2
 # The matrix multiply and score/softmax/output arithmetic remain verbatim.
 math_start=original.index('    float tile_max_a =')
 math_end=original.index('\ntemplate <int WARPS>\nvoid launch',math_start)
 original_math=original[math_start:math_end]
 candidate_math=cu[cu.index('    float tile_max_a ='):cu.index('\ntemplate <int WARPS>\nvoid launch')]
 if label=='async_pipeline':
  candidate_math=candidate_math.replace('    if (has_next) copy_wait();\n    __syncthreads();\n','')
 assert original_math==candidate_math,label+' altered floating-point arithmetic'
 (root/(label+'.cu')).write_text(cu)
 (root/(label+'.cu')).with_suffix('.json').write_text(json.dumps({'variant':label,'source':cu}))
(root/'variants.json').write_text(json.dumps(variants))
print('Generated two memory-only variants; score, softmax, accumulation and output arithmetic are verbatim.')
