"""Turing-compatible variants: change memory lifetimes, preserve arithmetic verbatim."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
original=json.loads((root.parent/'payload.json').read_text())['csrc/disentangled_attention_v4.cu']
helper='''template <int CHUNKS, int THREADS>
__device__ __forceinline__ void copy_rows_sync(__half* tile, const __half* base, int64_t stride, int row0, int limit) {
#pragma unroll
  for (int m = 0; m < CHUNKS; ++m) {
    const int index = threadIdx.x + m * THREADS, row = row0 + (index >> 3);
    uint4 value = make_uint4(0, 0, 0, 0);
    if (row >= 0 && row < limit)
      value = __ldg(reinterpret_cast<const uint4*>(base + row * stride + (index & 7) * 8));
    *reinterpret_cast<uint4*>(tile + swz(index >> 3, index & 7)) = value;
  }
}

'''
# Retain K/V prefetch; load positional operands immediately before use.
position=original.replace('template <int NT>\n__device__ __forceinline__ void warp_gemm_abt',helper+'template <int NT>\n__device__ __forceinline__ void warp_gemm_abt',1)
position=position.replace(', pk_regs[S::POS_CHUNKS], pq_regs[S::POS_CHUNKS]','')
position='\n'.join(line for line in position.split('\n') if not ('fetch_rows<S::POS_CHUNKS' in line))
position=position.replace('store_rows<S::POS_CHUNKS, S::THREADS>(pk_regs, pos_tile);','copy_rows_sync<S::POS_CHUNKS, S::THREADS>(pos_tile, pk_base, D, slot_lo, pos_rows);')
position=position.replace('store_rows<S::POS_CHUNKS, S::THREADS>(pq_regs, pos_tile);','copy_rows_sync<S::POS_CHUNKS, S::THREADS>(pos_tile, pq_base, D, slot_lo, pos_rows);')
assert 'pk_regs' not in position and 'pq_regs' not in position
# Also remove K/V prefetch in a second experiment to distinguish latency hiding from pressure.
sync=position
start=sync.index('  uint4 k_regs[');end=sync.index('  float o[8][4];',start)
sync=sync[:start]+sync[end:]
sync=sync.replace('const int slot_lo = slot_lo_next;','const int slot_lo = window_start(j0);')
sync=sync.replace('store_rows<S::KV_CHUNKS / 2, S::THREADS>(k_regs, k_tile);','copy_rows_sync<S::KV_CHUNKS / 2, S::THREADS>(k_tile, k_base, k_sl, j0, real);')
sync=sync.replace('store_rows<S::KV_CHUNKS / 2, S::THREADS>(v_regs, v_tile);','copy_rows_sync<S::KV_CHUNKS / 2, S::THREADS>(v_tile, v_base, v_sl, j0, real);')
start=sync.index('    if (has_next) {  // lands');end=sync.index('    float s[4][4];',start)
sync=sync[:start]+sync[end:]
assert 'k_regs' not in sync and 'slot_lo_next' not in sync
variants={'pos_sync':position,'all_sync':sync}
for name,cu in variants.items():
 a=original.index('    float tile_max_a =');b=original.index('\ntemplate <int WARPS>\nvoid launch',a)
 assert original[a:b]==cu[cu.index('    float tile_max_a ='):cu.index('\ntemplate <int WARPS>\nvoid launch')]
 (root/(name+'.cu')).write_text(cu)
(root/'variants.json').write_text(json.dumps(variants))
print('Verified score/softmax/output arithmetic unchanged; 2 variants generated')
