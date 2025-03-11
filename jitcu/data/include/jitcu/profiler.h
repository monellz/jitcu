// This file is modified from the original version,
// which is part of the flashinfer project
// (https://github.com/flashinfer-ai/flashinfer).

#ifndef _JITCU_PROFILER_H
#define _JITCU_PROFILER_H

#include "utils.h"

namespace jc::profiler {

// Init following fields in host code
// can be passed with const
struct ProfilerHostParams {
  uint64_t* buffer;
};

struct ProfilerContext {
  uint64_t* write_ptr;
  uint32_t write_stride;
  uint32_t entry_tag_base;
  bool write_thread_predicate;
};

struct ProfilerEntry {
  union {
    struct {
      uint32_t nblocks;
      uint32_t ngroups;
    };
    struct {
      uint32_t tag;
      uint32_t delta_time;
    };
    uint64_t raw;
  };
};

constexpr uint32_t BLOCK_GROUP_IDX_MASK = 0xFFFFF;
constexpr uint32_t EVENT_IDX_MASK = 0x3FF;
constexpr uint32_t BEGIN_END_MASK = 0x3;

constexpr uint32_t EVENT_IDX_SHIFT = 2;
constexpr uint32_t BLOCK_GROUP_IDX_SHIFT = 12;

constexpr uint32_t EVENT_BEGIN = 0x0;
constexpr uint32_t EVENT_END = 0x1;
constexpr uint32_t EVENT_INSTANT = 0x2;

#ifdef JC_ENABLE_PROFILER

#ifndef __CUDACC__
#error "CUDA should be enabled for profiler"
#endif

__device__ __forceinline__ uint32_t get_block_idx() {
  return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
}

__device__ __forceinline__ uint32_t get_num_blocks() { return gridDim.x * gridDim.y * gridDim.z; }

__device__ __forceinline__ uint32_t get_thread_idx() {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ uint32_t encode_tag(uint32_t block_group_idx, uint32_t event_idx,
                                               uint32_t event_type) {
  // (32..12]: block_group_idx (max: 1048576)
  // (12..2]: event_idx (max: 1024)
  // (2..0]: event_type
  return (block_group_idx << BLOCK_GROUP_IDX_SHIFT) | (event_idx << EVENT_IDX_SHIFT) | event_type;
}

__device__ __forceinline__ uint32_t get_timestamp() {
  volatile uint32_t ret;
  asm volatile("mov.u32 %0, %globaltimer_lo;" : "=r"(ret));
  return ret;
}

__device__ __forceinline__ ProfilerContext context_init(const ProfilerHostParams& params,
                                                        uint32_t group_idx, uint32_t num_groups,
                                                        bool write_thread_predicate) {
  if (get_block_idx() == 0 && get_thread_idx() == 0) {
    ProfilerEntry entry;
    entry.nblocks = get_num_blocks();
    entry.ngroups = num_groups;
    params.buffer[0] = entry.raw;
  }
  // [1] + [#event, #block, #warp_group]
  ProfilerContext ctx;
  ctx.write_ptr = params.buffer + 1 + get_block_idx() * num_groups + group_idx;
  ctx.write_stride = get_num_blocks() * num_groups;
  ctx.entry_tag_base = encode_tag(get_block_idx() * num_groups + group_idx, 0, 0);
  ctx.write_thread_predicate = write_thread_predicate;
  return ctx;
}

template <typename T>
__device__ __forceinline__ void event_start(ProfilerContext& ctx, T event) {
  if (ctx.write_thread_predicate) {
    ProfilerEntry entry;
    entry.tag = ctx.entry_tag_base | (uint32_t(event) << EVENT_IDX_SHIFT) | EVENT_BEGIN;
    entry.delta_time = get_timestamp();
    *ctx.write_ptr = entry.raw;
    ctx.write_ptr += ctx.write_stride;
  }
  __threadfence_block();
}

template <typename T>
__device__ __forceinline__ void event_end(ProfilerContext& ctx, T event) {
  __threadfence_block();
  if (ctx.write_thread_predicate) {
    ProfilerEntry entry;
    entry.tag = ctx.entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_END;
    entry.delta_time = get_timestamp();
    *ctx.write_ptr = entry.raw;
    ctx.write_ptr += ctx.write_stride;
  }
}

template <typename T>
__device__ __forceinline__ void event_instant(ProfilerContext& ctx, T event) {
  __threadfence_block();
  if (ctx.write_thread_predicate) {
    ProfilerEntry entry;
    entry.tag = ctx.entry_tag_base | ((uint32_t)event << EVENT_IDX_SHIFT) | EVENT_INSTANT;
    entry.delta_time = get_timestamp();
    *ctx.write_ptr = entry.raw;
  }
  __threadfence_block();
}

#else

_JITCU_DEVICE ProfilerContext context_init(const ProfilerHostParams& params, uint32_t group_idx,
                                           uint32_t num_groups, bool write_thread_predicate) {
  ProfilerContext ctx;
  ctx.write_ptr = nullptr;
  ctx.write_stride = ctx.entry_tag_base = 0;
  ctx.write_thread_predicate = false;
  return ctx;
}

template <typename T>
_JITCU_DEVICE void event_start(ProfilerContext& ctx, T event) {}

template <typename T>
_JITCU_DEVICE void event_end(ProfilerContext& ctx, T event) {}

template <typename T>
_JITCU_DEVICE void event_instant(ProfilerContext& ctx, T event) {}

#endif

}  // namespace jc::profiler

#endif  // _JITCU_PROFILER_H
