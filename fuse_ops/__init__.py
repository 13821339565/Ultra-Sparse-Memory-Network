import torch
from torch.utils import cpp_extension
import os
buildpath="fuse_ops/build"
try:
    os.mkdir(buildpath)
except OSError:
    if not os.path.isdir(buildpath):
        print(f"Creation of the build directory {buildpath} failed")

#ret = cpp_extension.load(
#            name="unsorted_segment_sum",
#            sources=['fuse_ops/unsorted_segment_sum.cpp', 'fuse_ops/unsorted_segment_sum.cu'],
#            build_directory=buildpath,
#            extra_cflags=['-O3',],
#            extra_cuda_cflags=['-O3',
#                               '-gencode', 'arch=compute_80,code=sm_80',
#                               '--use_fast_math'],
#            verbose=True
# )

ret = cpp_extension.load(
            name="fused_lookup",
            sources=['fuse_ops/fused_lookup.cpp', 'fuse_ops/fused_lookup.cu'],
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_80,code=sm_80',
                               '--use_fast_math'],
            verbose=True
 )


ret = cpp_extension.load(
            name="fused_glu",
            sources=['fuse_ops/fused_glu.cpp', 'fuse_ops/fused_glu.cu'],
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_80,code=sm_80',
                               '--use_fast_math'],
            verbose=True
 )


ret = cpp_extension.load(
            name="fused_topk",
            sources=['fuse_ops/fused_topk.cpp', 'fuse_ops/fused_topk.cu'],
            build_directory=buildpath,
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_80,code=sm_80',
                               '--use_fast_math'],
            verbose=True
 )