import torch
from torch.utils import cpp_extension

ret = cpp_extension.load(
            name="unsorted_segment_sum",
            sources=['unsorted_segment_sum.cpp', 'unsorted_segment_sum.cu'],
            build_directory="build",
            extra_cflags=['-O3',],
            extra_cuda_cflags=['-O3',
                               '-gencode', 'arch=compute_80,code=sm_80',
                               '--use_fast_math'],
            verbose=True
 )
import unsorted_segment_sum

class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, value_expand_time):
        import unsorted_segment_sum
        output = torch.zeros((indices.shape[0], value_expand_time, values.shape[2]),device=values.device, dtype=values.dtype)
        unsorted_segment_sum.forward(group_indice.to(torch.int32), values, output, -1)
        ctx.save_for_backward(indices)
        return output

    @staticmethod
    def backward(ctx, output_grads):
        import unsorted_segment_sum
        indices, = ctx.saved_tensors
        values_grad = torch.empty(indices.shape[0], indices.shape[1], output_grads.shape[2], device=output_grads.device, dtype=output_grads.dtype)
        torch.gather(output_grads, 1, indices.unsqueeze(dim=-1).expand(-1,-1,output_grads.shape[-1]), sparse_grad=False, out=values_grad)
        return None, values_grad, None


values = torch.load("../values.pt0")
values1 = values.detach().clone()
values2 = values.detach().clone()
values1.requires_grad=True
values2.requires_grad=True
group_indice = torch.load("../group_indice.pt0")
output = torch.load("../output.pt0")
output1 = output.detach().clone()
#bsz = 10
#gather_dim=4
#emb_dim=2
#out_dim=128
#values=torch.randn(bsz, out_dim, emb_dim).cuda().bfloat16()
#output=torch.zeros(bsz, gather_dim, emb_dim).cuda().bfloat16()
#group_indice=torch.randint(0, gather_dim, (bsz, out_dim)).cuda().bfloat16()
vdim=384
#import ipdb;ipdb.set_trace()
with torch.profiler.profile(with_stack=True, record_shapes=True) as prof:
    output1.scatter_add_(1, group_indice.unsqueeze(dim=-1).expand(-1,-1,vdim), values1)  # output shape: [bs, value_expand_time, vdim]
    #unsorted_segment_sum.forward(group_indice.to(torch.int32), values, output, -1)
    output = ScatterAdd.apply(group_indice, values2, 4)
print(prof.key_averages().table())
loss=(output * output).sum()
loss.backward()
loss1=(output1 * output1).sum()
loss1.backward()
import ipdb;ipdb.set_trace()
diff=output1-output
diff2=values1.grad-values2.grad
print("diff", diff.max(), diff.min())
print("diff2", diff2.max(), diff2.min())