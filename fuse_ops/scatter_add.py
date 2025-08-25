import torch


class ScatterAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, value_expand_time):
        import unsorted_segment_sum
        if values.dtype == torch.float32:
            values = values.to(torch.bfloat16)
        output = torch.zeros((indices.shape[0], value_expand_time, values.shape[2]),device=values.device, dtype=values.dtype)
        unsorted_segment_sum.forward(indices.to(torch.int32), values, output, -1)
        ctx.save_for_backward(indices)
        return output

    @staticmethod
    def backward(ctx, output_grads):
        import unsorted_segment_sum
        indices, = ctx.saved_tensors
        values_grad = torch.empty(indices.shape[0], indices.shape[1], output_grads.shape[2], device=output_grads.device, dtype=output_grads.dtype)
        torch.gather(output_grads, 1, indices.unsqueeze(dim=-1).expand(-1,-1,output_grads.shape[-1]), sparse_grad=False, out=values_grad)
        return None, values_grad, None