#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
fused_einsum_topk_step1(const torch::Tensor& scores, const torch::Tensor& tucker_core_uv, int64_t k1);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fused_einsum_topk_step1_balance_loss(const torch::Tensor& scores, const torch::Tensor& tucker_core_uv, int64_t k1);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
fused_einsum_topk_step2(const torch::Tensor& scores, const torch::Tensor& multi_tucker_core, int64_t k2);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_einsum_topk_step1", &fused_einsum_topk_step1,
        "fused_einsum_topk_step1 -- Forward.");
  m.def("fused_einsum_topk_step1_balance_loss", &fused_einsum_topk_step1_balance_loss,
        "fused_einsum_topk_step1_balance_loss -- Forward.");
  m.def("fused_einsum_topk_step2", &fused_einsum_topk_step2,
        "fused_einsum_topk_step2 -- Forward.");
}
