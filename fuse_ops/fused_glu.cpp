#include <torch/extension.h>

torch::Tensor GluForward(const torch::Tensor& indices, const torch::Tensor& weight, const torch::Tensor& p_input,
                         int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx);

torch::Tensor GluBackward(const torch::Tensor& indices, const torch::Tensor& weight, const torch::Tensor& p_input, const torch::Tensor& output_grad, torch::Tensor &weight_grad,
                          int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &GluForward,
        "GluForward -- Forward.");
  m.def("backward", &GluBackward,
        "GluBackward -- Backward.");
}
