#include <torch/extension.h>

torch::Tensor EmbedLookupReduceForward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& score,
                                       int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx);

torch::Tensor EmbedLookupReduceBackward(const torch::Tensor& input, const torch::Tensor& weight, const torch::Tensor& score, const torch::Tensor& output_grad, torch::Tensor &weight_grad,
                                        int vocab_size, int per_layer_vocab_size, int shift, int group_size, int padding_idx, bool has_padding_idx);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &EmbedLookupReduceForward,
        "EmbedLookupReduceForward -- Forward.");
  m.def("backward", &EmbedLookupReduceBackward,
        "EmbedLookupReduceBackward -- Backward.");
}
