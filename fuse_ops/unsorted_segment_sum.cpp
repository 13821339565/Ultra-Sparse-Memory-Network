#include <torch/extension.h>

void UnsortedSegmentSum(const torch::Tensor& input, const torch::Tensor& output_grad_, torch::Tensor &weight_grad, const int padding_idx);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &UnsortedSegmentSum,
        "UnsortedSegmentSum -- Forward.");
}
