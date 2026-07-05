#include <cstdint>
#include <torch/extension.h>

void compute_valid_udf_cuda(float* vertices, int* faces, int* udf, const int numTriangles, const int DIM=512, const float threshold=8);
void compute_valid_sdf_cuda(float* vertices, int* faces, int64_t* sdf, const int numTriangles, const int DIM=512, const float threshold=8);
void compute_sharp_mask_cuda(const int64_t* packed_sdf, uint8_t* sharp_mask, const int DIM, const float band, const float grad_dev_thresh);

extern "C" 
void compute_valid_udf_wrapper(torch::Tensor vertices, torch::Tensor faces, torch::Tensor udf, const int numTriangles, const int DIM=512, const float threshold=8.0) {
    compute_valid_udf_cuda(vertices.data_ptr<float>(), faces.data_ptr<int>(), udf.data_ptr<int>(), numTriangles, DIM, threshold);
}

extern "C"
void compute_valid_sdf_wrapper(torch::Tensor vertices, torch::Tensor faces, torch::Tensor sdf, const int numTriangles, const int DIM=512, const float threshold=8.0) {
    // Use int64_t / data_ptr<int64_t>() — PyTorch may not export TensorBase::data_ptr<long long>()
    // (undefined symbol at import time on some torch builds).
    compute_valid_sdf_cuda(vertices.data_ptr<float>(), faces.data_ptr<int>(), sdf.data_ptr<int64_t>(), numTriangles, DIM, threshold);
}

// Wrapper for the sharp-mask kernel.
// packed_sdf : int64 tensor [DIM^3] produced by compute_valid_sdf.
// sharp_mask : uint8 tensor [DIM^3] (output, must be pre-allocated with zeros).
// band       : normalised SDF threshold; only voxels with |sdf|<band are evaluated.
// grad_dev_thresh : |1 - |∇SDF|| > this => marked sharp (e.g. 0.3).
extern "C"
void compute_sharp_mask_wrapper(
    torch::Tensor packed_sdf,
    torch::Tensor sharp_mask,
    const int     DIM,
    const float   band,
    const float   grad_dev_thresh
) {
    compute_sharp_mask_cuda(
        packed_sdf.data_ptr<int64_t>(),
        sharp_mask.data_ptr<uint8_t>(),
        DIM,
        band,
        grad_dev_thresh
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_valid_udf",  &compute_valid_udf_wrapper,  "Compute UDF using CUDA");
    m.def("compute_valid_sdf",  &compute_valid_sdf_wrapper,  "Compute signed SDF using CUDA");
    m.def("compute_sharp_mask", &compute_sharp_mask_wrapper,
          "Compute sharp-edge mask from packed SDF via 6-neighbour gradient magnitude");
}

