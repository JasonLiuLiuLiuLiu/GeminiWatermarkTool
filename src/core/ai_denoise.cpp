/**
 * @file    ai_denoise.cpp
 * @brief   NCNN-based AI denoiser implementation
 * @author  AllenK (Kwyshell)
 * @license MIT
 */

#ifdef GWT_HAS_AI_DENOISE

// NCNN shim must be included FIRST (volk + simplevk.h hack)
#include "core/ncnn_shim.hpp"
#include "core/ai_denoise.hpp"

#include <spdlog/spdlog.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>

// Embedded model data (defined in ai_denoise_model.cpp)
namespace gwt::ai_model {
    const unsigned char* param_data();
    const unsigned char* bin_data();
}

namespace gwt {

// ============================================================================
// Impl (PIMPL idiom hides NCNN types from header)
// ============================================================================

struct NcnnDenoiser::Impl {
    ncnn::Net net;
    bool ready{false};
    bool gpu_enabled{false};
    int gpu_device_index{-1};
    std::string device_desc;

    bool load_model() {
        // Load from embedded memory (defined in ai_denoise_model.cpp)
        int ret_param = net.load_param(ai_model::param_data());
        if (ret_param != 0) {
            spdlog::error("NcnnDenoiser: Failed to load param (error {})", ret_param);
            return false;
        }

        int ret_model = net.load_model(ai_model::bin_data());
        if (ret_model != 0) {
            spdlog::error("NcnnDenoiser: Failed to load model (error {})", ret_model);
            return false;
        }

        spdlog::info("NcnnDenoiser: Model loaded from embedded data");
        return true;
    }

    bool init_gpu() {
        int gpu_count = ncnn::get_gpu_count();
        if (gpu_count <= 0) {
            spdlog::info("NcnnDenoiser: No Vulkan GPU detected, using CPU");
            return false;
        }

        // Use the first available GPU
        gpu_device_index = ncnn::get_default_gpu_index();
        const ncnn::GpuInfo& gpu_info = ncnn::get_gpu_info(gpu_device_index);
        device_desc = gpu_info.device_name();

        net.opt.use_vulkan_compute = true;

        spdlog::info("NcnnDenoiser: Vulkan GPU #{} - {}", gpu_device_index, device_desc);
        return true;
    }

    void init_cpu() {
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = std::max(1, ncnn::get_cpu_count());
        device_desc = fmt::format("CPU ({} threads)", net.opt.num_threads);
        spdlog::info("NcnnDenoiser: {}", device_desc);
    }
};

// ============================================================================
// NcnnDenoiser public API
// ============================================================================

NcnnDenoiser::NcnnDenoiser() : m_impl(std::make_unique<Impl>()) {}

NcnnDenoiser::~NcnnDenoiser() {
    // Impl destructor handles ncnn::Net cleanup
}

NcnnDenoiser::NcnnDenoiser(NcnnDenoiser&&) noexcept = default;
NcnnDenoiser& NcnnDenoiser::operator=(NcnnDenoiser&&) noexcept = default;

bool NcnnDenoiser::initialize() {
    if (m_impl->ready) {
        spdlog::warn("NcnnDenoiser: Already initialized");
        return true;
    }

    // Initialize Vulkan GPU instance
    ncnn::create_gpu_instance();

    // Configure network options
    m_impl->net.opt.use_fp16_packed = true;
    m_impl->net.opt.use_fp16_storage = true;
    m_impl->net.opt.use_fp16_arithmetic = false;  // FP16 storage, FP32 compute
    m_impl->net.opt.use_packing_layout = true;

    // Try GPU first, fall back to CPU
    m_impl->gpu_enabled = m_impl->init_gpu();
    if (!m_impl->gpu_enabled) {
        m_impl->init_cpu();
    }

    // Load model
    if (!m_impl->load_model()) {
        return false;
    }

    m_impl->ready = true;
    spdlog::info("NcnnDenoiser: Initialized ({})", m_impl->device_desc);
    return true;
}

bool NcnnDenoiser::is_ready() const {
    return m_impl->ready;
}

bool NcnnDenoiser::is_gpu_enabled() const {
    return m_impl->gpu_enabled;
}

std::string NcnnDenoiser::device_name() const {
    return m_impl->device_desc;
}

void NcnnDenoiser::denoise(
    cv::Mat& image,
    const cv::Rect& region,
    float sigma,
    float strength,
    int padding)
{
    if (!m_impl->ready) {
        spdlog::warn("NcnnDenoiser: Not initialized, skipping denoise");
        return;
    }

    if (image.empty() || image.type() != CV_8UC3) {
        spdlog::warn("NcnnDenoiser: Invalid image (empty or not BGR CV_8UC3)");
        return;
    }

    // Clamp parameters
    sigma = std::clamp(sigma, 0.0f, 75.0f);
    strength = std::clamp(strength, 0.0f, 1.0f);

    if (strength < 0.001f) return;  // Nothing to do

    // Compute padded ROI (clamp to image bounds)
    const int img_w = image.cols;
    const int img_h = image.rows;

    cv::Rect padded_roi(
        std::max(0, region.x - padding),
        std::max(0, region.y - padding),
        std::min(img_w, region.x + region.width + padding) - std::max(0, region.x - padding),
        std::min(img_h, region.y + region.height + padding) - std::max(0, region.y - padding)
    );

    if (padded_roi.width < 4 || padded_roi.height < 4) {
        spdlog::warn("NcnnDenoiser: ROI too small ({} x {})", padded_roi.width, padded_roi.height);
        return;
    }

    // TODO: Step 2 - Implement NCNN inference pipeline
    //   1. Extract padded ROI
    //   2. Convert BGR uint8 -> RGB float [0,1]
    //   3. Create sigma map [1, H, W] = sigma/255.0
    //   4. Build ncnn::Mat with 4 channels (RGB + sigma)
    //   5. Run inference: ncnn::Extractor ex = net.create_extractor()
    //   6. Residual subtraction: clean = input - output
    //   7. Clamp [0,1], convert back to BGR uint8
    //   8. Blend: result = strength * denoised + (1 - strength) * original
    //   9. Write back to image ROI

    spdlog::info("NcnnDenoiser: denoise() called - sigma={:.1f}, strength={:.0f}%, roi={}x{} (TODO: inference)",
                 sigma, strength * 100.0f, padded_roi.width, padded_roi.height);
}

} // namespace gwt

#endif // GWT_HAS_AI_DENOISE
