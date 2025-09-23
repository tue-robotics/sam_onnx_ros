#ifndef SAMINFERENCE_H
#define SAMINFERENCE_H


#define RET_OK nullptr
#include <memory>
#include <string>
#include <vector>
#include <cstdio>
#include "utils.h"
#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

class SAM
{
public:
    SAM();

    ~SAM();

public:
    const char *CreateSession(SEG::DL_INIT_PARAM &iParams);

    const char *RunSession(const cv::Mat &iImg, std::vector<SEG::DL_RESULT> &oResult, SEG::MODEL_TYPE modelType, SEG::DL_RESULT &result);

private:

    char *WarmUpSession_(SEG::MODEL_TYPE modelType);

    template <typename N>
    const char *TensorProcess_(clock_t &starttime_1, const cv::Mat &iImg, N &blob, std::vector<int64_t> &inputNodeDims,
                        SEG::MODEL_TYPE modelType, std::vector<SEG::DL_RESULT> &oResult, Utils &utilities, SEG::DL_RESULT &result);

    Ort::Env _env;
    std::unique_ptr<Ort::Session> _session;
    bool _cudaEnable;
    Ort::RunOptions _options;
    std::vector<const char *> _inputNodeNames;
    std::vector<const char *> _outputNodeNames;

    SEG::MODEL_TYPE _modelType;
    std::vector<int> _imgSize;
    float _rectConfidenceThreshold;
};

#endif // SAMINFERENCE_H
