#include "inference.h"
#include <regex>



void SegmentAnything() {

    SAM* samSegmentor = new SAM;

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_encoder.onnx";
    params.imgSize = { 1024, 1024 };

    samSegmentor->CreateSession(params);

    DL_INIT_PARAM params1;
    params1 = params;
    params.modelPath = "/home/amigo/Documents/repos/hero_sam/sam_inference/model/SAM_mask_decoder.onnx";
    samSegmentor->CreateSession(params1);
    //input_tensor_size = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount();
    //output_tensor_size = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementCount();
    //input_shape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    //output_shape = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

int main()
{
    SegmentAnything();
    //ClsTest();
}