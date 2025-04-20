#include "inference.h"
#include <regex>



void SegmentAnything() {

    SAM* samSegmentor = new SAM;

    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "model/SAM_encoder.onnx";
    params.imgSize = { 640, 640 };

    samSegmentor->CreateSession(params);

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