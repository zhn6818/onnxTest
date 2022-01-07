//
// Created by 张海宁 on 2022/1/7.
//

#include <iostream>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/providers/cuda/cuda_provider_factory.h>
#include <core/session/onnxruntime_c_api.h>
#include "Onnx.h"
#include "UTIL.h"


int main(int argc, char** argv)
{
    cv::Mat img = cv::Mat(100, 100, CV_8UC1, cv::Scalar::all(0));
    cv::Rect roi = cv::Rect(0, 0, img.cols - 1, img.rows - 1);

    cv::Mat img2 = img(roi);


    const char* filepath = "/Users/zhanghaining/CLionProjects/onnxTest/imgDir";
    HN_UTIL::VecString sdf = HN_UTIL::GetListFiles(filepath);
    const char* modelpath1 = "../test.onnx";
    std::string filename1 = "../imgDir/testImg.png";
    MyOnnxPack* tmpOnnx = new MyOnnxPack(modelpath1, sdf);

    //tmpOnnx->PrintfInputInfo();
    //tmpOnnx->ReadImg();
    tmpOnnx->InferenceVecImg();

    delete tmpOnnx;
    return 0;
}