//
// Created by 张海宁 on 2020/8/19.
//

#include "Onnx.h"

MyOnnxPack::MyOnnxPack(std::string modelpath, std::vector<std::string> &vecString) {
    this->strModelPath = modelpath;
    this->vecImglist = vecString;
    this->out_shape = {1, HEIGHT, WIDTH, 2};
    mPredict = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar::all(0));
    InitializeOnnxEnv();
    input_tensor_value = std::vector<float>(WIDTH * HEIGHT * CHANNEL);
    PrintfInputInfo();
    std::cout << "Construct Over! " << std::endl;
}

MyOnnxPack::MyOnnxPack(std::string modelpath, std::string testImg)
{
    this->strModelPath = modelpath;
    this->strTestImg = testImg;
    this->out_shape = {1, HEIGHT, WIDTH, 2};
    mPredict = cv::Mat(HEIGHT, WIDTH, CV_8UC1, cv::Scalar::all(0));
    InitializeOnnxEnv();
    input_tensor_value = std::vector<float>(WIDTH * HEIGHT * CHANNEL);
    PrintfInputInfo();
    std::cout << "Construct Over! " << std::endl;

    //std::cout << "inputImg size:  " << input_image.size() << std::endl;
}

void MyOnnxPack::InitializeOnnxEnv() {
    this->env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "test");
    this->sessionOptions = new Ort::SessionOptions();
    this->sessionOptions->SetIntraOpNumThreads(1);
    this->sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    this->session = new Ort::Session(*this->env, this->strModelPath.c_str(), *sessionOptions);
    this->allocator = new Ort::AllocatorWithDefaultOptions();
}

MyOnnxPack::~MyOnnxPack() {
    if(this->env != nullptr)
    {
        delete(env);
    }
    if(this->sessionOptions != nullptr)
    {
        delete(sessionOptions);
    }
    if(this->session != nullptr)
    {
        delete(session);
    }
    if(this->allocator != nullptr)
    {
        delete(allocator);
    }
}

void MyOnnxPack::PrintfInputInfo() {
    size_t num_input_nodes = this->session->GetInputCount();
//    std::cout << "Number of inputs " << num_input_nodes << std::endl;
    input_nodes_names = std::vector<const char*>(num_input_nodes);
    for(int i = 0; i < num_input_nodes; i++)
    {
        char* input_name = this->session->GetInputName(i, *allocator);
        //std::cout << "Input " << i << " : " << " name = " << input_name << std::endl;
        input_nodes_names[i] = input_name;

        Ort::TypeInfo type_info = this->session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        input_node_dims = tensor_info.GetShape();

//        for (int j = 0; j < input_node_dims.size(); j++)
//            printf("Input %d : dim %d=%lld\n", i, j, input_node_dims[j]);
    }
}




std::vector<const char*> MyOnnxPack::GetOutPutName() {

    size_t num_out_nodes = this->session->GetOutputCount();
    std::vector<const char*> out_vec_name(num_out_nodes);
   // std::cout << "Number of outputs " << num_out_nodes << std::endl;
    for(int i = 0; i < num_out_nodes; i++)
    {
        char* out_name = this->session->GetOutputName(i, *allocator);
        out_vec_name[i] = out_name;
    }
    return out_vec_name;
}

cv::Mat* MyOnnxPack::GetImgFromVector(ResultType &result) {
    if(result.size() <= 0)
    {
        return &mPredict;
    }
    assert(result.size() == WIDTH * HEIGHT * 2);
    for(int i = 0; i < WIDTH * HEIGHT; i++)
    {
        int row = i / WIDTH;
        int col = i % WIDTH;
        uchar mm = saturate_cast<uchar>(result[i * 2] * 255);
        mPredict.at<uchar>(row, col) =  mm;
    }
    return &mPredict;
}

void MyOnnxPack::Mat2ChannelLast(cv::Mat &src, float *p_input) {
    assert(!src.empty());

    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            for(int c = 0; c < CHANNEL; c++)
            {
                p_input[i * WIDTH * CHANNEL + j * CHANNEL + c] = (src.ptr<uchar>(i)[j * CHANNEL + c]) / 1.0;
            }
        }
    }
}

void MyOnnxPack::Mat2ChannelFirst(cv::Mat &src, float* p_input) {
    assert(!src.empty());
    for(int c = 0; c < CHANNEL; c++)
    {
        for(int i = 0; i < src.rows; i++)
        {
            for(int j = 0; j < src.cols; j++)
            {
                p_input[c * src.rows * src.cols + i * src.cols + j] = (src.ptr<uchar>(i)[j * CHANNEL + c]) / 1.0;
            }
        }
    }
}

void MyOnnxPack::InferenceImg() {
    assert(img.cols == WIDTH || img.rows == HEIGHT || img.channels() == CHANNEL);
    cv::Mat dst;
    cv::resize(img, dst, cv::Size(WIDTH, HEIGHT));
    float* p_input = this->input_image.data();
    std::fill(input_image.begin(), input_image.end(), 0.f);
    Mat2ChannelLast(dst, p_input);


    output_node_names = GetOutPutName();
    assert(output_node_names.size() == 1);
    memcpy(this->input_tensor_value.data(), p_input, sizeof(float) * WIDTH * HEIGHT * CHANNEL);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_node_dims[0] = 1;
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_value.data(), input_tensor_value.size(), input_node_dims.data(), 4);
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, result.data(), result.size(), out_shape.data(), out_shape.size());
    assert(input_tensor.IsTensor());

    session->Run(Ort::RunOptions(nullptr), input_nodes_names.data(), &input_tensor, 1, output_node_names.data(), &output_tensor, 1);

    cv::Mat img2 = *GetImgFromVector(result);

    std::string strPredictPath = ConstructFilePath();
    cv::imwrite(strPredictPath, img2);



}


void MyOnnxPack::InferenceVecImg() {
    assert(this->vecImglist.size() > 0);
    for(int i = 0; i < vecImglist.size(); i++)
    {
        ReadImg(vecImglist[i]);
        InferenceImg();
    }
}

void MyOnnxPack::ReadImg() {
    img  = cv::imread(this->strTestImg);
}

void MyOnnxPack::ReadImg(std::string path) {
    this->strTestImg = path;
    img = cv::imread(this->strTestImg, 1);
    assert(img.channels() == CHANNEL);
}

std::string MyOnnxPack::ConstructFilePath() {
    int pos = this->strTestImg.find_last_of(HN_UTIL::prefix);
    std::string filename = std::string(this->strTestImg.substr(pos + 1));
    std::string filepath = std::string(this->strTestImg.substr(0, pos + 1));
    int posPoint = filename.find_last_of(".");
    std::string name = std::string(filename.substr(0, posPoint));
    std::string hPrix = std::string(filename.substr(posPoint));
    return std::string(filepath + name + "_predict" + hPrix);
}


































