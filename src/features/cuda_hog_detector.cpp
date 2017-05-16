#include "cuda_hog_detector.hpp"


CUDA_HOGDetector::CUDA_HOGDetector(int group_threshold, double hit_threshold){
	args.make_gray = true;
    args.resize_src = true;
    args.width = 64;
    args.height = 128;
    args.scale = 1.0;
    args.nlevels = 13;
    args.gr_threshold = group_threshold;
    args.hit_threshold = hit_threshold;
    args.hit_threshold_auto = false;
    args.win_width = args.width ;
    args.win_stride_width = 8;
    args.win_stride_height = 8;
    args.block_width = 16;
    args.block_stride_width = 8;
    args.block_stride_height = 8;
    args.cell_width = 8;
    args.nbins = 9;
    args.overlap_threshold=0.5;
	Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.width, args.height);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);
    gpu_hog = cuda::HOG::create(win_size, block_size, block_stride, cell_size, args.nbins);
    gpu_hog->setWinStride(win_stride);
}

vector<Rect> CUDA_HOGDetector::detect(Mat &frame)
{
    cuda::GpuMat gpu_img;
    this->detections.clear();
    vector<double> weights;
    Mat img_aux;
    cvtColor(frame, img_aux, COLOR_BGR2BGRA);
    gpu_img.upload(img_aux);
    this->gpu_hog->detectMultiScale(gpu_img, this->detections,&weights);
	double* ptr = &weights[0];
	this->weights = Eigen::Map<Eigen::VectorXd>(ptr, weights.size());
	this->frame = frame;
	return this->detections;
}

void CUDA_HOGDetector::draw()
{
	for (size_t i = 0; i < this->detections.size(); i++)
    {
        Rect r = this->detections[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(this->frame, r.tl(), r.br(), cv::Scalar(255,0,0), 3);
    }
    //cout << "detections size: " << this->detections.size() << endl;
}

MatrixXd CUDA_HOGDetector::getFeatureValues(Mat &frame)
{
	int num_rows=frame.rows - this->args.height + this->args.win_stride_height;
	int num_cols=frame.cols - this->args.width + this->args.win_stride_width;
	MatrixXd hogFeatures(num_rows*num_cols, this->gpu_hog->getDescriptorSize());
	Mat subImage,hog_descriptor;
	vector<float> features;
	cuda::GpuMat gpu_img,hog_img;
	gpu_img.upload(img_aux);
	gpu_hog->compute(gpu_img,hog_img);
	hog_img.download(hog_descriptors);
	cv2eigen(hog_descriptors,hogFeatures);
	return hogFeatures;
}

VectorXd CUDA_HOGDetector::getDetectionWeights(){
	return this->weights;
}