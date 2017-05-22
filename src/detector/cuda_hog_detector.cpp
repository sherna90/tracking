#include "cuda_hog_detector.hpp"
CUDA_HOGDetector::CUDA_HOGDetector(){
	
}

CUDA_HOGDetector::CUDA_HOGDetector(int group_threshold, double hit_threshold){
	args.make_gray = true;
    args.resize_src = false;
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
    args.lambda = 1e-1;
    args.epsilon= 1e-2;
    args.tolerance = 1e-1;
    args.n_iterations = 1e2;
	Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.width, args.height);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);
    gpu_hog = cuda::HOG::create(win_size, block_size, block_stride, cell_size, args.nbins);
    gpu_hog->setWinStride(win_stride);
}

CUDA_HOGDetector::CUDA_HOGDetector(int group_threshold, double hit_threshold,Rect reference_roi){
	args.make_gray = true;
    args.resize_src = true;
    args.width = 64;
    args.height = 128;
    cout << "Width: hog=" << args.width << ", roi=" << reference_roi.width << endl;
    cout << "Height: hog=" << args.height << ", roi=" << reference_roi.height << endl;
    args.scale = max(args.width/reference_roi.width,args.height/reference_roi.height);
    args.nlevels = 13;
    args.gr_threshold = group_threshold;
    args.hit_threshold = hit_threshold;
    args.hit_threshold_auto = false;
    args.win_width = args.width ;
    args.win_stride_width = 16;
    args.win_stride_height = 16;
    args.block_width = 16;
    args.block_stride_width = 8;
    args.block_stride_height = 8;
    args.cell_width = 8;
    args.nbins = 9;
    args.overlap_threshold=0.5;
    args.lambda = 1e-2;
    args.epsilon= 1e-4;
    args.tolerance = 1e-1;
    args.n_iterations = 1e4;
	Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.width, args.height);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);
    gpu_hog = cuda::HOG::create(win_size, block_size, block_stride, cell_size, args.nbins);
    gpu_hog->setWinStride(win_stride);
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);
}
vector<Rect> CUDA_HOGDetector::detect(Mat &frame)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	if(args.resize_src) {
		Size image_size(cvRound(current_frame.cols*args.scale),cvRound(current_frame.rows*args.scale));
		resize(current_frame,current_frame,image_size,0,0,INTER_CUBIC);
	}
	MatrixXd features=this->getFeatureValues(current_frame);
	VectorXd predict_prob = this->logistic_regression.predict(features, true);
	this->feature_values=MatrixXd::Zero(0,features.cols());
	this->detections.clear();
	this->weights.resize(0);
	int idx=0;
	for(int row = 0; row < current_frame.rows - this->args.height + this->args.win_stride_height; row+=this->args.win_stride_height){
		for(int col = 0; col < current_frame.cols - this->args.width + this->args.win_stride_width; col+=this->args.win_stride_width){
			Rect current_window(col, row, this->args.width, this->args.height);
			if (predict_prob(idx) > args.hit_threshold)
			{
				rectangle( current_frame, current_window, Scalar(0,255,0), 2, LINE_AA );
				if(args.resize_src) {
					Size image_size(cvRound(current_window.width/args.scale),cvRound(current_window.height/args.scale));			
					current_window=Rect(Point2d(current_window.x,current_window.y) / args.scale,image_size );
				}
				this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				this->feature_values.row(this->feature_values.rows() - 1)=features.row(idx);
				this->weights.conservativeResize( this->weights.size() + 1 );
				this->weights(this->weights.size() - 1) = predict_prob(idx);
				this->detections.push_back(current_window);
				rectangle(frame, current_window, Scalar(0,255,0), 2, LINE_AA );
			}
			idx++;
		}
	}
	imwrite("resized_image.png", current_frame);
	return this->detections;
}

void CUDA_HOGDetector::train(Mat &frame,Rect reference_roi)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	if(args.resize_src) {
		Size image_size((int)current_frame.cols*args.scale,(int)current_frame.rows*args.scale);
		resize(current_frame,current_frame,image_size,0,0,INTER_CUBIC);
		reference_roi+=Point((int)reference_roi.width*args.scale,(int)reference_roi.height*args.scale);
		reference_roi+=Size((int)reference_roi.width*args.scale,(int)reference_roi.height*args.scale);
	}
	MatrixXd features=this->getFeatureValues(current_frame);
	this->feature_values=MatrixXd::Zero(0,features.cols());
	this->labels.resize(0);
	this->weights.resize(0);
	int idx=0;
	uniform_real_distribution<double> unif(0.0,1.0);
	for(int row = 0; row < current_frame.rows - this->args.height + this->args.win_stride_height; row+=this->args.win_stride_height){
		for(int col = 0; col < current_frame.cols - this->args.width + this->args.win_stride_width; col+=this->args.win_stride_width){
			Rect current_window(col, row, this->args.width, this->args.height);
			Rect intersection = reference_roi & current_window;
			float overlap=(float)intersection.area()/reference_roi.area();
			double uni_rand = (overlap > args.overlap_threshold) ? 1.0 : unif(this->generator);
			if(uni_rand>0.9){ 
				this->detections.push_back(current_window);
				this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				this->feature_values.row(this->feature_values.rows() - 1)=features.row(idx);
				this->labels.conservativeResize(this->labels.size() + 1 );
				this->labels(this->labels.size() - 1) = (overlap > args.overlap_threshold) ? 1.0 : 0.0;
				rectangle( current_frame, current_window, Scalar(255,255,255), 2, LINE_8  );
			}
			idx++;
		}
	}
	cout << "positive examples : " << (this->labels.array() > 0).count() << endl;
	cout << "negative examples : " << (this->labels.array() <= 0).count() << endl;
	rectangle( current_frame, reference_roi, Scalar(0,255,0), 2, LINE_AA );
	imwrite("resized_image.png", current_frame);
	this->logistic_regression = LogisticRegression(this->feature_values, this->labels, args.lambda);
	cout << this->feature_values.rows() << "," << this->feature_values.cols() << "," << this->labels.rows() << endl;
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
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
	Mat subImage,hog_descriptors;
	vector<float> features;
	cuda::GpuMat gpu_img,hog_img;
	gpu_img.upload(frame);
	gpu_hog->compute(gpu_img,hog_img);
	hog_img.download(hog_descriptors);
	cv2eigen(hog_descriptors,hogFeatures);
	return hogFeatures;
}

MatrixXd CUDA_HOGDetector::getFeatureValues()
{
	return this->feature_values;
}

VectorXd CUDA_HOGDetector::getDetectionWeights(){
	return this->weights;
}