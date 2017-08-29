#include "CPU_LR_hog_detector.hpp"

#ifndef PARAMS
const bool USE_COLOR=false;
#endif

void CPU_LR_HOGDetector::init(double group_threshold, double hit_threshold,Rect reference_roi){
	args.make_gray = true;
    args.resize_src = false;
    args.width = reference_roi.width;
    args.height = reference_roi.height;
    args.hog_width = 64;
    args.hog_height = 64;
    args.scale = 2;
    args.nlevels = 1;
    args.gr_threshold = group_threshold;
    args.hit_threshold = hit_threshold;
    args.hit_threshold_auto = false;
    args.win_width = args.width ;
    args.win_stride_width = 3;
    args.win_stride_height = 3;
    args.block_width = 16;
    args.block_stride_width = 8;
    args.block_stride_height = 8;
    args.cell_width = 8;
    args.nbins = 9;
    args.overlap_threshold=0.8;
    args.p_accept = 0.9;
    args.lambda = 10.0;
    args.epsilon= 1e-3;
    args.tolerance = 1e-1;
    args.n_iterations = 1e3;
    args.padding = 8;
    //this->n_descriptors = (args.width/args.cell_width-1)*(args.height/args.cell_width-1)*args.nbins*(args.block_width*args.block_width/(args.cell_width*args.cell_width));
    //this->n_descriptors = 3780;
	Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.hog_width, args.hog_height);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->hog = HOGDescriptor(win_size, block_size, block_stride, cell_size, args.nbins);
    if(USE_COLOR){
    	this->n_descriptors = this->hog.getDescriptorSize() + H_BINS*S_BINS;
    	//int channels = 3;
    	//this->n_descriptors=this->hog.getDescriptorSize() + (this->args.hog_width/2)*(this->args.hog_height/2)*channels);
    }
    else this->n_descriptors = this->hog.getDescriptorSize();
    this->generator.seed(seed1);
    this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	this->labels.resize(0);
	this->num_frame=0;
}



vector<Rect> CPU_LR_HOGDetector::detect(Mat &frame,Rect reference_roi)
{
	Mat cropped_frame,current_frame,cielab_image;
	int x_shift=50;
	int y_shift=50;
	Rect cropped_roi=reference_roi+Point(-x_shift,-y_shift);
	cropped_roi.x=MIN(MAX(cropped_roi.x, 0), frame.cols);
	cropped_roi.y=MIN(MAX(cropped_roi.y, 0), frame.rows);
	cropped_roi+=Size(100,100);
	current_frame=frame(cropped_roi);
	current_frame.copyTo(cropped_frame);
	current_frame.copyTo(cielab_image);
	cvtColor(cielab_image,cielab_image, CV_RGB2Lab);
	vector<Rect> raw_detections;
	vector<double> detection_weights;
	//copyMakeBorder( current_frame, current_frame, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	//copyMakeBorder( cielab_image, cielab_image, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	this->detections.clear();
	int channels = frame.channels();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
	this->weights.resize(0);
	this->penalty_weights.resize(0);
	for (int k=0;k<args.nlevels;k++){
		int num_rows=(current_frame.rows- this->args.height + this->args.win_stride_height)/this->args.win_stride_height;
		int num_cols=(current_frame.cols- this->args.width + this->args.win_stride_width)/this->args.win_stride_width;
		if (num_rows*num_cols<=0) break;
		//Mat resized_frame;
		//current_frame.copyTo(resized_frame);
		double scaleMult=pow(args.scale,k);
		for(int i=0;i<num_rows;i++){
			for(int j=0;j<num_cols;j++){
				int row=i*this->args.win_stride_height;
				int col=j*this->args.win_stride_width;
				Rect current_window(col,row, this->args.width,this->args.height);
				Mat subImage = current_frame(current_window);
				VectorXd hogFeatures = this->genHog(subImage);
				VectorXd temp;
				MatrixXd temp_features_matrix;
				if(USE_COLOR){
					Mat subLabImage = cielab_image(current_window);
					VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
					temp_features_matrix.resize(1, hogFeatures.rows()+rawPixelsFeatures.rows());
					temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
					temp << hogFeatures, rawPixelsFeatures;
				}
				else{
					temp_features_matrix.resize(1, hogFeatures.rows());//
					temp.resize(hogFeatures.rows());//
					temp << hogFeatures;//
				}	
				//
				temp.normalize();				
				temp_features_matrix.row(0) = temp;
				VectorXd predict_prob = this->logistic_regression.predict(temp_features_matrix, true);
				if (predict_prob(0)>args.hit_threshold) {
					stringstream ss;
        			ss << predict_prob(0);
        			this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
					this->feature_values.row(this->feature_values.rows() - 1)=temp_features_matrix.row(0);
					this->weights.conservativeResize(this->weights.size() + 1 );
					this->weights(this->weights.size() - 1) = predict_prob(0);
					this->penalty_weights.conservativeResize(this->penalty_weights.size() + 1 );
					this->penalty_weights(this->penalty_weights.size() - 1) = predict_prob(0);
        			string disp = ss.str().substr(0,4);
        			rectangle( current_frame, Point(col,row),Point(col+current_window.width,row+20), Scalar(0,0,255), -1, 8,0 );
        			putText(current_frame, disp, Point(col+5, row+12), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255),1);
					rectangle( current_frame, current_window, Scalar(0,0,255), 1, LINE_8  );
					raw_detections.push_back(current_window);
					detection_weights.push_back(predict_prob(0));
				}
			}	
		}
		//rectangle( resized_frame, reference_roi, Scalar(255,255,255), 2, LINE_8  );
		cout << "-----------------------" << endl;
		string name= to_string(this->num_frame)+"_detections_raw.png";
		imwrite(name, current_frame);
		pyrDown( current_frame, current_frame, Size( cvCeil(current_frame.cols/args.scale) , cvCeil(current_frame.rows/args.scale)));
	}
	if(this->args.gr_threshold > 0) {
		nms(raw_detections,this->detections, args.gr_threshold, 1);
		//DPP dpp = DPP();
		//VectorXd qualityTerm;
		//this->detections = dpp.run(raw_detections,this->weights, this->weights, this->feature_values, qualityTerm, 1.0, 0.5, 0.1);
 	}
	else {
		for (unsigned int i = 0; i < raw_detections.size(); ++i)
		{
			this->detections.push_back(raw_detections[i]);	
		}
	}
	for(int i=0;i<detections.size();i++){
		rectangle( cropped_frame, this->detections[i], Scalar(0,0,255), 2, LINE_8  );
		this->detections[i]+=Point(cropped_roi.x,cropped_roi.y);				
	}
	string name2= to_string(this->num_frame)+"_detections_nms.png";
	imwrite(name2, cropped_frame); 
	cout << "Frame : " << this->num_frame << endl; 
	cout << ", raw_detections: " << raw_detections.size() << endl; 
	cout << "detections: " << detections.size() << endl;
	this->num_frame++; 
	return this->detections;
}

vector<Rect> CPU_LR_HOGDetector::detect(Mat &frame, vector<Rect> samples)
{
	Mat current_frame,cielab_image;
	frame.copyTo(current_frame);
	frame.copyTo(cielab_image);
	cvtColor(cielab_image,cielab_image, CV_RGB2Lab);
	vector<Rect> raw_detections;
	vector<double> detection_weights;
	copyMakeBorder( current_frame, current_frame, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	copyMakeBorder( cielab_image, cielab_image, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	this->detections.clear();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
	this->weights.resize(0);
	this->penalty_weights.resize(0);
	for (int k=0;k<args.nlevels;k++){
		Mat resized_frame;
		current_frame.copyTo(resized_frame);
		double scaleMult=pow(args.scale,k);
		for(int i=0;i<samples.size();i++){
			Rect current_window=samples[i];
			Mat subImage = current_frame(current_window);
			VectorXd hogFeatures = this->genHog(subImage);
			VectorXd temp;
			MatrixXd temp_features_matrix;
			if(USE_COLOR){
				Mat subLabImage = cielab_image(current_window);
				VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
				temp_features_matrix.resize(1, hogFeatures.rows()+rawPixelsFeatures.rows());
				temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
				temp << hogFeatures, rawPixelsFeatures;
			}
			else{
				temp_features_matrix.resize(1, hogFeatures.rows());//
				temp.resize(hogFeatures.rows());//
				temp << hogFeatures;//
			}
			temp.normalize();				
			temp_features_matrix.row(0) = temp;
			VectorXd predict_prob = this->logistic_regression.predict(temp_features_matrix, true);
			stringstream ss;
        	ss << predict_prob(0);
        		this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				this->feature_values.row(this->feature_values.rows() - 1)=temp_features_matrix.row(0);
				this->weights.conservativeResize(this->weights.size() + 1 );
				this->weights(this->weights.size() - 1) = predict_prob(0);
				this->penalty_weights.conservativeResize(this->penalty_weights.size() + 1 );
				this->penalty_weights(this->penalty_weights.size() - 1) = predict_prob(0);
        		string disp = ss.str().substr(0,4);
        		putText(resized_frame, disp, Point(current_window.x+5, current_window.y+10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
				rectangle( resized_frame, current_window, Scalar(0,0,255), 2, LINE_8  );
				raw_detections.push_back(current_window);
				detection_weights.push_back(predict_prob(0));
		}	
		cout << "-----------------------" << endl;
		string name= "resized_image_"+to_string(k)+".png";
		imwrite(name, resized_frame);
		pyrDown( current_frame, current_frame, Size( cvCeil(current_frame.cols/args.scale) , cvCeil(current_frame.rows/args.scale)));
	}
	cout << "raw_detections: " << raw_detections.size() << endl; 
	cout << "detections: " << detections.size() << endl;
	return this->detections;
}

void CPU_LR_HOGDetector::train(Mat &frame,Rect reference_roi)
{
	Mat cropped_frame,current_frame,cielab_image;
	int x_shift=50;
	int y_shift=50;
	Rect cropped_roi=reference_roi+Point(-x_shift,-y_shift);
	cropped_roi.x=MIN(MAX(cropped_roi.x, 0), frame.cols);
	cropped_roi.y=MIN(MAX(cropped_roi.y, 0), frame.rows);
	cropped_roi+=Size(100,100);
	cout << cropped_roi << reference_roi << endl;
	reference_roi.x=MIN(x_shift,reference_roi.x);
	reference_roi.y=MIN(y_shift,reference_roi.y);
	cout << cropped_roi << reference_roi << endl;
	cropped_frame=frame(cropped_roi);
	cropped_frame.copyTo(current_frame);
	cropped_frame.copyTo(cielab_image);
	cvtColor(cielab_image,cielab_image, CV_RGB2Lab);
	int num_rows=(current_frame.rows- this->args.height + this->args.win_stride_height)/this->args.win_stride_height;
	int num_cols=(current_frame.cols- this->args.width + this->args.win_stride_width)/this->args.win_stride_width;
	this->detections.clear();
	int channels = frame.channels();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
	this->labels.resize(0);
	uniform_real_distribution<double> unif(0.0,1.0);
	for(int i=0;i<num_rows;i++){
		for(int j=0;j<num_cols;j++){
			int row=i*this->args.win_stride_height;
			int col=j*this->args.win_stride_width;
			Rect current_window(col,row,this->args.width,this->args.height);
			Rect intersection = reference_roi & current_window;
			double overlap=(double)intersection.area()/(double)reference_roi.area();
			double uni_rand = (overlap > args.overlap_threshold) ? 1.0 : unif(this->generator);
			Mat subImage = current_frame(current_window);
			if(uni_rand > args.p_accept ){
				VectorXd hogFeatures = this->genHog(subImage);
				VectorXd temp;
				MatrixXd temp_features_matrix;
				if(USE_COLOR){
					Mat subLabImage = cielab_image(current_window);		
					VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
					temp_features_matrix.resize(1, hogFeatures.rows()+rawPixelsFeatures.rows());
					temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
					temp << hogFeatures, rawPixelsFeatures;
				}
				else{
					temp_features_matrix.resize(1, hogFeatures.rows());//
					temp.resize(hogFeatures.rows());//
					temp << hogFeatures;//
				}	
				temp.normalize();		
				temp_features_matrix.row(0) = temp;
				this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				this->feature_values.row(this->feature_values.rows() - 1)=temp_features_matrix.row(0);
				this->labels.conservativeResize(this->labels.size() + 1 );
				this->labels(this->labels.size() - 1) = (overlap > args.overlap_threshold) ? 1.0 : 0.0;
				if(overlap > args.overlap_threshold) rectangle( current_frame, current_window, Scalar(255,255,255), 1, LINE_AA );
				//else rectangle( current_frame, current_window, Scalar(0,0,0), 1, LINE_AA );
			}
		}
	}
	cout << "positive examples : " << (this->labels.array() > 0).count() << endl;
	cout << "negative examples : " << (this->labels.array() <= 0).count() << endl;
	rectangle( current_frame, reference_roi, Scalar(0,255,0), 2, LINE_AA );
	imwrite("resized_image.png", current_frame);
	if(!this->logistic_regression.initialized){
		this->logistic_regression.init(this->feature_values, this->labels, args.lambda,false,true,true);	
	} 
	else{
		this->logistic_regression.setData(this->feature_values, this->labels);
	}
	cout << this->feature_values.rows() << "," << this->feature_values.cols() << "," << this->labels.rows() << endl;
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
	//exit(0);
}

/*
void CPU_LR_HOGDetector::train(Mat &frame,Rect reference_roi)
{
	Mat cropped_frame,current_frame,cielab_image;
	int x_shift=80;
	int y_shift=80;
	Rect cropped_roi=reference_roi+Point(-x_shift,-y_shift);
	cropped_roi.x=MIN(MAX(cropped_roi.x, 0), frame.cols);
	cropped_roi.y=MIN(MAX(cropped_roi.y, 0), frame.rows);
	cropped_roi+=Size(2*reference_roi.width,2*reference_roi.height);
	cout << cropped_roi << reference_roi << endl;
	reference_roi.x=y_shift;
	reference_roi.y=y_shift;
	cropped_frame=frame(cropped_roi);
	cropped_frame.copyTo(current_frame);
	cropped_frame.copyTo(cielab_image);
	cvtColor(cielab_image,cielab_image, CV_RGB2Lab);
	copyMakeBorder( current_frame, current_frame, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	copyMakeBorder( cielab_image, cielab_image, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	this->detections.clear();
	int channels = frame.channels();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
	this->labels.resize(0);
	vector<Rect> positiveBox;
	vector<Rect> negativeBox;
	samplerBox(current_frame, reference_roi, 100, positiveBox, negativeBox);

	for (int i = 0; i < positiveBox.size(); ++i){
		Rect current_window = positiveBox.at(i);
		Mat subImage = current_frame(current_window);
		VectorXd hogFeatures = this->genHog(subImage);
		VectorXd temp;
		MatrixXd temp_features_matrix;
		if(USE_COLOR){
			Mat subLabImage = cielab_image(current_window);
			VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
			temp_features_matrix.resize(1, hogFeatures.rows()+rawPixelsFeatures.rows());
			temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
			temp << hogFeatures, rawPixelsFeatures;
		}
		else{
			temp_features_matrix.resize(1, hogFeatures.rows());
			temp.resize(hogFeatures.rows());
			temp << hogFeatures;
		}
		temp.normalize();				
		temp_features_matrix.row(0) = temp;
		this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
		this->feature_values.row(this->feature_values.rows() - 1)=temp_features_matrix.row(0);
		this->labels.conservativeResize(this->labels.size() + 1 );
		this->labels(this->labels.size() - 1) = 1.0;
		rectangle( current_frame, current_window, Scalar(255,255,255), 1, LINE_AA );
	}

	for (int i = 0; i < negativeBox.size(); ++i){
		Rect current_window = negativeBox.at(i);
		Mat subImage = current_frame(current_window);
		VectorXd hogFeatures = this->genHog(subImage);
		VectorXd temp;
		MatrixXd temp_features_matrix;
		if(USE_COLOR){
			Mat subLabImage = cielab_image(current_window);
			VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
			temp_features_matrix.resize(1, hogFeatures.rows()+rawPixelsFeatures.rows());
			temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
			temp << hogFeatures, rawPixelsFeatures;
		}
		else{
			temp_features_matrix.resize(1, hogFeatures.rows());
			temp.resize(hogFeatures.rows());
			temp << hogFeatures;
		}
		//temp.normalize();				
		temp_features_matrix.row(0) = temp;
		this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
		this->feature_values.row(this->feature_values.rows() - 1)=temp_features_matrix.row(0);
		this->labels.conservativeResize(this->labels.size() + 1 );
		this->labels(this->labels.size() - 1) = 0.0;
		rectangle( current_frame, current_window, Scalar(0,0,0), 1, LINE_AA );
	}
	
	//else rectangle( current_frame, current_window, Scalar(0,0,0), 1, LINE_AA );


	cout << "positive examples : " << (this->labels.array() > 0).count() << endl;
	cout << "negative examples : " << (this->labels.array() <= 0).count() << endl;
	rectangle( current_frame, reference_roi, Scalar(0,255,0), 2, LINE_AA );
	imwrite("resized_image.png", current_frame);
	if(!this->logistic_regression.initialized){
		this->logistic_regression.init(this->feature_values, this->labels, args.lambda,false,true,true);	
	} 
	else{
		this->logistic_regression.setData(this->feature_values, this->labels);
	}
	cout << this->feature_values.rows() << "," << this->feature_values.cols() << "," << this->labels.rows() << endl;
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
	//exit(0);
}
*/

void CPU_LR_HOGDetector::train()
{
	this->logistic_regression.init(this->feature_values, this->labels, args.lambda, false,true,true);
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
	VectorXd weights = this->logistic_regression.getWeights();
	VectorXd bias(1);
	bias << this->logistic_regression.getBias();
	tools.writeToCSVfile("INRIA_Model_weights.csv", weights);
	tools.writeToCSVfile("INRIA_Model_means.csv", this->logistic_regression.featureMean.transpose());
	tools.writeToCSVfile("INRIA_Model_stds.csv", this->logistic_regression.featureStd.transpose());
	tools.writeToCSVfile("INRIA_Model_maxs.csv", this->logistic_regression.featureMax.transpose());
	tools.writeToCSVfile("INRIA_Model_mins.csv", this->logistic_regression.featureMin.transpose());
	tools.writeToCSVfile("INRIA_Model_bias.csv", bias);
}

/*VectorXd CPU_LR_HOGDetector::predict(MatrixXd data)
{
	return this->logistic_regression.predict(data, false);
}*/


MatrixXd CPU_LR_HOGDetector::getFeatureValues(Mat &current_frame)
{
	vector<float> temp_features;
	Size win_stride(args.win_stride_width, args.win_stride_height);
	this->hog.compute(current_frame, temp_features, win_stride);
	vector<double> features(temp_features.begin(), temp_features.end());
	double* ptr = &features[0];
	int rows = (int)(features.size()/this->hog.getDescriptorSize());
	Map<MatrixXd> hogFeatures(ptr, rows, this->hog.getDescriptorSize());
	//for (int k = 0; k < hogFeatures.rows(); ++k) hogFeatures.row(k).normalize();
	return hogFeatures;
}

VectorXd CPU_LR_HOGDetector::genHog(Mat &frame)
{	
	int interpolation;
	if(args.hog_width > frame.size().height){
        interpolation = INTER_LINEAR;
      }else{
        interpolation = INTER_AREA;
    }
	Mat current_frame;
	frame.convertTo( current_frame, CV_8U);
	cvtColor(current_frame, current_frame, CV_RGB2GRAY);
	resize(current_frame,current_frame,Size(args.hog_width, args.hog_height),0,0,interpolation);		
	vector<float> temp_features;
	this->hog.compute(current_frame, temp_features);
	vector<double> features(temp_features.begin(), temp_features.end());
	double* ptr = &features[0];
	Map<VectorXd> hogFeatures(ptr, this->hog.getDescriptorSize());
	hogFeatures.normalize();
	return hogFeatures;
}

VectorXd CPU_LR_HOGDetector::genRawPixels(Mat &frame)
{
	/*int interpolation;
	if(args.hog_width/2 > frame.size().height){
        interpolation = INTER_LINEAR;
      }else{
        interpolation = INTER_AREA;
    }
    Mat current_frame;
    frame.copyTo(current_frame);
	resize(current_frame,current_frame,Size(args.hog_width/2, args.hog_height/2),0,0,interpolation);
	//current_frame.convertTo( current_frame, CV_32F );
	int channels = current_frame.channels();
	vector<Mat> frame_channels(channels);
	split(current_frame, frame_channels); // get per channel
	VectorXd rawPixelsFeatures(current_frame.cols*current_frame.rows*channels);
	int cidx=0;
	for (int ch = 0; ch < channels; ++ch){
		for(int c = 0; c < current_frame.cols ; c++){
			for(int r = 0; r < current_frame.rows ; r++){   
		        rawPixelsFeatures(cidx) = frame_channels[ch].at<uchar>(r,c);
		        cidx++;
		    }
		}
	}
	double normTerm = rawPixelsFeatures.norm();
	if (normTerm > 1e-6){
		rawPixelsFeatures.normalize();
		//rawPixelsFeatures =  rawPixelsFeatures.array()/normTerm;
	}
	return rawPixelsFeatures;*/
	Mat hist;
	MatrixXd mat_hist;
	calc_hist_hsv(frame,hist);
	cv2eigen(hist,mat_hist);
	Map<VectorXd> color_hist(mat_hist.data(), mat_hist.size());
	return color_hist;
}


void CPU_LR_HOGDetector::loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias){
	this->logistic_regression.init(false, true, true);
	this->logistic_regression.setWeights(weights);
	this->logistic_regression.setBias(bias);
	this->logistic_regression.featureMean = featureMean;
	this->logistic_regression.featureStd = featureStd;
	this->logistic_regression.featureMax = featureMax;
	this->logistic_regression.featureMin = featureMin;
}


void CPU_LR_HOGDetector::samplerBox(Mat &current_frame, Rect ground_truth, int n_particles, vector<Rect>& sampleBox, vector<Rect>& negativeBox){
	const float POS_STD=1.0;
	const float SCALE_STD=1.0;
	const float DT=1.0;
	const float THRESHOLD=1.0;
	const float OVERLAP_RATIO=0.2;
	mt19937 generator;
	unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	generator.seed(seed1);
	vector<VectorXd> theta_x;
    theta_x.clear();
    RowVectorXd theta_x_pos(2);
    theta_x_pos << POS_STD,POS_STD;
    theta_x.push_back(theta_x_pos);
    RowVectorXd theta_x_scale(2);
    theta_x_scale << SCALE_STD,SCALE_STD;
    theta_x.push_back(theta_x_scale);
    normal_distribution<double> negative_random_pos(0.0,20.0);
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    sampleBox.clear();
    negativeBox.clear();
    Size im_size=current_frame.size();
    int left = MAX(ground_truth.x, 1);
    int top = MAX(ground_truth.y, 1);
    int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
    int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
    Rect reference_roi=Rect(left, top, right - left, bottom - top);
    if(reference_roi.width>0 && (reference_roi.x+reference_roi.width)<im_size.width &&
        reference_roi.height>0 && (reference_roi.y+reference_roi.height)<im_size.height){
        for (int i=0;i<n_particles;i++){
            Roi state;
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=position_random_y(generator);
            //float _dw=scale_random_width(generator);
            //float _dh=scale_random_height(generator);
            _x=MIN(MAX(cvRound(reference_roi.x+_dx),0),im_size.width);
            _y=MIN(MAX(cvRound(reference_roi.y+_dy),0),im_size.height);
            _width=MIN(MAX(cvRound(reference_roi.width),10.0),im_size.width);
            _height=MIN(MAX(cvRound(reference_roi.height),10.0),im_size.height);
            //_width=MIN(MAX(cvRound(state.width+state.scale),0),im_size.width);
            //_height=MIN(MAX(cvRound(state.height+state.scale),0),im_size.height);
            if( (_x+_width)<im_size.width
                && _x>0
                && (_y+_height)<im_size.height
                && _y>0
                && _width<im_size.width
                && _height<im_size.height
                && _width>0 && _height>0){
                state.x_p=reference_roi.x;
                state.y_p=reference_roi.y;
                state.width_p=reference_roi.width;
                state.height_p=reference_roi.height;
                state.x=_x;
                state.y=_y;
                state.width=_width;
                state.height=_height;
                state.scale_p=state.scale;
                state.scale=1.0;
            }
            else{
                state.x=reference_roi.x;
                state.y=reference_roi.y;
                state.width=cvRound(reference_roi.width);
                state.height=cvRound(reference_roi.height);
                state.x_p=reference_roi.x;
                state.y_p=reference_roi.y;
                state.width_p=cvRound(reference_roi.width);
                state.height_p=cvRound(reference_roi.height);
                state.scale=1.0;
            }
            Rect box(state.x, state.y, state.width, state.height);
            sampleBox.push_back(box);
        }
        for (int i=0;i<n_particles;i++){
            Rect box=reference_roi;
            Rect intersection=(box & reference_roi);
            while( double(intersection.area())/double(reference_roi.area()) > OVERLAP_RATIO ){
                float _dx=negative_random_pos(generator);
                float _dy=negative_random_pos(generator);
                box.x=MIN(MAX(cvRound(reference_roi.x+_dx),0),im_size.width);
                box.y=MIN(MAX(cvRound(reference_roi.y+_dy),0),im_size.height);
                box.width=MIN(MAX(cvRound(reference_roi.width),0),im_size.width-box.x);
                box.height=MIN(MAX(cvRound(reference_roi.height),0),im_size.height-box.y);
                intersection=(box & reference_roi);
            }
            negativeBox.push_back(box);
        }
    }
}
