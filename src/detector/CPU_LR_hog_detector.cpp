#include "CPU_LR_hog_detector.hpp"

void CPU_LR_HOGDetector::init(double group_threshold, double hit_threshold,Rect reference_roi){
	args.make_gray = true;
    args.resize_src = false;
    args.width = reference_roi.width;
    args.height = reference_roi.height;
    args.hog_width = 32;
    args.hog_height = 32;
    args.scale = 2;
    args.nlevels = 1;
    args.gr_threshold = group_threshold;
    args.hit_threshold = hit_threshold;
    args.hit_threshold_auto = false;
    args.win_width = args.width ;
    args.win_stride_width = 10;
    args.win_stride_height = 10;
    args.block_width = 16;
    args.block_stride_width = 8;
    args.block_stride_height = 8;
    args.cell_width = 8;
    args.nbins = 9;
    args.overlap_threshold=0.5;
    args.p_accept = 0.99;
    args.lambda = 1.0;
    args.epsilon= 1e-1;
    args.tolerance = 1e-1;
    args.n_iterations = 1e2;
    args.padding = 16;
    //this->n_descriptors = (args.width/args.cell_width-1)*(args.height/args.cell_width-1)*args.nbins*(args.block_width*args.block_width/(args.cell_width*args.cell_width));
    //this->n_descriptors = 3780;
	Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.hog_width, args.hog_height);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->hog = HOGDescriptor(win_size, block_size, block_stride, cell_size, args.nbins);
    this->n_descriptors = this->hog.getDescriptorSize();
    this->generator.seed(seed1);
    this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	this->labels.resize(0);
}



vector<Rect> CPU_LR_HOGDetector::detect(Mat &frame)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	vector<Rect> raw_detections;
	vector<double> detection_weights;
	copyMakeBorder( current_frame, current_frame, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	this->detections.clear();
	int channels = frame.channels();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors + (this->args.hog_width/2)*(this->args.hog_height/2)*channels);
	//this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
	this->weights.resize(0);
	this->penalty_weights.resize(0);
	for (int k=0;k<args.nlevels;k++){
		int num_rows=(current_frame.rows- this->args.height + this->args.win_stride_height)/this->args.win_stride_height;
		int num_cols=(current_frame.cols- this->args.width + this->args.win_stride_width)/this->args.win_stride_width;
		if (num_rows*num_cols<=0) break;
		Mat resized_frame;
		current_frame.copyTo(resized_frame);
		cout << "frame : " << current_frame.rows << "," << current_frame.cols << endl;
		cout << "num windows : " << num_rows << "," << num_cols << endl;
		double scaleMult=pow(args.scale,k);
		for(int i=0;i<num_rows;i++){
			for(int j=0;j<num_cols;j++){
				int row=i*this->args.win_stride_height;
				int col=j*this->args.win_stride_width;
				Rect current_window(col,row, this->args.width,this->args.height);
				Mat subImage = current_frame(current_window); 
				VectorXd hogFeatures = this->genHog(subImage);
				VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
				MatrixXd temp_features_matrix(1, hogFeatures.rows()+rawPixelsFeatures.rows());
				VectorXd temp(hogFeatures.rows()+rawPixelsFeatures.rows());
				temp << hogFeatures, rawPixelsFeatures;
				//MatrixXd temp_features_matrix(1, hogFeatures.rows());//
				//VectorXd temp(hogFeatures.rows());//
				//temp << hogFeatures;//
				//temp.normalize();				
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
        			putText(resized_frame, disp, Point(col+5, row+10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
					rectangle( resized_frame, current_window, Scalar(0,0,255), 2, LINE_8  );
					raw_detections.push_back(current_window);
					detection_weights.push_back(predict_prob(0));
				}
			}	
		}
		cout << "-----------------------" << endl;
		string name= "resized_image_"+to_string(k)+".png";
		imwrite(name, resized_frame);
		pyrDown( current_frame, current_frame, Size( cvCeil(current_frame.cols/args.scale) , cvCeil(current_frame.rows/args.scale)));
	}
	if(this->args.gr_threshold > 0) {
		nms2(raw_detections, detection_weights, this->detections, args.gr_threshold, 10);
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
	cout << "detections: " << detections.size() << endl;
	return this->detections;
}

vector<Rect> CPU_LR_HOGDetector::detect(Mat &frame, vector<Rect> samples)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	vector<Rect> raw_detections;
	vector<double> detection_weights;
	copyMakeBorder( current_frame, current_frame, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	this->detections.clear();
	int channels = frame.channels();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors + (this->args.hog_width/2)*(this->args.hog_height/2)*channels);
	//this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
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
			VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
			MatrixXd temp_features_matrix(1, hogFeatures.rows()+rawPixelsFeatures.rows());
			VectorXd temp(hogFeatures.rows()+rawPixelsFeatures.rows());
			temp << hogFeatures, rawPixelsFeatures;
			//MatrixXd temp_features_matrix(1, hogFeatures.rows()); //
			//VectorXd temp(hogFeatures.rows());//
			//temp << hogFeatures; //
			//temp.normalize();				
			temp_features_matrix.row(0) = temp;
			VectorXd predict_prob = this->logistic_regression.predict(temp_features_matrix, true);
			//cout << predict_prob.transpose() << endl;
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
        		putText(resized_frame, disp, Point(current_window.x+5, current_window.y+10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
				rectangle( resized_frame, current_window, Scalar(0,0,255), 2, LINE_8  );
				raw_detections.push_back(current_window);
				detection_weights.push_back(predict_prob(0));
			}
		}	
		cout << "-----------------------" << endl;
		string name= "resized_image_"+to_string(k)+".png";
		imwrite(name, resized_frame);
		pyrDown( current_frame, current_frame, Size( cvCeil(current_frame.cols/args.scale) , cvCeil(current_frame.rows/args.scale)));
	}
	if(this->args.gr_threshold > 0) {
		nms2(raw_detections, detection_weights, this->detections, args.gr_threshold, 10);
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
	cout << "detections: " << detections.size() << endl;
	return this->detections;
}

void CPU_LR_HOGDetector::train(Mat &frame,Rect reference_roi)
{
	
	Mat current_frame;
	frame.copyTo(current_frame);
	int num_rows=(frame.rows- this->args.height + this->args.win_stride_height)/this->args.win_stride_height;
	int num_cols=(frame.cols- this->args.width + this->args.win_stride_width)/this->args.win_stride_width;
	this->detections.clear();
	int channels = frame.channels();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors + (this->args.hog_width/2)*(this->args.hog_height/2)*channels);
	//this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
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
			if(uni_rand>args.p_accept){
				Mat subImage = current_frame(current_window);
				VectorXd hogFeatures = this->genHog(subImage);
				VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
				//cout << "["<< rawPixelsFeatures.transpose() << "]\n"<<endl;
				MatrixXd temp_features_matrix(1, hogFeatures.rows()+rawPixelsFeatures.rows());
				VectorXd temp(hogFeatures.rows()+rawPixelsFeatures.rows());
				temp << hogFeatures, rawPixelsFeatures;
				//MatrixXd temp_features_matrix(1, hogFeatures.rows()); //
				//VectorXd temp(hogFeatures.rows()); //
				//temp << hogFeatures; //
				//temp.normalize();				
				temp_features_matrix.row(0) = temp;
				this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				this->feature_values.row(this->feature_values.rows() - 1)=temp_features_matrix.row(0);
				this->labels.conservativeResize(this->labels.size() + 1 );
				this->labels(this->labels.size() - 1) = (overlap > args.overlap_threshold) ? 1.0 : 0.0;
			}
		}
	}
	cout << "positive examples : " << (this->labels.array() > 0).count() << endl;
	cout << "negative examples : " << (this->labels.array() <= 0).count() << endl;
	rectangle( current_frame, reference_roi, Scalar(0,255,0), 2, LINE_AA );
	imwrite("resized_image.png", current_frame);
	this->logistic_regression.init(this->feature_values, this->labels, args.lambda,true,true,true);
	cout << this->feature_values.rows() << "," << this->feature_values.cols() << "," << this->labels.rows() << endl;
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
}

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
	Map<VectorXd> hogFeatures(ptr, this->n_descriptors);
	//hogFeatures.normalize();
	return hogFeatures;
}

VectorXd CPU_LR_HOGDetector::genRawPixels(Mat &frame)
{
	int interpolation;
	if(args.hog_width/2 > frame.size().height){
        interpolation = INTER_LINEAR;
      }else{
        interpolation = INTER_AREA;
    }
    Mat current_frame;
    //frame.convertTo( current_frame, CV_8U );
	resize(frame,current_frame,Size(args.hog_width/2, args.hog_height/2),0,0,interpolation);
	//current_frame.convertTo( current_frame, CV_32F );
	int channels = current_frame.channels();
	vector<Mat> frame_channels(channels);
	split(current_frame, frame_channels); // get per channel
	if(channels == 3){
		if (norm(frame_channels[0]-frame_channels[1]) > 1e-6){
			cvtColor(current_frame,current_frame, CV_RGB2Lab);
		}
		//else{
		//	cvtColor(current_frame, current_frame, CV_RGB2GRAY);
		//}
		frame_channels.clear();
		channels = current_frame.channels();
  		split(current_frame, frame_channels);
	}
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
	return rawPixelsFeatures;
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