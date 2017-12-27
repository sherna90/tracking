#include "CPU_LR_hog_detector.hpp"

#ifndef PARAMS
const bool USE_COLOR=true;
#endif

void CPU_LR_HOGDetector::init(double group_threshold, double hit_threshold,Rect reference_roi){
	args.make_gray = true;
    args.resize_src = false;
    args.hog_width = 32;
    args.hog_height = 32;
    args.gr_threshold = group_threshold;
    args.hit_threshold = hit_threshold;
    args.n_orients = 9;
    args.bin_size = 8;
    args.overlap_threshold=0.7;
    args.p_accept = 0.999;
    args.lambda = 0.01;
    args.alpha= 0.9;
    args.step_size = 0.001;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
	args.n_iterations = 1000;
	//this->n_descriptors = (args.width/args.cell_width-1)*(args.height/args.cell_width-1)*args.nbins*(args.block_width*args.block_width/(args.cell_width*args.cell_width));
	if(USE_COLOR){
    	int channels = 3;
    	this->n_descriptors=(args.n_orients*3+5-1)*(args.hog_width/args.bin_size)*(args.hog_height/args.bin_size) + (this->args.hog_width/2)*(this->args.hog_height/2)*channels;
    }
    else this->n_descriptors = (args.n_orients*3+5-1)*(args.hog_width/args.bin_size)*(args.hog_height/args.bin_size);
    this->generator.seed(seed1);
    this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	this->labels.resize(0);
	this->num_frame=0;
	this->max_value=1.0;
}



vector<Rect> CPU_LR_HOGDetector::detect(Mat &frame,Rect reference_roi)
{
	Mat cropped_frame,current_frame;
	float scale_w,scale_h;
	frame.copyTo(current_frame);
	if(args.resize_src){
		scale_w=current_frame.cols/320.0f;
		scale_h=current_frame.rows/240.0f;
		resize(current_frame,current_frame,Size(320,240),0,0,INTER_LINEAR);
		reference_roi.x=cvRound(reference_roi.x/scale_w);
		reference_roi.y=cvRound(reference_roi.y/scale_h);
		reference_roi.width=cvRound(reference_roi.width/scale_w);
		reference_roi.height=cvRound(reference_roi.height/scale_h);
	}
	else{
		scale_w=1.0f;
		scale_h=1.0f;	
	}
	int x_shift=40;
	int y_shift=40;
	Rect cropped_roi=reference_roi+Point(-x_shift,-y_shift);
	int w_crop=(current_frame.cols-(cropped_roi.x+2*x_shift) >=0 )?  2*x_shift : current_frame.cols-cropped_roi.x;
	int h_crop=(current_frame.rows-(cropped_roi.y+2*y_shift) >=0 )?  2*y_shift : current_frame.rows-cropped_roi.y;
	cropped_roi+=Size(w_crop,h_crop);
	//current_frame=frame(cropped_roi);
	vector<Rect> samples,raw_detections;
	this->detections.clear();
	//samples=region_proposal(current_frame);
	samples=sliding_window(current_frame,reference_roi,10);
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	this->weights.clear();
	double max_prob=0.0;
	MatrixXd temp_features_matrix = MatrixXd::Zero(samples.size(),this->n_descriptors);
		for(int i=0;i<samples.size();i++){
			Rect current_window=samples[i];
			Mat subImage = current_frame(current_window);
			VectorXd hogFeatures = this->genHog(subImage);
			VectorXd temp;
			if(USE_COLOR){
				VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
				temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
				temp << hogFeatures, rawPixelsFeatures;
			}
			else{
				temp.resize(hogFeatures.rows());
				temp << hogFeatures;
			}			
			temp_features_matrix.row(i) = temp;	
		}
		VectorXd dataNorm = temp_features_matrix.rowwise().squaredNorm().array().sqrt();
		temp_features_matrix = temp_features_matrix.array().colwise() / dataNorm.array();
		VectorXd predict_prob = this->logistic_regression.predict(temp_features_matrix, true);
		for (int i = 0; i < predict_prob.rows(); ++i)
		{
			if(predict_prob(i)>args.hit_threshold){
				Rect current_window=samples[i];
				stringstream ss;
	    		ss << predict_prob(i);
				max_prob=MAX(max_prob,predict_prob(i));
				string disp = ss.str().substr(0,4);
	    		rectangle( current_frame, Point(current_window.x,current_window.y),Point(current_window.x+current_window.width,current_window.y+20), Scalar(0,0,255), -1, 8,0 );
	    		putText(current_frame, disp, Point(current_window.x+5, current_window.y+12), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255),1);
				rectangle( current_frame, current_window, Scalar(0,0,255), 1, LINE_8  );
				this->weights.push_back(predict_prob(i));
				this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				this->feature_values.row(this->feature_values.rows()-1) << temp_features_matrix.row(i);
				raw_detections.push_back(current_window);
			}
		}
	if(this->args.gr_threshold > 0) {
		nms2(raw_detections,this->weights,this->detections, args.gr_threshold,1);
	}
	else{
		this->detections=raw_detections;
	}
	rectangle( current_frame, reference_roi, Scalar(0,255,0), 2, LINE_AA );
	imwrite("resized_image.png", current_frame);
	this->num_frame++;
	return this->detections;
}

vector<double> CPU_LR_HOGDetector::detect(Mat &frame, vector<Rect> samples)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	float scale_w,scale_h;
	frame.copyTo(current_frame);
	if(args.resize_src){
		scale_w=current_frame.cols/320.0f;
		scale_h=current_frame.rows/240.0f;
		resize(frame,current_frame,Size(320,240),0,0,INTER_LINEAR);
	}
	this->weights.clear();
	double max_prob=0.0;
	MatrixXd temp_features_matrix = MatrixXd::Zero(samples.size(),this->n_descriptors);
		for(int i=0;i<samples.size();i++){
			Rect current_window=samples[i];
			if(args.resize_src){
				current_window.x=cvRound(current_window.x/scale_w);
				current_window.y=cvRound(current_window.y/scale_h);
				current_window.width=cvRound(current_window.width/scale_w);
				current_window.height=cvRound(current_window.height/scale_h);
			}
			Mat subImage = current_frame(current_window);
			VectorXd hogFeatures = this->genHog(subImage);
			VectorXd temp;
			if(USE_COLOR){
				VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
				temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
				temp << hogFeatures, rawPixelsFeatures;
			}
			else{
				temp.resize(hogFeatures.rows());
				temp << hogFeatures;
			}
			//temp.normalize();				
			temp_features_matrix.row(i) = temp;	
		}
		VectorXd dataNorm = temp_features_matrix.rowwise().squaredNorm().array().sqrt();
		temp_features_matrix = temp_features_matrix.array().colwise() / dataNorm.array();
		VectorXd predict_prob = this->logistic_regression.predict(temp_features_matrix, true);
		for (int i = 0; i < predict_prob.rows(); ++i)
		{
			Rect current_window=samples[i];
			stringstream ss;
	    	ss << predict_prob(i);
	    	max_prob=MAX(max_prob,predict_prob(i));
			//this->feature_values.row(i)=temp_features_matrix.row(i);
			this->weights.push_back(predict_prob(i));
			//string disp = ss.str().substr(0,4);
	    	//rectangle( current_frame, Point(current_window.x,current_window.y),Point(current_window.x+current_window.width,current_window.y+20), Scalar(0,0,255), -1, 8,0 );
	    	//putText(current_frame, disp, Point(current_window.x+5, current_window.y+12), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255),1);
			//rectangle( current_frame, current_window, Scalar(0,0,255), 1, LINE_8  );
		}
	this->num_frame++;
	return this->weights;
}

void CPU_LR_HOGDetector::train(Mat &frame,Rect reference_roi)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	vector<Rect> samples;
	int stride=1;
	MatrixXd positiveFeatures = MatrixXd::Zero(0,this->n_descriptors);
	MatrixXd negativeFeatures = MatrixXd::Zero(0,this->n_descriptors);
	VectorXd positiveLabels = VectorXd::Zero(0);
	VectorXd negativeLabels = VectorXd::Zero(0);
	uniform_real_distribution<double> unif(0.0,1.0);
	samples=sliding_window(current_frame,reference_roi,stride);
	for(unsigned int i=0;i<samples.size();i++){
		Rect current_window=samples.at(i);
		double Intersection = (double)(reference_roi & current_window).area();
		double Union=(double)reference_roi.area()+(double)current_window.area()-Intersection;
		double IoU=Intersection/Union;
		double uni_rand = (IoU > args.overlap_threshold) ? 1.0 : unif(this->generator);
		if(uni_rand>args.p_accept){
			Mat subImage = current_frame(current_window);
			VectorXd hogFeatures = this->genHog(subImage);
			VectorXd temp;
			if(USE_COLOR){
				VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
				temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
				temp << hogFeatures, rawPixelsFeatures;
			}
			else{
				temp.resize(hogFeatures.rows());//
				temp << hogFeatures;//
			}	
			//temp.normalize();
			if(IoU > args.overlap_threshold){
				positiveFeatures.conservativeResize(positiveFeatures.rows() + 1, NoChange);
				positiveFeatures.row(positiveFeatures.rows() - 1)=temp;
				positiveLabels.conservativeResize(positiveLabels.size() + 1 );
				positiveLabels(positiveLabels.size() - 1) = 1.0;
				rectangle( current_frame, current_window, Scalar(255,255,255), 1, LINE_AA );
			}
			else{
				negativeFeatures.conservativeResize(negativeFeatures.rows() + 1, NoChange);
				negativeFeatures.row(negativeFeatures.rows() - 1)=temp;
				negativeLabels.conservativeResize(negativeLabels.size() + 1 );
				negativeLabels(negativeLabels.size() - 1) = 0.0;
			}
			//else rectangle( current_frame, current_window, Scalar(0,0,0), 1, LINE_AA );
		}
	}
	VectorXd pdataNorm = positiveFeatures.rowwise().squaredNorm().array().sqrt();
	positiveFeatures = positiveFeatures.array().colwise() / pdataNorm.array();
	VectorXd ndataNorm = negativeFeatures.rowwise().squaredNorm().array().sqrt();
	negativeFeatures = negativeFeatures.array().colwise() / ndataNorm.array();
	
	int fRows = negativeFeatures.rows() + positiveFeatures.rows();
	this->feature_values.resize(fRows, this->n_descriptors);
	this->feature_values << positiveFeatures, negativeFeatures;
	this->labels.resize(fRows);
	this->labels << positiveLabels, negativeLabels;
	cout << "positive examples : " << (this->labels.array() > 0).count() << endl;
	cout << "negative examples : " << (this->labels.array() <= 0).count() << endl;
	rectangle( current_frame, reference_roi, Scalar(0,255,0), 2, LINE_AA );
	imwrite("resized_image.png", current_frame);
	if(!this->logistic_regression.initialized){
		this->logistic_regression.init(this->feature_values, this->labels, args.lambda,true,true,true);	
	} 
	else{
		this->logistic_regression.setData(this->feature_values, this->labels);
	}
	cout << this->feature_values.rows() << "," << this->feature_values.cols() << "," << this->labels.rows() << endl;
	int num_batches=this->feature_values.rows()/100;
	this->logistic_regression.train(num_batches*args.n_iterations,100,args.alpha, args.step_size);
}


void CPU_LR_HOGDetector::train()
{
	this->logistic_regression.init(this->feature_values, this->labels, args.lambda, false,false,true);
	this->logistic_regression.train(args.n_iterations, args.alpha, args.step_size);
	VectorXd weights = this->logistic_regression.getWeights();
	VectorXd bias(1);
	bias << this->logistic_regression.getBias();
	tools.writeToCSVfile("Model_weights.csv", weights);
	tools.writeToCSVfile("Model_means.csv", this->logistic_regression.featureMean.transpose());
	tools.writeToCSVfile("Model_stds.csv", this->logistic_regression.featureStd.transpose());
	tools.writeToCSVfile("Model_maxs.csv", this->logistic_regression.featureMax.transpose());
	tools.writeToCSVfile("Model_mins.csv", this->logistic_regression.featureMin.transpose());
	tools.writeToCSVfile("Model_bias.csv", bias);
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
	resize(frame,current_frame,Size(args.hog_width, args.hog_height),0,0,interpolation);
	cv::cvtColor(current_frame, current_frame, CV_BGR2GRAY);
	current_frame.convertTo(current_frame, CV_32FC1);
	//extract(const cv::Mat & img, int use_hog = 2, int bin_size = 4, int n_orients = 9, int soft_bin = -1, float clip = 0.2)
	vector<Mat> mat_hog_features=FHoG::extract(current_frame, 2, args.bin_size, args.n_orients);
	int hog_channels=mat_hog_features.size();
	int cidx=0;
	VectorXd hog_features=VectorXd::Zero((args.n_orients*3+5-1)*(args.hog_width/args.bin_size)*(args.hog_height/args.bin_size));
	for (unsigned int ch = 0; ch < hog_channels; ++ch){
		for (int i = 0; i < mat_hog_features[ch].rows; i++){   
			for (int j = 0; j < mat_hog_features[ch].cols; j++){
				hog_features(cidx) =mat_hog_features[ch].at<float>(i,j);
				cidx++;
			}
		}
	}
	hog_features.normalize();
	return hog_features;
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
  frame.copyTo(current_frame);
  resize(current_frame,current_frame,Size(args.hog_width/2, args.hog_height/2),0,0,interpolation);
  //cvtColor(current_frame, current_frame, COLOR_BGR2Lab);
  current_frame.convertTo( current_frame, CV_32FC1, 1. / 255., -0.5); //to double
  Mat ch1(current_frame.size(), CV_32FC1);
  Mat ch2(current_frame.size(), CV_32FC1);
  Mat ch3(current_frame.size(), CV_32FC1);
  vector<Mat> color_features = {ch1, ch2, ch3};
  cv::split(current_frame, color_features);
  split(current_frame, color_features);
  //vector<Mat> cn_feat = CNFeat::extract(current_frame);
  int channels = color_features.size();
  //cout << channels << "," << cn_feat[0].cols << "," << cn_feat[0].rows << endl;
  VectorXd rawPixelsFeatures(color_features[0].cols*color_features[0].rows*channels);
  int cidx=0;
  for (int ch = 0; ch < channels; ++ch){   
      for(int c = 0; c < color_features[ch].cols ; c++){
        for(int r = 0; r < color_features[ch].rows ; r++){
            rawPixelsFeatures(cidx) = (double)color_features[ch].at<float>(r,c);
            cidx++;
        }
    }
  }
  double normTerm = rawPixelsFeatures.norm();
  if (normTerm > 1e-6){
    rawPixelsFeatures.normalize();
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

vector<Rect> CPU_LR_HOGDetector::sliding_window(Mat frame,Rect reference_roi,int stride){
	vector<Rect> samples;
	int num_rows=(frame.rows- reference_roi.height + stride)/stride;
	int num_cols=(frame.cols- reference_roi.width + stride)/stride;
	for(int i=0;i<num_rows;i++){
		for(int j=0;j<num_cols;j++){
			int row=i*stride;
			int col=j*stride;
			Rect current_window(col,row, reference_roi.width, reference_roi.height);
			samples.push_back(current_window);
		}
	}
	return samples;	
}

vector<Rect> CPU_LR_HOGDetector::region_proposal(Mat frame){
	vector<Rect> samples;
	setUseOptimized(true);
    setNumThreads(8);
    Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
    ss->setBaseImage(frame);
    ss->switchToSelectiveSearchFast();
    ss->process(samples);
	return samples;	
}