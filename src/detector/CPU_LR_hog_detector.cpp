#include "CPU_LR_hog_detector.hpp"

#ifndef PARAMS
const bool USE_COLOR=true;
#endif

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
    args.test_stride_width = 15;
    args.test_stride_height = 15;
    args.train_stride_width = 2;
    args.train_stride_height = 2;
    args.block_width = 16;
    args.block_stride_width = 8;
    args.block_stride_height = 8;
    args.cell_width = 8;
    args.nbins = 9;
    args.overlap_threshold=0.9;
    args.p_accept = 0.99;
    args.lambda = 10.0;
    args.epsilon= 0.99;
    args.tolerance = 1e-1;
    args.n_iterations = 1e3;
    args.padding = 8;
    //this->n_descriptors = (args.width/args.cell_width-1)*(args.height/args.cell_width-1)*args.nbins*(args.block_width*args.block_width/(args.cell_width*args.cell_width));
	Size win_stride(args.train_stride_width, args.train_stride_width);
    Size win_size(args.hog_width, args.hog_height);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->hog = HOGDescriptor(win_size, block_size, block_stride, cell_size, args.nbins);
    if(USE_COLOR){
    	//this->n_descriptors = args.hog_width/8 * args.hog_height/8 * (3*args.nbins+5) + H_BINS*S_BINS;
    	int channels = 3;
    	this->n_descriptors=args.hog_width/8 * args.hog_height/8 * (3*args.nbins+5) + (this->args.hog_width/2)*(this->args.hog_height/2)*channels;
    }
    //else this->n_descriptors = this->hog.getDescriptorSize();
    else this->n_descriptors = args.hog_width/8 * args.hog_height/8 * (3*args.nbins+5);
    this->generator.seed(seed1);
    this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	this->labels.resize(0);
	this->num_frame=0;
}



vector<Rect> CPU_LR_HOGDetector::detect(Mat &frame,Rect reference_roi)
{
Mat cropped_frame,current_frame;
	int x_shift=50;
	int y_shift=50;
	Rect cropped_roi=reference_roi+Point(-x_shift,-y_shift);
	cropped_roi.x=MIN(MAX(cropped_roi.x, 0), frame.cols);
	cropped_roi.y=MIN(MAX(cropped_roi.y, 0), frame.rows);
	int w_crop=(frame.cols > (cropped_roi.x+cropped_roi.width+2*x_shift) )?  2*x_shift : x_shift ;
	int h_crop=(frame.rows > (cropped_roi.y+cropped_roi.height+2*y_shift) )?  2*y_shift : x_shift;
	cropped_roi+=Size(w_crop,h_crop);
	//frame.copyTo(current_frame);
	current_frame=frame(cropped_roi);
	current_frame.copyTo(cropped_frame);
	vector<Rect> raw_detections;
	this->detections.clear();
	int channels = frame.channels();
	//this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
	this->weights.clear();
	double max_prob=0;
	for (int k=0;k<args.nlevels;k++){
		int num_rows=(current_frame.rows- this->args.height + this->args.test_stride_height)/this->args.test_stride_height;
		int num_cols=(current_frame.cols- this->args.width + this->args.test_stride_width)/this->args.test_stride_width;
		if (num_rows*num_cols<=0) break;
		double scaleMult=pow(args.scale,k);
		int idx = 0;
		MatrixXd temp_features_matrix = MatrixXd::Zero(0,this->n_descriptors); //
		vector<Rect> windows;
		for(int i=0;i<num_rows;i++){
			for(int j=0;j<num_cols;j++){
				int row=i*this->args.test_stride_height;
				int col=j*this->args.test_stride_width;
				int w_shift=(current_frame.cols-(col+this->args.width) >=0 )?  this->args.width : current_frame.cols-col;
				int h_shift=(current_frame.rows-(row+this->args.height) >=0 )?  this->args.height : current_frame.rows-row;
				Rect current_window(col,row, w_shift,h_shift);
				windows.push_back(current_window);
				Mat subImage = current_frame(current_window);
				VectorXd hogFeatures = this->genHog(subImage);
				VectorXd temp;
				temp_features_matrix.conservativeResize(temp_features_matrix.rows() + 1, NoChange);
				if(USE_COLOR){
					VectorXd rawPixelsFeatures = this->genRawPixels(subImage);
					temp.resize(hogFeatures.rows()+rawPixelsFeatures.rows());
					temp << hogFeatures, rawPixelsFeatures;
				}
				else{
					temp.resize(hogFeatures.rows());
					temp << hogFeatures;
				}	
				temp.normalize();				
				temp_features_matrix.row(idx) = temp;
				idx++;
				
			}	
		}
		VectorXd predict_prob = this->logistic_regression.predict(temp_features_matrix, true);
		tools.writeToCSVfile("test_feaures.csv", temp_features_matrix);
		tools.writeToCSVfile("test_labels.csv", predict_prob);
		//cout << predict_prob.transpose() << endl;
		for (int i = 0; i < predict_prob.rows(); ++i){
			max_prob=MAX(max_prob,predict_prob(i));
			if (predict_prob(i)>args.hit_threshold) {
				stringstream ss;
				ss << predict_prob(i);
				//this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				//this->feature_values.row(this->feature_values.rows() - 1)=temp_features_matrix.row(0);
				string disp = ss.str().substr(0,4);
				Rect current_window = windows.at(i);
				rectangle( current_frame, Point(current_window.x,current_window.y),Point(current_window.x+current_window.width,current_window.y+20), Scalar(0,0,255), -1, 8,0 );
				putText(current_frame, disp, Point(current_window.x+5, current_window.y+12), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255),1);
				rectangle( current_frame, current_window, Scalar(0,0,255), 1, LINE_8  );
				raw_detections.push_back(current_window);
				this->weights.push_back(predict_prob(i));
			}

		}
		
		//rectangle( resized_frame, reference_roi, Scalar(255,255,255), 2, LINE_8  );
		cout << "max prob: " << max_prob << "," << this->weights.size() << endl;
		string name= to_string(this->num_frame)+"_detections_raw.png";
		imwrite(name, current_frame);
		pyrDown( current_frame, current_frame, Size( cvCeil(current_frame.cols/args.scale) , cvCeil(current_frame.rows/args.scale)));
	}
	if(this->args.gr_threshold > 0) {
		nms2(raw_detections,this->weights,this->detections, args.gr_threshold, 0);
		//DPP dpp = DPP();
		//VectorXd qualityTerm;
		//double* ptr = &this->weights[0];
		//Map<VectorXd> eigen_weights(ptr,this->weights.size());
		//this->detections = dpp.run(raw_detections,eigen_weights, eigen_weights, this->feature_values, qualityTerm, 1.0, 0.5, 0.1);
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
	//cout << "Frame : " << this->num_frame << endl; 
	//cout << "raw_detections: " << raw_detections.size() << endl; 
	//cout << "detections: " << detections.size() << endl;
	this->num_frame++; 
	exit(0);
	return this->detections;
}

vector<double> CPU_LR_HOGDetector::detect(Mat &frame, vector<Rect> samples)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	this->feature_values=MatrixXd::Zero(samples.size(),this->n_descriptors); //
	this->weights.clear();
	double max_prob=0.0;
	for (int k=0;k<args.nlevels;k++){
		double scaleMult=pow(args.scale,k);
		for(int i=0;i<samples.size();i++){
			Rect current_window=samples[i];
			int col=current_window.x;
			int row=current_window.y;
			Mat subImage = current_frame(current_window);
			VectorXd hogFeatures = this->genHog(subImage);
			VectorXd temp;
			MatrixXd temp_features_matrix;
			if(USE_COLOR){
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
        	max_prob=MAX(max_prob,predict_prob(0));
        	//cout << predict_prob(0) << ",";
    		this->feature_values.row(i)=temp_features_matrix.row(0);
			this->weights.push_back(predict_prob(0));
			string disp = ss.str().substr(0,4);
        	rectangle( current_frame, Point(col,row),Point(col+current_window.width,row+20), Scalar(0,0,255), -1, 8,0 );
        	putText(current_frame, disp, Point(col+5, row+12), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255),1);
			rectangle( current_frame, current_window, Scalar(0,0,255), 1, LINE_8  );
		}	
		//cout << "-----------------------" << endl;
		string name= to_string(this->num_frame)+"_particle_filter.png";
		imwrite(name, current_frame);
		pyrDown( current_frame, current_frame, Size( cvCeil(current_frame.cols/args.scale) , cvCeil(current_frame.rows/args.scale)));
	}
	cout << "-----------------------" << endl;
	cout << "Frame : " << this->num_frame << endl; 
	cout << "max prob: " << max_prob << endl;
	this->num_frame++;
	return this->weights;
}

void CPU_LR_HOGDetector::train(Mat &frame,Rect reference_roi)
{
	cout << reference_roi << endl;
	Mat cropped_frame,current_frame;
	int x_shift=50;
	int y_shift=50;
	Rect cropped_roi=reference_roi+Point(-x_shift,-y_shift);
	cropped_roi.x=MIN(MAX(cropped_roi.x, 0), frame.cols);
	cropped_roi.y=MIN(MAX(cropped_roi.y, 0), frame.rows);
	int w_crop=(frame.cols-(cropped_roi.x+2*x_shift) >=0 )?  2*x_shift : frame.cols-cropped_roi.x;
	int h_crop=(frame.rows-(cropped_roi.y+2*y_shift) >=0 )?  2*y_shift : frame.rows-cropped_roi.y;
	cropped_roi+=Size(w_crop,h_crop);
	reference_roi.x=MIN(x_shift,reference_roi.x);
	reference_roi.y=MIN(y_shift,reference_roi.y);
	cropped_frame=frame(cropped_roi);
	cropped_frame.copyTo(current_frame);
	int num_rows=(current_frame.rows- this->args.height + this->args.train_stride_height)/this->args.train_stride_height;
	int num_cols=(current_frame.cols- this->args.width + this->args.train_stride_width)/this->args.train_stride_width;
	this->detections.clear();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors); //
	this->labels.resize(0);
	uniform_real_distribution<double> unif(0.0,1.0);
	for(int i=0;i<num_rows;i++){
		for(int j=0;j<num_cols;j++){
			int row=i*this->args.train_stride_height;
			int col=j*this->args.train_stride_width;
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
	tools.writeToCSVfile("train_feaures.csv", this->feature_values);
	tools.writeToCSVfile("train_labels.csv", this->labels);
	rectangle( current_frame, reference_roi, Scalar(0,255,0), 2, LINE_AA );
	imwrite("resized_image.png", current_frame);
	if(!this->logistic_regression.initialized){
		this->logistic_regression.init(this->feature_values, this->labels, args.lambda,false,true,true);	
	} 
	else{
		this->logistic_regression.setData(this->feature_values, this->labels);
	}
	cout << this->feature_values.rows() << "," << this->feature_values.cols() << "," << this->labels.rows() << endl;
	this->logistic_regression.train((int)args.n_iterations, args.epsilon, args.tolerance);
	//exit(0);
}


void CPU_LR_HOGDetector::train()
{
	this->logistic_regression.init(this->feature_values, this->labels, args.lambda, false,true,true);
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
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




MatrixXd CPU_LR_HOGDetector::getFeatureValues(Mat &current_frame)
{
	vector<float> temp_features;
	Size win_stride(args.test_stride_width, args.test_stride_height);
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
	Mat current_frame,mat_hog_features;
	resize(frame,current_frame,Size(args.hog_width, args.hog_height),0,0,interpolation);
	
	current_frame.convertTo( current_frame, CV_32FC(3), 1.0/255.0); //to double
	//current_frame.convertTo( current_frame, CV_32FC(3));
	current_frame *= 255.;
	piotr::fhogToCol(current_frame,mat_hog_features,8,0,0);
	VectorXd hog_features=VectorXd::Zero(mat_hog_features.cols);
	for (int j = 0; j < mat_hog_features.cols; j++){
		hog_features(j) =mat_hog_features.at<float>(0,j);
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
  current_frame.convertTo( current_frame, CV_32FC(3), 1.0/255.0); //to double
  
  Mat cieLabFrame = tools.RGBtoLAB(current_frame);

  resize(cieLabFrame,cieLabFrame,Size(args.hog_width/2, args.hog_height/2),0,0,interpolation);

  int channels = cieLabFrame.channels();
  vector<Mat> frame_channels(channels);
  split(cieLabFrame, frame_channels);
  VectorXd rawPixelsFeatures(cieLabFrame.cols*cieLabFrame.rows*channels);
  int cidx=0;
  for (int ch = 0; ch < channels; ++ch){   
      for(int c = 0; c < cieLabFrame.cols ; c++){
        for(int r = 0; r < cieLabFrame.rows ; r++){
            rawPixelsFeatures(cidx) = (double)frame_channels[ch].at<double>(r,c);
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

