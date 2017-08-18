#include "cpu_hog_detector.hpp"
CPU_HOGDetector::CPU_HOGDetector(){
	
}

CPU_HOGDetector::CPU_HOGDetector(double group_threshold, double hit_threshold){
	args.make_gray = true;
    args.resize_src = false;
    args.width = 64;
    args.height = 128;
    args.scale = 2;
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
    args.p_accept = 0.99;
    args.lambda = 100;
    args.epsilon= 1e-2;
    args.tolerance = 1e-1;
    args.n_iterations = 1e4;
    args.padding = 16;
    this->n_descriptors = (args.width/args.cell_width-1)*(args.height/args.cell_width-1)*args.nbins*(args.block_width*args.block_width/(args.cell_width*args.cell_width));
    //this->n_descriptors = (pow((int)(args.block_width/args.cell_width),2)* args.nbins) * (int)((args.width/args.block_width)-1) * (int)((args.height/args.block_width)-1);
    //this->n_descriptors = 3780;
   	int num_rows=frame.rows - this->args.height + this->args.win_stride_height;
	int num_cols=frame.cols - this->args.width + this->args.win_stride_width;
	this->n_data = num_cols*num_rows;
	Size win_stride(args.win_stride_width, args.win_stride_height);
    Size win_size(args.width, args.height);
    Size block_size(args.block_width, args.block_width);
    Size block_stride(args.block_stride_width, args.block_stride_height);
    Size cell_size(args.cell_width, args.cell_width);
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    this->generator.seed(seed1);
    this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	this->labels.resize(0);
}

vector<Rect> CPU_HOGDetector::detect(Mat &frame)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	vector<Rect> raw_detections;
	vector<double> detection_weights;
	copyMakeBorder( current_frame, current_frame, args.padding, args.padding,args.padding,args.padding,BORDER_REPLICATE);
	this->detections.clear();
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
	for (int k=0;k<args.nlevels;k++){
		int num_rows=(current_frame.rows- this->args.height + this->args.win_stride_height)/this->args.win_stride_height;
		int num_cols=(current_frame.cols- this->args.width + this->args.win_stride_width)/this->args.win_stride_width;
		if (num_rows*num_cols<=0) break;
		Mat resized_frame;
		current_frame.copyTo(resized_frame);
		cvtColor(resized_frame, resized_frame, COLOR_GRAY2BGR);
		MatrixXd features=this->getFeatureValues(current_frame);
		VectorXd predict_prob = this->logistic_regression.predict(features, true);
		//VectorXd predict_prob = this->hmc.predict(features, true);
		cout << "frame : " << current_frame.rows << "," << current_frame.cols << endl;
		cout << "num windows : " << num_rows << "," << num_cols << endl;
		cout << "features : " << features.rows() << "," << features.cols() << ", prob : " << predict_prob.rows() << endl;
		int idx=0;
		double scaleMult=pow(args.scale,k);
		for(int i=0;i<num_rows;i++){
			for(int j=0;j<num_cols;j++){
				int row=i*this->args.win_stride_height;
				int col=j*this->args.win_stride_width;
				int x1 = (k>0) ? int(scaleMult  * col)  : col-this->args.padding;
            	int y1 = (k>0) ? int(scaleMult  * row)  : row-this->args.padding;
            	int x2 = (k>0) ? int(x1 + scaleMult*(this->args.width))-2*this->args.padding : int(x1 + this->args.width);
            	int y2 = (k>0) ? int(y1 + scaleMult*(this->args.height))-2*this->args.padding : int(y1 + this->args.height);
				Rect current_window(Point(x1,y1),Point(x2,y2));
				if (predict_prob(idx)>args.hit_threshold) {
					Rect current_resized_window(col,row,this->args.width,this->args.height);
					cout << current_resized_window << endl;
					stringstream ss;
        			ss << predict_prob(idx);
        			string disp = ss.str().substr(0,4);
        			putText(resized_frame, disp, Point(col+5, row+10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 2);
					rectangle( resized_frame, current_resized_window, Scalar(0,0,255), 2, LINE_8  );
					raw_detections.push_back(current_window);
					detection_weights.push_back(predict_prob(idx));
				}
				idx++;
			}	
		}
		cout << "-----------------------" << endl;
		string name= "resized_image_"+to_string(k)+".png";
		imwrite(name, resized_frame);
		pyrDown( current_frame, current_frame, Size( cvCeil(current_frame.cols/args.scale) , cvCeil(current_frame.rows/args.scale)));
	}
	if(this->args.gr_threshold >= 0) {
		nms2(raw_detections, detection_weights, this->detections, args.gr_threshold, 1);
	}
	else {
		for (int i = 0; i < raw_detections.size(); ++i)
		{
			this->detections.push_back(raw_detections[i]);	
		}
	} 
	cout << "detections: " << detections.size() << endl;
	return this->detections;
}

void CPU_HOGDetector::train(Mat &frame,Rect reference_roi)
{
	Mat current_frame;
	frame.copyTo(current_frame);
	if(args.resize_src) {
		Size image_size((int)current_frame.cols*args.scale,(int)current_frame.rows*args.scale);
		resize(current_frame,current_frame,image_size,0,0,INTER_CUBIC);
		reference_roi+=Point((int)reference_roi.width*args.scale,(int)reference_roi.height*args.scale);
		reference_roi+=Size((int)reference_roi.width*args.scale,(int)reference_roi.height*args.scale);
	}
	this->feature_values=MatrixXd::Zero(0,this->n_descriptors);
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
			if(uni_rand>args.p_accept){ 
				this->detections.push_back(current_window);
				this->feature_values.conservativeResize(this->feature_values.rows() + 1, NoChange);
				Mat subImage = current_frame(current_window);
				this->feature_values.row(this->feature_values.rows() - 1)= this->getFeatureValues(subImage).row(0);
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
	this->logistic_regression = LogisticRegression(this->feature_values, this->labels, args.lambda,false,true,true);
	//this->hmc = Hamiltonian_MC(this->feature_values, this->labels, args.lambda, 100, args.n_iterations, 0.01, 100 ,0.0, false);
	cout << this->feature_values.rows() << "," << this->feature_values.cols() << "," << this->labels.rows() << endl;
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
	//this->hmc.run();
}

void CPU_HOGDetector::generateFeatures(Mat &frame, double label)
{	
	MatrixXd hogFeatures = this->getFeatureValues(frame);
	MatrixXd oldhogFeatures = this->feature_values;
	this->feature_values.conservativeResize(this->feature_values.rows() + hogFeatures.rows(), NoChange);
	this->feature_values << oldhogFeatures, hogFeatures;
	VectorXd hogLabels = VectorXd::Ones(feature_values.rows())*label;
	VectorXd oldhogLabels = this->labels;
	this->labels.conservativeResize(this->labels.rows() + hogLabels.size());
	this->labels << oldhogLabels, hogLabels;
}

void CPU_HOGDetector::generateFeature(Mat &frame, double label)
{
	this->feature_values= this->getFeatureValues(frame);
	this->labels = VectorXd::Ones(feature_values.rows())*label;
}

void CPU_HOGDetector::dataClean(){
	this->feature_values.resize(0,0);
	this->labels.resize(0);
}

void CPU_HOGDetector::train()
{
	this->logistic_regression = LogisticRegression(this->feature_values, this->labels, args.lambda, false,true,true);
	//this->hmc = Hamiltonian_MC(this->feature_values, this->labels, args.lambda, 1000, args.n_iterations, 0.01, 100 ,0.0, false);
	this->logistic_regression.train(args.n_iterations, args.epsilon, args.tolerance);
	//this->hmc.run();
	VectorXd weights = this->logistic_regression.getWeights();
	VectorXd bias(1);
	bias << this->logistic_regression.getBias();
	tools.writeToCSVfile("INRIA_Model_weights.csv", weights);
	tools.writeToCSVfile("INRIA_Model_means.csv", this->logistic_regression.featureMean.transpose());
	tools.writeToCSVfile("INRIA_Model_stds.csv", this->logistic_regression.featureStd.transpose());
	tools.writeToCSVfile("INRIA_Model_maxs.csv", this->logistic_regression.featureMax.transpose());
	tools.writeToCSVfile("INRIA_Model_mins.csv", this->logistic_regression.featureMin.transpose());
	tools.writeToCSVfile("INRIA_Model_bias.csv", bias);

	/*VectorXd weights, featureMean, featureStd, featureMax, featureMin;
	double bias;
	this->hmc.getModel(weights, featureMean, featureStd, featureMax, featureMin, bias);
	VectorXd v_bias(1);
	v_bias << bias;
	
	tools.writeToCSVfile("INRIA_Model_weights.csv", weights);
	tools.writeToCSVfile("INRIA_Model_means.csv", featureMean);
	tools.writeToCSVfile("INRIA_Model_stds.csv", featureStd);
	tools.writeToCSVfile("INRIA_Model_maxs.csv", featureMax);
	tools.writeToCSVfile("INRIA_Model_mins.csv", featureMin);
	tools.writeToCSVfile("INRIA_Model_bias.csv", v_bias);*/

}

VectorXd CPU_HOGDetector::predict(MatrixXd data)
{
	return this->logistic_regression.predict(data, false);
	//return this->hmc.predict(data, false);
}

void CPU_HOGDetector::draw()
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
}

MatrixXd CPU_HOGDetector::getFeatureValues(Mat &current_frame)
{
	vector<float> temp_features;
	Size win_stride(args.win_stride_width, args.win_stride_height);
	this->hog.compute(current_frame, temp_features, win_stride);
	vector<double> features(temp_features.begin(), temp_features.end());
	double* ptr = &features[0];
	int rows = (int)(features.size()/this->hog.getDescriptorSize());
	Map<MatrixXd> hogFeatures(ptr, rows, this->hog.getDescriptorSize());
	return hogFeatures;
}

MatrixXd CPU_HOGDetector::getFeatures()
{
	return this->feature_values;
}

VectorXd CPU_HOGDetector::getDetectionWeights(){
	return this->weights;
}

void CPU_HOGDetector::saveToCSV(string name, bool append){
	tools.writeToCSVfile(name+"_values.csv", this->feature_values, append);
	tools.writeToCSVfile(name+"_labels.csv", this->labels, append);
}

void CPU_HOGDetector::loadFeatures(MatrixXd features, VectorXd labels){
	this->dataClean();
	this->feature_values = features;
	this->labels = labels;
}
void CPU_HOGDetector::loadModel(VectorXd weights,VectorXd featureMean, VectorXd featureStd, VectorXd featureMax, VectorXd featureMin, double bias){
	//this->hmc.loadModel(weights, featureMean, featureStd, featureMax, featureMin, bias);
	this->logistic_regression = LogisticRegression(false, true, true);
	this->logistic_regression.setWeights(weights);
	this->logistic_regression.setBias(bias);
	this->logistic_regression.featureMean = featureMean;
	this->logistic_regression.featureStd = featureStd;
	this->logistic_regression.featureMax = featureMax;
	this->logistic_regression.featureMin = featureMin;
}