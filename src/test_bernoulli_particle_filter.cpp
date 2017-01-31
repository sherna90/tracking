#include "test_bernoulli_particle_filter.hpp"

#ifndef PARAMS
const int STEPSLIDE = 10;
//DPP's parameters
const double ALPHA = 0.9;
const double LAMBDA = -0.1;
const double BETA = 1.1;
const double MU = 0;
const double EPSILON = 0.4;
#endif

TestBernoulliParticleFilter::TestBernoulliParticleFilter(string _firstFrameFilename, string _gtFilename, int _num_particles){
	imageGenerator generator(_firstFrameFilename,_gtFilename);
	num_particles = _num_particles;
	num_frames = generator.getDatasetSize();
	gt_vec = generator.ground_truth;
	images = generator.images;
}

void TestBernoulliParticleFilter::run(){
	BernoulliParticleFilter filter(num_particles);
	Rect ground_truth;
	Mat current_frame; 
	string current_gt;
	reinit_rate = 0.0;
	time_t start, end;
	time(&start);
	Performance performance;
	namedWindow("Tracker");

	VectorXd weights;
	MatrixXd featureValues;
	vector<Rect> dppResults, preDetections;	

	for(int k = 0; k <num_frames; ++k){
		current_gt = gt_vec[k];
		ground_truth = generator.stringToRect(current_gt);
		current_frame = images[k].clone();

		/*************** DPP ***************/
		Mat grayImg;
		cvtColor(current_frame, grayImg, CV_RGB2GRAY);
		int left = MAX(ground_truth.x, 1);
		int top = MAX(ground_truth.y, 1);
		int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
		int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
		Rect reference_roi = Rect(left, top, right - left, bottom - top);
		weights.resize(0);
		preDetections.clear();
		dppResults.clear();
		for(int row = 0; row <= grayImg.rows - reference_roi.height; row+=STEPSLIDE){
			for(int col = 0; col <= grayImg.cols - reference_roi.width; col+=STEPSLIDE){
				Rect current_window(col, row, reference_roi.width, reference_roi.height);
				Rect intersection = reference_roi & current_window;
				preDetections.push_back(current_window);
				weights.conservativeResize( weights.size() + 1 );
				weights(weights.size() - 1) = intersection.area();
			}
		}
		featureValues = MatrixXd(this->haar.featureNum, preDetections.size());
	   	this->haar.init(grayImg, reference_roi, preDetections);
	   	cv2eigen(this->haar.sampleFeatureValue, featureValues);
	   	featureValues.transposeInPlace();

	   	dppResults = this->dpp.run(preDetections, weights, featureValues, ALPHA, LAMBDA, BETA, MU, EPSILON);
	   	/***********************************/

	   	if(!filter.is_initialized()){
	   		filter.initialize(current_frame, ground_truth);
	   		filter.draw_particles(current_frame, Scalar(0,255,255));
	   		/*string s;
	   		s = "../../images/"+std::to_string(k)+".png";
	   		imwrite(s, current_frame);*/
		}else{
			filter.predict();
			filter.update(current_frame, dppResults);
			filter.draw_particles(current_frame, Scalar(0,255,255));
			rectangle( current_frame, ground_truth, Scalar(0,255,0), 1, LINE_AA );
			Rect estimate = filter.estimate(current_frame, true);
			for (size_t i = 0; i < dppResults.size(); i++){
	        	rectangle( current_frame, dppResults.at(i), Scalar(255,0,0), 1, LINE_AA );
	    	}
			//double r1 = performance.calc(ground_truth, estimate);
			/*if(r1 < 0.1) {
				filter.reinitialize();
				reinit_rate += 1.0;
	  		}*/
			/*string s;
	   		s = "../../images/"+std::to_string(k)+".png";
	   		imwrite(s, current_frame);*/
		}
		imshow("Tracker", current_frame);
		waitKey(1);
  	}
	time(&end);
	double sec = difftime (end, start);
	cout  << performance.get_avg_precision()/(num_frames - reinit_rate);
	cout << "," << performance.get_avg_recall()/(num_frames - reinit_rate);
	cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;
};

int main(int argc, char* argv[]){
	if(argc != 7) {
		cerr <<"Incorrect input list" << endl;
		cerr <<"exiting..." << endl;
		return EXIT_FAILURE;
	}
	else{
		string _firstFrameFilename,_gtFilename;
		int _num_particles;
		if(strcmp(argv[1], "-img") == 0) {
			_firstFrameFilename=argv[2];
		}
		else{
			cerr <<"No images given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[3], "-gt") == 0) {
			_gtFilename=argv[4];
		}
		else{
			cerr <<"No ground truth given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[5], "-npart") == 0) {
			_num_particles=atoi(argv[6]);
		}
		else{
			_num_particles=300;
		}
		TestBernoulliParticleFilter tracker(_firstFrameFilename,_gtFilename,_num_particles);
		tracker.run();
	}
}

