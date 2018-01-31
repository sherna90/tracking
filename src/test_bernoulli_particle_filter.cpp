#include "test_bernoulli_particle_filter.hpp"

TestBernoulliParticleFilter::TestBernoulliParticleFilter(string firstFrameFilename, string gtFilename, string dtFilename, int num_particles){
	imageGenerator generator(firstFrameFilename,gtFilename,dtFilename);
	this->num_particles = num_particles;
	this->num_frames = generator.getDatasetSize();
	this->gt_vec = generator.ground_truth;
	this->detections = generator.detections;
	this->images = generator.images;
}

void TestBernoulliParticleFilter::run(){
	BernoulliParticleFilter filter(this->num_particles);
	Rect ground_truth;
	Mat current_frame; 
	string current_gt;
	reinit_rate = 0.0;
	time_t start, end;
	time(&start);
	Performance performance;
	namedWindow("Tracker");
	/*VectorXd weights;
	MatrixXd featureValues;
	vector<Rect> dppResults, preDetections;*/

	for(int k = 0; k < num_frames; ++k){
		current_gt = this->gt_vec[k];
		ground_truth = generator.stringToRect(current_gt);
		current_frame = this->images[k].clone();

	   	if(!filter.is_initialized())
	   	{
	   		filter.initialize(current_frame, ground_truth);
	   		cout <<  ground_truth.x << "," << ground_truth.y << "," << ground_truth.width << "," << ground_truth.height << endl;
		}
		else{
			filter.predict();
			filter.update(current_frame,this->detections[k]);
			rectangle( current_frame, ground_truth, Scalar(0,255,0), 2, LINE_AA );
			Rect estimate = filter.estimate(current_frame, true);
			cout <<  estimate.x << "," << estimate.y << "," << estimate.width << "," << estimate.height << endl;
			performance.calc(ground_truth, estimate);
		}
		imshow("Tracker", current_frame);
		//imwrite(to_string(k)+".png", current_frame );
		waitKey(100);
  	}
	time(&end);
	double sec = difftime (end, start);
	//cout << performance.get_avg_precision()/(num_frames - reinit_rate);
	//cout << "," << performance.get_avg_recall()/(num_frames - reinit_rate);
	//cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;
};

int main(int argc, char* argv[]){
	if(argc != 9) {
		cerr <<"Incorrect input list" << endl;
		cerr <<"exiting..." << endl;
		return EXIT_FAILURE;
	}
	else{
		string firstFrameFilename,gtFilename,dtFilename;
		int num_particles;
		double lambda, mu, epsilon;
		if(strcmp(argv[1], "-img") == 0) {
			firstFrameFilename = argv[2];
		}
		else{
			cerr <<"No images given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[3], "-gt") == 0) {
			gtFilename = argv[4];
		}
		else{
			cerr <<"No ground truth given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[5], "-dt") == 0) {
			dtFilename = argv[6];
		}
		else{
			cerr <<"No detections truth given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[7], "-npart") == 0) {
			num_particles = atoi(argv[8]);
		}
		else{
			num_particles = 300;
		}
		TestBernoulliParticleFilter tracker(firstFrameFilename, gtFilename,dtFilename, num_particles);
		tracker.run();
	}
}

