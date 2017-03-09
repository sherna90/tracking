#include "test_bernoulli_particle_filter.hpp"

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

	for(int k = 0; k < num_frames; ++k){
		current_gt = gt_vec[k];
		ground_truth = generator.stringToRect(current_gt);
		current_frame = images[k].clone();

	   	if(!filter.is_initialized())
	   	{
	   		filter.initialize(current_frame, ground_truth);
	   		//filter.draw_particles(current_frame, Scalar(0,255,255));
		}
		else{
			filter.predict();
			filter.update(current_frame);
			//filter.draw_particles(current_frame, Scalar(0,255,255));
			rectangle( current_frame, ground_truth, Scalar(0,255,0), 1, LINE_AA );
			Rect estimate = filter.estimate(current_frame, true);
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

