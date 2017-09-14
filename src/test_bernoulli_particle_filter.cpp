#include "test_bernoulli_particle_filter.hpp"

TestBernoulliParticleFilter::TestBernoulliParticleFilter(string firstFrameFilename, string gtFilename, 
	int num_particles, double lambda, double mu, double epsilon){
	imageGenerator generator(firstFrameFilename,gtFilename);
	this->num_particles = num_particles;
	this->num_frames = generator.getDatasetSize();
	this->gt_vec = generator.ground_truth;
	this->images = generator.images;
	this->lambda = lambda;
	this->mu = mu;
	this->epsilon = epsilon;
}

void TestBernoulliParticleFilter::run(){
	BernoulliParticleFilter filter(this->num_particles, this->lambda, this->mu, this->epsilon);
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
	   		//filter.draw_particles(current_frame, Scalar(0,255,255));
		}
		else{
			filter.predict();
			filter.update(current_frame);
			//filter.draw_particles(current_frame, Scalar(0,255,255));
			rectangle( current_frame, ground_truth, Scalar(0,255,0), 2, LINE_AA );
			Rect estimate = filter.estimate(current_frame, true);
			cout <<  estimate.x << "," << estimate.y << "," << estimate.width << "," << estimate.height << endl;
			performance.calc(ground_truth, estimate);
			/*if(r1 < 0.1) {
				filter.reinitialize();
				reinit_rate+=1.0;
			}*/
		}
		imshow("Tracker", current_frame);
		//imwrite(to_string(k)+".png", current_frame );
		waitKey(1);
  	}
	time(&end);
	double sec = difftime (end, start);
	//cout << performance.get_avg_precision()/(num_frames - reinit_rate);
	//cout << "," << performance.get_avg_recall()/(num_frames - reinit_rate);
	//cout << "," << num_frames/sec << "," << reinit_rate <<  "," << num_frames << endl;
};

int main(int argc, char* argv[]){
	if(argc != 13) {
		cerr <<"Incorrect input list" << endl;
		cerr <<"exiting..." << endl;
		return EXIT_FAILURE;
	}
	else{
		string firstFrameFilename,gtFilename;
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
		if(strcmp(argv[5], "-npart") == 0) {
			num_particles = atoi(argv[6]);
		}
		else{
			num_particles = 300;
		}
		if(strcmp(argv[7], "-lambda") == 0) {
			lambda = stod(argv[8]);
		}
		else{
			cerr <<"No lambda given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[9], "-mu") == 0) {
			mu = stod(argv[10]);
		}
		else{
			cerr <<"No mu given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[11], "-epsilon") == 0) {
			epsilon = stod(argv[12]);
		}
		else{
			cerr <<"No epsilon given" << endl;
			cerr <<"exiting..." << endl;
			return EXIT_FAILURE;
		}
		TestBernoulliParticleFilter tracker(firstFrameFilename, gtFilename, num_particles, lambda, mu, epsilon);
		tracker.run();
	}
}

