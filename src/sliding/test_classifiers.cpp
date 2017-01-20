#include "test_classifiers.hpp"

#ifndef PARAMS
const bool GAUSSIAN_NAIVEBAYES = true;
const bool MULTINOMIAL_NAIVEBAYES = false;
const bool LOGISTIC_REGRESSION = false;

const bool HAAR_FEATURE = true;
const bool HOG_FEATURE = false;
const bool LBP_FEATURE = false;

const double RATIO_TRAINTEST = 0.8;
const float OVERLAP_THRESHOLD = 0.8;
const int STEPSLIDE = 1;
#endif


TestClassifiers::TestClassifiers(string _firstFrame, string _gtFileName){
	imageGenerator imgGenerator(_firstFrame, _gtFileName);
	this->numFrames = imgGenerator.getDatasetSize();
	this->groundtruth = imgGenerator.ground_truth;
	this->images = imgGenerator.images;
	this->numTrain = cvRound(this->numFrames*RATIO_TRAINTEST);
	this->numTest = this->numFrames - this->numTrain;
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	this->generator.seed(seed);
}

void TestClassifiers::initialize(){
	//namedWindow("Test Classifiers");

	//Shuffle images and grountruth
	vector<int> indexes;
	indexes.reserve(this->numFrames);
	for (unsigned int i = 0; i < this->numFrames; ++i)
		indexes.push_back(i);

	random_shuffle(indexes.begin(), indexes.end());
	vector<Mat> auxImages;
	vector<string> auxGt;
	for(vector<int>::iterator it = indexes.begin(); it != indexes.end(); ++it)
	{	
		auxImages.push_back(this->images[*it]);
		auxGt.push_back(this->groundtruth[*it]);
	}
	this->images.clear(); this->groundtruth.clear();
	this->images = auxImages; this->groundtruth = auxGt;
	auxImages.clear(); auxGt.clear();

	//Get positive and negative box to train model
	if(HAAR_FEATURE){
		this->xTrain = MatrixXd(50, this->numTrain * 2);
	}

	if(HOG_FEATURE){
		this->xTrain = MatrixXd(this->numTrain * 2, 3780);;
	}

	if(LBP_FEATURE){
		this->xTrain = MatrixXd(this->numTrain * 2, 236);
	}

	normal_distribution<double> negative_random_position(0.0,20.0);
	Mat grayImg;
	for(unsigned int i = 0; i < this->numTrain; i++){
		Rect positiveBox = this->imgGenerator.stringToRect(this->groundtruth[i]);
		
		int left = MAX(positiveBox.x, 1);
		int top = MAX(positiveBox.y, 1);
		int right = MIN(positiveBox.x + positiveBox.width, this->images[i].cols -1);
		int bottom = MIN(positiveBox.y + positiveBox.height, this->images[i].rows - 1);
		Rect reference = Rect(left, top, right - left, bottom - top);
		
		Rect negativeBox = reference;
		Rect intersection = positiveBox & reference;

		while( double(intersection.area())/double(reference.area()) > OVERLAP_THRESHOLD){
			float _dx = negative_random_position(generator);
			float _dy = negative_random_position(generator);

			negativeBox.x = MIN(MAX(cvRound(reference.x + _dx), 0), this->images[i].size().width);
			negativeBox.y = MIN(MAX(cvRound(reference.y + _dy), 0), this->images[i].size().height);
			negativeBox.width = MIN(MAX(cvRound(reference.width), 0), this->images[i].size().width - negativeBox.x);
			negativeBox.height = MIN(MAX(cvRound(reference.height), 0), this->images[i].size().height - negativeBox.y);
			intersection = (negativeBox & reference);
		}
		
		cvtColor(this->images[i], grayImg, CV_RGB2GRAY);
		
		if(HAAR_FEATURE){
			MatrixXd positiveSample, negativeSample;
			vector<Rect> auxBoxes;

			auxBoxes.clear();
			auxBoxes.push_back(positiveBox);
			haar.init(grayImg, positiveBox, auxBoxes);
			cv2eigen(haar.sampleFeatureValue, positiveSample);
			this->xTrain.col(i) = positiveSample.col(0);

			auxBoxes.clear();
			auxBoxes.push_back(negativeBox);
			haar.getFeatureValue(grayImg, auxBoxes);
			cv2eigen(haar.sampleFeatureValue, negativeSample);
			this->xTrain.col(this->numTrain + i) = negativeSample.col(0);
		}

		if(HOG_FEATURE){
			VectorXd hist;
			Mat subImage;

			subImage = grayImg(positiveBox);
			calc_hog(subImage, hist, Size(positiveBox.width, positiveBox.height));
			this->xTrain.row(i) = hist;

			subImage = grayImg(negativeBox);
			calc_hog(subImage, hist, Size(positiveBox.width, positiveBox.height));
			this->xTrain.row(this->numTrain + i) = hist;	
		}

		if(LBP_FEATURE){
			VectorXd hist;
			vector<Rect> auxBoxes;

			auxBoxes.clear();
			auxBoxes.push_back(positiveBox);
			this->local_binary_pattern.init(grayImg, auxBoxes);

			auxBoxes.clear();
			auxBoxes.push_back(negativeBox);
			this->local_binary_pattern.getFeatureValue(grayImg, auxBoxes, false);

			this->xTrain.row(i) = this->local_binary_pattern.sampleFeatureValue.row(0);
			this->xTrain.row(this->numTrain + i) = this->local_binary_pattern.negativeFeatureValue.row(0);
		}

		/*Mat showImage = this->images[i].clone();
		rectangle(showImage, positiveBox, Scalar(0,255,0), 1, LINE_AA);
		rectangle(showImage, negativeBox, Scalar(0,0,255), 1, LINE_AA);
		imshow("Test Classifiers", showImage);
		waitKey(1);*/
	}

	//Sliding Window to get test samples
	srand(time(NULL));
	int randFrame = rand() % this->numTest + this->numTrain;

	cvtColor(this->images[randFrame], grayImg, CV_RGB2GRAY);
	Rect gtTest = this->imgGenerator.stringToRect(this->groundtruth[randFrame]);
	int window_rows = gtTest.height;
  	int window_cols = gtTest.width;

  	//namedWindow("Sliding Window");
  	//vector<Rect> this->positiveTestBoxes, this->negativeTestBoxes;

  	for(int row = 0; row <= grayImg.rows - window_rows; row+=STEPSLIDE){
  		for(int col = 0; col <= grayImg.cols - window_cols; col+=STEPSLIDE){
  			Rect window(col, row, window_cols, window_rows);
  			Mat auxImage = this->images[randFrame].clone();
  			//rectangle(auxImage, window, Scalar(255), 1, 8, 0);
  			//imshow("Sliding Window", auxImage);
  			waitKey(1);
  			Rect intersection = gtTest & window;

  			if( double(intersection.area())/double(gtTest.area()) > OVERLAP_THRESHOLD ){
  				// positive sample
  				rectangle(auxImage, window, Scalar(255), 1, 8, 0);
  				//imshow("Sliding Window", auxImage);
  				this->positiveTestBoxes.push_back(window);
  			}
  			else{
  				// negative sample
  				rectangle(auxImage, window, Scalar(255), 1, 8, 0);
  				//imshow("Sliding Window", auxImage);
  				this->negativeTestBoxes.push_back(window);
  			}
  		}
  	}

  	//this->yTest = VectorXi(this->positiveTestBoxes.size() + this->negativeTestBoxes.size());
	//this->yTest << VectorXi::Ones(this->positiveTestBoxes.size()), VectorXi::Zero(this->negativeTestBoxes.size());

  	if(HAAR_FEATURE){
  		this->xTest = MatrixXd(50, this->positiveTestBoxes.size() + this->negativeTestBoxes.size());
  		MatrixXd positiveSampleFeatureValues, negativeSampleFeatureValues;
  		haar.init(grayImg, gtTest, this->positiveTestBoxes);
  		cv2eigen(haar.sampleFeatureValue, positiveSampleFeatureValues);
  		haar.getFeatureValue(grayImg, this->negativeTestBoxes);
  		cv2eigen(haar.sampleFeatureValue, negativeSampleFeatureValues);

  		this->xTest << positiveSampleFeatureValues,
  						negativeSampleFeatureValues;
  		this->xTest.transposeInPlace();
  		this->xTrain.transposeInPlace();
  	}

  	if(HOG_FEATURE){
  		this->xTest = MatrixXd(this->positiveTestBoxes.size() + this->negativeTestBoxes.size(), 3780);
  		VectorXd hist;

  		for(unsigned int i = 0; i < this->positiveTestBoxes.size(); i++){
  			Mat subImage = grayImg(this->positiveTestBoxes.at(i));
  			calc_hog(subImage, hist, Size(gtTest.width, gtTest.height));
  			this->xTest.row(i) = hist;
  		}

  		for(unsigned int i = 0; i < this->negativeTestBoxes.size(); i++){
  			Mat subImage = grayImg(this->negativeTestBoxes.at(i));
  			calc_hog(subImage, hist, Size(gtTest.width, gtTest.height));
  			this->xTest.row(this->positiveTestBoxes.size() + i) = hist;
  		}
  	}

  	if(LBP_FEATURE){
  		this->xTest = MatrixXd(this->positiveTestBoxes.size() + this->negativeTestBoxes.size(), 236);
  		VectorXd hist;

  		this->local_binary_pattern.init(grayImg, this->positiveTestBoxes);
  		this->local_binary_pattern.getFeatureValue(grayImg, this->negativeTestBoxes, false);

  		this->xTest << this->local_binary_pattern.sampleFeatureValue,
  						this->local_binary_pattern.negativeFeatureValue;
  	}

  	cout << "PositiveTestBoxes: " << this->positiveTestBoxes.size() << endl;
  	cout << "NegativeTestBoxes: " << this->negativeTestBoxes.size() << endl;
	//cout << "numTrain: " << this->numTrain << endl;

	//cout << "initialized!!!" << endl;
}

void TestClassifiers::fit(){

	if(GAUSSIAN_NAIVEBAYES){
		VectorXd yTrain = VectorXd(2 * this->numTrain);
		yTrain << VectorXd::Ones(this->numTrain), VectorXd::Zero(this->numTrain);
		this->gaussian_naivebayes = GaussianNaiveBayes(this->xTrain, yTrain);
		this->gaussian_naivebayes.fit();
	}

	if(MULTINOMIAL_NAIVEBAYES){
		VectorXd yTrain = VectorXd(2 * this->numTrain);
		yTrain << VectorXd::Ones(this->numTrain), VectorXd::Zero(this->numTrain);
		double lambda = 0.1;
		this->multinomial_naivebayes = MultinomialNaiveBayes(this->xTrain, yTrain);
		this->multinomial_naivebayes.fit(lambda);
	}

	if(LOGISTIC_REGRESSION){
		VectorXd yTrain = VectorXd(2 * this->numTrain);
		yTrain << VectorXd::Ones(this->numTrain), VectorXd::Constant(this->numTrain, -1.0);
		double lambda = 0.1;
		this->hamiltonian_monte_carlo = Hamiltonian_MC(this->xTrain, yTrain, lambda);
		this->hamiltonian_monte_carlo.run(1e3,13-2,10);
	}

	//cout << "fitted!!!" << endl;
}

void TestClassifiers::predict(){
	ut = C_utils();
	
	if(GAUSSIAN_NAIVEBAYES){
		VectorXd yTest = VectorXd(this->positiveTestBoxes.size() + this->negativeTestBoxes.size());
		yTest << VectorXd::Ones(this->positiveTestBoxes.size()), VectorXd::Zero(this->negativeTestBoxes.size());
		VectorXd yHat;
		yHat = this->gaussian_naivebayes.predict(this->xTest);
		/*//ut.classification_Report(yTest, yHat);
		ut.confusion_matrix(yTest, yHat);
		cout << "Precision:" << endl;
		ut.precision_score(yTest, yHat);
		cout << "Accuracy:" << endl;
		ut.accuracy_score(yTest, yHat);
		cout << "Recall:" << endl;
		ut.recall_score(yTest, yHat);*/
		cout << "Report" << endl;
		ut.report(yTest, yHat);
	}
	
	if(MULTINOMIAL_NAIVEBAYES){
		VectorXd yTest = VectorXd(this->positiveTestBoxes.size() + this->negativeTestBoxes.size());
		yTest << VectorXd::Ones(this->positiveTestBoxes.size()), VectorXd::Zero(this->negativeTestBoxes.size());
		VectorXd yHat;
		yHat = this->multinomial_naivebayes.test(this->xTest);
		//ut.classification_Report(yTest, yHat);
		cout << "Precision:" << endl;
		//mcout << ut.precision_score(yTest, yHat) << endl;
	}

	if(LOGISTIC_REGRESSION){
		VectorXd yTest = VectorXd(this->positiveTestBoxes.size() + this->negativeTestBoxes.size());
		yTest << VectorXd::Ones(this->positiveTestBoxes.size()), VectorXd::Constant(this->negativeTestBoxes.size(), -1);
		VectorXd yHat;
		//VectorXd yHat;
		yHat = this->hamiltonian_monte_carlo.predict(this->xTest, false);
		//ut.classification_Report_d(yTest, yHat);
		cout << "Precision:" << endl;
		ut.precision_score(yTest, yHat);
	}

	//cout << "predicted!!!" << endl;
}

void TestClassifiers::run(){
	initialize();
	fit();
	predict();
}

int main(int argc, char* argv[]){
	if(argc != 5){
		cout << "Incorrect input list!!!" << endl;
		cout << "exiting ..." << endl;
		return EXIT_FAILURE;
	}
	else{
		string firstFrameFileName, gtFileName;
		if(strcmp(argv[1], "-img") == 0){
			firstFrameFileName = argv[2];
		}
		else{
			cout << "No images given" << endl;
			cout << "exiting ..." << endl;
			return EXIT_FAILURE;
		}
		if(strcmp(argv[3],"-gt") == 0){
			gtFileName = argv[4];
		}
		else{
			cout << "No ground truth given" << endl;
			cout << "exiting ..." << endl;
			return EXIT_FAILURE;
		}
		TestClassifiers testClassifiers(firstFrameFileName, gtFileName);
		testClassifiers.run();
	}
}