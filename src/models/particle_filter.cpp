/**
 * @file particle_filter.cpp
 * @brief particle filter
 * @author Sergio Hernandez
 */
#include "particle_filter.hpp"

#ifndef PARAMS
const float POS_STD=1.0;
const float SCALE_STD=1.0;
const float DT=1.0;
const float THRESHOLD=1.0;
const float OVERLAP_RATIO=0.8;

//const bool INCREMENTAL_GAUSSIAN_NAIVEBAYES=true;
const bool GAUSSIAN_NAIVEBAYES=false;
const bool LOGISTIC_REGRESSION=true;
const bool MULTINOMIAL_NAIVEBAYES=false;

const bool HAAR_FEATURE=true;
const bool LBP_FEATURE=false;
const bool HOG_FEATURE=false;
const bool MB_LBP_FEATURE=false;
#endif

particle_filter::particle_filter() {
}

particle_filter::~particle_filter() {
    states.clear();
    weights.clear();
}

particle_filter::particle_filter(int _n_particles) {
    states.clear();
    weights.clear();
    positive_likelihood.clear();
    n_particles = _n_particles;
    time_stamp=0;
    initialized=false;
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed1);
    theta_x.clear();
    RowVectorXd theta_x_pos(2);
    theta_x_pos << POS_STD,POS_STD;
    theta_x.push_back(theta_x_pos);
    RowVectorXd theta_x_scale(2);
    theta_x_scale << SCALE_STD,SCALE_STD;
    theta_x.push_back(theta_x_scale);
    eps= std::numeric_limits<double>::epsilon();
}

bool particle_filter::is_initialized() {
    return initialized;
}

void particle_filter::reinitialize() {
    initialized=false;
}

void particle_filter::initialize(Mat& current_frame, Rect ground_truth) {
    normal_distribution<double> negative_random_pos(0.0,20.0);
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    marginal_likelihood=0.0;
    vector<Rect> negativeBox;
    states.clear();
    weights.clear();
    estimates.clear();
    sampleBox.clear();
    estimates.push_back(ground_truth);
    //cout << "INIT!!!!!" << endl;
    //cout << ground_truth << endl;

    im_size=current_frame.size();
    int left = MAX(ground_truth.x, 1);
    int top = MAX(ground_truth.y, 1);
    int right = MIN(ground_truth.x + ground_truth.width, current_frame.cols - 1);
    int bottom = MIN(ground_truth.y + ground_truth.height, current_frame.rows - 1);
    reference_roi=Rect(left, top, right - left, bottom - top);
    if(reference_roi.width>0 && (reference_roi.x+reference_roi.width)<im_size.width && 
        reference_roi.height>0 && (reference_roi.y+reference_roi.height)<im_size.height){
        marginal_likelihood=0.0;
        float weight=log(1.0/n_particles);
        for (int i=0;i<n_particles;i++){
            particle state;
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
            states.push_back(state);
            weights.push_back(weight);
            ESS=0.0f;   
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
        Mat grayImg;
        cvtColor(current_frame, grayImg, CV_RGB2GRAY);
        //equalizeHist( grayImg, grayImg );
        haar.init(grayImg,reference_roi,sampleBox);

        if(GAUSSIAN_NAIVEBAYES){
            VectorXi labels(2*n_particles);
            labels << VectorXi::Ones(n_particles), VectorXi::Zero(n_particles);
            
            if(HAAR_FEATURE){
                MatrixXd eigen_sample_positive_feature_value, eigen_sample_negative_feature_value;
                cv2eigen(haar.sampleFeatureValue, eigen_sample_positive_feature_value);
                haar.getFeatureValue(grayImg,negativeBox);
                cv2eigen(haar.sampleFeatureValue, eigen_sample_negative_feature_value);
                MatrixXd eigen_sample_feature_value( eigen_sample_positive_feature_value.rows(),
                    eigen_sample_positive_feature_value.cols() + eigen_sample_negative_feature_value.cols());
                eigen_sample_feature_value <<   eigen_sample_positive_feature_value,
                                                eigen_sample_negative_feature_value;
                eigen_sample_feature_value.transposeInPlace();
                gaussian_naivebayes = GaussianNaiveBayes(eigen_sample_feature_value, labels);
                gaussian_naivebayes.fit();
            }
            if(LBP_FEATURE){
                local_binary_pattern.init(grayImg, sampleBox);
                local_binary_pattern.getFeatureValue(grayImg, negativeBox, false);
                MatrixXd eigen_sample_feature_value(local_binary_pattern.sampleFeatureValue.rows() +
                local_binary_pattern.negativeFeatureValue.rows(), local_binary_pattern.sampleFeatureValue.cols());
                eigen_sample_feature_value << local_binary_pattern.sampleFeatureValue,
                                              local_binary_pattern.negativeFeatureValue;
                gaussian_naivebayes = GaussianNaiveBayes(eigen_sample_feature_value, labels);
                gaussian_naivebayes.fit();
            }
            if(MB_LBP_FEATURE){
                multiblock_local_binary_patterns = MultiScaleBlockLBP(3,59,2,true,false,3,3);
                multiblock_local_binary_patterns.init(grayImg, sampleBox);
                multiblock_local_binary_patterns.getFeatureValue(grayImg, negativeBox, false);
                MatrixXd eigen_sample_feature_value(multiblock_local_binary_patterns.sampleFeatureValue.rows() +
                    multiblock_local_binary_patterns.negativeFeatureValue.rows(), multiblock_local_binary_patterns.sampleFeatureValue.cols());
                eigen_sample_feature_value << multiblock_local_binary_patterns.sampleFeatureValue,
                                              multiblock_local_binary_patterns.negativeFeatureValue;
                gaussian_naivebayes = GaussianNaiveBayes(eigen_sample_feature_value, labels);
                gaussian_naivebayes.fit();
            }
            if(HOG_FEATURE){
                MatrixXd hog_descriptors(0, 3780);
                VectorXd hist;
                for (unsigned int i = 0; i < sampleBox.size(); ++i)
                {
                    Mat subImage = grayImg(sampleBox.at(i));
                    calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                    hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                    hog_descriptors.row(hog_descriptors.rows()-1) = hist;
                }

                for (unsigned int i = 0; i < negativeBox.size(); ++i)
                {
                    Mat subImage = grayImg(negativeBox.at(i));
                    calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                    hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                    hog_descriptors.row(hog_descriptors.rows()-1) = hist;
                }
                gaussian_naivebayes = GaussianNaiveBayes(hog_descriptors, labels);
                gaussian_naivebayes.fit();
            }
        }

        if(LOGISTIC_REGRESSION){
            VectorXd labels(2*n_particles);
            labels << VectorXd::Ones(n_particles), VectorXd::Constant(n_particles,-1.0);
            hamiltonian_monte_carlo=Hamiltonian_MC();
            /*int num_iter=1e2;
            double step_size=1e-3;
            int leapgrog=10;*/ 
            double lambda=0.1;
            int num_steps=10;
            if(HAAR_FEATURE){
                MatrixXd eigen_sample_positive_feature_value, eigen_sample_negative_feature_value;
                cv2eigen(haar.sampleFeatureValue, eigen_sample_positive_feature_value);
                haar.getFeatureValue(grayImg,negativeBox);
                cv2eigen(haar.sampleFeatureValue, eigen_sample_negative_feature_value);
                MatrixXd eigen_sample_feature_value( eigen_sample_positive_feature_value.rows(),
                    eigen_sample_positive_feature_value.cols() + eigen_sample_negative_feature_value.cols());
                eigen_sample_feature_value <<   eigen_sample_positive_feature_value,
                                                eigen_sample_negative_feature_value;
                eigen_sample_feature_value.transposeInPlace();
                hamiltonian_monte_carlo = Hamiltonian_MC(eigen_sample_feature_value, labels,lambda);
                //hamiltonian_monte_carlo.run(1e3,1e-2,10);
                hamiltonian_monte_carlo.fit_map(3);
            }

            if(LBP_FEATURE){
                //local_binary_pattern = LocalBinaryPattern();
                local_binary_pattern.init(grayImg, sampleBox);
                local_binary_pattern.getFeatureValue(grayImg, negativeBox, false);
                MatrixXd eigen_sample_feature_value(local_binary_pattern.sampleFeatureValue.rows() +
                local_binary_pattern.negativeFeatureValue.rows(), local_binary_pattern.sampleFeatureValue.cols());
                eigen_sample_feature_value << local_binary_pattern.sampleFeatureValue,
                                              local_binary_pattern.negativeFeatureValue;
                hamiltonian_monte_carlo = Hamiltonian_MC(eigen_sample_feature_value, labels,lambda);
                hamiltonian_monte_carlo.fit_map(num_steps);
            }

            if(MB_LBP_FEATURE){
                multiblock_local_binary_patterns = MultiScaleBlockLBP(3,59,2,true,false,3,3);
                multiblock_local_binary_patterns.init(grayImg, sampleBox);
                multiblock_local_binary_patterns.getFeatureValue(grayImg, negativeBox, false);
                MatrixXd eigen_sample_feature_value(multiblock_local_binary_patterns.sampleFeatureValue.rows() +
                    multiblock_local_binary_patterns.negativeFeatureValue.rows(), multiblock_local_binary_patterns.sampleFeatureValue.cols());
                eigen_sample_feature_value << multiblock_local_binary_patterns.sampleFeatureValue,
                                              multiblock_local_binary_patterns.negativeFeatureValue;
                hamiltonian_monte_carlo = Hamiltonian_MC(eigen_sample_feature_value, labels,lambda);
                hamiltonian_monte_carlo.fit_map(num_steps);
            }

            if(HOG_FEATURE){
                //MatrixXd hog_descriptors(sampleBox.size() + negativeBox.size(), 7040);
                MatrixXd hog_descriptors(0, 3780);
                VectorXd hist;
                for (unsigned int i = 0; i < sampleBox.size(); ++i)
                {
                    Mat subImage = grayImg(sampleBox.at(i));
                    calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                    hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                    hog_descriptors.row(hog_descriptors.rows()-1) = hist;
                }

                for (unsigned int i = 0; i < negativeBox.size(); ++i)
                {
                    Mat subImage = grayImg(negativeBox.at(i));
                    calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                    hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                    hog_descriptors.row(hog_descriptors.rows()-1) = hist;
                }
                hamiltonian_monte_carlo = Hamiltonian_MC(hog_descriptors, labels,lambda);
                hamiltonian_monte_carlo.fit_map(num_steps);
                //logistic_regression = LogisticRegression(eigen_sample_feature_value, labels,lambda);
                //logistic_regression.Train(num_iter,step_size);
            }
            
        }

        if(MULTINOMIAL_NAIVEBAYES){
            VectorXd labels(2*n_particles);
            labels << VectorXd::Ones(n_particles), VectorXd::Zero(n_particles);
            double lambda=0.1;
            if(HAAR_FEATURE){
                MatrixXd eigen_sample_positive_feature_value, eigen_sample_negative_feature_value;
                cv2eigen(haar.sampleFeatureValue, eigen_sample_positive_feature_value);
                haar.getFeatureValue(grayImg,negativeBox);
                cv2eigen(haar.sampleFeatureValue, eigen_sample_negative_feature_value);
                MatrixXd eigen_sample_feature_value( eigen_sample_positive_feature_value.rows(),
                    eigen_sample_positive_feature_value.cols() + eigen_sample_negative_feature_value.cols());
                eigen_sample_feature_value <<   eigen_sample_positive_feature_value,
                                                eigen_sample_negative_feature_value;
                eigen_sample_feature_value.transposeInPlace();
                multinomial_naivebayes = MultinomialNaiveBayes(eigen_sample_feature_value, labels);
                multinomial_naivebayes.fit(lambda);
            }

            if(LBP_FEATURE){
                local_binary_pattern.init(grayImg, sampleBox);
                local_binary_pattern.getFeatureValue(grayImg, negativeBox, false);
                MatrixXd eigen_sample_feature_value(local_binary_pattern.sampleFeatureValue.rows() +
                local_binary_pattern.negativeFeatureValue.rows(), local_binary_pattern.sampleFeatureValue.cols());
                eigen_sample_feature_value << local_binary_pattern.sampleFeatureValue,
                                              local_binary_pattern.negativeFeatureValue;
                multinomial_naivebayes = MultinomialNaiveBayes(eigen_sample_feature_value, labels);
                multinomial_naivebayes.fit(lambda);
            }

            if(MB_LBP_FEATURE){
                multiblock_local_binary_patterns = MultiScaleBlockLBP(3,59,2,true,false,3,3);
                multiblock_local_binary_patterns.init(grayImg, sampleBox);
                multiblock_local_binary_patterns.getFeatureValue(grayImg, negativeBox, false);
                MatrixXd eigen_sample_feature_value(multiblock_local_binary_patterns.sampleFeatureValue.rows() +
                    multiblock_local_binary_patterns.negativeFeatureValue.rows(), multiblock_local_binary_patterns.sampleFeatureValue.cols());
                eigen_sample_feature_value << multiblock_local_binary_patterns.sampleFeatureValue,
                                              multiblock_local_binary_patterns.negativeFeatureValue;
                multinomial_naivebayes = MultinomialNaiveBayes(eigen_sample_feature_value, labels);
                multinomial_naivebayes.fit(lambda);
            }

            if(HOG_FEATURE){
                MatrixXd hog_descriptors(0, 3780);
                VectorXd hist;
                for (unsigned int i = 0; i < sampleBox.size(); ++i)
                {
                    Mat subImage = grayImg(sampleBox.at(i));
                    calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                    hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                    hog_descriptors.row(hog_descriptors.rows()-1) = hist;
                }

                for (unsigned int i = 0; i < negativeBox.size(); ++i)
                {
                    Mat subImage = grayImg(negativeBox.at(i));
                    calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                    hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                    hog_descriptors.row(hog_descriptors.rows()-1) = hist;
                }
                multinomial_naivebayes=MultinomialNaiveBayes(hog_descriptors, labels);
                multinomial_naivebayes.fit(lambda);
            }
        }
        initialized=true;
    }
}

void particle_filter::predict(){
    normal_distribution<double> position_random_x(0.0,theta_x.at(0)(0));
    normal_distribution<double> position_random_y(0.0,theta_x.at(0)(1));
    normal_distribution<double> scale_random_width(0.0,theta_x.at(1)(0));
    normal_distribution<double> scale_random_height(0.0,theta_x.at(1)(1));
    sampleBox.clear();//important
    //cout << "predicted particles!" <<endl;
    if(initialized==true){
        time_stamp++;
        vector<particle> tmp_new_states(n_particles);
        for (int i=0;i<n_particles;i++){
            particle state=states[i];
            float _x,_y,_width,_height;
            float _dx=position_random_x(generator);
            float _dy=position_random_y(generator);
            //float _dw=scale_random_width(generator);
            //float _dh=scale_random_height(generator);
            _x=MIN(MAX(cvRound(state.x),0),im_size.width);
            _y=MIN(MAX(cvRound(state.y),0),im_size.height);
            _width=MIN(MAX(cvRound(state.width),10),im_size.width);
            _height=MIN(MAX(cvRound(state.height),10),im_size.height);
            //_width=MIN(MAX(cvRound(state.width+state.scale),0),im_size.width);
            //_height=MIN(MAX(cvRound(state.height+state.scale),0),im_size.height);
            if( (_x+_width)<im_size.width 
                && _x>0 
                && (_y+_height)<im_size.height 
                && _y>0 
                && _width<im_size.width 
                && _height<im_size.height 
                && _width>0 && _height>0 ){
                state.x_p=state.x;
                state.y_p=state.y;
                state.x=cvRound(2*_x-state.x_p+_dx);
                state.y=cvRound(2*_y-state.y_p+_dy);
                state.width_p=state.width;
                state.height_p=state.height;       
                state.width=cvRound(2*_width-state.width_p);
                state.height=cvRound(2*_height-state.height_p);
                state.scale=((float)state.width/(float)reference_roi.width)/2.0f+((float)state.height/(float)reference_roi.height)/2.0f;
            }
            else{
                state.x=state.x;
                state.y=state.y;
                state.x_p=reference_roi.x;
                state.y_p=reference_roi.y;
                state.width=cvRound(reference_roi.width);
                state.height=cvRound(reference_roi.height);
                state.width_p=cvRound(reference_roi.width);
                state.height_p=cvRound(reference_roi.height);
                state.scale=1.0f;
            }
            Rect box(state.x, state.y, state.width, state.height);
            box.x=MIN(MAX(cvRound(box.x),0),im_size.width);
            box.y=MIN(MAX(cvRound(box.y),0),im_size.height);
            box.width=MIN(MAX(cvRound(box.width),0),im_size.width-box.x);
            box.height=MIN(MAX(cvRound(box.height),0),im_size.height-box.y);
            sampleBox.push_back(box);
            
            //cout << "box " << box.height << " " << box.width << endl;

            //cout << "reference " << reference_roi << endl;
            //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height <<",weight:" << weights[i] << endl;
            tmp_new_states[i]=state;
        }
        states.swap(tmp_new_states);
        tmp_new_states = vector<particle>();
    }
}


void particle_filter::draw_particles(Mat& image, Scalar color=Scalar(0,255,255)){
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        Point pt1,pt2;
        pt1.x=cvRound(state.x);
        pt1.y=cvRound(state.y);
        pt2.x=cvRound(state.x+state.width);
        pt2.y=cvRound(state.y+state.height);
        rectangle( image, pt1,pt2, color, 1, LINE_AA );
    }
}

Rect particle_filter::estimate(Mat& image,bool draw=false){
    float _x=0.0,_y=0.0,_width=0.0,_height=0.0,norm=0.0;
    Rect estimate;
    //cout << "estimated particles!" <<endl;
    for (int i=0;i<n_particles;i++){
        particle state=states[i];
        //double weight=weights.at(i);
        if(state.x>0 && state.x<im_size.width 
            && state.y>0  && state.y<im_size.height 
            && state.width>0 && state.width<im_size.height 
            && state.height>0 && state.height<im_size.height){
            _x+= state.x; 
            _y+= state.y; 
            _width+= state.width; 
            _height+= state.height;
            norm++;
        }
        //else{
        //    cout << "weight:"  << weights[i] <<", x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
        //} 
        //cout << "weight:" << weight << endl;
        //cout << "ref x:" << reference_roi.x << ",y:" << reference_roi.y <<",w:" << reference_roi.width <<",h:" << reference_roi.height << endl;
        //
    }
    Point pt1,pt2;
    pt1.x=cvRound(_x/norm);
    pt1.y=cvRound(_y/norm);
    _width=cvRound(_width/norm);
    _height=cvRound(_height/norm);
    pt2.x=cvRound(pt1.x+_width);
    pt2.y=cvRound(pt1.y+_height); 
    if(pt2.x<im_size.width && pt1.x>=0 && pt2.y<im_size.height && pt1.y>=0){
        if(draw) rectangle( image, pt1,pt2, Scalar(0,0,255), 2, LINE_AA );
        estimate=Rect(pt1.x,pt1.y,_width,_height);
    }
    //cout << " estimate x:" << estimate.x << ",y:" << estimate.y <<",w:" << estimate.width <<",h:" << estimate.height << endl;
    estimates.push_back(estimate);
    return estimate;

}

void particle_filter::update(Mat& image)
{
    vector<float> tmp_weights;
    //uniform_int_distribution<int> random_feature(0,haar.featureNum-1);
    Mat grayImg;
    cvtColor(image, grayImg, CV_RGB2GRAY);
    //equalizeHist( grayImg, grayImg );

    if(GAUSSIAN_NAIVEBAYES){
        MatrixXd Phi;
        if(HAAR_FEATURE){
            haar.getFeatureValue(grayImg,sampleBox);
            MatrixXd eigen_sample_feature_value;
            cv2eigen(haar.sampleFeatureValue, eigen_sample_feature_value);
            eigen_sample_feature_value.transposeInPlace();
            Phi = gaussian_naivebayes.get_proba(eigen_sample_feature_value);
            //cout << "in loop" << endl;
        }
        //cout << "update" << endl;
        /*for (int i = 0; i < n_particles; ++i)
        {
            states[i] = update_state(states[i], image);
            tmp_weights.push_back(Phi(i,1)-Phi(i,0));
        }*/
        //cout << log(Phi.col(1)) << endl;
        if(LBP_FEATURE){
            local_binary_pattern.getFeatureValue(grayImg, sampleBox);
            Phi = gaussian_naivebayes.get_proba(local_binary_pattern.sampleFeatureValue);
        }

        if(MB_LBP_FEATURE){
            multiblock_local_binary_patterns.getFeatureValue(grayImg, sampleBox, true);
            Phi = gaussian_naivebayes.get_proba(multiblock_local_binary_patterns.sampleFeatureValue);
        }

        if(HOG_FEATURE){
            MatrixXd hog_descriptors(0,3780);
            VectorXd hist;
            for (unsigned int i = 0; i < sampleBox.size(); ++i)
            {
                Mat subImage = grayImg(sampleBox.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize(hog_descriptors.rows()+1, hog_descriptors.cols());
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }
            Phi = gaussian_naivebayes.get_proba(hog_descriptors);
        }
        //cout << "update" << endl;
        for (int i = 0; i < n_particles; ++i)
        {
            states[i] = update_state(states[i], image);
            weights[i]=Phi(i,1)-Phi(i,0);
        }
    }

    if(LOGISTIC_REGRESSION){
        VectorXd phi;
        
        if(HAAR_FEATURE){
            haar.getFeatureValue(grayImg,sampleBox);
            MatrixXd eigen_sample_feature_value;
            cv2eigen(haar.sampleFeatureValue, eigen_sample_feature_value);
            eigen_sample_feature_value.transposeInPlace();
            phi = hamiltonian_monte_carlo.predict(eigen_sample_feature_value);
            //phi = logistic_regression.Predict(eigen_sample_feature_value);
            //cout << "phi: " << phi.transpose() << endl;
        }

        if(LBP_FEATURE){
            local_binary_pattern.getFeatureValue(grayImg,sampleBox);
            phi = hamiltonian_monte_carlo.predict(local_binary_pattern.sampleFeatureValue);
        }

        if(MB_LBP_FEATURE){
            multiblock_local_binary_patterns.getFeatureValue(grayImg, sampleBox, true);
            phi = hamiltonian_monte_carlo.predict(multiblock_local_binary_patterns.sampleFeatureValue);
        }

        if(HOG_FEATURE){
            //MatrixXd hog_descriptors(sampleBox.size(),7040);
            MatrixXd hog_descriptors(0,3780);
            VectorXd hist;
            for (unsigned int i = 0; i < sampleBox.size(); ++i)
            {
                Mat subImage = grayImg(sampleBox.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize(hog_descriptors.rows()+1, hog_descriptors.cols());
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
                //hog_descriptors.row(i) = hist;
            }
            phi = hamiltonian_monte_carlo.predict(hog_descriptors);
        }
        //double max_value=phi.maxCoeff(); 
        //cout << phi.transpose() << ", max value: "<< max_value << ", prob: "<< max_value+log((phi.array()-max_value).exp().sum())-log(n_particles) << endl;
        for (int i = 0; i < n_particles; ++i)
        {
            states[i] = update_state(states[i], image);
            weights[i]=phi(i);
        }
    }

    if(MULTINOMIAL_NAIVEBAYES){
        MatrixXd Phi;
        if(HAAR_FEATURE){
            haar.getFeatureValue(grayImg,sampleBox);
            MatrixXd eigen_sample_feature_value;
            cv2eigen(haar.sampleFeatureValue, eigen_sample_feature_value);
            eigen_sample_feature_value.transposeInPlace();
            Phi = multinomial_naivebayes.get_proba(eigen_sample_feature_value);
            //cout << "in loop" << endl;
        }

        if(LBP_FEATURE){
            local_binary_pattern.getFeatureValue(grayImg, sampleBox);
            Phi = multinomial_naivebayes.get_proba(local_binary_pattern.sampleFeatureValue);
        }

        if(MB_LBP_FEATURE){
            multiblock_local_binary_patterns.getFeatureValue(grayImg, sampleBox, true);
            Phi = multinomial_naivebayes.get_proba(multiblock_local_binary_patterns.sampleFeatureValue);
        }

        if(HOG_FEATURE){
            MatrixXd hog_descriptors(0,3780);
            VectorXd hist;
            for (unsigned int i = 0; i < sampleBox.size(); ++i)
            {
                Mat subImage = grayImg(sampleBox.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize(hog_descriptors.rows()+1, hog_descriptors.cols());
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }
            Phi = multinomial_naivebayes.get_proba(hog_descriptors);
        }
        //cout << "update" << endl;
        for (int i = 0; i < n_particles; ++i)
        {
            states[i] = update_state(states[i], image);
            weights[i]=(Phi(i,1)-Phi(i,0));
        }
        //cout << log(Phi.col(1)) << endl;
    }

    //weights.swap(tmp_weights);
    tmp_weights.clear();
    resample();

}

float particle_filter::resample(){
    vector<float> cumulative_sum(n_particles);
    vector<float> normalized_weights(n_particles);
    vector<float> squared_normalized_weights(n_particles);
    uniform_real_distribution<float> unif_rnd(0.0,1.0); 
    float max_value = *max_element(weights.begin(), weights.end());
    float logsumexp=0.0f;
    for (int i=0; i<n_particles; i++) {
        normalized_weights.at(i) = exp(weights.at(i)-max_value);
        logsumexp+=normalized_weights.at(i);
    }
    Scalar sum_weights=sum(normalized_weights);
    for (int i=0; i<n_particles; i++) {
        normalized_weights.at(i) = normalized_weights.at(i)/sum_weights[0];
    }
    for (int i=0; i<n_particles; i++) {
        squared_normalized_weights.at(i)=normalized_weights.at(i)*normalized_weights.at(i);
        if (i==0) {
            cumulative_sum.at(i) = normalized_weights.at(i);
        } else {
            cumulative_sum.at(i) = cumulative_sum.at(i-1) + normalized_weights.at(i);
        }
        //cout << " cumsum: " << normalized_weights.at(i) << "," <<cumulative_sum.at(i) << endl;
    }
    Scalar sum_squared_weights=sum(squared_normalized_weights);
    marginal_likelihood+=max_value+log(sum_weights[0])-log(n_particles); 
    ESS=1/sum_squared_weights[0]/n_particles;
    cout  << "ESS :" << ESS << ",marginal_likelihood :" << marginal_likelihood <<  endl;
    //cout << "resampled particles!" << ESS << endl;
    if(isless(ESS,(float)THRESHOLD)){
        vector<particle> new_states(n_particles);
        for (int i=0; i<n_particles; i++) {
            float uni_rand = unif_rnd(generator);
            vector<float>::iterator pos = lower_bound(cumulative_sum.begin(), cumulative_sum.end(), uni_rand);
            unsigned int ipos = distance(cumulative_sum.begin(), pos);
            
            particle state=states[ipos];
            
            //cout << "x:" << state.x << ",y:" << state.y <<",w:" << state.width <<",h:" << state.height << endl;
            new_states[i]=state;
            weights[i]=log(1.0f/n_particles);
        }
        states.swap(new_states);
        new_states = vector<particle>();
    }
    else{
        //weights.swap(log_normalized_weights);
    }
    cumulative_sum.clear();
    normalized_weights.clear();
    squared_normalized_weights.clear();
    return marginal_likelihood;
}

float particle_filter::getESS(){
    return ESS;
}

void particle_filter::update_model(vector<VectorXd> theta_x_new){
    theta_x.clear();
    VectorXd theta_x_pos=theta_x_new.at(0);
    VectorXd theta_x_scale=theta_x_new.at(1);
    theta_x.push_back(theta_x_pos);
    theta_x.push_back(theta_x_scale);
}

void particle_filter::update_model(Mat& current_frame,vector<Rect> positive_examples,vector<Rect> negative_examples){
    /*Mat grayImg;
    cvtColor(current_frame, grayImg, CV_RGB2GRAY);
    if(LOGISTIC_REGRESSION){
        VectorXd labels(positive_examples.size()+negative_examples.size());
        labels << VectorXd::Ones(positive_examples.size()), VectorXd::Constant(negative_examples.size(),-1.0);
        if(HAAR_FEATURE){
            haar.init(grayImg,reference_roi,positive_examples);
            MatrixXd eigen_sample_positive_feature_value, eigen_sample_negative_feature_value;
            cv2eigen(haar.sampleFeatureValue, eigen_sample_positive_feature_value);
            haar.getFeatureValue(grayImg,negative_examples);
            cv2eigen(haar.sampleFeatureValue, eigen_sample_negative_feature_value);
            MatrixXd eigen_sample_feature_value( eigen_sample_positive_feature_value.rows(),
                eigen_sample_positive_feature_value.cols() + eigen_sample_negative_feature_value.cols());
            eigen_sample_feature_value <<   eigen_sample_positive_feature_value,
                                            eigen_sample_negative_feature_value;
            eigen_sample_feature_value.transposeInPlace();
            //logistic_regression.setData(eigen_sample_feature_value, labels);
            //logistic_regression.Train(1e2,1e-3,1);
        }
        if(HOG_FEATURE){
            //MatrixXd hog_descriptors(sampleBox.size() + negativeBox.size(), 7040);
            MatrixXd hog_descriptors(0, 3780);
            VectorXd hist;
            for (unsigned int i = 0; i < positive_examples.size(); ++i)
            {
                Mat subImage = grayImg(positive_examples.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }

            for (unsigned int i = 0; i < negative_examples.size(); ++i)
            {
                Mat subImage = grayImg(negative_examples.at(i));
                calc_hog(subImage, hist,Size(reference_roi.width,reference_roi.height));
                hog_descriptors.conservativeResize( hog_descriptors.rows()+1, hog_descriptors.cols() );
                hog_descriptors.row(hog_descriptors.rows()-1) = hist;
            }
            //logistic_regression.Train(1e3,1e-3,1);
        }
     }*/   
}

vector<VectorXd> particle_filter::get_dynamic_model(){
    return theta_x;
}

vector<VectorXd> particle_filter::get_observation_model(){
    return theta_y;
}
float particle_filter::getMarginalLikelihood(){
    return marginal_likelihood;
}

particle particle_filter::update_state(particle state, Mat& image){
    if (state.width < 0 || state.width>image.cols){
        state.width = reference_roi.width;
    }
    if (state.height < 0 || state.height>image.rows){
        state.height = reference_roi.width;
    }
    if (state.x < 0 || state.x>image.cols){
        state.x = reference_roi.x;
    }
    if (state.y < 0 || state.y>image.rows){
        state.y = reference_roi.y;
    }
    return state;
}
