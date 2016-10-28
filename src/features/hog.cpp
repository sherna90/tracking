/**
 * @file hist.cpp
 * @brief texture histogram handling
 * @author Sergio Hernandez
 */
 #include "hog.hpp"

using namespace cv;
using namespace std;

//Parameters
#define N_BINS 16           //Number of bins
#define N_DIVS 3            //Number of cells = N_DIVS*N_DIVS
#define N_PHOG N_DIVS*N_DIVS*N_BINS
#define BIN_RANGE (2*CV_PI)/N_BINS

void calc_hog(Mat& image,Mat& hist){
    // default opencv implementation
    Mat part_hog;
    std::vector<float> descriptors;
    std::vector<Point> points;
    HOGDescriptor descriptor;
    descriptor.winSize=Size(48,96);
    descriptor.nbins=32;
    if(image.cols>0 && image.rows>0){
        resize(image,part_hog,descriptor.winSize,0,0,INTER_LINEAR);
        cvtColor(part_hog, part_hog, COLOR_RGB2GRAY);
        descriptor.compute(part_hog,descriptors,Size(0,0), Size(0,0),points);
        hist.create(1,descriptors.size(),CV_32FC1);
        for(unsigned int i=0;i<descriptors.size();i++){
            hist.at<float>(0,i)=descriptors.at(i);
        }
    }
    //normalize(hist, hist, 0, 1, NORM_MINMAX);
}

void calc_hog(Mat& image,Eigen::VectorXd& hist,cv::Size reference_size){
    // default opencv implementation
    Mat part_hog;
    std::vector<float> descriptors;
    std::vector<Point> points;
    HOGDescriptor descriptor;
    descriptor.winSize=Size(64,128);
    //descriptor.nbins=32;
    if(image.cols>0 && image.rows>0){
        resize(image,part_hog,descriptor.winSize,0,0,INTER_LINEAR);
        //cvtColor(part_hog, part_hog, COLOR_RGB2GRAY);
        descriptor.compute(part_hog,descriptors,Size(0,0), Size(0,0),points);
        hist.setOnes(descriptors.size());
        for(unsigned int i=0;i<descriptors.size();i++){
            hist[i]=descriptors.at(i);
        }
    }
}

/*void calc_hog_gpu(Mat& image,Eigen::VectorXd& hist){
    // default opencv implementation
    Mat part_hog;
    cuda::GpuMat gpu_img;
    cuda::GpuMat gpu_descriptor_temp;
    std::vector<float> descriptors;
    cv::Ptr<cv::cuda::HOG> descriptor = cv::cuda::HOG::create(Size(48,96), Size(16, 16), Size(8, 8), Size(8, 8), 32);
    if(image.cols>0 && image.rows>0){
        resize(image,part_hog,Size(48,96),0,0,INTER_LINEAR);
        gpu_img.upload(part_hog);
        descriptor->compute(gpu_img,gpu_descriptor_temp);
        Mat gpu_descriptor(gpu_descriptor_temp);
        gpu_descriptor.copyTo(descriptors);
        hist.setOnes(descriptors.size());
        for(unsigned int i=0;i<descriptors.size();i++){
            hist[i]=descriptors.at(i);
        }
    }
    //normalize(hist, hist, 0, 1, NORM_MINMAX);
}*/
