/**
 * @file hist.cpp
 * @brief texture histogram handling
 * @author Sergio Hernandez
 */
 #include "../include/hog.hpp"

using namespace cv;
using namespace std;

//Parameters
#define N_BINS 16           //Number of bins
#define N_DIVS 3            //Number of cells = N_DIVS*N_DIVS
#define N_PHOG N_DIVS*N_DIVS*N_BINS
#define BIN_RANGE (2*CV_PI)/N_BINS

void calc_hog(Mat& image,Mat& hist){
    // default opencv implmentation
    Mat part_hog;
    vector<float> descriptors;
    vector<Point> points;
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

void calc_fhog(Mat &Img,Mat& hist){
    // https://github.com/lastlegion/hog
    hist = Mat::zeros(1, N_PHOG, CV_32FC1);

    Mat Ix, Iy,grayImg;
    cvtColor( Img, grayImg, COLOR_BGR2GRAY );
    //Find orientation gradients in x and y directions
    Sobel(grayImg, Ix, CV_16S, 1, 0, 3);
    Sobel(grayImg, Iy, CV_16S, 0, 1, 3);
    
    int cellx = grayImg.cols/N_DIVS;
    int celly = grayImg.rows/N_DIVS;
    
    int img_area = grayImg.rows * grayImg.cols;

    for(int m=0; m < N_DIVS; m++)
    {
        for(int n=0; n < N_DIVS; n++)
        {
             for(int i=0; i<cellx; i++)
             {
                for(int j=0; j<celly; j++)
                {
                    
                    float px, py, grad, norm_grad, angle;

                    //px = Ix.at(m*cellx+i, n*celly+j);
                    //py = Iy.at(m*cellx+i, n*celly+j);
                    px = static_cast<float>(Ix.at<int16_t>((m*cellx)+i, (n*celly)+j ));
                    py = static_cast<float>(Iy.at<int16_t>((m*cellx)+i, (n*celly)+j ));

                    grad = static_cast<float>(std::sqrt(1.0*px*px + py*py));
                    norm_grad = grad/img_area;

                    //Orientation
                    angle = std::atan2(py,px);
                    
                    //convert to 0 to 360 (0 to 2*pi)
                    if( angle < 0)
                        angle+= 2*CV_PI;
                    
                    //find appropriate bin for angle
                    //nth_bin = angle/BIN_RANGE;
                    
                    //add magnitude of the edges in the hog matrix
                    hist.at<float>(0,(m*N_DIVS +n)*N_BINS + static_cast<int>(angle)) += norm_grad;

                }
             }
        }
    }  
    //Normalization
    for(int i=0; i< N_DIVS*N_DIVS; i++)
    { 
        float max=0;
        int j;
        for(j=0; j<N_BINS; j++)
        {
            if(hist.at<float>(0, i*N_BINS+j) > max)
                max = hist.at<float>(0,i*N_BINS+j);
        }
        for(j=0; j<N_BINS; j++)
            hist.at<float>(0, i*N_BINS+j)/=max;
    }
}