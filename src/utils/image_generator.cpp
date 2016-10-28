#include "image_generator.hpp"

using namespace std;
using namespace cv;

imageGenerator::imageGenerator(){
}


imageGenerator::imageGenerator(string _firstFrameFilename, string _groundTruthFile){
  frame_id = 0;
  string FrameFilename,gtFilename;
  FrameFilename = _firstFrameFilename;
  gtFilename=_groundTruthFile;
  Mat current_frame = imread(FrameFilename);
  images.push_back(current_frame);
  while(1){
    getNextFilename(FrameFilename);
    current_frame = imread(FrameFilename );
    if(current_frame.empty()){
      break;
    }
    else{
      images.push_back(current_frame);
    }
  }
  ifstream gt_file(gtFilename.c_str(), ios::in);
  string line;
  while (getline(gt_file, line)) ground_truth.push_back(line);
  if(images.size() != ground_truth.size()){
        cerr << "There is not the same quantity of images and ground-truth data" << endl;
        cerr << "Maybe you typed wrong filenames" << endl;
        exit(EXIT_FAILURE);
  }
}

Mat imageGenerator::getFrame(){
  Mat current_frame=images[frame_id].clone();
  return current_frame;
}

Rect imageGenerator::getRegion(){
  string current_gt = ground_truth[frame_id];
  return stringToRect(current_gt);
}

bool imageGenerator::hasEnded(){
  if(frame_id >= (int) images.size()){
    return true;
  }else{
    return false;
  }
}

void imageGenerator::moveNext(){
  cout << frame_id << endl;
  frame_id++;
}

int imageGenerator::getDatasetSize(){
  return (int) images.size();
}

void imageGenerator::getNextFilename(string& fn){
    size_t index = fn.find_last_of("/");
    if(index == string::npos) {
        index = fn.find_last_of("\\");
    }
    //size_t index1 = fn.find_last_of("0");
    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    string frameNumberString = fn.substr(index+1, index2-index-1);
    istringstream iss(frameNumberString);
    int frameNumber = 0;
    iss >> frameNumber;
    ostringstream oss;
    oss << (frameNumber + 1);
    string zeros ("000000000");
    string nextFrameNumberString = oss.str();
    string nextFrameFilename = prefix + zeros.substr(0,zeros.length()-1-nextFrameNumberString.length())+nextFrameNumberString + suffix;
    fn.assign(nextFrameFilename);
}

Rect imageGenerator::stringToRect(string str){
    const int NUMBER=4;
    Point pt[1][NUMBER];
    size_t index1=0;
    size_t index2=-1;
    for (int i = 0; i < NUMBER; i++){
        index1=str.find(",",index2+1);
        string str_x1 = str.substr(index2+1, index1-index2-1);
        istringstream iss(str_x1);
        int x1 = 0;
        iss >> x1;
        index2=str.find(",",index1+1);
        string str_y1 = str.substr(index1+1, index2-index1-1);
        istringstream iss2(str_y1);
        int y1 = 0;
        iss2 >> y1;
        pt[0][i].x = cvRound(x1);
        pt[0][i].y = cvRound(y1);
    }

    //Make ground truth rect positive independently of point ordering
    int minx = pt[0][0].x;
    int maxx = pt[0][0].x;
    int miny = pt[0][0].y;
    int maxy = pt[0][0].y;
    for(int i = 0; i < NUMBER; i++){
      if(pt[0][i].x < minx)
        minx = pt[0][i].x;
      if(pt[0][i].x > maxx)
        maxx = pt[0][i].x;
      if(pt[0][i].y < miny)
        miny = pt[0][i].y;
      if(pt[0][i].y > maxy)
        maxy = pt[0][i].y;
    }
    return Rect(minx,miny,cvRound(maxx-minx),cvRound(maxy-miny));
}
