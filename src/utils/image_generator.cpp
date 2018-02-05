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
  readGroundTruth(_groundTruthFile);
}

imageGenerator::imageGenerator(string _firstFrameFilename, string _groundTruthFile, string _detectionsFile){
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
  readGroundTruth(_groundTruthFile);
  readDetections(_detectionsFile);
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
    int frameNumber = 1;
    iss >> frameNumber;
    ostringstream oss;
    oss << (frameNumber + 1);
    // VOT
    //string zeros ("000000000");
    // OTB
    string zeros ("00000");
    string nextFrameNumberString = oss.str();
    string nextFrameFilename = prefix + zeros.substr(0,zeros.length()-1-nextFrameNumberString.length())+nextFrameNumberString + suffix;
    fn.assign(nextFrameFilename);
}

Rect imageGenerator::stringToRect(string line){
    const int NUMBER=4;
    vector<int> coords(4,0);
    size_t pos1=0;
    size_t pos2=line.find(",");
    for (int i = 0; i < NUMBER; i++){
        coords[i] = stoi(line.substr(pos1,pos2 - pos1));
        pos1 = pos2+1;
        pos2 = line.find(",", pos1 + 1);
    }
    /*Point pt[1][NUMBER];
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
    return Rect(minx,miny,cvRound(maxx-minx),cvRound(maxy-miny));*/
    return Rect(coords[0],coords[1],coords[2],coords[3]);
}

void imageGenerator::readDetections(string detFilename){
  ifstream dt_file(detFilename.c_str(), ios::in);
  string line;
  this->detections.resize(getDatasetSize());
  this->detection_weights.resize(getDatasetSize());

  vector<double> coords(4,0);
  int frame_num;

  while (getline(dt_file, line)) {
    Rect rect;
    size_t pos2 = line.find(",");
    size_t pos1 = 0;
    if(pos2 > pos1){
      frame_num = stoi(line.substr(pos1, pos2)) - 1;
      //pos1 = line.find(",",pos2 + 1);
      //pos2 = line.find(",",pos1 + 1);
      //coords[0] = stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
      for(int j = 0; j < 4; j++){
        pos1 = pos2;
        pos2 = line.find(",", pos1 + 1);
        coords[j] = stoi(line.substr(pos1 + 1,pos2 - pos1 - 1));
      }
      rect.x = coords[0];
      rect.y = coords[1];
      rect.width = coords[2];
      rect.height = coords[3];
      this->detections[frame_num].push_back(rect);
      
      pos1 = pos2;
      pos2 = line.find(",", pos1 + 1);
      this->detection_weights[frame_num].conservativeResize( this->detection_weights[frame_num].size() + 1 );
      this->detection_weights[frame_num](this->detection_weights[frame_num].size() - 1) = stod(line.substr(pos1 + 1, pos2 - pos1 - 1));
      //cout << frame_num << "," << rect.x << "," << rect.y << "," << rect.width << "," << rect.height << "," << stod(line.substr(pos1 + 1, pos2 - pos1 - 1)) << endl;
    }
  }
}

void imageGenerator::readGroundTruth(string gtFilename){
 ifstream gt_file(gtFilename.c_str(), ios::in);
  string line;
  while (getline(gt_file, line)) ground_truth.push_back(line);
  if(images.size() != ground_truth.size()){
        cerr << "There is not the same quantity of images and ground-truth data" << endl;
        cerr << "Maybe you typed wrong filenames" << endl;
        exit(EXIT_FAILURE);
  }
}