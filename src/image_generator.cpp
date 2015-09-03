#include "../include/image_generator.hpp"

using namespace std;
using namespace cv;

ImageGenerator::ImageGenerator(){
  cout << "ImageGenerator using VOT" << endl;
  mode = 1;

  frame_id = 0;
}

ImageGenerator::ImageGenerator(string _firstFrameFilename, string _groundTruthFile){
  // Initializes the Image Generator with a list of files in a VOT style
  cout << "ImageGenerator using local dataset." << endl;
  mode = 0;
  string filename = _firstFrameFilename;
  filenames.push_back(filename);
  while(1){
    getNextFilename(filename);
    Mat current_frame = imread(filename);
    if(current_frame.empty()){
        break;
    }else{
      filenames.push_back(filename);
    }
  }
  cout << "Loaded " << filenames.size() << " files." << endl;

  frame_id = 0;

  // Loads ground-truth data
  ifstream gt_file(_groundTruthFile.c_str(), ios::in);
  string line;
  while (getline(gt_file, line)) ground_truths.push_back(line);
  cout << "Stored " << int(ground_truths.size()) << " ground-truth data" << endl;
}

string ImageGenerator::getFrame(){
  string fn;
  if(frame_id < filenames.size()){
    fn = filenames.at(frame_id);
  }
  frame_id++;
  return fn;
}

string ImageGenerator::getRegion(){
  string gt;
  if(frame_id < filenames.size()){
    gt = ground_truths.at(frame_id);
  }
  return gt;
}

bool ImageGenerator::isEnded(){
  if(frame_id >= filenames.size()){
    return true;
  }else{
    return false;
  }
}

void ImageGenerator::getNextFilename(string& fn){
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
