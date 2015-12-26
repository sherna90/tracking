/**
 * @file app.cpp
 * @brief visual tracking main application
 * @author Sergio Hernandez
 */

#include "../include/tracker.hpp"

Tracker::Tracker(){
}

Rect Tracker::stringToRect(string str){
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

