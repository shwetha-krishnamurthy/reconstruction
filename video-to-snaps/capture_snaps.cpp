#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <string> 
#include <sstream>

//using namespace std;
using namespace cv;

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

int main()
{
    VideoCapture cap("/home/shwetha/btp/video/ehd.mp4"); // open the default camera
    if(!cap.isOpened())  // check if we succeeded
        return -1;
    
    int count = -1;

    namedWindow("video",1);
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera
        Size dsize=Size(round(0.5*frame.rows), round(0.5*frame.cols));
        resize(frame, frame, dsize);

        ++count;

        std::string name = "ehd" + patch::to_string(count) + ".jpg";
        if (count%25==0)
        {
            imwrite("/home/shwetha/btp/video/demosnaps/" + name, frame);
        }

        if (waitKey(1) >= 0) break;
        imshow("video", frame);
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}