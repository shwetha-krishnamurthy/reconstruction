#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void readme();

Mat img_object, img_scene;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);
void thresh_callback(int, void* );
vector<vector<Point> > contours_poly_global;

/* @function main */
int main()
{
  img_object = imread( "obj1.jpg");
  img_scene = imread( "scene2.jpg");
  cvtColor(img_object, img_object, CV_BGR2HSV);
  cvtColor(img_scene, img_scene, CV_BGR2HSV);

  Mat channel[3];
  split(img_object, channel);
  channel[2] = Mat(img_object.rows, img_object.cols, CV_8UC1, 130);//Set V

  //Merge channels
  merge(channel, 3, img_object);

  Mat channel1[3];
  split(img_scene, channel1);
  channel1[2] = Mat(img_scene.rows, img_scene.cols, CV_8UC1, 130);//Set V

  //Merge channels
  merge(channel1, 3, img_scene);
  
  cvtColor(img_object, img_object, CV_HSV2BGR);
  cvtColor(img_scene, img_scene, CV_HSV2BGR);

  cvtColor(img_object, img_object, CV_BGR2GRAY);
  cvtColor(img_scene, img_scene, CV_BGR2GRAY);

  imshow("Object Shadow Removed", img_object);
  imwrite("shadow_removed_object.jpg", img_object);
  imwrite("shadow_removed_scene.jpg", img_scene);

  if( !img_object.data || !img_scene.data )
  { cout<< " --(!) Error reading images " << std::endl; return -1; }
  Size dsize=Size(round(0.75*img_object.rows), round(0.75*img_object.cols));
  resize(img_object, img_object, dsize);
  dsize=Size(round(0.75*img_scene.rows), round(0.75*img_scene.cols));
  resize(img_scene, img_scene, dsize);

  const char* source_window = "Source";
  namedWindow( source_window, WINDOW_AUTOSIZE );
  imshow( source_window, img_scene );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );
  thresh_callback( 0, 0 );

  waitKey(0);
  return 0;
  }
 
  /* @function readme */
void readme()
{ cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }

void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  threshold( img_scene, threshold_output, thresh, 255, THRESH_BINARY );
  findContours( threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );
  for( size_t i = 0; i < contours.size(); i++ )
     { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
       contours_poly_global.push_back(contours_poly[i]);
       boundRect[i] = boundingRect( Mat(contours_poly[i]) );
       minEnclosingCircle( contours_poly[i], center[i], radius[i] );
     }
  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
  for( size_t i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
    //   rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
    //   circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
     }
  namedWindow( "Contours", WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
  imwrite("contours_poly.jpg", drawing);

  //-- Step 1: Detect the keypoints and extract descriptors using SURF
  int minHessian = 900;
  Ptr<SURF> detector = SURF::create( minHessian );
  vector<KeyPoint> keypoints_object, keypoints_scene;
  Mat descriptors_object, descriptors_scene;
  detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );
  detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );
 
  //-- Step 2: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );
  double max_dist = 0; double min_dist = 100;
 
  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  vector< DMatch > good_matches;
  vector<int> keypoint_match_count(contours_poly_global.size());
  for (int i = 0; i < contours_poly_global.size(); i++)
  {
  	for (int j = 0; j <  good_matches.size(); j++)
  	{
  		if( pointPolygonTest(contours_poly_global[i], keypoints_scene[ matches[j].trainIdx ].pt, false) >= 0)
  		{
  			keypoint_match_count[i]++;
  		}
  	}
  }
  
  int max_index = max_element(keypoint_match_count.begin(), keypoint_match_count.end()) - keypoint_match_count.begin();
  for (int i = 0; i < keypoint_match_count.size(); i++)
  {
   cout << keypoint_match_count[i]<< ", " << i << endl; 
  }

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( (matches[i].distance <= max(2*min_dist, 0.20))  && ( pointPolygonTest(contours_poly_global[max_index], keypoints_scene[ matches[i].trainIdx ].pt, false) >= 0))
     { good_matches.push_back( matches[i]); }
  }
  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  //-- Localize the object
  vector<Point2f> obj;
  vector<Point2f> scene;
  for( size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, RANSAC );
  //-- Get the corners from the image_1 ( the object to be "detected" )
  vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  vector<Point2f> scene_corners(4);
  perspectiveTransform( obj_corners, scene_corners, H);
 
  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
 
  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );
  imwrite("keypoints_matches.jpg", img_matches);
}