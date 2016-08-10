#ifndef _WINDOW_
#define _WINDOW_
#include "opencv2/opencv.hpp"
#include "observable.h"
#include <vector>
#include <tuple>
using namespace cv ;

namespace SmartVision {
	class window{
		string name ;
	public :
		window(string nname):name(nname){
			namedWindow(name);		
		}
		void update(observable<Mat> &obs , const Mat& im){
			imshow(name,im);
		}
		void update(observable<tuple<Mat,Mat,vector<Rect_<int> > > > &obs , const tuple<Mat,Mat,vector<Rect_<int> > >& in){
			Mat show = get<0>(in);
			for(auto face : get<2>(in)){
				rectangle(show,face,CV_RGB(255,0,0),3) ;
			}
			
			imshow(name,show);
		}

	} ;
}


#endif

