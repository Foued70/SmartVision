#ifndef _FACEDETECTION_MODULE_
#define _FACEDETECTION_MODULE_

#include "observable.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <tuple>
#include <queue>
using namespace cv;
using namespace std;

namespace SmartVision{ 
// enable polymorphic inheritance : use public key word after the inheritance symbol ":"
	class faceDetectionModule :public observable<tuple<Mat,Mat,vector<Rect_<int> > > > {
		CascadeClassifier haarCascade ;
		queue<Mat> input;		
	public 	:
		faceDetectionModule(string file_haar){
		 	haarCascade.load(file_haar);
		}
		void update(observable<Mat> &obs , const Mat& im){
			input.push(im);
		}
		void detectFaces(){
			while(1){
				while(!input.empty()){
					Mat color,gray;
					color=input.front();
					input.pop();
					cvtColor(color,gray,CV_BGR2GRAY);
					vector<Rect_<int> > faces;
					haarCascade.detectMultiScale(gray,faces);
					this->notifyObservers(make_tuple(color,gray,faces));
				}
			}
		}
	};
}

#endif
