#include "acquisitionModule.h"
#include "observerDelegator.h"
#include "window.h"
#include "configurator.h"
#include <thread>
#include "faceDetectionModule.h"
using namespace SmartVision ;

string filePath ;
int cameraId ; 
method_type method;
istream& operator>>(istream& is , method_type& i){
	int tmp ;
	if(is >> tmp){
		i= static_cast<method_type> (tmp);
	}
	return is ;
}

void configure(char *filename){
	string fileName(filename);
	configurator conf(fileName);
	conf.addKeys("acquisition",{"method", "cameraId" ,"filePath"  });
	conf.read();
	conf.getProperty(string("acquisition"),string("method"),method);
	conf.getProperty(string("acquisition"),string("cameraId"),cameraId);
	conf.getProperty(string("acquisition"),string("filePath"),filePath);
	
}

int main(int argc, char** argv)
{
	if(argc!=2) {
		cout << "Usage:"<<endl << argv[0] <<" path_to_configuration_file"<<endl ;	
		return -1;
	}
	configure(argv[1]);
	acquisitionModule acq(method , cameraId, filePath);	
	//window w("AcquisitionModule") ;
	faceDetectionModule faces("/home/ziedom/Desktop/SmartVision/faceDetection/haarcascade_frontalface_default.xml");
	// the object acq will be converted to observable (polymorphism) 
	//observerDelegator<Mat,window> deleg(acq,w);
	window z("FaceDetectionModule") ;
	observerDelegator<Mat,faceDetectionModule> deleg1(acq,faces);
	observerDelegator<tuple<Mat,Mat,vector<Rect_<int> > >,window> deleg2(faces,z);	
	
	thread t1(&acquisitionModule::acquire,&acq) ;
	thread t2(&faceDetectionModule::detectFaces,&faces) ;
	// wait for the "t" thread to terminate before terminating the main thread
	t1.join();
	t2.join();
    return 0;
}
