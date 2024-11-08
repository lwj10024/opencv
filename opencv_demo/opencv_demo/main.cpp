#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// ��������
CascadeClassifier loadFaceCascade(const string& cascadePath);
void processFrame(Mat& frame, CascadeClassifier& face_cascade);
void drawFaces(Mat& frame, vector<Rect>& faces);
void showImage(const Mat& frame);
void runFaceDetection();

/**
 * @brief. �����������ķ�����
 * @param const string&  cascadePath ��������·��
 * 
 * \return 
 */
CascadeClassifier loadFaceCascade(const string& cascadePath) {
	CascadeClassifier face_cascade;
	if (!face_cascade.load(cascadePath)) {
		cerr << "Error loading face cascade\n";
		exit(-1);  // ����ʧ�����˳�
	}
	return face_cascade;
}
/**
 * .@brief ��ÿһ֡ͼ������������
 * @param Mat &farmeͼƬ��ַ
 * @param face_cascade������������
 * 
 * \return
 */
void processFrame(Mat& frame, CascadeClassifier& face_cascade) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, gray);
	// �������
	vector<Rect> faces;
	face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
	std::cout << "Type of frame: " << typeid(frame).name() << std::endl;
	std::cout << "Type of faces: " << typeid(faces).name() << std::endl;

	// ���ƾ��ο�
	drawFaces(frame, faces);

}


/**
 * @brief �ڼ�⵽������������ƾ��ο�.
 * @param Mat& frameͼƬ��ַ,vector<Rect>& faces��������
 * \return 
 */
void drawFaces(Mat& frame, vector<Rect>& faces) {
	for (size_t i = 0; i < faces.size(); i++) {
		rectangle(frame, faces[i], Scalar(255, 0, 0), 2);  // ���ƾ��ο�
	}

}
/**
 * @brief.// ��ʾͼ��
 * 
 * \return 
 */
void showImage(const Mat& frame) {
	imshow("Face Detection", frame);
}
/**
 * @brief.�����������
 * 
 * \return 
 */
void runFaceDetection() {
	string cascadePath =samples::findFile("D:/Libs/opencv-4.10.0/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml",true,true);
	CascadeClassifier face_cascade = loadFaceCascade(cascadePath);
	
	   // ������ͷ
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Error opening video stream or file\n";
		return;
	}
	// ��ȡ��Ƶ֡�����д���
	while (true) {
		Mat frame;
		cap >> frame;  // ������ͷ��ȡһ֡ͼ��

		if (frame.empty()) {
			break;  // ���û�в���ͼ���˳�
		}

		// ����ǰ֡�������������
		processFrame(frame, face_cascade);

		// ��ʾ�����
		showImage(frame);

		// �� 'q' ���˳�
		if (waitKey(1) == 'q') {
			break;
		}
	}

	// �ͷ�����ͷ��Դ
	cap.release();
	destroyAllWindows();
}

int main()
{
	
	/*Mat img = imread("../resources/images/cat1.bmp");
	namedWindow("test", WINDOW_NORMAL);
	imshow("test", img);*/
	runFaceDetection();
	//waitKey(0);
	return 0;
}
