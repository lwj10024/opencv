#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


// 函数声明
CascadeClassifier loadFaceCascade(const string& cascadePath);
void processFrame(Mat& frame, CascadeClassifier& face_cascade);
void drawFaces(Mat& frame, vector<Rect>& faces);
void showImage(const Mat& frame);
void runFaceDetection();

/**
 * @brief. 加载人脸检测的分类器
 * @param const string&  cascadePath 分类器的路径
 * 
 * \return 
 */
CascadeClassifier loadFaceCascade(const string& cascadePath) {
	CascadeClassifier face_cascade;
	if (!face_cascade.load(cascadePath)) {
		cerr << "Error loading face cascade\n";
		exit(-1);  // 加载失败则退出
	}
	return face_cascade;
}
/**
 * .@brief 对每一帧图像进行人脸检测
 * @param Mat &farme图片地址
 * @param face_cascade人脸检测分类器
 * 
 * \return
 */
void processFrame(Mat& frame, CascadeClassifier& face_cascade) {
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	equalizeHist(gray, gray);
	// 检测人脸
	vector<Rect> faces;
	face_cascade.detectMultiScale(gray, faces, 1.1, 3, 0, Size(30, 30));
	std::cout << "Type of frame: " << typeid(frame).name() << std::endl;
	std::cout << "Type of faces: " << typeid(faces).name() << std::endl;

	// 绘制矩形框
	drawFaces(frame, faces);

}


/**
 * @brief 在检测到的人脸区域绘制矩形框.
 * @param Mat& frame图片地址,vector<Rect>& faces人脸数组
 * \return 
 */
void drawFaces(Mat& frame, vector<Rect>& faces) {
	for (size_t i = 0; i < faces.size(); i++) {
		rectangle(frame, faces[i], Scalar(255, 0, 0), 2);  // 绘制矩形框
	}

}
/**
 * @brief.// 显示图像
 * 
 * \return 
 */
void showImage(const Mat& frame) {
	imshow("Face Detection", frame);
}
/**
 * @brief.运行人脸检测
 * 
 * \return 
 */
void runFaceDetection() {
	string cascadePath =samples::findFile("D:/Libs/opencv-4.10.0/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml",true,true);
	CascadeClassifier face_cascade = loadFaceCascade(cascadePath);
	
	   // 打开摄像头
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Error opening video stream or file\n";
		return;
	}
	// 读取视频帧并进行处理
	while (true) {
		Mat frame;
		cap >> frame;  // 从摄像头获取一帧图像

		if (frame.empty()) {
			break;  // 如果没有捕获到图像，退出
		}

		// 处理当前帧：进行人脸检测
		processFrame(frame, face_cascade);

		// 显示检测结果
		showImage(frame);

		// 按 'q' 键退出
		if (waitKey(1) == 'q') {
			break;
		}
	}

	// 释放摄像头资源
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
