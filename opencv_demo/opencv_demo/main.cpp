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
void drawRects(Mat& frame, vector<Rect>& object);
void showImage(const Mat& frame);
void runFaceDetection();

void drawHistline(InputArray src, OutputArray des, int size, const vector<float>& range = {0,256});

void thresBinary(InputArray src, OutputArray des,double value);
void CannyDetec(InputArray src, OutputArray des,double value,double times);
void Contours(InputArray src);
void templateMatching(Mat image, Mat templ, vector<Rect>& object);

void templates(vector<Mat>& templates);
string recognizeGesture(Mat& frame, const vector<Mat>& templates);
void runGestureRecognition();
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
	drawRects(frame, faces);
}


/**
 * @brief 在检测到的人脸区域绘制矩形框.
 * @param Mat& frame图片地址,vector<Rect>& objects人脸数组
 * \return 
 */
void drawRects(Mat& frame, vector<Rect>& objects) {
	for (size_t i = 0; i < objects.size(); i++) {
		rectangle(frame, objects[i], Scalar(255, 0, 0), 2);  // 绘制矩形框
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
	string cascadePath =samples::findFile("D:/Libs/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml",true,true);
	cout << "CascadePath: " << cascadePath << endl;
	CascadeClassifier face_cascade = loadFaceCascade(cascadePath);
	//模板图片
	vector<Mat> templateImages;
	templates(templateImages);
	// 打开摄像头
	VideoCapture cap(0,CAP_DSHOW);
	if (!cap.isOpened()) {
		cerr << "Error opening video stream or file\n";
		return;
	}
	// 读取视频帧并进行处理
	while (true) {
		Mat frame;
		cap >> frame;  // 从摄像头获取一帧图像

		if (frame.empty()) {
			//break;  // 如果没有捕获到图像，退出
			cerr << "Captured empty frame, skipping..." << endl;
			continue; // 跳过该帧
		}
		imshow("video",frame);

		// 处理当前帧：进行人脸检测
		processFrame(frame, face_cascade);
		showImage(frame);
	
		if (waitKey(1) == 'q') {
			break;
		}
	}

	// 释放摄像头资源
	cap.release();
	destroyAllWindows(); 
}
/**
 * .@brief 计算灰度直方图
 * 
 * \param frame
 * \return Mat
 */
Mat calcGrayHist(Mat& frame)
{
	Mat histogram = Mat::zeros(Size(256, 1), CV_32SC1);
	int rows = frame.rows;
	int cols = frame.cols;
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			int index = int(frame.at<uchar>(r, c));
			histogram.at<int>(0, index) += 1;
		}
	}

	return histogram;
}

void drawHistline(InputArray src, OutputArray des, int size, const vector<float>& range )
{
	Mat1f Hist;//直方图
	calcHist(vector<Mat>{src.getMat()}, { 0 }, Mat(), Hist, { size }, range);//计算直方图（图像集，通道，像素，直方图，各维大小，范围）
	Mat& im = des.getMatRef();//结果图像
	normalize(Hist, Hist, im.rows - 1, 0, NORM_INF);//对图像对比度增强，直方图高度与结果图一致
	im.setTo(255);
	float dx = (float)im.cols / size;
	for (int i = 0; i < size; i++) {
		int x = i *(int) dx + (int)dx / 2, y = Hist(i);
		Point pt1{ x,0 }, pt2{ x,y };
		line(im, pt1, pt2, {0});
	}
	flip(im, im, {0});//结果图垂直翻转

}

void thresBinary(InputArray src, OutputArray des,double value)
{
	threshold(src, des, value, 255, THRESH_BINARY);
	
	/*threshold(src, des, 192, 255, THRESH_BINARY_INV);
	adaptiveThreshold(src, des, 255, ADAPTIVE_THRESH_MEAN_C,
		THRESH_BINARY, 3, 5);
	adaptiveThreshold(src, des, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
		THRESH_BINARY, 3, 7.5);*/
}
/**
 * .@brief边缘检测
 * 
 * \param src
 * \param des
 * \param value
 * \param times
 */
void CannyDetec(InputArray src, OutputArray des, double value, double times)
{
	Canny(src, des, value, value * times);
}
/**
 * .
 * @brief轮廓检测
 * \param src
 */
void Contours(InputArray src)
{
	vector<vector<Point>> cont;
	vector<Vec4i> hierarchy; //标记轮廓间的层次关系
	int mode = RETR_CCOMP; // 提取所有轮廓,
	int method = CHAIN_APPROX_SIMPLE; // 压缩水平, 垂直和对角

	// 查找轮廓(源图像, 轮廓线数组, 层次信息, 组织方式, 存储方法)
	findContours(src, cont, hierarchy, mode, method);
	// 绘制轮廓线
	Mat3b  des(src.size()); // 结果图像(彩色图像)
	 des = 0; // 设置背景(黑色)
	Scalar outer(255, 255, 128), inner(128, 255, 255);// 外部轮廓颜色, 内部轮廓颜色

	for (int i = 0; i >= 0; i = hierarchy[i][0])	// 从0开始依次选取下一个外部轮廓
	{ // 绘制外层轮廓线
		drawContours(des, cont, i, outer); // 只绘制轮廓i
		// 绘制i的所有儿子(内层轮廓线)
			// 从首儿子开始依次选取下一个儿子
			for (int j = hierarchy[i][2]; j >= 0; j = hierarchy[j][0])
				drawContours(des, cont, j, inner); // 只绘制轮廓j
	}
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", des); 
	// 填充轮廓
	des= 0; // 重新绘制
	drawContours(des, cont, -1, outer, FILLED); // 所有轮廓, 填充
	namedWindow("Fill", WINDOW_NORMAL);
	imshow("Fill", des);

}
/**
 * @brief 差平方 模板匹配.
 * 
 * \param image
 * \param templ
 */
void templateMatching(Mat image, Mat templ, vector<Rect> &object)
{
	if (image.empty() || templ.empty()) {
		cerr << "Error: Input image or template is empty!" << endl;
		return; // 检查输入是否为空
	}
	Mat grayImage, grayTemplate, resultMat;
	if (image.channels() == 3) {
		cvtColor(image, grayImage, COLOR_BGR2GRAY);
	}
	else {
		grayImage = image.clone();
	}
	if (templ.channels() == 3) {
		cvtColor(templ, grayTemplate, COLOR_BGR2GRAY);
	}
	else {
		grayTemplate = templ.clone();
	}
	if (grayImage.rows < grayTemplate.rows || grayImage.cols < grayTemplate.cols) {
		cerr << "Error: Template size is larger than input image size!" << endl;
		return;
	}
	resultMat.create(grayImage.rows - grayTemplate.rows + 1, grayImage.cols - grayTemplate.cols + 1, CV_32FC1);

	matchTemplate(grayImage, grayTemplate, resultMat, TM_SQDIFF_NORMED); // 模板匹配(差平方匹配)
	Point minLoc; // 保存最小值位置，即矩形起点
	double minVal;
	minMaxLoc(resultMat, &minVal, nullptr, &minLoc, nullptr); // 最小值位置
	Rect rect{ minLoc, grayTemplate.size() }; //创建Rect对象
	object.push_back(rect);

	drawRects(image, object);
	namedWindow("模板", WINDOW_NORMAL);
	imshow("模板", templ);
	namedWindow("匹配", WINDOW_NORMAL);
	imshow("匹配", image);
}
/**
 * .模板加载
 * 
 * \param templates
 */
void templates(vector<Mat>& templates)
{
	Mat Rock = imread("../resources/images/石头.jpg");
	Mat	Paper = imread("../resources/images/布.jpg");
	Mat Scissors= imread("../resources/images/剪刀.jpg");
	if (Rock.empty() || Paper.empty() || Scissors.empty()) {
		cerr << "Error: Could not load one or more template images!" << endl;
		return;
	}
	templates.push_back(Rock);
	templates.push_back(Paper);
	templates.push_back(Scissors);
}
/**
 * .手势识别，多模板匹配
 * 
 * \param frame
 * \param templates
 * \return 
 */
string recognizeGesture(Mat& frame, const vector<Mat>& templates)
{
	Mat gray, frameBinary,resultMat;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	thresBinary(gray, frameBinary, 100);

	string gesture = "Unknown";
	double maxVal = 0;// 记录最高匹配度
	vector<Rect> objects; // 用于存放匹配到的矩形

	for (int i = 0; i < templates.size(); i++) {
		Mat grayTempl, binaryTempl;
		cvtColor(templates[i], grayTempl, COLOR_BGR2GRAY);
		thresBinary(grayTempl, binaryTempl, 100);
		//差平方匹配
		templateMatching(frameBinary, binaryTempl, objects);

		if (!objects.empty()) {
			double currentVal = 1.0 - (double)(norm(objects[0].area()) / (frameBinary.rows * frameBinary.cols)); // 计算当前匹配度
			if (currentVal > maxVal) {
				maxVal = currentVal;
				switch (i) {
				case 0: gesture = "Rock"; break; // 石头
				case 1: gesture = "Scissors"; break; // 剪刀
				case 2: gesture = "Paper"; break; // 布
				}
			}
		}
	}
	return gesture;
}

void runGestureRecognition()
{
	// 加载手势模板
	vector<Mat> templateImages;
	templates(templateImages);

	// 打开摄像头
	VideoCapture cap(0, cv::CAP_DSHOW);
	if (!cap.isOpened()) {
		cerr << "Error opening video stream or file" << endl;
		return;
	}

	while (true) {
		Mat frame;
		cap >> frame;
		if (frame.empty()) {
			cerr << "Captured an empty frame" << endl;
			break; // 若捕获到空帧则退出
		}
		// 识别手势
		string gesture = recognizeGesture(frame, templateImages);
		// 在图像上显示识别结果
		putText(frame, "Gesture: " + gesture, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
		imshow("Gesture Recognition", frame);
		if (waitKey(1) == 'q') {
			break; // 按 'q' 键退出
		}
	}
	cap.release();
	destroyAllWindows();
}

int main(int argc,char* argv[])
{	
	

	//Mat img = imread("../resources/images/cat1.bmp");//原图片
	//Mat templ = imread("../resources/images/eyes.png ");
	//if (img.empty()|| templ.empty())return -1;
	/*namedWindow("test", WINDOW_NORMAL);
	imshow("test", img);*/

	

	//runFaceDetection();//人脸检测
	// 转换为灰度图像
	//Mat gray;
	//cvtColor(img, gray, COLOR_BGR2GRAY);

	//Mat histimg(300, 384, CV_8U);//直方图
	//drawHistline(img, histimg, 128);
	//imshow("Hist",histimg);

	//Mat thresh;//二值化
	//thresBinary(gray,thresh,100);
	//namedWindow("thresh", WINDOW_NORMAL);
	//imshow("thresh", thresh);

	//Mat cannydete;//边界检测
	//CannyDetec(img, cannydete, 100,2);
	//namedWindow("cannydete", WINDOW_NORMAL);
	//imshow("cannydete", cannydete);

	//Contours(thresh);//轮廓检测

	//vector<Rect> object;
	//templateMatching(img, templ,object);//模板检测
	
	//runGestureRecognition(); // 启动手势识别程序
	//waitKey(0);//阻塞
	//return 0;

	VideoCapture cap(0, CAP_DSHOW); // 打开默认摄像头
	if (!cap.isOpened()) {
		std::cerr << "Error: Unable to access camera." << std::endl;
		return -1;
	}

	Mat frame;
	while (true) {
		cap >> frame; // 捕获帧
		if (frame.empty()) {
			std::cerr << "Error: Captured empty frame." << std::endl;
			break;
		}
		imshow("Camera", frame); // 显示捕获的帧
		if (waitKey(30) >= 0) break; // 按任意键退出
	}

	cap.release(); // 释放摄像头
	destroyAllWindows();
	return 0;
}
