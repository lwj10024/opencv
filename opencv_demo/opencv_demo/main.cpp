#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <random>

using namespace cv;
using namespace std;

bool startGame = false;
bool stateResult = false;
double initialTime = 0;
string playerGesture = "";
Mat imgAI;
vector<int> scores = { 0, 0 }; // [AI, Player]
vector<string> gestures = { "Rock", "Scissors", "Paper" };
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

void templatesLoad(vector<Mat>& templates);
string recognizeGesture(Mat& frame, const vector<Mat>& templates);
void runGestureRecognition();

int aiAct();
void playGame(Mat& frame, const vector<Mat>& templates);

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
	templatesLoad(templateImages);
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
 * .模板加载
 * 
 * \param templates
 */
void templatesLoad(vector<Mat>& templates)
{
	Mat Rock = imread("../resources/images/Rock.png");
	Mat	Paper = imread("../resources/images/Paper.png");
	Mat Scissors= imread("../resources/images/Scissors.png");
	if (Rock.empty() || Paper.empty() || Scissors.empty()) {
		cout << "Error: Could not load one or more template images!" << endl;
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
	double maxVal = 0;// 记录最高匹配度
	vector<Rect> objects; // 用于存放匹配到的矩形
	
	// 转换为灰度图像进行匹配
	Mat grayFrame;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

	for (int i = 0; i < templates.size(); i++) {
		Mat result;
		//匹配
		Mat grayTemplate;
		cvtColor(templates[i], grayTemplate, COLOR_BGR2GRAY);

		matchTemplate(grayFrame, grayTemplate, result, TM_CCOEFF_NORMED);
		double minVal, maxValLocal;
		Point minLoc, maxLoc;
		minMaxLoc(result, &minVal, &maxValLocal, &minLoc, &maxLoc);
		cout << "Match result for gesture " << gestures[i] << ": " << maxValLocal << endl; // 调试输出匹配值
		
		if (maxValLocal > maxVal && maxValLocal > 0.4) {
				maxVal = maxValLocal;
				switch (i) {
				case 0: playerGesture = "Rock"; break; // 石头
				case 1: playerGesture = "Scissors"; break; // 剪刀
				case 2: playerGesture = "Paper"; break; // 布
				}		
				rectangle(frame, maxLoc, Point(maxLoc.x + templates[i].cols, maxLoc.y + templates[i].rows), Scalar(0, 255, 0), 2);
				
		}
	}
	cout << "playerGesture:" << playerGesture << endl;
	return playerGesture;
}

void runGestureRecognition()
{
	// 加载手势模板
	vector<Mat> templateImages;
	templatesLoad(templateImages);

	// 打开摄像头
	VideoCapture cap(1, cv::CAP_DSHOW);
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
		
		// 在图像上显示识别结果		
		playGame(frame, templateImages);
		imshow("Gesture Recognition", frame);
		char key = (char)waitKey(1);
		if (key == 32) {  // 空格键开始游戏
			startGame = true;
			stateResult = false;
			initialTime = getTickCount();
			playerGesture = "";
		}
		else if (key == 27) {  // ESC 键退出
			break;
		}
	}
	cap.release();
	destroyAllWindows();
}

// AI 随机数选择手势
int aiAct() {
	//伪随机数生成器
	random_device rd;
	unsigned int seed = rd();//种子
	mt19937 gen(seed);// 使用随机的种子值创建一个伪随机数生成器
	//随机数分布类
	uniform_int_distribution<> distrib(0, 2);

	int aiMove = distrib(gen);
	return aiMove;
}
void playGame(Mat& frame, const vector<Mat>& templates)
{
	//前
	if (startGame) {
		if (!stateResult) {
			double timer = (getTickCount() - initialTime) / getTickFrequency();//数数3-2-1
			putText(frame, "Time: " + to_string(int(3-timer)), Point(50, 50), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
	
			if (timer > 3) {
				stateResult = true;
				// 识别手势
				playerGesture = recognizeGesture(frame, templates);
				int aiMove = aiAct();
				imgAI = templates[aiMove];
				if (aiMove == 0) {
					cout << "AI:石头"<< aiMove << endl;
				}
				else if (aiMove == 1) {
					cout << "AI:布" << aiMove << endl;
				}
				else  if (aiMove == 2) {
					cout << "AI:剪刀" << aiMove << endl;
				}

				if (!playerGesture.empty()) {
				
					//规则
					if ((playerGesture == "Rock" && aiMove == 2) || (playerGesture == "Scissors" && aiMove == 1) || (playerGesture == "Paper" && aiMove == 0)) {
						scores[1]++; // Player wins
					}
					else if ((playerGesture == "Rock" && aiMove == 1) || (playerGesture == "Scissors" && aiMove == 0) || (playerGesture == "Paper" && aiMove == 2)) {
						scores[0]++; // AI wins
					}
				}
			}
		}
	}
	// 显示 AI 手势
	if (stateResult && !imgAI.empty()) {
		Mat resizedImgAI;
		resize(imgAI, resizedImgAI, Size(100, 100));
		resizedImgAI.copyTo(frame(Rect(300, 50, resizedImgAI.cols, resizedImgAI.rows)));
	}
	// 显示分数
	putText(frame, "Player: " + to_string(scores[1]), Point(50, 100), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
	putText(frame, "AI: " + to_string(scores[0]), Point(50, 150), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
	if (stateResult && !playerGesture.empty()) {
		putText(frame, "Player Gesture: " + playerGesture, Point(50, 200), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 2);
	}

	if (stateResult) {
		if (scores[1] > scores[0]) {
			putText(frame, "Player Wins!", Point(50, 250), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 2);
		}
		else if (scores[0] > scores[1]) {
			putText(frame, "AI Wins!", Point(50, 250), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 2);
		}
		else {
			putText(frame, "It's a Tie!", Point(50, 250), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2);
		}
	}
}
int main(int argc,char* argv[])
{	
	runGestureRecognition(); // 启动手势识别程序
	waitKey(0);//阻塞
	return 0;
}
