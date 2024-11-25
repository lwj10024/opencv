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
// ��������
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
	drawRects(frame, faces);
}


/**
 * @brief �ڼ�⵽������������ƾ��ο�.
 * @param Mat& frameͼƬ��ַ,vector<Rect>& objects��������
 * \return 
 */
void drawRects(Mat& frame, vector<Rect>& objects) {
	for (size_t i = 0; i < objects.size(); i++) {
		rectangle(frame, objects[i], Scalar(255, 0, 0), 2);  // ���ƾ��ο�
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
	string cascadePath =samples::findFile("D:/Libs/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml",true,true);
	cout << "CascadePath: " << cascadePath << endl;
	CascadeClassifier face_cascade = loadFaceCascade(cascadePath);
	//ģ��ͼƬ
	vector<Mat> templateImages;
	templatesLoad(templateImages);
	// ������ͷ
	VideoCapture cap(0,CAP_DSHOW);
	if (!cap.isOpened()) {
		cerr << "Error opening video stream or file\n";
		return;
	}
	// ��ȡ��Ƶ֡�����д���
	while (true) {
		Mat frame;
		cap >> frame;  // ������ͷ��ȡһ֡ͼ��

		if (frame.empty()) {
			//break;  // ���û�в���ͼ���˳�
			cerr << "Captured empty frame, skipping..." << endl;
			continue; // ������֡
		}
		imshow("video",frame);

		// ����ǰ֡�������������
		processFrame(frame, face_cascade);
		showImage(frame);
	
		if (waitKey(1) == 'q') {
			break;
		}
	}

	// �ͷ�����ͷ��Դ
	cap.release();
	destroyAllWindows(); 
}
/**
 * .@brief ����Ҷ�ֱ��ͼ
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
	Mat1f Hist;//ֱ��ͼ
	calcHist(vector<Mat>{src.getMat()}, { 0 }, Mat(), Hist, { size }, range);//����ֱ��ͼ��ͼ�񼯣�ͨ�������أ�ֱ��ͼ����ά��С����Χ��
	Mat& im = des.getMatRef();//���ͼ��
	normalize(Hist, Hist, im.rows - 1, 0, NORM_INF);//��ͼ��Աȶ���ǿ��ֱ��ͼ�߶�����ͼһ��
	im.setTo(255);
	float dx = (float)im.cols / size;
	for (int i = 0; i < size; i++) {
		int x = i *(int) dx + (int)dx / 2, y = Hist(i);
		Point pt1{ x,0 }, pt2{ x,y };
		line(im, pt1, pt2, {0});
	}
	flip(im, im, {0});//���ͼ��ֱ��ת

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
 * .@brief��Ե���
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
 * @brief�������
 * \param src
 */
void Contours(InputArray src)
{
	vector<vector<Point>> cont;
	vector<Vec4i> hierarchy; //���������Ĳ�ι�ϵ
	int mode = RETR_CCOMP; // ��ȡ��������,
	int method = CHAIN_APPROX_SIMPLE; // ѹ��ˮƽ, ��ֱ�ͶԽ�

	// ��������(Դͼ��, ����������, �����Ϣ, ��֯��ʽ, �洢����)
	findContours(src, cont, hierarchy, mode, method);
	// ����������
	Mat3b  des(src.size()); // ���ͼ��(��ɫͼ��)
	 des = 0; // ���ñ���(��ɫ)
	Scalar outer(255, 255, 128), inner(128, 255, 255);// �ⲿ������ɫ, �ڲ�������ɫ

	for (int i = 0; i >= 0; i = hierarchy[i][0])	// ��0��ʼ����ѡȡ��һ���ⲿ����
	{ // �������������
		drawContours(des, cont, i, outer); // ֻ��������i
		// ����i�����ж���(�ڲ�������)
			// ���׶��ӿ�ʼ����ѡȡ��һ������
			for (int j = hierarchy[i][2]; j >= 0; j = hierarchy[j][0])
				drawContours(des, cont, j, inner); // ֻ��������j
	}
	namedWindow("Contours", WINDOW_NORMAL);
	imshow("Contours", des); 
	// �������
	des= 0; // ���»���
	drawContours(des, cont, -1, outer, FILLED); // ��������, ���
	namedWindow("Fill", WINDOW_NORMAL);
	imshow("Fill", des);

}

/**
 * .ģ�����
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
 * .����ʶ�𣬶�ģ��ƥ��
 * 
 * \param frame
 * \param templates
 * \return 
 */
string recognizeGesture(Mat& frame, const vector<Mat>& templates)
{
	double maxVal = 0;// ��¼���ƥ���
	vector<Rect> objects; // ���ڴ��ƥ�䵽�ľ���
	
	// ת��Ϊ�Ҷ�ͼ�����ƥ��
	Mat grayFrame;
	cvtColor(frame, grayFrame, COLOR_BGR2GRAY);

	for (int i = 0; i < templates.size(); i++) {
		Mat result;
		//ƥ��
		Mat grayTemplate;
		cvtColor(templates[i], grayTemplate, COLOR_BGR2GRAY);

		matchTemplate(grayFrame, grayTemplate, result, TM_CCOEFF_NORMED);
		double minVal, maxValLocal;
		Point minLoc, maxLoc;
		minMaxLoc(result, &minVal, &maxValLocal, &minLoc, &maxLoc);
		cout << "Match result for gesture " << gestures[i] << ": " << maxValLocal << endl; // �������ƥ��ֵ
		
		if (maxValLocal > maxVal && maxValLocal > 0.4) {
				maxVal = maxValLocal;
				switch (i) {
				case 0: playerGesture = "Rock"; break; // ʯͷ
				case 1: playerGesture = "Scissors"; break; // ����
				case 2: playerGesture = "Paper"; break; // ��
				}		
				rectangle(frame, maxLoc, Point(maxLoc.x + templates[i].cols, maxLoc.y + templates[i].rows), Scalar(0, 255, 0), 2);
				
		}
	}
	cout << "playerGesture:" << playerGesture << endl;
	return playerGesture;
}

void runGestureRecognition()
{
	// ��������ģ��
	vector<Mat> templateImages;
	templatesLoad(templateImages);

	// ������ͷ
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
			break; // �����񵽿�֡���˳�
		}
		
		// ��ͼ������ʾʶ����		
		playGame(frame, templateImages);
		imshow("Gesture Recognition", frame);
		char key = (char)waitKey(1);
		if (key == 32) {  // �ո����ʼ��Ϸ
			startGame = true;
			stateResult = false;
			initialTime = getTickCount();
			playerGesture = "";
		}
		else if (key == 27) {  // ESC ���˳�
			break;
		}
	}
	cap.release();
	destroyAllWindows();
}

// AI �����ѡ������
int aiAct() {
	//α�����������
	random_device rd;
	unsigned int seed = rd();//����
	mt19937 gen(seed);// ʹ�����������ֵ����һ��α�����������
	//������ֲ���
	uniform_int_distribution<> distrib(0, 2);

	int aiMove = distrib(gen);
	return aiMove;
}
void playGame(Mat& frame, const vector<Mat>& templates)
{
	//ǰ
	if (startGame) {
		if (!stateResult) {
			double timer = (getTickCount() - initialTime) / getTickFrequency();//����3-2-1
			putText(frame, "Time: " + to_string(int(3-timer)), Point(50, 50), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 255), 2);
	
			if (timer > 3) {
				stateResult = true;
				// ʶ������
				playerGesture = recognizeGesture(frame, templates);
				int aiMove = aiAct();
				imgAI = templates[aiMove];
				if (aiMove == 0) {
					cout << "AI:ʯͷ"<< aiMove << endl;
				}
				else if (aiMove == 1) {
					cout << "AI:��" << aiMove << endl;
				}
				else  if (aiMove == 2) {
					cout << "AI:����" << aiMove << endl;
				}

				if (!playerGesture.empty()) {
				
					//����
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
	// ��ʾ AI ����
	if (stateResult && !imgAI.empty()) {
		Mat resizedImgAI;
		resize(imgAI, resizedImgAI, Size(100, 100));
		resizedImgAI.copyTo(frame(Rect(300, 50, resizedImgAI.cols, resizedImgAI.rows)));
	}
	// ��ʾ����
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
	runGestureRecognition(); // ��������ʶ�����
	waitKey(0);//����
	return 0;
}
