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
	CascadeClassifier face_cascade = loadFaceCascade(cascadePath);
	//ģ��ͼƬ
	vector<Mat> templateImages;
	templates(templateImages);
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

		//// ����ǰ֡�������������
		//processFrame(frame, face_cascade);
		// //������ʾ�����
		//showImage(frame);
		
		//����ʶ��
		string gesture=recognizeGesture(frame, templateImages);
		// ��ͼ������ʾʶ����
		putText(frame, "Gesture: " + gesture, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 255, 255), 2);
		imshow("Gesture Recognition", frame);

		//// ����˳�����
		//if (waitKey(30) >= 0)
		//	break;
		//

		// �� 'q' ���˳�
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
		int x = i * dx + dx / 2, y = Hist(i);
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
 * @brief ��ƽ�� ģ��ƥ��.
 * 
 * \param image
 * \param templ
 */
void templateMatching(Mat image, Mat templ, vector<Rect> &object)
{

	Mat resultMat; // �������
	matchTemplate(image, templ, resultMat, TM_SQDIFF_NORMED); // ģ��ƥ��(��ƽ��ƥ��)
	Point minLoc; // ������Сֵλ�ã����������
	minMaxLoc(resultMat, 0, 0, &minLoc); // ��Сֵλ��
	Rect rect{ minLoc, templ.size() }; //����Rect����
	object.push_back(rect);

	drawRects(image, object);
	namedWindow("ģ��", WINDOW_NORMAL);
	imshow("ģ��", templ);
	namedWindow("ƥ��", WINDOW_NORMAL);
	imshow("ƥ��", image);
}
/**
 * .ģ�����
 * 
 * \param templates
 */
void templates(vector<Mat>& templates)
{
	Mat Rock = imread("../resources/images/ʯͷ.jpg");
	Mat	Paper = imread("../resources/images/��.jpg");
	Mat Scissors= imread("../resources/images/����.jpg");
	if (Rock.empty() || Paper.empty() || Scissors.empty()) {
		cerr << "Error: Could not load one or more template images!" << endl;
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
	Mat gray,binary,resultMat;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	thresBinary(gray, binary, 100);

	string gesture = "Unknown";
	double maxVal = 0;
	for (int i = 0; i < templates.size(); i++) {
		//��ƽ��ƥ��
		matchTemplate(binary, templates[i], resultMat, TM_CCOEFF_NORMED);

		// �ҵ�ͼ���е���С�����Ҷ�ֵ���Լ����ǵ�λ��
		Point minLoc, maxLoc;
		double minVal, maxValTemp ,maxVal=0;
		minMaxLoc(resultMat, &minVal, &maxValTemp, &minLoc, &maxLoc);
		//����Rect����,���ƾ���
		Rect rect{ minLoc, templates[i].size() };
		vector<Rect> object;
		object.push_back(rect);
		drawRects(frame, object);

		if (maxValTemp > maxVal) {
			maxVal = maxValTemp;
			switch (i) {
			case 0: gesture = "Rock"; break; // ʯͷ
			case 1: gesture = "Scissors"; break; // ����
			case 2: gesture = "Paper"; break; // ��
			}
		}
	}
	return gesture;
}

int main(int argc,char* argv[])
{	
	VideoCapture cap(0);
	if (!cap.isOpened()) {
		cerr << "Error opening video stream or file\n";
		return -1;
	}
	cap.release();
	destroyAllWindows();
	//Mat img = imread("../resources/images/cat1.bmp");//ԭͼƬ
	//Mat templ = imread("../resources/images/eyes.png ");
	//if (img.empty()|| templ.empty())return -1;
	///*namedWindow("test", WINDOW_NORMAL);
	//imshow("test", img);*/

	//vector<Mat> templates;
	//for (const auto& templ : templates) {
	//	if (templ.empty()) {
	//		cerr << "Error loading template images!" << endl;
	//		return -1;
	//	}
	//}
	//runFaceDetection();//�������
	// 
	
	// ת��Ϊ�Ҷ�ͼ��
	//Mat gray;
	//cvtColor(img, gray, COLOR_BGR2GRAY);

	//Mat histimg(300, 384, CV_8U);//ֱ��ͼ
	//drawHistline(img, histimg, 128);
	//imshow("Hist",histimg);

	//Mat thresh;//��ֵ��
	//thresBinary(gray,thresh,100);
	//namedWindow("thresh", WINDOW_NORMAL);
	//imshow("thresh", thresh);

	//Mat cannydete;//�߽���
	//CannyDetec(img, cannydete, 100,2);
	//namedWindow("cannydete", WINDOW_NORMAL);
	//imshow("cannydete", cannydete);

	//Contours(thresh);//�������

	//vector<Rect> object;
	//templateMatching(img, templ,object);//ģ����

	
	waitKey(0);//����
	return 0;
}
