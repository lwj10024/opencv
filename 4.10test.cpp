#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int main()
{
	//����Ҫ��ӳ������ͼƬ�ļ�������Ŀ��ǰĿ¼��
	Mat img = imread("img\\cat1.bmp");
	namedWindow("test", WINDOW_NORMAL);
	imshow("test", img);
	waitKey(0);
	return 0;
}
