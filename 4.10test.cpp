#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int main()
{
	//����Ҫ��ӳ������ͼƬ�ļ�������Ŀ��ǰĿ¼��
	Mat img = imread("C:\\Users\\LAPTOP2021\\Desktop\\1.png");
	namedWindow("test");
	imshow("test", img);
	waitKey(0);
	return 0;
}
