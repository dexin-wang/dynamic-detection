#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include "iostream"
#include "func.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

String videoFile = "./woman.mp4";
/*�Ż���Ŀ����ɢ�鲢*/
int main(int a, uchar** b) {
	Func func;
	func.attention();
	func.readVideo(videoFile);		//����Ƶ

	int num = 0;
	double timer[8];	//��ʱ
	double time[7];	//��ʱ

	while (true) {
		num++;
		printf("��%d֡\n", num);

		timer[0] = static_cast<double>(getTickCount());

		func.readFrame();			//��frame
		timer[1] = static_cast<double>(getTickCount());

		func.surfDetect();			//surf���
		timer[2] = static_cast<double>(getTickCount());

		func.BfMatch();				//BBF���Գ�Լ����������ɸѡ
		timer[3] = static_cast<double>(getTickCount());

		func.filtForePts();			//�˳�ǰ����
		timer[4] = static_cast<double>(getTickCount());

		func.makeBGD();				//��������
		timer[5] = static_cast<double>(getTickCount());

		func.getFGD();				//ǰ����ȡ��֡�
		timer[6] = static_cast<double>(getTickCount());

		func.drawContours();		//���ƶ�̬Ŀ��������ʶ��̬Ŀ��
		timer[7] = static_cast<double>(getTickCount());

		for (int i = 0; i < 7; i++)
		{
			time[i] = (timer[i + 1] - timer[i]) * 1000.0 / getTickFrequency();
		}
		printf("surf���: %.1f, ������ɸѡ: %.1f, �˳�ǰ����: %.1f, ��������: %.1f, ǰ����ȡ(֡��): %.1f, ���ƶ�̬Ŀ������: %.1f \n",
			time[1], time[2], time[3], time[4], time[5], time[6]);

		func.showAndUpdate();
		if (!func.readKey())
			break;

		//if (num == 19)
			//cv::waitKey(0);
	}

	//func.capture.release();
	cv::waitKey(0);
	return 0;
}

