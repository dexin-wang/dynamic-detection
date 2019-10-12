#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include "iostream"
#include "func.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

String videoFile = "./woman.mp4";
/*优化了目标离散归并*/
int main(int a, uchar** b) {
	Func func;
	func.attention();
	func.readVideo(videoFile);		//读视频

	int num = 0;
	double timer[8];	//计时
	double time[7];	//耗时

	while (true) {
		num++;
		printf("第%d帧\n", num);

		timer[0] = static_cast<double>(getTickCount());

		func.readFrame();			//读frame
		timer[1] = static_cast<double>(getTickCount());

		func.surfDetect();			//surf检测
		timer[2] = static_cast<double>(getTickCount());

		func.BfMatch();				//BBF、对称约束，特征点筛选
		timer[3] = static_cast<double>(getTickCount());

		func.filtForePts();			//滤除前景点
		timer[4] = static_cast<double>(getTickCount());

		func.makeBGD();				//背景补偿
		timer[5] = static_cast<double>(getTickCount());

		func.getFGD();				//前景提取（帧差）
		timer[6] = static_cast<double>(getTickCount());

		func.drawContours();		//绘制动态目标轮廓、识别动态目标
		timer[7] = static_cast<double>(getTickCount());

		for (int i = 0; i < 7; i++)
		{
			time[i] = (timer[i + 1] - timer[i]) * 1000.0 / getTickFrequency();
		}
		printf("surf检测: %.1f, 特征点筛选: %.1f, 滤除前景点: %.1f, 背景补偿: %.1f, 前景提取(帧差): %.1f, 绘制动态目标轮廓: %.1f \n",
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

