#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include "iostream"
#include "func.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

#define pi 3.1415926


//使用说明
void Func::attention()
{
	printf("[0]表示前一帧，[1]表示当前帧\n\n");
}
//读视频文件
void Func::readVideo(String &videoFile)
{
	capture.open(videoFile);
	//capture.open(0);
	if (!capture.isOpened())
	{
		cout << "Could not initialize capturing...\n";
		system("pause");
	}
	width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

	surfDetector = SURF::create(1000);
}

//读frame
void Func::readFrame()
{
	capture.read(cur_frame);							//读当前帧
	cvtColor(cur_frame, cur_gray, COLOR_BGR2GRAY);
	if (prev_gray.empty()) {
		cur_gray.copyTo(prev_gray);						//更新前一帧
		cur_frame.copyTo(prev_frame);					//更新前一帧
	}
}

//SURF特征检测
void Func::surfDetect()
{
	Pts[0].clear(); Pts[1].clear();
	desc[0].release(); desc[1].release();
	surfDetector->detectAndCompute(prev_gray, Mat(), Pts[0], desc[0]);
	surfDetector->detectAndCompute(cur_gray, Mat(), Pts[1], desc[1]);
	//drawKeypoints(cur_frame, Pts[1], cur_frame, Scalar(0, 0, 255));

	//printf("SURF检测点数：%d、%d \n", Pts[0].size(), Pts[1].size());
}

//KNN匹配
void Func::BfMatch()
{
	knnMatches[0].clear();
	matcher.knnMatch(desc[0], desc[1], knnMatches[0], 2);		//正向匹配
	printf("KNN匹配点对数：%d \n", knnMatches[0].size());
	//BBF搜索算法
	//BBFmatches.clear();
	BBFpts[0].clear(); BBFpts[1].clear();
	for (size_t i = 0; i < knnMatches[0].size(); i++) {
		const DMatch& bestMatch = knnMatches[0][i][0];			//汉明距离最小
		const DMatch& betterMatch = knnMatches[0][i][1];		//汉明距离次小

		float  distanceRatio = bestMatch.distance / betterMatch.distance;
		//利用汉明距离粗略过滤匹配点对
		if (distanceRatio < minRatio) {
			//BBFmatches.push_back(bestMatch);
			BBFpts[0].push_back(Pts[0][bestMatch.queryIdx].pt);
			BBFpts[1].push_back(Pts[1][bestMatch.trainIdx].pt);
		}
	}
	printf("对称约束前：BBFpts[0].size = %d \n", BBFpts[0].size());

	knnMatches[1].clear();
	matcher.knnMatch(desc[1], desc[0], knnMatches[1], 2);		//反向匹配

	BBFpts_rev[0].clear();
	BBFpts_rev[1].clear();
	for (size_t i = 0; i < knnMatches[1].size(); i++) {
		const DMatch& bestMatch_rev = knnMatches[1][i][0];			//汉明距离最小
		const DMatch& betterMatch_rev = knnMatches[1][i][1];		//汉明距离次小

		float distanceRatio = bestMatch_rev.distance / betterMatch_rev.distance;
		//利用汉明距离粗略过滤匹配点对
		if (distanceRatio < minRatio) {
			//BBFmatches.push_back(bestMatch);
			BBFpts_rev[0].push_back(Pts[0][bestMatch_rev.trainIdx].pt);
			BBFpts_rev[1].push_back(Pts[1][bestMatch_rev.queryIdx].pt);
		}
	}
	//printf("对称约束前：BBFpts_rev[0].size = %d, BBFpts_rev[1].size = %d \n", BBFpts_rev[0].size(), BBFpts_rev[1].size());

	/*printf("---------------------------------------------------------\n");
	for (size_t i = 0; i < BBFpts[0].size(); i++)
	{
		printf("(%d, %d) (%d, %d)\n", BBFpts[0][i].x, BBFpts[0][i].y, BBFpts[1][i].x, BBFpts[1][i].y);
	}
	printf("~~~~~~~~~~~~~~\n");
	for (size_t j = 0; j < BBFpts_rev[0].size(); j++)
	{
		printf("(%d, %d) (%d, %d)\n", BBFpts_rev[0][j].x, BBFpts_rev[0][j].y, BBFpts_rev[1][j].x, BBFpts_rev[1][j].y);
	}
	printf("---------------------------------------------------------\n");*/
	//对称约束
	pts_opti[0].clear(); pts_opti[1].clear();
	for (size_t i = 0; i < BBFpts[0].size(); i++)
	{
		for (size_t j = 0; j < BBFpts_rev[0].size(); j++)
		{
			//满足对称约束，则跳过
			if ((BBFpts[0][i].x == BBFpts_rev[0][j].x) && (BBFpts[0][i].y == BBFpts_rev[0][j].y)) {

				if ((BBFpts[1][i].x == BBFpts_rev[1][j].x) && (BBFpts[1][i].y == BBFpts_rev[1][j].y)) {

					//位于四角且边界向内10像素范围内的点保留 
					if (BBFpts[1][i].x >= 20 && BBFpts[1][i].x <= (width - 20) && BBFpts[1][i].y >= 20 && BBFpts[1][i].y <= (height - 20)) {

						pts_opti[0].push_back(Point2f(BBFpts[0][i].x, BBFpts[0][i].y));
						pts_opti[1].push_back(Point2f(BBFpts[1][i].x, BBFpts[1][i].y));
					}
				}
			}
		}
	}
	printf("对称约束后：pts_opti[1].size = %d \n", pts_opti[1].size());
	/*for (size_t i = 0; i < pts_opti[1].size(); i++)
	{
		printf("(%d, %d) \n", pts_opti[1][i].x, pts_opti[1][i].y);
	}
	*/
}

/***********************  待优化  **********************/
//自适应外点滤除
void Func::filtForePts()
{
	//获取边角处的特征点
	getCornerPts(0.2, 0.2);
	if (pts_corner[1].size() < 10) {
		getCornerPts(0.3, 0.3);
	}
	printf("四角的点数：%d \n", pts_corner[1].size());
	
	//自适应外点滤除
	pts_BGD[0].clear(); pts_BGD[1].clear();
	pts_FGD[0].clear(); pts_FGD[1].clear();
	if (pts_corner[1].size() < 10) {	//不做外点滤除
		if (pts_corner[1].size() <= 3) {
			for (size_t i = 0; i < pts_opti[1].size(); i++)
			{
				pts_BGD[0].push_back(pts_opti[0][i]);
				pts_BGD[1].push_back(pts_opti[1][i]);
			}
		}
		else {
			for (size_t i = 0; i < pts_corner[1].size(); i++)
			{
				pts_BGD[0].push_back(pts_corner[0][i]);
				pts_BGD[1].push_back(pts_corner[1][i]);
			}
		}
	}
	else {	//外点滤除
		H = myGetAffineTransform(pts_corner[0], pts_corner[1]);	//计算初始 H
		cout << "初始 H：" << endl; cout << H << endl;

		calcPtsErrs(); //计算距离残差

		int errN = ptsErrs.size();
		//printf("errN: %d \n", errN);

		//迭代min max 距离残差
		float minPtsErr = 100.0, maxPtsErr = 0.0;
		for (size_t i = 0; i < ptsErrs.size(); i++)
		{
			float ptsErr = ptsErrs[i];
			if (ptsErr > maxPtsErr)
				maxPtsErr = ptsErr;
			if (ptsErr < minPtsErr)
				minPtsErr = ptsErr;
		}
		//printf("minPtsErr：%.2f, maxPtsErr:%.2f \n", minPtsErr, maxPtsErr);

		//ptsErrs_num分配空间
		ptsErrs_num.clear();
		for (int i = 0; i <= L; i++)
		{
			ptsErrs_num.push_back(0);
		}

		float ptsErr_element = (maxPtsErr - minPtsErr) / ((L + 1) * 1.0);		//分级因子

		if (ptsErr_element != 0.0) {
			//统计每级（L）个数
			for (size_t i = 0; i < errN; i++)
			{
				int a = int((ptsErrs[i] - minPtsErr) / ptsErr_element);		//商即为级数
				//printf(" %d", a);
				if (a <= L) {
					ptsErrs_num[a]++;
				}
				else {
					ptsErrs_num[L]++;
				}
			}

			//计算距离残差各等级概率
			ptsErrs_prob.clear();
			for (size_t i = 0; i <= L; i++)
			{
				ptsErrs_prob.push_back(ptsErrs_num[i] * 1.0 / (errN*1.0));
				//printf("%d -- %.2f\n", ptsErrs_num[i], ptsErrs_prob[i]);
				//printf("%d ", ptsErrs_num[i]);
			}
			//printf("\n");

			//从0-L之间遍历找到使sigma最大的m
			float maxSigma = 0.0;
			for (int m = 1; m < L; m++)
			{
				//计算背景（前景）点集概率和
				PtsBGD_prob = 0.0;
				PtsFGD_prob = 0.0;
				for (size_t i = 0; i <= L; i++)
				{
					if (i <= m) {		//背景
						PtsBGD_prob += ptsErrs_prob[i];
					}
					else if (i > m) {	//前景
						PtsFGD_prob += ptsErrs_prob[i];
					}
				}

				//计算背景（前景）点集的均值
				PtsBGD_aver = 0.0;
				PtsFGD_aver = 0.0;
				for (size_t i = 0; i <= L; i++)
				{
					if (i <= m) {
						PtsBGD_aver += i * ptsErrs_prob[i] / PtsBGD_prob;
					}
					else if (i > m) {
						PtsFGD_aver += i * ptsErrs_prob[i] / PtsFGD_prob;
					}
				}

				//计算类间方差
				float sigma = PtsBGD_prob * pow(PtsBGD_aver, 2) + PtsFGD_prob * pow(PtsFGD_aver, 2);

				//更新前(背)景阈值
				if (sigma > maxSigma) {
					maxSigma = sigma;
					th = m;
				}
			}
			//printf("th = %d \n", th);

			//分离前景和背景点
			for (size_t i = 0; i < errN; i++)
			{
				if (ptsErrs[i] < ((th + 1)*ptsErr_element + minPtsErr)) {
					//背景点
					pts_BGD[0].push_back(pts_opti[0][i]);
					pts_BGD[1].push_back(pts_opti[1][i]);
				}
				else {
					//前景点
					pts_FGD[0].push_back(pts_opti[0][i]);
					pts_FGD[1].push_back(pts_opti[1][i]);
				}
			}
		}
	}
	printf("背景点：%d, 前景点数：%d \n", pts_BGD[1].size(), pts_FGD[1].size());

	float s = calPtsS();	//计算背景点方差
	printf("背景点方差%.2f \n", s);
}

//利用背景点进行背景补偿
void Func::makeBGD()
{
	if (pts_BGD[0].size() >= 5) {
		H = myGetAffineTransform(pts_BGD[0], pts_BGD[1]);
	}
	else {
		H = (Mat_<double>(2, 3) << 1.0, 0, 0,
			0, 1.0, 0);
	}
	cout << "最终 H：" << endl;
	cout << H << endl;

	warpAffine(prev_frame, prev_frame_tranf, H, prev_frame.size(), INTER_LINEAR);//双线性插值
	cv::imshow("背景补偿", prev_frame_tranf);
}

//前景提取（帧差）
void Func::getFGD()
{
	/*
	论文方案：//帧差-》滤波-》二值化-》膨胀-》区域标记-》小区域滤除->二次膨胀-》离散区域滤除-》
	*/
	cvtColor(prev_frame_tranf, prev_gray_tranf, CV_BGR2GRAY);	//转灰度

	absdiff(cur_gray, prev_gray_tranf, differ);  //帧差
	//imshow("帧差", differ);

	medianBlur(differ, differ, 3);			//滤波
	//imshow("滤波", differ);

	threshold(differ, differ, 60, 255, CV_THRESH_BINARY);	//二值化
	//imshow("二值化", differ);

	//边界向内20像素置为0
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if ((row > 20) && (row < (height - 20)) && (col > 20) && (col < (width - 20))) {
				//不作处理
			}
			else {
				differ.at<uchar>(row, col) = int(0);
			}
		}
	}
	
	morphologyEx(differ, differ, MORPH_DILATE, kernal3, Point(-1, -1), 1);	//膨胀
	
	smallTargetFilte(differ, 20);  //区域标记  小区域滤除
	//imshow("小区域滤除", differ);

	morphologyEx(differ, differ_dst, MORPH_CLOSE, kernal3, Point(-1, -1), 1);	//闭操作
	//cv::imshow("形态学操作结果", differ_dst);
	//目标离散区域归并
	combinTarget(cur_frame, differ_dst);
}

//绘制动态目标轮廓
void Func::drawContours()
{
	findContours(differ_dst, movContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);	//寻找轮廓
	cur_frame.copyTo(frame_contours);
	for (size_t i = 0; i < movContours.size(); i++)
	{
		Rect roi= boundingRect(movContours[i]);

		//目标分类轮廓比原始轮廓四周扩大10像素
		Rect ROI = Rect(roi.x - 10, roi.y - 10, roi.width + 10, roi.height + 10);
		rectangle(frame_contours, roi, Scalar(0, 255, 0), 1, 8);
		//目标分类
		//for (size_t j = 0; j < recogMov(cur_frame(ROI)).size(); j++)
		//{
		//	//在frame_dst上写出轮廓内目标分类标签
		//	putText(frame_dst, recogMov(cur_frame(ROI))[j], Point(roi.x, roi.y+i*10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, 8);
		//}
	}
	cv::imshow("轮廓", frame_contours);
}
//识别动态目标
//vector<String> Func::recogMov(const Mat &img)
//{
//	vector<String> resultLabels;
//	//图像预处理
//	Mat blobImg = blobFromImage(img, scaleFactor, Size(bolbWidth, bolbHeight), meanVal, false);
//	//输入图像
//	net.setInput(blobImg, "data");
//	//读取输出
//	Mat detection = net.forward("detection_out");
//	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
//	
//	for (int i = 0; i < detectionMat.rows; i++)
//	{
//		//可能性
//		float confidence = detectionMat.at<float>(i, 2);
//		if (confidence > confidence_threshold) {
//			//可能性大于阈值，认为预测正确
//			// t:top  l:left  b:buttom  r:right
//			int labelIdx = (int)(detectionMat.at<float>(i, 1));
//			int tl_x = (int)(detectionMat.at<float>(i, 3) * img.cols);
//			int tl_y = (int)(detectionMat.at<float>(i, 4) * img.rows);
//			int br_x = (int)(detectionMat.at<float>(i, 5) * img.cols);
//			int br_y = (int)(detectionMat.at<float>(i, 6) * img.rows);
//
//			resultLabels.push_back(labels[labelIdx]);
//		}
//	}
//	return resultLabels;
//}

//显示及数据更新
void Func::showAndUpdate()
{
	//更新前一帧
	cur_gray.copyTo(prev_gray);
	cur_frame.copyTo(prev_frame);

	//标记背景点
	cur_frame.copyTo(frame_features);
	//line(frame_features, )
	if (pts_BGD[1].size() > 0) {
		for (size_t i = 0; i < pts_BGD[1].size(); i++)
		{
			circle(frame_features, pts_BGD[1][i], 2, Scalar(0, 0, 255), -1, 8);
		}
	}
	//标记前景点
	if (pts_FGD[1].size() > 0) {
		for (size_t i = 0; i < pts_FGD[1].size(); i++)
		{
			circle(frame_features, pts_FGD[1][i], 2, Scalar(0, 255, 0), -1, 8);
		}
	}
	cv::imshow("自适应外点滤除", frame_features);
	//printf("背景点数(红色)：%d, 前景点数(绿色)：%d \n", pts_BGD[1].size(), pts_FGD[1].size());

	imshow("cur_frame", cur_frame);	
	//imshow("prev_frame_tranf", prev_frame_tranf);
	//imshow("differ", differ);
	//imshow("differ_dst", differ_dst);
	//imshow("frame_dst", frame_dst);

	printf("********************************************\n\n");
} 

//按键检测
int Func::readKey()
{
	char c = (char)waitKey(10);
	if (c == 27)
		return 0;
	else if (c == 32) {
		//暂停
		for (;;) {
			c = waitKey(50);
			if (c == 13)
				break;
		}
	}
	return 1;
}

//显示鼠标当前位置
void Func::onMouse(int event, int x, int y, int flags, void* ustc)
{
	if ((event == CV_EVENT_MOUSEMOVE) && (flags))
	{
		//format(mouseLoc, "(%d, %d)", x, y);
	}
}

//最小二乘法求解仿射矩阵
Mat Func::myGetAffineTransform(vector<Point2f> src, vector<Point2f> dst)
{
	int m = src.size();
	Mat_<float> X = Mat(m, 3, CV_32FC1, Scalar(0));
	Mat_<float> Y = Mat(m, 2, CV_32FC1, Scalar(0));

	for (int i = 0; i < m; i++)
	{
		float x0 = src[i].x, x1 = src[i].y;
		float y0 = dst[i].x, y1 = dst[i].y;

		X(i, 0) = x0;
		X(i, 1) = x1;
		X(i, 2) = 1;

		Y(i, 0) = y0;
		Y(i, 1) = y1;
	}

	cv::Mat_<float> F = (X.t()*X).inv()*(X.t()*Y);

	// cout << F << endl;

	return F.t();
}

//计算距离残差
void Func::calcPtsErrs()
{
	ptsErrs.clear();
	float x0, y0;
	float _x0, _y0;
	float x1, y1;
	float A, B, C, D, E, F;
	for (size_t i = 0; i < pts_opti[0].size(); i++)
	{
		x0 = pts_opti[0][i].x;  y0 = pts_opti[0][i].y;
		x1 = pts_opti[1][i].x;  y1 = pts_opti[1][i].y;

		A = H.at<float>(0, 0);
		B = H.at<float>(0, 1);
		C = H.at<float>(0, 2);
		D = H.at<float>(1, 0);
		E = H.at<float>(1, 1);
		F = H.at<float>(1, 2);

		_x0 = A * x0 + B * y0 + C;
		_y0 = D * x0 + E * y0 + F;

		ptsErrs.push_back(pow(pow(_x0 - x1, 2) + pow(_y0 - y1, 2), 0.5));
	}
}

//获取边角处的特征点
void Func::getCornerPts(double dx, double dy)
{
	pts_corner[0].clear();
	pts_corner[1].clear();
	for (size_t i = 0; i < pts_opti[1].size(); i++)
	{
		if ((pts_opti[1][i].x <= dx * width && pts_opti[1][i].y <= dy * height)
			|| (pts_opti[1][i].x <= dx * width && pts_opti[1][i].y >= (1 - dy) * height)
			|| (pts_opti[1][i].x >= (1 - dx) * width && pts_opti[1][i].y <= dy * height)
			|| (pts_opti[1][i].x >= (1 - dx) * width && pts_opti[1][i].y >= (1 - dy) * height)) {

			pts_corner[0].push_back(pts_opti[0][i]);
			pts_corner[1].push_back(pts_opti[1][i]);
		}
	}
}

//计算边角点标准差
double Func::calPtsS()
{
	//平均值
	double x_ave = 0.0, y_ave = 0.0;
	for (size_t i = 0; i < pts_BGD[1].size(); i++)
	{
		x_ave += pts_BGD[1][i].x * 1.0 / pts_BGD[1].size();
		y_ave += pts_BGD[1][i].y * 1.0 / pts_BGD[1].size();
	}
	//方差
	double x_ss = 0.0, y_ss = 0.0;
	for (size_t i = 0; i < pts_BGD[1].size(); i++)
	{
		x_ss += pow(x_ave - pts_BGD[1][i].x, 2) * 1.0 / pts_BGD[1].size();
		y_ss += pow(y_ave - pts_BGD[1][i].y, 2) * 1.0 / pts_BGD[1].size();
	}

	return (double)pow(pow(x_ss, 2) + pow(y_ss, 2), 0.5);
}

//小区域滤除
/* img:二值化图 */
void Func::smallTargetFilte(Mat &img, int thresh)
{
	//发现轮廓
	Mat img_m;
	img.convertTo(img_m, CV_8UC1);
	vector<vector<Point>> contours;
	findContours(img_m, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//填充各目标区域
	vector<double> areas;
	//printf("area: ");
	for (size_t i = 0; i < contours.size(); i++)
	{
		//方便后面根据像素值统计各区域像素大小
		cv::drawContours(img_m, contours, i, Scalar::all(i + 1), -1);
		areas.push_back(contourArea(contours[i]));
		//printf("%.2f ", areas[i]);
	}
	//printf("\n");
	imshow("markers", img_m * 20);

	//区域面积小于30的
	for (size_t row = 0; row < height; row++)
	{
		for (size_t col = 0; col < width; col++)
		{
			int pixel = static_cast<int>(img_m.at<uchar>(row, col));
			if (pixel > 0) {
				if (areas[pixel-1] < thresh) {
					img.at<uchar>(row, col) = 0;
				}
				else {
					img.at<uchar>(row, col) = 255;
				}
			}
		}
	}
}

//目标离散区域归并
/*
img_rgb:	RGB图
img_abs:	提取目标的二值化图
*/
void Func::combinTarget(Mat &img_rgb, Mat &img_abs)
{
	Mat img_rgb1, img_abs1;
	img_rgb.copyTo(img_rgb1);
	img_abs.copyTo(img_abs1);	//保存区域连接后的二值化图
	cvtColor(img_rgb1, img_rgb1, CV_BGR2HSV);	//RGB-》hsv
	vector<vector<Point>> contours;
	findContours(img_abs, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() > 1) {
		//contour_center[0].clear();
		contour_center[1].clear();	//初始化contour_center
		//hsiMsg[0].clear();
		hsiMsg[1].clear();
		for (size_t i = 0; i < contours.size(); i++)	//初始化hsiMsg
		{
			vector<double> msg(6);
			msg[0] = 0.0;	//H 均值
			msg[0] = 0.0;	//S 均值
			msg[0] = 0.0;	//I 均值
			msg[0] = 0.0;	//H 方差
			msg[0] = 0.0;	//S 方差
			msg[0] = 0.0;	//I 方差
			hsiMsg[1].push_back(msg);
		}
		
		Rect rect;	//缓存每个区域的外接矩形
		//遍历每个轮廓
		for (size_t i = 0; i < contours.size(); i++)
		{
			//获取轮廓质心
			Moments mu = moments(contours[i]);
			contour_center[1].push_back(Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00));
			//printf("质心%d: (%.1f, %.1f) \n", i, contour_center[i].x, contour_center[i].y);
			//获取轮廓内部像素点个数
			int pixelNum = 0;	//缓存轮廓内部像素点个数
			//获取轮廓内部颜色信息
			Mat mat = Mat::zeros(img_rgb.size(), CV_8UC1);
			cv::drawContours(mat, contours, i, Scalar::all(255), -1);
			rect = boundingRect(contours[i]);	//获取区域外接矩形
			vector<double> hsi(3);	// HSI
			//在区域矩形内遍历目标点
			//计算区域内HSI的均值
			for (int row = rect.y; row < (rect.y+rect.height); row++)
			{
				const uchar* imgRow = img_rgb1.ptr<uchar>(row);
				for (int col = rect.x; col < (rect.x+rect.width); col++)
				{
					if ((int)mat.at<uchar>(row, col) == 255) {
						//获取HSV分量
						hsi[0] = img_rgb1.at<Vec3b>(row, col)[0] * 2;
						hsi[1] = img_rgb1.at<Vec3b>(row, col)[1] / 255;
						hsi[2] = img_rgb1.at<Vec3b>(row, col)[2] / 255;

						//计算区域内HSI的均值
						hsiMsg[1][i][0] += hsi[0];	//H均值
						hsiMsg[1][i][1] += hsi[1];	//S均值
						hsiMsg[1][i][2] += hsi[2];	//I均值

						pixelNum++;	//记录轮廓内像素点个数
					}
				}
			}
			hsiMsg[1][i][0] /= pixelNum;	//H均值
			hsiMsg[1][i][1] /= pixelNum;	//S均值
			hsiMsg[1][i][2] /= pixelNum;	//I均值
			//计算区域内HSI的方差
			for (size_t row = rect.y; row < (rect.y + rect.height); row++)
			{
				for (size_t col = rect.x; col < (rect.x + rect.width); col++)
				{
					if (mat.at<uchar>(row, col) == 255) {
						//获取HSV分量
						hsi[0] = img_rgb1.at<Vec3b>(row, col)[0] * 2;
						hsi[1] = img_rgb1.at<Vec3b>(row, col)[1] / 255;
						hsi[2] = img_rgb1.at<Vec3b>(row, col)[2] / 255;

						//计算区域内HSI的均值
						hsiMsg[1][i][3] += pow(hsi[0] - hsiMsg[1][i][0], 2) / pixelNum;	//H方差
						hsiMsg[1][i][4] += pow(hsi[1] - hsiMsg[1][i][1], 2) / pixelNum;	//S方差
						hsiMsg[1][i][5] += pow(hsi[2] - hsiMsg[1][i][2], 2) / pixelNum;	//I方差
					}
				}
			}
			//printf("轮廓%d: HSI均值:%.2f, %.2f, %.2f   HSI方差:%.2f, %.2f, %.2f\n", 
				//i, hsiMsg[1][i][0], hsiMsg[1][i][1], hsiMsg[1][i][2], hsiMsg[1][i][3], hsiMsg[1][i][4], hsiMsg[1][i][5]);
		}
		/*********** 到这里区域内的HSI颜色信息已统计完毕 **********/
		//根据颜色信息归并
		for (size_t i = 0; i < contours.size()-1; i++)
		{
			for (size_t j = i+1; j < contours.size(); j++)
			{
				//1.质心相距50以内视为临近
				if (pts2fDist(contour_center[1][i], contour_center[1][j]) < 50.0) {
					//比较颜色特征距离
					double d = colorDist(hsiMsg[1][i], hsiMsg[1][j]);	//计算颜色特征距离
					//printf("颜色特征距离：%.2f\n", d);
					//putText(img_rgb, to_string(int(d)), Point((int)((contour_center[1][i].x + contour_center[1][j].x) / 2), ((int)(contour_center[1][i].y + contour_center[1][j].y) / 2)),
						 //FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
					//2.颜色特征相差3000以内视为同一目标
					if (d < 3000) {
						line(img_abs1, Point2f(contour_center[1][i].x, contour_center[1][i].y),	//连接归并区域
							Point2f(contour_center[1][j].x, contour_center[1][j].y), Scalar::all(255), 3, 8);
						//printf("画线");
					}
				}
			}
		}
		/******************* 根据位置和目标位移信息归并 *****************/
		if (contour_center[0].empty()) {
			contour_center[0].assign(contour_center[1].begin(), contour_center[1].end());
		}
		if (hsiMsg[0].empty()) {
			hsiMsg[0].assign(hsiMsg[1].begin(), hsiMsg[1].end());
		}
		//下面是归并方法
		//1.提取当前帧与前一帧位置接近的区域对
		//2.计算区域对的颜色特征距离确定前后帧对应目标区域
		vector<vector<double>> targetsMovMsg;	//缓存相邻两帧对应目标区域的移动方向、距离和当前帧质心(x y)
		vector<double> targetMovMsg(4);	//缓存相邻两帧某一目标区域的移动方向、距离和当前帧质心(x y)
		for (size_t i = 0; i < contour_center[1].size(); i++)
		{
			for (size_t j = 0; j < contour_center[0].size(); j++)
			{
				//1.距离小于20认为临近
				if (pts2fDist(contour_center[1][i], contour_center[0][j]) < 30.0) {
					//2.颜色特征距离小于10000认为是同一目标
					if (colorDist(hsiMsg[1][i], hsiMsg[0][j]) < 8000.0) {
						//3.计算目标移动方向、距离和当前帧质心
						targetMovMsg[0] = tgtMovDirec(contour_center[0][j], contour_center[1][i]);//移动方向
						targetMovMsg[1] = pts2fDist(contour_center[1][i], contour_center[0][j]);  //移动距离
						targetMovMsg[2] = (double)contour_center[1][i].x;	//当前帧位置 X
						targetMovMsg[3] = (double)contour_center[1][i].y;	//当前帧位置 Y
						targetsMovMsg.push_back(targetMovMsg);	//记录

						//将移动轨迹画在图上
						//line(img_rgb, Point(int(contour_center[0][j].x), int(contour_center[0][j].y)),
							//Point(int(contour_center[1][i].x), int(contour_center[1][i].y)), Scalar(0, 0, 255), 2, 8);
						//printf("由(%d, %d)->(%d, %d)移动了%.2f \n", int(contour_center[0][j].x), int(contour_center[0][j].y), 
							//int(contour_center[1][i].x), int(contour_center[1][i].y), targetMovMsg[1]);
					}
				}
			}
		}
		//比较相邻两帧目标区域的位置和移动信息
		if (targetsMovMsg.size() > 1) {
			for (size_t i = 0; i < targetsMovMsg.size() - 1; i++)
			{
				for (size_t j = i +1; j < targetsMovMsg.size(); j++)
				{
					//1.区域质心在当前帧相距小于50
					Point2f p1 = Point2f((float)targetsMovMsg[i][2], (float)targetsMovMsg[i][3]);
					Point2f p2 = Point2f((float)targetsMovMsg[j][2], (float)targetsMovMsg[j][3]);
					//printf("区域质心在当前帧距离：%.2f\n", pts2fDist(p1, p2));
					if (pts2fDist(p1, p2) < 55.0) {
						//2.评价不同物体的移动一致性
						double movOffset = getmovOffset(targetsMovMsg[i], targetsMovMsg[j]);
						//printf("移动一致性: %.2f \n", movOffset);
						if (movOffset <= 30) {
							//移动一致性小于20，认为是同一目标
							line(img_abs1, Point(int(targetsMovMsg[i][2]), int(targetsMovMsg[i][3])),	//连接归并区域
								Point(int(targetsMovMsg[j][2]), int(targetsMovMsg[j][3])), Scalar::all(255), 3, 8);
						}			
					}
				}
			}
		}

		//更新前一帧区域位置和颜色信息
		contour_center[0].assign(contour_center[1].begin(), contour_center[1].end());	//更新位置信息
		hsiMsg[0].assign(hsiMsg[1].begin(), hsiMsg[1].end());	//更新HSI颜色信息
	}
	cv::imshow("目标归并", img_abs1);
	//cv::imshow("区域移动轨迹", img_rgb);

	if(prev_con.empty())
		img_abs.copyTo(prev_con);
	cv::imshow("前一帧二值图", prev_con);

	img_abs.copyTo(prev_con);

	img_abs1.copyTo(img_abs);
}

//BGR转HSI
void Func::BGR2HSI(vector<double> &bgr, vector<double> &hsi)
{
	double b = bgr[0], g = bgr[1], r = bgr[2];	//BGR
	double h = 0.0, s = 0.0, i = 0.0;			//HSI
	//坐标变换法
	double k = (2 * r - g - b) / (pow(3.0, 0.5) * (g - b));
	double sita = pi / 2 - pow(tan(k), -1);
	//获取H
	if (g >= b) {
		h = sita;
	}
	else if (g < b) {
		h = sita + pi;
	}
	//获取S
	s = 2.0 / pow(6.0, 0.5) * pow((r - g)*(r - g) + (r - b)*(g - b), 0.5);
	//获取I
	i = (r + g + b) / pow(3, 0.5);



	//分段定义法
	/*
	double minBGR = *min_element(bgr.begin(), bgr.end());	//最小值
	double maxBGR = *max_element(bgr.begin(), bgr.end());	//最大值
	int minPos = min_element(bgr.begin(), bgr.end()) - bgr.begin();	//最小值下标
	int maxPos = max_element(bgr.begin(), bgr.end()) - bgr.begin();	//最大值下标

	if (maxPos == 2) {		//max = R
		h = pi / 3.0 * (g - b) / (maxBGR - minBGR);
	}
	else if (maxPos == 1) {	//max = G
		h = pi / 3.0 * (b - r) / (maxBGR - minBGR) + 2.0 * pi / 3.0;
	}
	else if (maxPos == 0) {	//max = B
		h = pi / 3.0 * (r - g) / (maxBGR - minBGR) + 4.0 * pi / 3.0;
	}
	if (h < 0) 
		h += 2.0 * pi;			//获得H
	i = 0.5*(maxBGR + minBGR);	//获得I
	if (i > 0 && i <= 0.5) {	//获得S
		s = (maxBGR - minBGR) / (maxBGR + minBGR);
	}
	else if (i > 0.5) {
		s = (maxBGR - minBGR) / (2 - maxBGR - minBGR);
	}
	*/

	hsi[0] = h;
	hsi[1] = s;
	hsi[2] = i;
}

//计算两点间距离
double Func::pts2fDist(Point2f &pts1, Point2f &pts2)
{
	return pow(pow(pts1.x - pts2.x, 2) + pow(pts1.y - pts2.y, 2), 0.5);
}
//计算从pts1到pts2的移动方向
/*
假定向右为正方向，角度为0，沿逆时针增加至360
*/
double Func::tgtMovDirec(Point2f &pts1, Point2f &pts2)
{
	double sita;	//角度 0-360
	if (pts2.x < pts1.x && pts2.y <= pts1.y) {		//p2在p1左上
		sita = 180 - atan((pts1.y - pts2.y) / (pts1.x - pts2.x)) * 180.0 / pi;
	}
	else if (pts2.x > pts1.x && pts2.y <= pts1.y) {	//p2在p1右上
		sita = atan((pts1.y - pts2.y) / (pts2.x - pts1.x)) * 180.0 / pi;
	}
	else if (pts2.x < pts1.x && pts2.y > pts1.y) {	//p2在p1左下
		sita = 270 - atan((pts2.y - pts1.y) / (pts1.x - pts2.x)) * 180.0 / pi;
	}
	else if (pts2.x > pts1.x && pts2.y > pts1.y) {	//p2在p1右下
		sita = 360 - atan((pts2.y - pts1.y) / (pts2.x - pts1.x)) * 180.0 / pi;
	}
	else if (pts2.x == pts1.x && pts2.y < pts1.y) {	//p2在p1上
		sita = 90.0;
	}
	else if (pts2.x == pts1.x && pts2.y > pts1.y) {	//p2在p1下
		sita = 270.0;
	}
	else if (pts2.x == pts1.x && pts2.y == pts1.y) {//p2 == p1
		sita = 0.0;
	}
	return sita;
}
//比较颜色特征距离
double Func::colorDist(vector<double> &hsiMsg1, vector<double> &hsiMsg2)
{
	return pow(pow(hsiMsg1[0] - hsiMsg2[0], 2) +
				pow(hsiMsg1[1] - hsiMsg2[1], 2) +
				pow(hsiMsg1[2] - hsiMsg2[2], 2) +
				pow(hsiMsg1[3] - hsiMsg2[3], 2) +
				pow(hsiMsg1[4] - hsiMsg2[4], 2) +
				pow(hsiMsg1[5] - hsiMsg2[5], 2), 0.5);
}
//评价不同物体的移动一致性
/*
targetsMovMsg1: 四个元素：方向、距离、x、y
*/
double Func::getmovOffset(vector<double> targetsMovMsg1, vector<double> targetsMovMsg2)
{
	double dx1 = 0.0, dy1 = 0.0;
	double dx2 = 0.0, dy2 = 0.0;
	double k = pi / 180.0;	//角度->弧度

	dx1 = targetsMovMsg1[1] * cos(targetsMovMsg1[0] * k);
	dy1 = targetsMovMsg1[1] * sin(targetsMovMsg1[0] * k) * -1;

	dx2 = targetsMovMsg2[1] * cos(targetsMovMsg2[0] * k);
	dy2 = targetsMovMsg2[1] * sin(targetsMovMsg2[0] * k) * -1;

	return pow(pow(dx1 - dx2, 2) + pow(dy1 - dy2, 2), 0.5);
}



