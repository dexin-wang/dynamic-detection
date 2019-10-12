#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include "iostream"
#include "func.h"

using namespace cv;
using namespace std;
using namespace xfeatures2d;

#define pi 3.1415926


//ʹ��˵��
void Func::attention()
{
	printf("[0]��ʾǰһ֡��[1]��ʾ��ǰ֡\n\n");
}
//����Ƶ�ļ�
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

//��frame
void Func::readFrame()
{
	capture.read(cur_frame);							//����ǰ֡
	cvtColor(cur_frame, cur_gray, COLOR_BGR2GRAY);
	if (prev_gray.empty()) {
		cur_gray.copyTo(prev_gray);						//����ǰһ֡
		cur_frame.copyTo(prev_frame);					//����ǰһ֡
	}
}

//SURF�������
void Func::surfDetect()
{
	Pts[0].clear(); Pts[1].clear();
	desc[0].release(); desc[1].release();
	surfDetector->detectAndCompute(prev_gray, Mat(), Pts[0], desc[0]);
	surfDetector->detectAndCompute(cur_gray, Mat(), Pts[1], desc[1]);
	//drawKeypoints(cur_frame, Pts[1], cur_frame, Scalar(0, 0, 255));

	//printf("SURF��������%d��%d \n", Pts[0].size(), Pts[1].size());
}

//KNNƥ��
void Func::BfMatch()
{
	knnMatches[0].clear();
	matcher.knnMatch(desc[0], desc[1], knnMatches[0], 2);		//����ƥ��
	printf("KNNƥ��������%d \n", knnMatches[0].size());
	//BBF�����㷨
	//BBFmatches.clear();
	BBFpts[0].clear(); BBFpts[1].clear();
	for (size_t i = 0; i < knnMatches[0].size(); i++) {
		const DMatch& bestMatch = knnMatches[0][i][0];			//����������С
		const DMatch& betterMatch = knnMatches[0][i][1];		//���������С

		float  distanceRatio = bestMatch.distance / betterMatch.distance;
		//���ú���������Թ���ƥ����
		if (distanceRatio < minRatio) {
			//BBFmatches.push_back(bestMatch);
			BBFpts[0].push_back(Pts[0][bestMatch.queryIdx].pt);
			BBFpts[1].push_back(Pts[1][bestMatch.trainIdx].pt);
		}
	}
	printf("�Գ�Լ��ǰ��BBFpts[0].size = %d \n", BBFpts[0].size());

	knnMatches[1].clear();
	matcher.knnMatch(desc[1], desc[0], knnMatches[1], 2);		//����ƥ��

	BBFpts_rev[0].clear();
	BBFpts_rev[1].clear();
	for (size_t i = 0; i < knnMatches[1].size(); i++) {
		const DMatch& bestMatch_rev = knnMatches[1][i][0];			//����������С
		const DMatch& betterMatch_rev = knnMatches[1][i][1];		//���������С

		float distanceRatio = bestMatch_rev.distance / betterMatch_rev.distance;
		//���ú���������Թ���ƥ����
		if (distanceRatio < minRatio) {
			//BBFmatches.push_back(bestMatch);
			BBFpts_rev[0].push_back(Pts[0][bestMatch_rev.trainIdx].pt);
			BBFpts_rev[1].push_back(Pts[1][bestMatch_rev.queryIdx].pt);
		}
	}
	//printf("�Գ�Լ��ǰ��BBFpts_rev[0].size = %d, BBFpts_rev[1].size = %d \n", BBFpts_rev[0].size(), BBFpts_rev[1].size());

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
	//�Գ�Լ��
	pts_opti[0].clear(); pts_opti[1].clear();
	for (size_t i = 0; i < BBFpts[0].size(); i++)
	{
		for (size_t j = 0; j < BBFpts_rev[0].size(); j++)
		{
			//����Գ�Լ����������
			if ((BBFpts[0][i].x == BBFpts_rev[0][j].x) && (BBFpts[0][i].y == BBFpts_rev[0][j].y)) {

				if ((BBFpts[1][i].x == BBFpts_rev[1][j].x) && (BBFpts[1][i].y == BBFpts_rev[1][j].y)) {

					//λ���Ľ��ұ߽�����10���ط�Χ�ڵĵ㱣�� 
					if (BBFpts[1][i].x >= 20 && BBFpts[1][i].x <= (width - 20) && BBFpts[1][i].y >= 20 && BBFpts[1][i].y <= (height - 20)) {

						pts_opti[0].push_back(Point2f(BBFpts[0][i].x, BBFpts[0][i].y));
						pts_opti[1].push_back(Point2f(BBFpts[1][i].x, BBFpts[1][i].y));
					}
				}
			}
		}
	}
	printf("�Գ�Լ����pts_opti[1].size = %d \n", pts_opti[1].size());
	/*for (size_t i = 0; i < pts_opti[1].size(); i++)
	{
		printf("(%d, %d) \n", pts_opti[1][i].x, pts_opti[1][i].y);
	}
	*/
}

/***********************  ���Ż�  **********************/
//����Ӧ����˳�
void Func::filtForePts()
{
	//��ȡ�߽Ǵ���������
	getCornerPts(0.2, 0.2);
	if (pts_corner[1].size() < 10) {
		getCornerPts(0.3, 0.3);
	}
	printf("�Ľǵĵ�����%d \n", pts_corner[1].size());
	
	//����Ӧ����˳�
	pts_BGD[0].clear(); pts_BGD[1].clear();
	pts_FGD[0].clear(); pts_FGD[1].clear();
	if (pts_corner[1].size() < 10) {	//��������˳�
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
	else {	//����˳�
		H = myGetAffineTransform(pts_corner[0], pts_corner[1]);	//�����ʼ H
		cout << "��ʼ H��" << endl; cout << H << endl;

		calcPtsErrs(); //�������в�

		int errN = ptsErrs.size();
		//printf("errN: %d \n", errN);

		//����min max ����в�
		float minPtsErr = 100.0, maxPtsErr = 0.0;
		for (size_t i = 0; i < ptsErrs.size(); i++)
		{
			float ptsErr = ptsErrs[i];
			if (ptsErr > maxPtsErr)
				maxPtsErr = ptsErr;
			if (ptsErr < minPtsErr)
				minPtsErr = ptsErr;
		}
		//printf("minPtsErr��%.2f, maxPtsErr:%.2f \n", minPtsErr, maxPtsErr);

		//ptsErrs_num����ռ�
		ptsErrs_num.clear();
		for (int i = 0; i <= L; i++)
		{
			ptsErrs_num.push_back(0);
		}

		float ptsErr_element = (maxPtsErr - minPtsErr) / ((L + 1) * 1.0);		//�ּ�����

		if (ptsErr_element != 0.0) {
			//ͳ��ÿ����L������
			for (size_t i = 0; i < errN; i++)
			{
				int a = int((ptsErrs[i] - minPtsErr) / ptsErr_element);		//�̼�Ϊ����
				//printf(" %d", a);
				if (a <= L) {
					ptsErrs_num[a]++;
				}
				else {
					ptsErrs_num[L]++;
				}
			}

			//�������в���ȼ�����
			ptsErrs_prob.clear();
			for (size_t i = 0; i <= L; i++)
			{
				ptsErrs_prob.push_back(ptsErrs_num[i] * 1.0 / (errN*1.0));
				//printf("%d -- %.2f\n", ptsErrs_num[i], ptsErrs_prob[i]);
				//printf("%d ", ptsErrs_num[i]);
			}
			//printf("\n");

			//��0-L֮������ҵ�ʹsigma����m
			float maxSigma = 0.0;
			for (int m = 1; m < L; m++)
			{
				//���㱳����ǰ�����㼯���ʺ�
				PtsBGD_prob = 0.0;
				PtsFGD_prob = 0.0;
				for (size_t i = 0; i <= L; i++)
				{
					if (i <= m) {		//����
						PtsBGD_prob += ptsErrs_prob[i];
					}
					else if (i > m) {	//ǰ��
						PtsFGD_prob += ptsErrs_prob[i];
					}
				}

				//���㱳����ǰ�����㼯�ľ�ֵ
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

				//������䷽��
				float sigma = PtsBGD_prob * pow(PtsBGD_aver, 2) + PtsFGD_prob * pow(PtsFGD_aver, 2);

				//����ǰ(��)����ֵ
				if (sigma > maxSigma) {
					maxSigma = sigma;
					th = m;
				}
			}
			//printf("th = %d \n", th);

			//����ǰ���ͱ�����
			for (size_t i = 0; i < errN; i++)
			{
				if (ptsErrs[i] < ((th + 1)*ptsErr_element + minPtsErr)) {
					//������
					pts_BGD[0].push_back(pts_opti[0][i]);
					pts_BGD[1].push_back(pts_opti[1][i]);
				}
				else {
					//ǰ����
					pts_FGD[0].push_back(pts_opti[0][i]);
					pts_FGD[1].push_back(pts_opti[1][i]);
				}
			}
		}
	}
	printf("�����㣺%d, ǰ��������%d \n", pts_BGD[1].size(), pts_FGD[1].size());

	float s = calPtsS();	//���㱳���㷽��
	printf("�����㷽��%.2f \n", s);
}

//���ñ�������б�������
void Func::makeBGD()
{
	if (pts_BGD[0].size() >= 5) {
		H = myGetAffineTransform(pts_BGD[0], pts_BGD[1]);
	}
	else {
		H = (Mat_<double>(2, 3) << 1.0, 0, 0,
			0, 1.0, 0);
	}
	cout << "���� H��" << endl;
	cout << H << endl;

	warpAffine(prev_frame, prev_frame_tranf, H, prev_frame.size(), INTER_LINEAR);//˫���Բ�ֵ
	cv::imshow("��������", prev_frame_tranf);
}

//ǰ����ȡ��֡�
void Func::getFGD()
{
	/*
	���ķ�����//֡��-���˲�-����ֵ��-������-��������-��С�����˳�->��������-����ɢ�����˳�-��
	*/
	cvtColor(prev_frame_tranf, prev_gray_tranf, CV_BGR2GRAY);	//ת�Ҷ�

	absdiff(cur_gray, prev_gray_tranf, differ);  //֡��
	//imshow("֡��", differ);

	medianBlur(differ, differ, 3);			//�˲�
	//imshow("�˲�", differ);

	threshold(differ, differ, 60, 255, CV_THRESH_BINARY);	//��ֵ��
	//imshow("��ֵ��", differ);

	//�߽�����20������Ϊ0
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if ((row > 20) && (row < (height - 20)) && (col > 20) && (col < (width - 20))) {
				//��������
			}
			else {
				differ.at<uchar>(row, col) = int(0);
			}
		}
	}
	
	morphologyEx(differ, differ, MORPH_DILATE, kernal3, Point(-1, -1), 1);	//����
	
	smallTargetFilte(differ, 20);  //������  С�����˳�
	//imshow("С�����˳�", differ);

	morphologyEx(differ, differ_dst, MORPH_CLOSE, kernal3, Point(-1, -1), 1);	//�ղ���
	//cv::imshow("��̬ѧ�������", differ_dst);
	//Ŀ����ɢ����鲢
	combinTarget(cur_frame, differ_dst);
}

//���ƶ�̬Ŀ������
void Func::drawContours()
{
	findContours(differ_dst, movContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);	//Ѱ������
	cur_frame.copyTo(frame_contours);
	for (size_t i = 0; i < movContours.size(); i++)
	{
		Rect roi= boundingRect(movContours[i]);

		//Ŀ�����������ԭʼ������������10����
		Rect ROI = Rect(roi.x - 10, roi.y - 10, roi.width + 10, roi.height + 10);
		rectangle(frame_contours, roi, Scalar(0, 255, 0), 1, 8);
		//Ŀ�����
		//for (size_t j = 0; j < recogMov(cur_frame(ROI)).size(); j++)
		//{
		//	//��frame_dst��д��������Ŀ������ǩ
		//	putText(frame_dst, recogMov(cur_frame(ROI))[j], Point(roi.x, roi.y+i*10), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2, 8);
		//}
	}
	cv::imshow("����", frame_contours);
}
//ʶ��̬Ŀ��
//vector<String> Func::recogMov(const Mat &img)
//{
//	vector<String> resultLabels;
//	//ͼ��Ԥ����
//	Mat blobImg = blobFromImage(img, scaleFactor, Size(bolbWidth, bolbHeight), meanVal, false);
//	//����ͼ��
//	net.setInput(blobImg, "data");
//	//��ȡ���
//	Mat detection = net.forward("detection_out");
//	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
//	
//	for (int i = 0; i < detectionMat.rows; i++)
//	{
//		//������
//		float confidence = detectionMat.at<float>(i, 2);
//		if (confidence > confidence_threshold) {
//			//�����Դ�����ֵ����ΪԤ����ȷ
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

//��ʾ�����ݸ���
void Func::showAndUpdate()
{
	//����ǰһ֡
	cur_gray.copyTo(prev_gray);
	cur_frame.copyTo(prev_frame);

	//��Ǳ�����
	cur_frame.copyTo(frame_features);
	//line(frame_features, )
	if (pts_BGD[1].size() > 0) {
		for (size_t i = 0; i < pts_BGD[1].size(); i++)
		{
			circle(frame_features, pts_BGD[1][i], 2, Scalar(0, 0, 255), -1, 8);
		}
	}
	//���ǰ����
	if (pts_FGD[1].size() > 0) {
		for (size_t i = 0; i < pts_FGD[1].size(); i++)
		{
			circle(frame_features, pts_FGD[1][i], 2, Scalar(0, 255, 0), -1, 8);
		}
	}
	cv::imshow("����Ӧ����˳�", frame_features);
	//printf("��������(��ɫ)��%d, ǰ������(��ɫ)��%d \n", pts_BGD[1].size(), pts_FGD[1].size());

	imshow("cur_frame", cur_frame);	
	//imshow("prev_frame_tranf", prev_frame_tranf);
	//imshow("differ", differ);
	//imshow("differ_dst", differ_dst);
	//imshow("frame_dst", frame_dst);

	printf("********************************************\n\n");
} 

//�������
int Func::readKey()
{
	char c = (char)waitKey(10);
	if (c == 27)
		return 0;
	else if (c == 32) {
		//��ͣ
		for (;;) {
			c = waitKey(50);
			if (c == 13)
				break;
		}
	}
	return 1;
}

//��ʾ��굱ǰλ��
void Func::onMouse(int event, int x, int y, int flags, void* ustc)
{
	if ((event == CV_EVENT_MOUSEMOVE) && (flags))
	{
		//format(mouseLoc, "(%d, %d)", x, y);
	}
}

//��С���˷����������
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

//�������в�
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

//��ȡ�߽Ǵ���������
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

//����߽ǵ��׼��
double Func::calPtsS()
{
	//ƽ��ֵ
	double x_ave = 0.0, y_ave = 0.0;
	for (size_t i = 0; i < pts_BGD[1].size(); i++)
	{
		x_ave += pts_BGD[1][i].x * 1.0 / pts_BGD[1].size();
		y_ave += pts_BGD[1][i].y * 1.0 / pts_BGD[1].size();
	}
	//����
	double x_ss = 0.0, y_ss = 0.0;
	for (size_t i = 0; i < pts_BGD[1].size(); i++)
	{
		x_ss += pow(x_ave - pts_BGD[1][i].x, 2) * 1.0 / pts_BGD[1].size();
		y_ss += pow(y_ave - pts_BGD[1][i].y, 2) * 1.0 / pts_BGD[1].size();
	}

	return (double)pow(pow(x_ss, 2) + pow(y_ss, 2), 0.5);
}

//С�����˳�
/* img:��ֵ��ͼ */
void Func::smallTargetFilte(Mat &img, int thresh)
{
	//��������
	Mat img_m;
	img.convertTo(img_m, CV_8UC1);
	vector<vector<Point>> contours;
	findContours(img_m, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	//����Ŀ������
	vector<double> areas;
	//printf("area: ");
	for (size_t i = 0; i < contours.size(); i++)
	{
		//��������������ֵͳ�Ƹ��������ش�С
		cv::drawContours(img_m, contours, i, Scalar::all(i + 1), -1);
		areas.push_back(contourArea(contours[i]));
		//printf("%.2f ", areas[i]);
	}
	//printf("\n");
	imshow("markers", img_m * 20);

	//�������С��30��
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

//Ŀ����ɢ����鲢
/*
img_rgb:	RGBͼ
img_abs:	��ȡĿ��Ķ�ֵ��ͼ
*/
void Func::combinTarget(Mat &img_rgb, Mat &img_abs)
{
	Mat img_rgb1, img_abs1;
	img_rgb.copyTo(img_rgb1);
	img_abs.copyTo(img_abs1);	//�����������Ӻ�Ķ�ֵ��ͼ
	cvtColor(img_rgb1, img_rgb1, CV_BGR2HSV);	//RGB-��hsv
	vector<vector<Point>> contours;
	findContours(img_abs, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() > 1) {
		//contour_center[0].clear();
		contour_center[1].clear();	//��ʼ��contour_center
		//hsiMsg[0].clear();
		hsiMsg[1].clear();
		for (size_t i = 0; i < contours.size(); i++)	//��ʼ��hsiMsg
		{
			vector<double> msg(6);
			msg[0] = 0.0;	//H ��ֵ
			msg[0] = 0.0;	//S ��ֵ
			msg[0] = 0.0;	//I ��ֵ
			msg[0] = 0.0;	//H ����
			msg[0] = 0.0;	//S ����
			msg[0] = 0.0;	//I ����
			hsiMsg[1].push_back(msg);
		}
		
		Rect rect;	//����ÿ���������Ӿ���
		//����ÿ������
		for (size_t i = 0; i < contours.size(); i++)
		{
			//��ȡ��������
			Moments mu = moments(contours[i]);
			contour_center[1].push_back(Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00));
			//printf("����%d: (%.1f, %.1f) \n", i, contour_center[i].x, contour_center[i].y);
			//��ȡ�����ڲ����ص����
			int pixelNum = 0;	//���������ڲ����ص����
			//��ȡ�����ڲ���ɫ��Ϣ
			Mat mat = Mat::zeros(img_rgb.size(), CV_8UC1);
			cv::drawContours(mat, contours, i, Scalar::all(255), -1);
			rect = boundingRect(contours[i]);	//��ȡ������Ӿ���
			vector<double> hsi(3);	// HSI
			//����������ڱ���Ŀ���
			//����������HSI�ľ�ֵ
			for (int row = rect.y; row < (rect.y+rect.height); row++)
			{
				const uchar* imgRow = img_rgb1.ptr<uchar>(row);
				for (int col = rect.x; col < (rect.x+rect.width); col++)
				{
					if ((int)mat.at<uchar>(row, col) == 255) {
						//��ȡHSV����
						hsi[0] = img_rgb1.at<Vec3b>(row, col)[0] * 2;
						hsi[1] = img_rgb1.at<Vec3b>(row, col)[1] / 255;
						hsi[2] = img_rgb1.at<Vec3b>(row, col)[2] / 255;

						//����������HSI�ľ�ֵ
						hsiMsg[1][i][0] += hsi[0];	//H��ֵ
						hsiMsg[1][i][1] += hsi[1];	//S��ֵ
						hsiMsg[1][i][2] += hsi[2];	//I��ֵ

						pixelNum++;	//��¼���������ص����
					}
				}
			}
			hsiMsg[1][i][0] /= pixelNum;	//H��ֵ
			hsiMsg[1][i][1] /= pixelNum;	//S��ֵ
			hsiMsg[1][i][2] /= pixelNum;	//I��ֵ
			//����������HSI�ķ���
			for (size_t row = rect.y; row < (rect.y + rect.height); row++)
			{
				for (size_t col = rect.x; col < (rect.x + rect.width); col++)
				{
					if (mat.at<uchar>(row, col) == 255) {
						//��ȡHSV����
						hsi[0] = img_rgb1.at<Vec3b>(row, col)[0] * 2;
						hsi[1] = img_rgb1.at<Vec3b>(row, col)[1] / 255;
						hsi[2] = img_rgb1.at<Vec3b>(row, col)[2] / 255;

						//����������HSI�ľ�ֵ
						hsiMsg[1][i][3] += pow(hsi[0] - hsiMsg[1][i][0], 2) / pixelNum;	//H����
						hsiMsg[1][i][4] += pow(hsi[1] - hsiMsg[1][i][1], 2) / pixelNum;	//S����
						hsiMsg[1][i][5] += pow(hsi[2] - hsiMsg[1][i][2], 2) / pixelNum;	//I����
					}
				}
			}
			//printf("����%d: HSI��ֵ:%.2f, %.2f, %.2f   HSI����:%.2f, %.2f, %.2f\n", 
				//i, hsiMsg[1][i][0], hsiMsg[1][i][1], hsiMsg[1][i][2], hsiMsg[1][i][3], hsiMsg[1][i][4], hsiMsg[1][i][5]);
		}
		/*********** �����������ڵ�HSI��ɫ��Ϣ��ͳ����� **********/
		//������ɫ��Ϣ�鲢
		for (size_t i = 0; i < contours.size()-1; i++)
		{
			for (size_t j = i+1; j < contours.size(); j++)
			{
				//1.�������50������Ϊ�ٽ�
				if (pts2fDist(contour_center[1][i], contour_center[1][j]) < 50.0) {
					//�Ƚ���ɫ��������
					double d = colorDist(hsiMsg[1][i], hsiMsg[1][j]);	//������ɫ��������
					//printf("��ɫ�������룺%.2f\n", d);
					//putText(img_rgb, to_string(int(d)), Point((int)((contour_center[1][i].x + contour_center[1][j].x) / 2), ((int)(contour_center[1][i].y + contour_center[1][j].y) / 2)),
						 //FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
					//2.��ɫ�������3000������ΪͬһĿ��
					if (d < 3000) {
						line(img_abs1, Point2f(contour_center[1][i].x, contour_center[1][i].y),	//���ӹ鲢����
							Point2f(contour_center[1][j].x, contour_center[1][j].y), Scalar::all(255), 3, 8);
						//printf("����");
					}
				}
			}
		}
		/******************* ����λ�ú�Ŀ��λ����Ϣ�鲢 *****************/
		if (contour_center[0].empty()) {
			contour_center[0].assign(contour_center[1].begin(), contour_center[1].end());
		}
		if (hsiMsg[0].empty()) {
			hsiMsg[0].assign(hsiMsg[1].begin(), hsiMsg[1].end());
		}
		//�����ǹ鲢����
		//1.��ȡ��ǰ֡��ǰһ֡λ�ýӽ��������
		//2.��������Ե���ɫ��������ȷ��ǰ��֡��ӦĿ������
		vector<vector<double>> targetsMovMsg;	//����������֡��ӦĿ��������ƶ����򡢾���͵�ǰ֡����(x y)
		vector<double> targetMovMsg(4);	//����������֡ĳһĿ��������ƶ����򡢾���͵�ǰ֡����(x y)
		for (size_t i = 0; i < contour_center[1].size(); i++)
		{
			for (size_t j = 0; j < contour_center[0].size(); j++)
			{
				//1.����С��20��Ϊ�ٽ�
				if (pts2fDist(contour_center[1][i], contour_center[0][j]) < 30.0) {
					//2.��ɫ��������С��10000��Ϊ��ͬһĿ��
					if (colorDist(hsiMsg[1][i], hsiMsg[0][j]) < 8000.0) {
						//3.����Ŀ���ƶ����򡢾���͵�ǰ֡����
						targetMovMsg[0] = tgtMovDirec(contour_center[0][j], contour_center[1][i]);//�ƶ�����
						targetMovMsg[1] = pts2fDist(contour_center[1][i], contour_center[0][j]);  //�ƶ�����
						targetMovMsg[2] = (double)contour_center[1][i].x;	//��ǰ֡λ�� X
						targetMovMsg[3] = (double)contour_center[1][i].y;	//��ǰ֡λ�� Y
						targetsMovMsg.push_back(targetMovMsg);	//��¼

						//���ƶ��켣����ͼ��
						//line(img_rgb, Point(int(contour_center[0][j].x), int(contour_center[0][j].y)),
							//Point(int(contour_center[1][i].x), int(contour_center[1][i].y)), Scalar(0, 0, 255), 2, 8);
						//printf("��(%d, %d)->(%d, %d)�ƶ���%.2f \n", int(contour_center[0][j].x), int(contour_center[0][j].y), 
							//int(contour_center[1][i].x), int(contour_center[1][i].y), targetMovMsg[1]);
					}
				}
			}
		}
		//�Ƚ�������֡Ŀ�������λ�ú��ƶ���Ϣ
		if (targetsMovMsg.size() > 1) {
			for (size_t i = 0; i < targetsMovMsg.size() - 1; i++)
			{
				for (size_t j = i +1; j < targetsMovMsg.size(); j++)
				{
					//1.���������ڵ�ǰ֡���С��50
					Point2f p1 = Point2f((float)targetsMovMsg[i][2], (float)targetsMovMsg[i][3]);
					Point2f p2 = Point2f((float)targetsMovMsg[j][2], (float)targetsMovMsg[j][3]);
					//printf("���������ڵ�ǰ֡���룺%.2f\n", pts2fDist(p1, p2));
					if (pts2fDist(p1, p2) < 55.0) {
						//2.���۲�ͬ������ƶ�һ����
						double movOffset = getmovOffset(targetsMovMsg[i], targetsMovMsg[j]);
						//printf("�ƶ�һ����: %.2f \n", movOffset);
						if (movOffset <= 30) {
							//�ƶ�һ����С��20����Ϊ��ͬһĿ��
							line(img_abs1, Point(int(targetsMovMsg[i][2]), int(targetsMovMsg[i][3])),	//���ӹ鲢����
								Point(int(targetsMovMsg[j][2]), int(targetsMovMsg[j][3])), Scalar::all(255), 3, 8);
						}			
					}
				}
			}
		}

		//����ǰһ֡����λ�ú���ɫ��Ϣ
		contour_center[0].assign(contour_center[1].begin(), contour_center[1].end());	//����λ����Ϣ
		hsiMsg[0].assign(hsiMsg[1].begin(), hsiMsg[1].end());	//����HSI��ɫ��Ϣ
	}
	cv::imshow("Ŀ��鲢", img_abs1);
	//cv::imshow("�����ƶ��켣", img_rgb);

	if(prev_con.empty())
		img_abs.copyTo(prev_con);
	cv::imshow("ǰһ֡��ֵͼ", prev_con);

	img_abs.copyTo(prev_con);

	img_abs1.copyTo(img_abs);
}

//BGRתHSI
void Func::BGR2HSI(vector<double> &bgr, vector<double> &hsi)
{
	double b = bgr[0], g = bgr[1], r = bgr[2];	//BGR
	double h = 0.0, s = 0.0, i = 0.0;			//HSI
	//����任��
	double k = (2 * r - g - b) / (pow(3.0, 0.5) * (g - b));
	double sita = pi / 2 - pow(tan(k), -1);
	//��ȡH
	if (g >= b) {
		h = sita;
	}
	else if (g < b) {
		h = sita + pi;
	}
	//��ȡS
	s = 2.0 / pow(6.0, 0.5) * pow((r - g)*(r - g) + (r - b)*(g - b), 0.5);
	//��ȡI
	i = (r + g + b) / pow(3, 0.5);



	//�ֶζ��巨
	/*
	double minBGR = *min_element(bgr.begin(), bgr.end());	//��Сֵ
	double maxBGR = *max_element(bgr.begin(), bgr.end());	//���ֵ
	int minPos = min_element(bgr.begin(), bgr.end()) - bgr.begin();	//��Сֵ�±�
	int maxPos = max_element(bgr.begin(), bgr.end()) - bgr.begin();	//���ֵ�±�

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
		h += 2.0 * pi;			//���H
	i = 0.5*(maxBGR + minBGR);	//���I
	if (i > 0 && i <= 0.5) {	//���S
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

//������������
double Func::pts2fDist(Point2f &pts1, Point2f &pts2)
{
	return pow(pow(pts1.x - pts2.x, 2) + pow(pts1.y - pts2.y, 2), 0.5);
}
//�����pts1��pts2���ƶ�����
/*
�ٶ�����Ϊ�����򣬽Ƕ�Ϊ0������ʱ��������360
*/
double Func::tgtMovDirec(Point2f &pts1, Point2f &pts2)
{
	double sita;	//�Ƕ� 0-360
	if (pts2.x < pts1.x && pts2.y <= pts1.y) {		//p2��p1����
		sita = 180 - atan((pts1.y - pts2.y) / (pts1.x - pts2.x)) * 180.0 / pi;
	}
	else if (pts2.x > pts1.x && pts2.y <= pts1.y) {	//p2��p1����
		sita = atan((pts1.y - pts2.y) / (pts2.x - pts1.x)) * 180.0 / pi;
	}
	else if (pts2.x < pts1.x && pts2.y > pts1.y) {	//p2��p1����
		sita = 270 - atan((pts2.y - pts1.y) / (pts1.x - pts2.x)) * 180.0 / pi;
	}
	else if (pts2.x > pts1.x && pts2.y > pts1.y) {	//p2��p1����
		sita = 360 - atan((pts2.y - pts1.y) / (pts2.x - pts1.x)) * 180.0 / pi;
	}
	else if (pts2.x == pts1.x && pts2.y < pts1.y) {	//p2��p1��
		sita = 90.0;
	}
	else if (pts2.x == pts1.x && pts2.y > pts1.y) {	//p2��p1��
		sita = 270.0;
	}
	else if (pts2.x == pts1.x && pts2.y == pts1.y) {//p2 == p1
		sita = 0.0;
	}
	return sita;
}
//�Ƚ���ɫ��������
double Func::colorDist(vector<double> &hsiMsg1, vector<double> &hsiMsg2)
{
	return pow(pow(hsiMsg1[0] - hsiMsg2[0], 2) +
				pow(hsiMsg1[1] - hsiMsg2[1], 2) +
				pow(hsiMsg1[2] - hsiMsg2[2], 2) +
				pow(hsiMsg1[3] - hsiMsg2[3], 2) +
				pow(hsiMsg1[4] - hsiMsg2[4], 2) +
				pow(hsiMsg1[5] - hsiMsg2[5], 2), 0.5);
}
//���۲�ͬ������ƶ�һ����
/*
targetsMovMsg1: �ĸ�Ԫ�أ����򡢾��롢x��y
*/
double Func::getmovOffset(vector<double> targetsMovMsg1, vector<double> targetsMovMsg2)
{
	double dx1 = 0.0, dy1 = 0.0;
	double dx2 = 0.0, dy2 = 0.0;
	double k = pi / 180.0;	//�Ƕ�->����

	dx1 = targetsMovMsg1[1] * cos(targetsMovMsg1[0] * k);
	dy1 = targetsMovMsg1[1] * sin(targetsMovMsg1[0] * k) * -1;

	dx2 = targetsMovMsg2[1] * cos(targetsMovMsg2[0] * k);
	dy2 = targetsMovMsg2[1] * sin(targetsMovMsg2[0] * k) * -1;

	return pow(pow(dx1 - dx2, 2) + pow(dy1 - dy2, 2), 0.5);
}



