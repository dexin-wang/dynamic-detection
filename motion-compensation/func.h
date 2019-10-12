#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include "iostream"

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace xfeatures2d;

class Func
{
public:
	Func() {};
	~Func() {};
	//使用说明
	void attention();
	//读视频文件
	void readVideo(String &videoFile);
	//读frame
	void readFrame();
	//SURF检测
	void surfDetect();
	//Knn匹配
	void BfMatch();
	//自适应外点滤除
	void filtForePts();
	//利用背景点进行背景补偿
	void makeBGD();
	//前景提取（帧差）
	void getFGD();	
	//绘制动态目标轮廓
	void drawContours();	

	//识别动态目标, 返回值为分类标签
	vector<String> recogMov(const Mat &img);


	//显示及数据更新
	void showAndUpdate();
	//按键检测
	int readKey();

	//显示鼠标当前位置
	void onMouse(int event, int x, int y, int flags, void* ustc);

	//最小二乘法求解仿射矩阵
	Mat myGetAffineTransform(vector<Point2f> src, vector<Point2f> dst);
	//计算距离残差
	void calcPtsErrs();
	//获取边角处特征点
	void getCornerPts(double dx, double dy);
	//计算边角点标准差
	double calPtsS();
	//小区域滤除
	void smallTargetFilte(Mat &img, int thresh);
	//目标离散区域归并
	void combinTarget(Mat &img_rgb, Mat &img_abs);
	//BGR转HSI
	void BGR2HSI(vector<double> &bgr, vector<double> &hsi);
	//计算两点间距离
	double pts2fDist(Point2f &pts1, Point2f &pts2);
	//比较颜色特征距离
	double colorDist(vector<double> &hsiMsg1, vector<double> &hsiMsg2);
	//计算从pts1到pts2的移动方向
	double tgtMovDirec(Point2f &pts1, Point2f &pts2);
	//评价不同物体的移动一致性
	double getmovOffset(vector<double> targetsMovMsg1, vector<double> targetsMovMsg2);

private:
	RNG rng;

	const float minRatio = 0.5f;						//BBF搜索参数
	const int L = 20;									//距离残差等级
	//const int m = 5;									//距离残差等级阈值

	VideoCapture capture;

	int width;			//视频帧宽
	int height;			//视频帧高

	Ptr<SURF> surfDetector;// = SURF::create(1000);

	Mat prev_frame, prev_gray;							//前一帧彩图、灰度图
	Mat prev_frame_tranf, prev_gray_tranf;				//前一帧背景补偿后的彩图、灰度图
	Mat cur_frame, cur_gray;							//当前帧
	Mat differ;											//帧差结果
	Mat differ_dst;										//帧差结果
	Mat frame_contours;									//cur_frame的复制，用于绘制轮廓
	Mat frame_features;									//cur_frame的复制，用于绘制特征点

	vector<KeyPoint> Pts[2];							//surf检测的特征点
	Mat desc[2];										//surf描述子

	FlannBasedMatcher matcher;							//匹配模型
	vector<vector<DMatch>> knnMatches[2];				//初始匹配结果

	vector<DMatch> BBFmatches;							//BBF搜索后的match
	vector<Point> BBFpts[2];							//BBF正向匹配点对（前一帧-》当前帧）
	vector<Point> BBFpts_rev[2];						//BBF正向匹配点对（当前帧-》前一帧）
	vector<Point2f> pts_opti[2];						//对称约束之后的点对
	vector<Point2f> pts_corner[2];						//四角的点对
	vector<Point2f> pts_BGD[2];							//自适应外点滤除后的背景点
	vector<Point2f> pts_FGD[2];							//自适应外点滤除后的前景点

	//char* mouseLoc;

	Mat H;												//六参数仿射矩阵

	vector<float> ptsErrs;								//距离残差
	vector<int>   ptsErrs_num;							//距离残差各级数量 n
	vector<float> ptsErrs_prob;							//距离残差各等级概率
	float PtsBGD_prob;									//背景点集概率和
	float PtsFGD_prob;									//前景点集概率和
	float PtsBGD_aver;									//背景点集均值
	float PtsFGD_aver;									//前景点集均值
	int th;												//距离残差阈值

	Mat kernal3 = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));		//形态学因子
	Mat kernal5 = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));		//形态学因子

	vector<vector<Point>> movContours;					//动态目标轮廓

	const vector<String> labels = { "background", "aeroplane", "bicycle",					//分类标签
						"bird", "boat", "bottle", "bus", "car",
						"cat", "chair", "cow", "diningtable",
						"dog", "horse", "motorbike", "person",
						"pottedplant", "sheep", "sofa", "train",
						"tvmonitor" };	
	const String SSD_model_file = "D:/develop_software/opencv_3.3.0/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.caffemodel";	//SSD网络模型
	const String SSD_txt_file = "D:/develop_software/opencv_3.3.0/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.prototxt";		//SSD描述文件
	//Net net = readNetFromCaffe(SSD_txt_file, SSD_model_file);
	const float meanVal = 127.5;
	const float scaleFactor = 0.00783f;
	int bolbWidth = 300;
	int bolbHeight = 300;
	const float confidence_threshold = 0.4;					//SSD自信阈值

	/*********** 目标区域预处理 ************/
	Mat masks;					//区域标记
	int border[5];				//缓存当前像素点的 左 右 左下 下 右下 五个像素值

	/**************** 离散区域归并 ***************/
	vector<Point2f> contour_center[2];		//记录相邻两帧每个区域的质心
	vector<vector<double>> hsiMsg[2];		//记录相邻两帧各轮廓的HSI颜色信息
	Mat prev_con;		//记录前一帧框图
};