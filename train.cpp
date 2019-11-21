


#include <Windows.h>
#include "cc_nb.h"
#include <iostream>
#include <pa_file\pa_file.h>
#include <highgui.h>
#include <fstream>
#include <mutex>
#include <thread>
#include "visualize.h"

using namespace cv;
using namespace std;
using namespace cc;
namespace L = cc::layers;

#define trainbatch		8
#define valbatch		10
#define numcache        8
#define threadnum		16
#define maxdatanum      80000
#define imageSize		112

string datarootdir = "./class_front"; 
bool g_pause = true;

void errorExit(const char* fmt, ...){
	va_list vl;
	va_start(vl, fmt);
	
	printf("error exit.\n");
	vprintf(fmt, vl);

	if (g_pause){
		system("pause");
		exit(0);
	}
}

//第一节单独卷积
cc::Tensor resnet_conv(const cc::Tensor& input, const vector<int>& kernel, const string& name, int stride = 1, bool has_relu = true, bool bias_term = true){

	auto x = L::conv2d(input, kernel, "same", { stride, stride }, { 1, 1 }, name);
	L::OConv2D* layer = (L::OConv2D*)x->owner.get();
	layer->bias_term = bias_term;
	layer->kernel_initializer.reset(new cc::Initializer());
	layer->kernel_initializer->type = "msra";

	x = L::batch_norm_only(x, "bn_" + name);
	x = L::scale(x, true, "scale_" + name);
	if (has_relu)
		x = L::relu(x, name + "_relu");
	return x;
}

//残差块卷积模块.构造一个分支卷积模块，左右通用
cc::Tensor resnet_conv_block(const cc::Tensor& input, const vector<int>& kernel, int innum1, string part1, int innum2, string part2, int stride = 1, bool has_relu = true, bool bias_term = true){

	auto x = L::conv2d(input, kernel, "same", { stride, stride }, { 1, 1 }, format("res%d", innum1) + part1 + format("_branch%d%s", innum2, part2.c_str()));
	L::OConv2D* layer = (L::OConv2D*)x->owner.get();
	layer->bias_term = bias_term;
	layer->kernel_initializer.reset(new cc::Initializer());
	layer->kernel_initializer->type = "msra";

	x = L::batch_norm_only(x, format("bn%d", innum1) + part1 + format("_branch%d%s", innum2, part2.c_str()));
	x = L::scale(x, true, format("scale%d", innum1) + part1 + format("_branch%d%s", innum2, part2.c_str()));
	if (has_relu)
		x = L::relu(x, format("res%d", innum1) + part1 + format("_branch%d%s_relu", innum2, part2.c_str()));

	return x;
}

//右侧分支模块，由三个卷积模块组成
cc::Tensor resnet_branch2(const cc::Tensor& input, int stride, int innum, int outnum, int stage, string part){

	auto right = input;
	right = resnet_conv_block(right, { 1, 1, innum }, stage, part, 2, "a", stride, true);
	right = resnet_conv_block(right, { 3, 3, innum }, stage, part, 2, "b", 1, true);
	right = resnet_conv_block(right, { 1, 1, outnum }, stage, part, 2, "c", 1, false);
	return right;
}

cc::Tensor resnet_block(const cc::Tensor& input, int n, int stride, int innum, int outnum, int numinner){

	auto x = input;
	{
		auto branch1 = resnet_conv_block(x, { 1, 1, outnum }, n, "a", 1, "", stride, false, true); //构建左侧分支
		auto branch2 = resnet_branch2(x, stride, innum, outnum, n, "a");
		auto out = L::add(branch1, branch2, format("res%da", n));
		x = L::relu(out, format("res%da_relu", n));
	};

	string buff[5] = { "b", "c", "d", "e", "f" };
	for (int i = 0; i < numinner; ++i){
		auto branch1 = x;
		auto branch2 = resnet_branch2(x, 1, innum, outnum, n, buff[i]);
		auto out = L::add(branch1, branch2, format("res%d%s", n, buff[i].c_str()));
		x = L::relu(out, format("res%d%s_relu", n, buff[i].c_str()));
	};
	return x;
}

cc::Tensor resnet50(const cc::Tensor& input, int numunit){

	auto x = input;
	{
		//cc::name_scope n("input");
		x = resnet_conv(x, { 7, 7, 64 }, "conv1", 2, true, true);
		x = L::max_pooling2d(x, { 3, 3 }, { 2, 2 }, { 0, 0 }, false, "pool1");
	};

	x = resnet_block(x, 2, 1, 64, 256, 2);
	x = resnet_block(x, 3, 2, 128, 512, 3);
	x = resnet_block(x, 4, 2, 256, 1024, 5);
	x = resnet_block(x, 5, 2, 512, 2048, 2);

	x = L::avg_pooling2d(x, { 1, 1 }, { 1, 1 }, { 0, 0 }, true, "pool5");
	x = L::dense(x, numunit, "fc1000", true);
	return x;
}

cc::Tensor vgg(const cc::Tensor& input){

	cc::Tensor x = input;
	int num_output = 64;
	for (int i = 1; i <= 5; ++i){
		int n = i <= 2 ? 2 : 3;
		n++;
		for (int j = 1; j < n; ++j){
			x = L::conv2d(x, { 3, 3, num_output }, "same", { 1, 1 }, { 1, 1 }, cc::f("conv%d_%d", i, j));
			x = L::relu(x, cc::f("relu%d_%d", i, j));
		}
		if (i != 5)
			x = L::max_pooling2d(x, { 2, 2 }, { 2, 2 }, { 0, 0 }, false, cc::f("pool%d", i));

		if (i < 4){
			num_output *= 2;
		}
	}
	return x;
}

struct DataItem{
	shared_ptr<mutex> lock_;
	shared_ptr<Blob> top0;
	shared_ptr<Blob> top1;
};


#define min(a, b)  ((a)<(b)?(a):(b))
#define max(a, b)  ((a)>(b)?(a):(b))
float randr(float mi, float mx){
	float acc = rand() / (float)RAND_MAX;
	return acc * (mx - mi) + mi;
}

int randr(int mi, int mx){
	if (mi > mx) std::swap(mi, mx);
	int r = mx - mi + 1;
	return rand() % r + mi;
}

namespace DataEnlarge{


	void adBrightness(Mat& img_src, float min, float max)
	{
		float alpha = randr((float)min, (float)max);
		int   beta = randr((int)-8, (int)8);
		img_src.convertTo(img_src, CV_8U, alpha, beta);
	}

	void cropImage(Mat& frame, float pad_scalar, int type){
		Point center;
		float minval;
		float maxval;
		Mat pad_image;
		if (type == 0){
			pad_image = Mat::zeros((1 + pad_scalar)*frame.rows, (1 + pad_scalar)*frame.cols, CV_8UC3);
			center.x = pad_image.cols*0.5;
			center.y = pad_image.rows*0.5;
			Rect roi = Rect(center.x - frame.cols*0.5, center.y - frame.rows*0.5, frame.cols, frame.rows)&Rect(0, 0, pad_image.cols, pad_image.rows);
			frame.copyTo(pad_image(roi));
			minval = min(1 - pad_scalar * 2, 1 + pad_scalar);
			maxval = max(1 - pad_scalar * 2, 1 + pad_scalar);
		}
		else{
			pad_image = frame.clone();
			center.x = pad_image.cols*0.5;
			center.y = pad_image.rows*0.5;
			minval = min(1 - pad_scalar * 2, 1);
			maxval = max(1 - pad_scalar * 2, 1);
		}

		Rect rcOut;
		rcOut.x = center.x - 0.5*randr((float)minval, (float)maxval)*frame.cols;
		rcOut.y = center.y - 0.5*randr((float)minval, (float)maxval)*frame.rows;
		rcOut.width = center.x + 0.5*randr((float)minval, (float)maxval)*frame.cols - rcOut.x;
		rcOut.height = center.y + 0.5*randr((float)minval, (float)maxval)*frame.rows - rcOut.y;
		frame = pad_image(rcOut&Rect(0, 0, pad_image.cols, pad_image.rows));
	}

	void addNoise(Mat& image, float min_scalar, float max_scalar){

		int min = int(100 * min_scalar);
		int max = int(100 * max_scalar);

		int type = randr(0, 1);
		int p = randr(min, max);
		for (int i = 0; i < image.rows; i++){
			for (int j = 0; j < image.cols; j++){
				if (randr(0, p) != 0)
					continue;

				if (type == 1){
					if (randr(0, 1) == 1){
						image.at<cv::Vec3b>(i, j)[0] = (uchar)(0);
						image.at<cv::Vec3b>(i, j)[1] = (uchar)(0);
						image.at<cv::Vec3b>(i, j)[2] = (uchar)(0);
					}
					else{
						image.at<cv::Vec3b>(i, j)[0] = (uchar)(255);
						image.at<cv::Vec3b>(i, j)[1] = (uchar)(255);
						image.at<cv::Vec3b>(i, j)[2] = (uchar)(255);
					}
				}
				else{
					image.at<cv::Vec3b>(i, j)[0] = (uchar)(randr(0, 255));
					image.at<cv::Vec3b>(i, j)[1] = (uchar)(randr(0, 255));
					image.at<cv::Vec3b>(i, j)[2] = (uchar)(randr(0, 255));
				}
			}
		}
	}

	void addCutOut(Mat& image, float min_scalar, float max_scalar, int maxNum){
		Mat dst = image.clone();
		int loop = randr(1, maxNum);
		for (int i = 0; i < loop; i++){
			Rect rcTmp;
			rcTmp.x = randr(0, image.cols);
			rcTmp.y = randr(0, image.rows);
			rcTmp.width = randr(min_scalar*image.cols, max_scalar*image.cols);
			rcTmp.height = randr(min_scalar*image.rows, max_scalar*image.rows);
			rcTmp = rcTmp&Rect(0, 0, image.cols, image.rows);
			image(rcTmp).setTo(0);
		}
	}

	void addRotateImage(Mat& image){
		double angle = randr(0, 360.0);
		cv::Point2f center(image.cols / 2, image.rows / 2);
		cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1);
		cv::Rect bbox = cv::RotatedRect(center, image.size(), angle).boundingRect();

		rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
		rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;

		cv::Mat dst;
		cv::warpAffine(image, image, rot, bbox.size());
	}

	void addMedianBlur(Mat& image, int filterSize){
		cv::medianBlur(image, image, filterSize);
	}

	void addGaussianBlur(Mat& image, int filterSize){
		float sigma = randr((float)0.1, (float)1.0);
		cv::GaussianBlur(image, image, Size(filterSize, filterSize), sigma, sigma);

	}

	void addMotionBlur(Mat& srcImg, int filterSize) {
		int size = filterSize;
		cv::Mat filter = cv::Mat::zeros(size, size, CV_8UC1);
		for (int i = 0; i < size; i++)filter.at<uchar>(i, i) = (uchar)1;
		int len = size / 2;
		for (int r = 0; r < srcImg.rows; r++) {
			for (int c = 0; c < srcImg.cols; c++) {
				//mask
				int red = 0, green = 0, blue = 0;
				for (int i = r - len; i <= r + len; i++) {
					for (int j = c - len; j <= c + len; j++) {
						if (i < 0 || j < 0 || i >= srcImg.rows || j >= srcImg.cols) continue;
						blue += ((int)srcImg.at<cv::Vec3b>(i, j)[0]) * ((int)filter.at<uchar>(i - (r - len), j - (c - len)));
						green += ((int)srcImg.at<cv::Vec3b>(i, j)[1]) * ((int)filter.at<uchar>(i - (r - len), j - (c - len)));
						red += ((int)srcImg.at<cv::Vec3b>(i, j)[2]) * ((int)filter.at<uchar>(i - (r - len), j - (c - len)));
					}
				}
				srcImg.at<cv::Vec3b>(r, c)[0] = (uchar)(blue / size);
				srcImg.at<cv::Vec3b>(r, c)[1] = (uchar)(green / size);
				srcImg.at<cv::Vec3b>(r, c)[2] = (uchar)(red / size);
			}
		}
	}

	void adBlur(Mat& img_src, int maxKenerl)
	{
		int nCount = 0;// randr(0, 1);
		if (nCount == 0){
			int kenerl_size = randr(1, 5);
			if (kenerl_size % 2 == 0)
				kenerl_size++;
			addMotionBlur(img_src, kenerl_size);
		}
		else{
			int kenerl_size = randr(1, maxKenerl);
			if (kenerl_size % 2 == 0)
				kenerl_size++;

			if (randr(0, 1) == 0){
				addMedianBlur(img_src, kenerl_size);
			}
			else{
				addGaussianBlur(img_src, kenerl_size);
			}
		}
	}

	void dataEnlarge(Mat &im){
		//一半概率做图像flip ,级联往下做
		if (randr(0, 1) == 1){
			if (randr(0, 2) == 1){
				flip(im, im, 0);
			}
			if (randr(0, 2) == 1){
				flip(im, im, 1);
			}
			if (randr(0, 2) == 1){
				flip(im, im, -1);
			}
			if (randr(0, 1) == 1){
				im = im.t();
			}
		}

		//对图像进行运动模糊,中值模糊,高斯模糊等操作
		if (randr(0, 1) == 0){
			adBlur(im, 5);
		}

		//对图像进行cutout操作
		/*if (randr(0, 1) == 0){
		addCutOut(im, 0.1, 0.4, 2);
		}*/

		//对图像进行亮度调整
		adBrightness(im, 0.9, 1.1);

		//对图像添加椒盐或者随机噪点
		/*if (randr(0, 2) == 0){
		addNoise(im, 0.1, 0.5);
		}*/

		//对图像进行旋转
		bool brotate = false;
		if (randr(0, 1) == 0){
			addRotateImage(im);
			brotate = true;
		}

		//对图像做crop
		int ntype = 0;
		if (brotate == true){
			ntype = 1;
		}
		if (randr(0, 1) == 0 || ntype == 1){
			cropImage(im, 0.2, ntype);
		}
	}

	bool mixFrame(Mat& frameIn, const Mat& framemix, float min_scalar, float max_scalar){
		if (min(frameIn.cols, frameIn.rows) <= 10 || framemix.empty()){
			return false;
		}

		float alpha = randr(min_scalar, max_scalar);
		float beta = (1.0 - alpha);
		Mat mixTmp = framemix.clone();
		if (mixTmp.size() != frameIn.size()){
			resize(mixTmp, mixTmp, frameIn.size());
		}
		addWeighted(frameIn, beta, mixTmp, alpha, 0.0, frameIn);
		return true;
	}
};


map<int, string> keymap = {
	{ 0, "电子秤" },
	{ 1, "铁架台" },
	{ 2, "橡皮导管" },
	{ 3, "试管" },
	{ 4, "导气管" },
	{ 5, "手" },
	{ 6, "夹子" },
	{ 7, "ph培养皿" },
	{ 8, "毛玻璃片" },
	{ 9, "酒精灯" },
	{ 10, "滴管" },
	{ 11, "烧杯" },
	{ 12, "无ph培养皿" },
	{ 13, "量筒" },
	{ 14, "称量纸" },
	{ 15, "硫酸瓶" },
	{ 16, "玻璃棒" },
	{ 17, "试管夹" },
	{ 18, "盆" },
	{ 19, "纸槽" },
	{ 20, "滤纸漏斗" },
	{ 21, "无滤纸漏斗" },
	{ 22, "试管口" },
	{ 23, "试管尾" },
	{ 24, "量筒口" },
	{ 25, "量筒尾" },
	{ 26, "人脸" }

};

map<string, int> labmap_ = {
	{ "电子秤",0 },
	{"铁架台", 1 },
	{ "橡皮导管", 2 },
	{ "试管", 3 },
	{ "导气管", 4 },
	{ "手", 5 },
	{ "夹子", 6 },
	{ "ph培养皿", 7 },
	{ "毛玻璃片", 8 },
	{ "酒精灯", 9 },
	{ "滴管", 10 },
	{ "烧杯", 11 },
	{ "无ph培养皿", 12 },
	{ "量筒", 13 },
	{ "称量纸", 14 },
	{ "硫酸瓶", 15 },
	{ "玻璃棒", 16 },
	{ "试管夹", 17 },
	{ "盆", 18 },
	{ "纸槽", 19 },
	{ "滤纸漏斗", 20 },
	{ "无滤纸漏斗", 21 },
	{ "试管口", 22 },
	{ "试管尾", 23 },
	{ "量筒口", 24 },
	{ "量筒尾", 25 },
	{ "人脸", 26 }
};


class trainData
{
public:
	trainData(){
		this->maplabel = labmap_;
		trainnum_ = 0;
		testnum_ = 0;

		float fTrainScalar = 0.9;     //训练测试比例
		vector<string> vec_floderpath;//文件夹路径
		vec_floderpath.push_back(datarootdir);



		string labelfile = datarootdir + "/labels.txt"; //label路径
		ifstream fin(labelfile, ios::in|ios::binary);
		if (!fin.is_open()){
			errorExit("无法加载labels文件: %s\n", labelfile.c_str());
		}

		//int label = 1;
		string line;
		//maplabel[background] = 0;


		int jpg_index = 0;
		map<string, vector<string>> mapSample;
		vector<string> pathset;
		while (getline(fin, line)){
			//标签+1
			if (jpg_index == 78203 || jpg_index == 78056 || jpg_index == 44619 || jpg_index == 47922 || jpg_index == 75952
				|| jpg_index == 76976 || jpg_index == 77686){
				jpg_index++;
				continue;
			}
			int k = stoi(line);
			string lab = keymap[k];
			string path = datarootdir + "/data/" + to_string(jpg_index) + ".jpg";
			if (mapSample.find(lab) == mapSample.end()){
				vector<string> pathset_tmp;
				pathset_tmp.emplace_back(path);
				mapSample.emplace(make_pair(lab, pathset_tmp));
			}
			else mapSample[lab].emplace_back(path);
			jpg_index++;
		}

		//读取每个文件夹下的所有样本，并拆分训练测试集

 
		for (auto& obj : mapSample){
			PaVfiles vfsjpg;

			random_shuffle(obj.second.begin(), obj.second.end());

			vector<string>vallist;
			int start = obj.second.size() - 1;
			int end = fTrainScalar*obj.second.size();
			for (int i = start; i >= end; i--){
				vallist.push_back(obj.second[i]);
				obj.second.erase(obj.second.end() - 1);
			}
			maptrainset[obj.first] = obj.second;
			mapvalset[obj.first] = vallist;

			trainnum_ += obj.second.size();
			testnum_ += vallist.size();
		}

		if (trainnum_ < 1 || testnum_ < 1){

			errorExit("测试或者训练图片为空.\n");
		}
	}

	map<string, vector<string>> & train(){
		return maptrainset;
	}

	map<string, vector<string>> & val(){
		return mapvalset;
	}

	int  trainnum(){
		return trainnum_;
	}

	int  valnum(){
		return testnum_;
	}

	int numClass(){
		return maplabel.size();
	}

	map<string, int> & labelMap(){
		return maplabel;
	}

private:
	int trainnum_, testnum_;
	map<string, int> maplabel;
	map<string, vector<string>> maptrainset;
	map<string, vector<string>> mapvalset;
};

shared_ptr<trainData> g_dataset;
class MyData : public cc::BaseLayer{

public:
	SETUP_LAYERFUNC(MyData);

	MyData()
		:BaseLayer(), batchs(5){
	}

	virtual ~MyData(){
		batchs.setEOF();

		if (work_.joinable())
			work_.join();
	}

	void readPartData(map<string, vector<string>> datasetinput, int phase){
		if (phase == PhaseTrain){
			datas.clear();
		}
		else{
			if (datas.size() != 0)
				return;
		}

		//有放回抽样,实现样本均衡化
		int perSingleNum = int(maxdatanum / (datasetinput.size() + 1e-5));
		map<string, int> labelmap = g_dataset->labelMap();
		vector<pair<string, int>> dataset;
		for (auto& obj : datasetinput){
			random_shuffle(obj.second.begin(), obj.second.end());
			if (labelmap.find(obj.first) == labelmap.end()){
				errorExit("并没有找到对应label: %s\n", obj.first.c_str());
			}

			int label = labelmap.at(obj.first);
			if (phase == PhaseTrain){
				for (int i = 0; i < min(perSingleNum, obj.second.size()); i++){
					string jpgpath = obj.second[i];
					dataset.push_back(make_pair(jpgpath, label));
				}
			}
			else{
				for (int i = 0; i < obj.second.size(); i++){
					string jpgpath = obj.second[i];
					dataset.push_back(make_pair(jpgpath, label));
				}
			}
		}

		vector<int> vec_index;
		for (int i = 0; i < dataset.size(); i++){
			vec_index.push_back(i);
		}
		random_shuffle(vec_index.begin(), vec_index.end());

		if (!ignoreFirstOutput_){
			if (phase == PhaseTrain){
				printf("Load Train [");
			}
			else{
				printf("Load Val");
			}
		}

		int lastlen = 0;
		int numWillLoadData = min(maxdatanum, dataset.size());

		auto printProcess = [&](){
			if (ignoreFirstOutput_)
				return;

			for (int i = 0; i < lastlen; ++i)
				printf("\b");

			float rate = datas.size() / (float)numWillLoadData;
			int numSymbol = 30 * rate;
			for (int i = 0; i < numSymbol; ++i)
				printf("=");

			if (numSymbol < 30){
				printf(">");
			}
			else{
				printf("=");
			}

			for (int i = numSymbol; i < 30; ++i)
				printf(" ");

			lastlen = 30 + 1;
			char outbuf[1000];
			sprintf(outbuf, "] %d / %d (%.2f %%)", datas.size(), numWillLoadData, rate * 100);
			lastlen += strlen(outbuf);
			printf("%s", outbuf);
		};
		//vector<Mat> c;
#pragma omp parallel for num_threads(threadnum)
		for (int i = 0; i < numWillLoadData; ++i){
			auto& item = dataset[vec_index[i]];
			Mat im = imread(item.first);
			//c.emplace_back(im);
			//cout << "read finished" + item.first << endl;
			int label = item.second;
			if (phase == PhaseTrain)
				DataEnlarge::dataEnlarge(im);

#pragma omp critical
			{
				//im.convertTo(im, CV_32F, 1 / 127.5, -1.0);
				im.convertTo(im, CV_32F, 1 / 128.0, -127.5 / 128.0);
				if (phase == PhaseTrain && randr(0, 2) == 0 && datas.size() > 1){
					DataEnlarge::mixFrame(im, datas[randr(0, datas.size() - 1)].first, 0.1, 0.2);
				}
				if (datas.size() % 1000 == 0 && datas.size()>0){
					printProcess();
				}
				datas.push_back(make_pair(im, label));
			}
		}
		printProcess();
		ignoreFirstOutput_ = false;
		printf("\n");
	}

	virtual void setup(const char* name, const char* type, const char* param_str, int phase, Blob** bottom, int numBottom, Blob** top, int numTop){

		map<string, vector<string>>  dataset;
		if (phase == PhaseTest){
			batch_size = valbatch;
			dataset = g_dataset->val();
			cout << "加载测试数据" << endl;
		}
		else{
			batch_size = trainbatch;
			dataset = g_dataset->train();
			cout << "加载训练数据" << endl;
		}

		top[0]->reshape(batch_size, 3, imageSize, imageSize);
		top[1]->reshape(batch_size, 1);

		allbatch.resize(numcache);
		for (int i = 0; i < numcache; ++i){
			allbatch[i].top0 = newBlob();
			allbatch[i].top1 = newBlob();
			allbatch[i].top0->reshapeLike(top[0]);
			allbatch[i].top1->reshapeLike(top[1]);
			allbatch[i].lock_.reset(new mutex());
		}

		work_ = thread([](MyData* _this, map<string, vector<string>> dataset, int phase){
			while (!_this->batchs.eof()){
				_this->readPartData(dataset, phase);

				vector<int> inds;
				for (int i = 0; i < _this->datas.size(); ++i)
					inds.push_back(i);
				std::random_shuffle(inds.begin(), inds.end());

				int cursor = 0;
				int epochs = 0;
				int threshold = 1;
				if (phase == PhaseTest)
					threshold = 1;

				while (epochs < threshold && !_this->batchs.eof()){
					for (int n = 0; n < numcache; ++n){
						std::unique_lock<mutex> l(*_this->allbatch[n].lock_.get());
						Blob* top0 = _this->allbatch[n].top0.get();
						Blob* top1 = _this->allbatch[n].top1.get();
						for (int i = 0; i < _this->batch_size; ++i){
							auto& item = _this->datas[inds[cursor]];

							top0->setData(i, item.first);

							*(top1->mutable_cpu_data() + i) = item.second;

							cursor++;
							if (cursor == inds.size()){
								cursor = 0;
								epochs++;
								std::random_shuffle(inds.begin(), inds.end());
							}
						}
						_this->batchs.push(&_this->allbatch[n]);
					}
				}
			}
		}, this, dataset, phase);
	}

	virtual void forward(Blob** bottom, int numBottom, Blob** top, int numTop){

		DataItem* out = nullptr;
		while (!batchs.eof() && !batchs.pull(out))
			std::this_thread::sleep_for(std::chrono::milliseconds(1));

		if (!batchs.eof() && out){
			std::unique_lock<mutex> l(*out->lock_.get());
			top[0]->copyFrom(out->top0.get());
			top[1]->copyFrom(out->top1.get());
		}
	}

private:
	bool ignoreFirstOutput_ = true;
	int cursor = 0;
	vector<pair<Mat, int>> datas;
	int batch_size = 0;
	ThreadSafetyQueue<DataItem*> batchs;
	vector<DataItem> allbatch;
	thread work_;
};

void deloldmodel(){

	PaVfiles vfs;
	paFindFiles(datarootdir.c_str(), vfs, "saved_*.caffemodel", false);

	if (!vfs.empty()){
		printf("是否要删除%d个模型:%s？(yes/no)：", vfs.size(), vfs[0].c_str());
		char input[100] = { 0 };
		if (scanf("%s", &input)){
			if (strcmp(input, "yes") == 0){
				for (int i = 0; i < vfs.size(); ++i){
					printf("delete: %s\n", vfs[i].c_str());
					remove(vfs[i].c_str());
				}
			}
		}
	}
}

//第一节单独卷积
cc::Tensor resnet18_conv(const cc::Tensor& input, const vector<int>& kernel, const string& name, int stride = 1, bool has_relu = true, bool bias_term = true){

	auto x = L::conv2d(input, kernel, "same", { stride, stride }, { 1, 1 }, name);
	L::OConv2D* layer = (L::OConv2D*)x->owner.get();
	layer->bias_term = bias_term;
	layer->kernel_initializer.reset(new cc::Initializer());
	layer->kernel_initializer->type = "msra";

	x = L::batch_norm_only(x, "bn_" + name);
	L::OBatchNorm* batchnorm = (L::OBatchNorm*)x->owner.get();
	batchnorm->moving_average_fraction = 0.9f;

	x = L::scale(x, true, "scale_" + name);
	if (has_relu)
		x = L::relu(x, name + "_relu");
	return x;
}

//残差块卷积模块.构造一个分支卷积模块，左右通用
cc::Tensor resnet18_conv_block(const cc::Tensor& input, const vector<int>& kernel, int innum1, string part1, int innum2, string part2, int stride = 1, bool has_relu = true, bool bias_term = false){

	auto x = L::conv2d(input, kernel, "same", { stride, stride }, { 1, 1 }, format("res%d", innum1) + part1 + format("_branch%d%s", innum2, part2.c_str()));
	L::OConv2D* layer = (L::OConv2D*)x->owner.get();
	layer->bias_term = bias_term;
	layer->kernel_initializer.reset(new cc::Initializer());
	layer->kernel_initializer->type = "msra";

	x = L::batch_norm_only(x, format("bn%d", innum1) + part1 + format("_branch%d%s", innum2, part2.c_str()));
	L::OBatchNorm* batchnorm = (L::OBatchNorm*)x->owner.get();
	batchnorm->moving_average_fraction = 0.9f;

	x = L::scale(x, true, format("scale%d", innum1) + part1 + format("_branch%d%s", innum2, part2.c_str()));
	if (has_relu)
		x = L::relu(x, format("res%d", innum1) + part1 + format("_branch%d%s_relu", innum2, part2.c_str()));

	return x;
}

//右侧分支模块，由三个卷积模块组成
cc::Tensor resnet18_branch2(const cc::Tensor& input, int stride, int innum, int outnum, int stage, string part){

	auto right = input;
	right = resnet18_conv_block(right, { 3, 3, innum }, stage, part, 2, "a", stride, true);
	right = resnet18_conv_block(right, { 3, 3, innum }, stage, part, 2, "b", 1, false);
	return right;
}

cc::Tensor resnet18_block(const cc::Tensor& input, int n, int stride, int innum, int numinner){

	auto x = input;
	{
		auto branch1 = resnet18_conv_block(x, { 1, 1, innum }, n, "a", 1, "", stride, false, false); //构建左侧分支
		auto branch2 = resnet18_branch2(x, stride, innum, innum, n, "a");
		auto out = L::add(branch1, branch2, format("res%da", n));
		x = L::relu(out, format("res%da_relu", n));
	};

	string buff[5] = { "b", "c", "d", "e", "f" };
	for (int i = 0; i < numinner; ++i){
		auto branch1 = x;
		auto branch2 = resnet18_branch2(x, 1, innum, innum, n, buff[i]);
		auto out = L::add(branch1, branch2, format("res%d%s", n, buff[i].c_str()));
		x = L::relu(out, format("res%d%s_relu", n, buff[i].c_str()));
	};
	return x;
}

cc::Tensor resnet18(const cc::Tensor& input, int numunit){

	auto x = input;
	{
		//cc::name_scope n("input");
		x = resnet18_conv(x, { 7, 7, 64 }, "conv1", 2, true, false);
		x = L::max_pooling2d(x, { 3, 3 }, { 2, 2 }, { 0, 0 }, false, "pool1");
	};

	x = resnet18_block(x, 2, 1, 64, 1);
	x = resnet18_block(x, 3, 2, 128, 1);
	x = resnet18_block(x, 4, 2, 256, 1);
	x = resnet18_block(x, 5, 2, 512, 1);

	x = L::avg_pooling2d(x, { 1, 1 }, { 1, 1 }, { 0, 0 }, true, "pool5");

#if 0
	x = L::dense(x, 1024, "fc5", true);
	L::ODense* denseLayer = (L::ODense*)x->owner.get();
	denseLayer->kernel_mult.reset(new cc::ParamSpecMult(1.0f, 1.0f));
	denseLayer->bias_mult.reset(new cc::ParamSpecMult(2.0f, 1.0f));
	denseLayer->weight_initializer.reset(new cc::Initializer("xavier"));
	denseLayer->bias_initializer.reset(new cc::Initializer(0, "constant"));
#endif

	x = L::dense(x, numunit, "fc", true);
	L::ODense* denseLayer = (L::ODense*)x->owner.get();
	denseLayer->kernel_mult.reset(new cc::ParamSpecMult(1.0f, 1.0f));
	denseLayer->bias_mult.reset(new cc::ParamSpecMult(2.0f, 1.0f));
	denseLayer->weight_initializer.reset(new cc::Initializer("xavier"));
	denseLayer->bias_initializer.reset(new cc::Initializer(0, "constant"));
	return x;
}


void stepEnd(OThreadContextSession* session, int step, float smoothed_loss){

	auto solver = session->solver();
	auto net = solver->net();
	Blob* a = net->blob("res5c_branch2c");
	//postBlob(a, "res5c_branch2c");

}

void main(){
	initializeVisualze();
	//Mat im = imread("./class_side/data/1171.jpg");

	system(format("title 训练resnet18，%s", datarootdir.c_str()).c_str());

	srand(3);
	cc::installRegister();
	INSTALL_LAYER(MyData);

	g_dataset.reset(new trainData());
	printf(
		"numClass: %d\n"
		"numTrain: %d\n"
		"numVal: %d\n",
		g_dataset->numClass(), g_dataset->trainnum(), g_dataset->valnum());

	auto data = L::data("MyData", { "image", "label" }, "data");
	auto image = data[0];
	auto label = data[1];
	int numClass = g_dataset->numClass();
	auto x = image;
	x = resnet50(x, numClass);

	auto deploy = L::input({ 1, 3, imageSize, imageSize }, "image");
	deploy = resnet50(deploy, numClass);
	deploy = L::softmax(deploy, "prob", false);

	auto loss = cc::loss::softmax_cross_entropy(x, label, "loss");
	auto accuracy = cc::metric::classifyAccuracy(x, label, "accuracy");
	vector<int> step_size = { 140000, 280000, 420000 };
	auto op = cc::optimizer::momentumStochasticGradientDescent(cc::learningrate::multistep(0.001, 0.5, step_size), 0.9);
	int epoch_iters = (int)(cvRound(g_dataset->trainnum() / (float)trainbatch));
	int train_epochs = max(100, ceil(1000 / (float)epoch_iters));
	op->max_iter = train_epochs * epoch_iters;
	op->display = 100;
	//op->snapshot = 10000;
	op->test_interval = epoch_iters;
	op->test_iter = (int)(ceil(g_dataset->valnum() / (float)valbatch));
	op->weight_decay = 0.0002f;
	op->test_initialization = false;
	op->device_ids = { 0 };
	//op->snapshot_prefix = "model_";
	//op->reload_weights = "E:/globaldata/resnet50/saved_iter[9230]_class[24]_loss[0.040126]_accuracy[0.987008].caffemodel";
	//op->minimizeFromFile("Resnet_train_val.prototxt");
	op->minimize({ loss, accuracy });
	//op->minimize({ loss});
	printf("%s\n", op->seril().c_str());
	printf("epoch_iters = %d\n", epoch_iters);
	system(format("title 训练resnet18[maxiter: %d, epochs: %d]，%s", op->max_iter.intval(), train_epochs, datarootdir.c_str()).c_str());


	if (!engine::caffe::buildGraphToFile({ deploy }, datarootdir + "/deploy.prototxt")){
		errorExit("deploy 无法保存.\n");
	}

	float lastQuality = 0;
	float lastScore = 0;
	int lastiter = 0;
	float lastTestLoss = 0;
	string lastsavedname;

	registerOnTestClassificationFunction([&](Solver* solver, float testloss, int index, const char* itemname, float itemscore){
		if (strcmp(itemname, "accuracy") == 0){

			float quality = itemscore - testloss * 0.1;
			if (quality >= lastQuality){
				if (!lastsavedname.empty())
					remove(lastsavedname.c_str());

				int iter = solver->iter();
				lastiter = iter;
				lastScore = itemscore;
				lastTestLoss = testloss;
				printf("iter: %d, loss: %f, %s: %f\n", iter, testloss, itemname, itemscore);
				string savedname = format("%s/saved_iter[%d]_class[%d]_loss[%f]_accuracy[%f].caffemodel", datarootdir.c_str(), iter, numClass, testloss, itemscore);
				solver->net()->saveToCaffemodel(savedname.c_str());
				lastsavedname = savedname;
				lastQuality = quality;

				if (itemscore >= 1){
					printf("accuracy 提前满足要求，退出.\n");
					solver->postEarlyStopSignal();
				}
			}
		}
	});

#if 1
	deloldmodel();
	double tick = getTickCount();

	cc::train::caffe::run(op, stepEnd, [&](OThreadContextSession* session){
		
	});



	tick = (getTickCount() - tick) / getTickFrequency() * 1000;
	printf("done，总耗时：%.2f 分钟, epoch: %d，最好结果：loss[%f], accuracy[%f], iter[%d]\n", tick / (1000 * 60), train_epochs, lastTestLoss, lastScore, lastiter);
	
	if(g_pause){
		system("pause");
	}
	destoryVisualze();
#else

	auto net = engine::caffe::buildNet({ loss, accuracy }, PhaseTest);
	net->weightsFromFile((datarootdir + "/saved_3420_loss[0.011347]_accuracy[0.993902].caffemodel").c_str());

	float allacc = 0;

	for (int i = 0; i < op->test_iter.intval(); ++i){
		net->forward();
		allacc += *net->blob("accuracy")->cpu_data();
	}
	allacc /= op->test_iter.intval();
	printf("accuracy: %f\n", allacc);
#endif
}




//int main(){
//	PaVfiles vfs;
//	string datarootdir = "D:/CC5.0-project/导线抽象实验/original正面/data/";
//	string resnetdeployPath = "D:/CC5.0-project/导线抽象实验/original正面/deploy.prototxt";
//	string resnetmodelPath = "D:/CC5.0-project/导线抽象实验/original正面/saved_iter[4002]_class[24]_loss[0.034976]_accuracy[1.000000].caffemodel";
//	paFindFiles(datarootdir.c_str(), vfs, "*.jpg");
//	shared_ptr<Net> net_cls = loadNetFromPrototxt(resnetdeployPath.c_str());
//	net_cls->weightsFromFile(resnetmodelPath.c_str());
//	int count = 0;
//	for (auto vs : vfs){
//		count++;
//		Mat image = imread(vs.c_str());
//		resize(image, image, Size(512, 512));
//		image.convertTo(image, CV_32F, 1 / 128.0, -127.5 / 128.0);
//		Blob* input_blob2 = net_cls->input_blob(0);
//		Blob* detection_out2 = net_cls->blob("prob");
//		input_blob2->setData(0, image);
//		net_cls->forward();
//		vector<float> vec_cls = vector<float>(detection_out2->mutable_cpu_data(), detection_out2->mutable_cpu_data() + detection_out2->count());
//
//		auto x = distance(vec_cls.begin(), max_element(vec_cls.begin(), vec_cls.end()));
//		string label_k = keymap[x];
//
//		cout << label_k << endl;
//		cout << vs << endl;
//		cout << count << endl;
//	}
//	vfs;
//}


