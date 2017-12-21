#include<iostream>
#include<cmath>
#include<time.h>

using std::cout;
using std::endl;

const int first = 784;	//784 = 28 * 28
const int second = 100;
const int third = 10;
const double alpha = 0.35;

int input[first];
int target[third];
double weight1[first][second];
double weight2[second][third];
double output1[second];
double output2[third];
double delta1[second];
double delta2[third];
double threshold1[second];	//ãÐÖµ
double threshold2[third];	//ãÐÖµ

double cal(double x) {
	return 1.0 / (1.0 + exp(-x));
}

int main() {
	//Initialize
	srand(time(0));
	for (int i = 0; i < first; i++) {
		for (int j = 0; j < second; j++) {
			weight1[i][j] = rand() % 1000 * 0.001 - 0.5;
		}
	}
	for (int i = 0; i < second; i++) {
		for (int j = 0; j < third; j++) {
			weight2[i][j] = rand() % 1000 * 0.001 - 0.5;
		}
	}
	for (int i = 0; i < second; i++) {
		threshold1[i] = rand() % 1000 * 0.001 - 0.5;
	}
	for (int i = 0; i < third; i++) {
		threshold2[i] = rand() % 1000 * 0.001 - 0.5;
	}

	//training
	FILE *image_train;
	FILE *image_label;
	image_train = fopen("train-images.idx3-ubyte", "rb");
	image_label = fopen("train-labels.idx1-ubyte", "rb");
	if (image_train == NULL || image_label == NULL) {
		cout << "can't open the file!" << endl;
		exit(0);
	}

	unsigned char image_buf[784];
	unsigned char label_buf[10];

	int otherinf[1000];
	fread(otherinf, 1, 16, image_train);
	fread(otherinf, 1, 8, image_label);

	int cnt = 0;
	cout << "Lighting Charm!" << endl;
	while (!feof(image_train) && !feof(image_label)) {
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, image_train);
		fread(label_buf, 1, 1, image_label);

		for (int i = 0; i < 784; i++) {
			if ((unsigned int)image_buf[i] < 128)
				input[i] = 0;
			else
				input[i] = 1;
		}
		
		//initialize the target output
		int answer = (unsigned int)label_buf[0];
		for (int i = 0; i < third; i++) {
			target[i] = 0;
		}
		target[answer] = 1;

		for (int i = 0; i < second; i++) {
			double sum = 0;
			for (int j = 0; j < first; j++) {
				sum += input[j] * weight1[j][i];
			}
			double x = sum + threshold1[i];
			output1[i] = cal(x);
		}

		for (int i = 0; i < third; i++) {
			double sum = 0;
			for (int j = 0; j < second; j++) {
				sum += output1[j] * weight2[j][i];
			}
			double x = sum + threshold2[i];
			output2[i] = cal(x);
		}

		for (int i = 0; i < third; i++) {
			delta2[i] = (output2[i]) * (1.0 - output2[i]) * (output2[i] - target[i]);
		}

		for (int i = 0; i < second; i++) {
			double sum = 0;
			for (int j = 0; j < third; j++) {
				sum += weight2[i][j] * delta2[j];
			}
			delta1[i] = (output1[i]) * (1.0 - output1[i]) * sum;
		}

		for (int i = 0; i < second; i++) {
			for (int j = 0; j < first; j++) {
				weight1[j][i] = weight1[j][i] - alpha * input[j] * delta1[i];
			}
			threshold1[i] = threshold1[i] - alpha * delta1[i];
		}

		for (int i = 0; i < third; i++) {
			for (int j = 0; j < second; j++) {
				weight2[j][i] = weight2[j][i] - alpha * output1[j] * delta2[i];
			}
			threshold2[i] = threshold2[i] - alpha * delta2[i];
		}

		cnt++;
		if (cnt % 1000 == 0) {
			cout << "training image: " << cnt << endl;
		}
	}
	cout << endl;

	//testing
	FILE *image_test;
	FILE *image_test_label;
	image_test = fopen("t10k-images.idx3-ubyte", "rb");
	image_test_label = fopen("t10k-labels.idx1-ubyte", "rb");
	if (image_test == NULL || image_test_label == NULL) {
		cout << "can't open the test file!" << endl;
		exit(0);
	}

	unsigned char test_image_buf[784];
	unsigned char test_label_buf[10];

	int _otherinf[1000];
	fread(_otherinf, 1, 16, image_test);
	fread(_otherinf, 1, 8, image_test_label);

	int test_count = 0;
	int success_count = 0;
	while (!feof(image_test) && !feof(image_test_label)) {
		memset(test_image_buf, 0, 784);
		memset(test_label_buf, 0, 10);
		fread(test_image_buf, 1, 784, image_test);
		fread(test_label_buf, 1, 1, image_test_label);

		for (int i = 0; i < 784; i++) {
			if ((unsigned int)test_image_buf[i] < 128)
				input[i] = 0;
			else
				input[i] = 1;
		}

		for (int i = 0; i < third; i++) {
			target[i] = 0;
		}
		int answer = (unsigned int)test_label_buf[0];
		target[answer] = 1;

		for (int i = 0; i < second; i++) {
			double sum = 0;
			for (int j = 0; j < first; j++) {
				sum += input[j] * weight1[j][i];
			}
			double x = sum + threshold1[i];
			output1[i] = cal(x);
		}

		for (int i = 0; i < third; i++) {
			double sum = 0;
			for (int j = 0; j < second; j++) {
				sum += output1[j] * weight2[j][i];
			}
			double x = sum + threshold2[i];
			output2[i] = cal(x);
		}

		double max_value = -9999999;
		int max_index = 0;
		for (int i = 0; i < third; i++) {
			if (output2[i] > max_value) {
				max_value = output2[i];
				max_index = i;
			}
		}
		if (target[max_index] == 1) {
			success_count++;
		}

		test_count++;

		if (test_count % 1000 == 0) {
			cout << "test num: " << test_count << " success: " << success_count << endl;
		}
	}
	cout << endl;
	cout << "The success rate: " << success_count * 1.0 / test_count * 1.0 << endl;
	return 0;
}