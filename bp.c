#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "head.h"

#define TRAINC 10000                                               /* 训练次数上限 */
#define LEARN  0.2                                                 /* 学习率 */
#define ERROR 0.001                                                /* 误差 */
/*
 * 存放训练数据的文件 
 */
#define TRAIN_FILE_INPUT "./train_in.txt"
#define TRAIN_FILE_OUTPUT "./train_out.txt"

#define TEST_FILE "./test.txt"                                     /* 存放训练数据的文件 */
#define NEURON_WEIGHT "./neuron.txt"                               /* 存放训练后的权值 */
#define CMD_SIZE 10                                                /* 输入命令的最大长度 */

static double data_in[DATA][IN];                                   /* 存储DATA个样本，每个样本IN个输入 */
double data_out[DATA][OUT];                                        /* 存储DATA个样本，每个样本OUT个输出 */
double input_weight[NEURON][IN];                                   /* 输入对神经元的权重 */
double output_weight[OUT][NEURON];                                 /* 神经元对输出的权重 */
static double input_delta[NEURON][IN];                             /* 输入权重的修正量 */
static double output_delta[OUT][NEURON];                           /* 输出权重的修正量 */
static double activate[NEURON];                                    /* 神经元激活函数对外的输出 */
double output_data[OUT];                                           /* BP神经网络的输出 */
static double max_in[IN], min_in[IN], max_out[OUT], min_out[OUT];  /* 训练数据的最值，用于归一化 */
static double test_data[ALL_DATA - DATA][IN + OUT];                /* 存放测试数据 */
static double bp_out[ALL_DATA - DATA];                             /* 单bp的训练输出 */
static double bp_ga_out[ALL_DATA - DATA];                          /* bp-ga的训练输出 */
static double bp_ga_sa_out[ALL_DATA - DATA];                       /* bp-ga-sa的训练输出 */

/* 
 * 读训练数据 
 */
static void read_data(void) 
{
	FILE *fp_tmp;
	int i, j;
	if ((fp_tmp = fopen(TRAIN_FILE_INPUT, "r")) == NULL) {
		fprintf(stderr, "can not open the in file\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < DATA; i++) {
		for (j = 0; j < IN; j++)
			fscanf(fp_tmp, "%lf", &data_in[i][j]);
	}
	fclose(fp_tmp);

	if ((fp_tmp = fopen(TRAIN_FILE_OUTPUT, "r")) == NULL) {
		fprintf(stderr, "can not open the out file\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < DATA; i++) {
		for (j = 0; j < OUT; j++)
			fscanf(fp_tmp, "%lf", &data_out[i][j]);
	}
	fclose(fp_tmp);
}

/*
 *读测试数据
 */
static void read_test(void)
{
	FILE *fp_tmp;
	int i, j;
	if ((fp_tmp = fopen(TEST_FILE, "r")) == NULL) {
		fprintf(stderr, "can not open the test file\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < ALL_DATA - DATA; i++) {
		for (j = 0; j < IN + OUT; j++)
			fscanf(fp_tmp, "%lf", &test_data[i][j]);
	}
	fclose(fp_tmp);
}

/* 
 * 初始化BP神经网络 
 */
static void init_bpnetwork(void) 
{
	int i, j;

	for (i = 0; i < IN; i++) {
		min_in[i] = max_in[i] = data_in[0][i];
		for (j = 0; j < DATA; j++) {
			max_in[i]=max_in[i] > data_in[j][i] ? max_in[i] : data_in[j][i];
			min_in[i]=min_in[i] < data_in[j][i] ? min_in[i] : data_in[j][i];
		}
	}

	for (i = 0; i < OUT; i++) {
		min_out[i] = max_out[i] = data_out[0][i];
		for (j = 0; j < DATA; j++) {
			max_out[i] = max_out[i] > data_out[j][i] ? max_out[i] : data_out[j][i];
			min_out[i] = min_out[i] < data_out[j][i] ? min_out[i] : data_out[j][i];
		}
	}

	for (i = 0; i < IN; i++) {
		for (j = 0; j < DATA; j++) {
			data_in[j][i] = (data_in[j][i] - min_in[i] + 1) / (max_in[i] - min_in[i] + 1);
			//printf("%f ", data_in[j][i]);
		}
		//printf("\n");
	}

	for (i = 0; i < OUT; i++) {
		for (j = 0; j < DATA; j++) {
			data_out[j][i] = (data_out[j][i] - min_out[i] + 1) / (max_out[i] - min_out[i] + 1);
			//printf("init: %lf\n", data_out[j][i]);
		}
	}

	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {	
			input_weight[i][j] = rand() * 2.0 / RAND_MAX - 1;
			input_delta[i][j] = 0;
		}
	}

	for (i = 0; i < OUT; i++) {
		for (j = 0; j < NEURON; j++) {
			output_weight[i][j] = rand() * 2.0 / RAND_MAX - 1;
			output_delta[i][j] = 0;
		}
	}
}

/*
 * 计算输出 
 */
void comput_output(int var) 
{
	int i,j;
	double sum;
	for (i = 0; i < NEURON; i++) {
		sum = 0;
		for (j = 0; j < IN; j++) {
			sum += input_weight[i][j] * data_in[var][j];
		}
		activate[i] = 1 / (1 + exp(-1 * sum));
	}

	for (i = 0; i < OUT; i++) {
		sum = 0;
		for (j = 0; j < NEURON; j++) {
			sum += output_weight[i][j] * activate[j];
		}
		output_data[i]= sum;
	}
}

/* 
 * 反馈学习 
 */
static void back_update(int var) 
{
	int i, j;
	double tmp;
	for (i = 0; i < NEURON; i++) {
		tmp = 0;
		for (j = 0; j < OUT; j++) {
			tmp += (output_data[j] - data_out[var][j]) * output_weight[j][i];

			output_delta[j][i] = LEARN * output_delta[j][i] + LEARN * (output_data[j]-data_out[var][j]) * activate[i];
			output_weight[j][i] -= output_delta[j][i];
		}

		for (j = 0; j < IN; j++) {
			input_delta[i][j] = LEARN * input_delta[i][j] + LEARN * tmp * activate[i] * (1-activate[i]) * data_in[var][j];
			input_weight[i][j] -= input_delta[i][j];
		}
	}
}

/* 
 * 训练神经网络 
 */
static void  train_network(void) 
{
	int i, j, time = 0;
	double error;  /* 误差 */
	do {
		error = 0.0;
		for (i = 0; i < DATA; i++) {
			comput_output(i);
			for (j = 0; j < OUT; j++) {
				error += fabs((output_data[j] - data_out[i][j]) / data_out[i][j]);
			}
			back_update(i);
		}
		time++;
		//printf("BP: %d %lf\n",time, error / DATA);
	} while (time < TRAINC /* && error / DATA > ERROR */);
}

/* 
 * 将训练后的权值写入到文件中 
 */
static void write_neuron(void) 
{
	int i, j;
	FILE *fp;
	if ((fp = fopen(NEURON_WEIGHT, "w")) == NULL) {
		fprintf(stderr, "can not open the neuron file\n");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {
			fprintf(fp, "%lf ", input_weight[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	for (i = 0; i < OUT; i++) {	
		for (j = 0; j < NEURON; j++) {
			fprintf(fp, "%lf ", output_weight[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	for (i = 0; i < IN; i++) {
		fprintf(fp, "%lf ", max_in[i]);
	}
	fprintf(fp, "\n\n");

	for (i = 0; i < IN; i++) {
		fprintf(fp, "%lf ", min_in[i]);
	}
	fprintf(fp, "\n\n");

	for (i = 0; i < OUT; i++) {
		fprintf(fp, "%lf ", max_out[i]);
	}
	fprintf(fp, "\n\n");

	for (i = 0; i < OUT; i++) {
		fprintf(fp, "%lf ", min_out[i]);
	}
	fprintf(fp, "\n\n");

	fclose(fp);
}

/* 
 * 从文件中读取训练好的权值 
 */
static void read_neuron(void) 
{
	int i, j;
	FILE *fp;
	if ((fp = fopen(NEURON_WEIGHT, "r")) == NULL) {
		fprintf(stderr, "can not open the neuron file\n");
		exit(EXIT_FAILURE);
	}

	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {
			fscanf(fp, "%lf", &input_weight[i][j]);
		}
	}
	for (i = 0; i < OUT; i++) {	
		for (j = 0; j < NEURON; j++) {
			fscanf(fp, "%lf", &output_weight[i][j]);
		}
	}
	for (i = 0; i < IN; i++) {
		fscanf(fp, "%lf", &max_in[i]);
	}

	for (i = 0; i < IN; i++) {
		fscanf(fp, "%lf", &min_in[i]);
	}

	for (i = 0; i < OUT; i++) {
		fscanf(fp, "%lf", &max_out[i]);
	}

	for (i = 0; i < OUT; i++) {
		fscanf(fp, "%lf", &min_out[i]);
	}
	fclose(fp);
}

/* 
 * 输出权值，用于调试 
 */
static void print_weight(void) 
{
	int i, j;
	for (i = 0; i < NEURON; i++) {	
		for (j = 0; j < IN; j++) {
			printf("%lf ", input_weight[i][j]);
		}
		printf("\n");
	}
	printf("\n");

	for (i = 0; i < OUT; i++) {	
		for (j = 0; j < NEURON; j++) {
			printf("%lf ", output_weight[i][j]);
		}
		printf("\n");
	}
}

/* 
 * 测试训练后的网络 
 */
static double test_network(double *test_in) 
{
	int i, j;
	double sum;
	
	for (i = 0; i< IN; i++) {
		//printf("%lf\t", test_in[i]);
		test_in[i] = (test_in[i] - min_in[i] + 1) / (max_in[i] - min_in[i] + 1);
		//printf("%lf\t%lf\t%lf\n", test_in[i], min_in[i], max_in[i]);
	}
	for (i = 0; i < NEURON; i++) {
		sum = 0;
		for (j = 0; j < IN; j++) {
			sum += input_weight[i][j] * test_in[j];
		}
		activate[i] = 1 / (1 + exp(-1 * sum));
	}

	for (i = 0; i < OUT; i++) {
		sum = 0;
		for (j = 0; j < NEURON; j++) {
			sum += output_weight[i][j] * activate[j];
		}
		//printf("%lf\t%lf\t%lf\t%lf\n", sum, data_out[i], max_out[i], min_out[i]);
		//printf("%lf\n", sum * (max_out[i] - min_out[i] + 1) + min_out[i] - 1);
		return sum * (max_out[i] - min_out[i] + 1) + min_out[i] - 1;
	}
}

int main(int argc, char *argv[]) 
{
	char cmd[CMD_SIZE];	
	int i, j;
	double test_in[IN];
	FILE *fp_tmp;
	/*
	 * 测试程序运行时间
	 */
	struct timeval tpstart, tpend;
	float timeuse;

	printf("********** Bpnetwork Console **********\n");
	while (TRUE) {
		scanf("%s", cmd);
		if (!strcmp(cmd, "help")) {
			printf("train  训练神经网络\n");
			printf("test  展示测试结果\n");
			printf("draw  画出结果对比图\n");
			printf("exit  退出程序\n");
		} else if (!strcmp(cmd, "train")) {
			read_data();
			//printf("read_test\n");
			read_test();
			//printf("read_test_end\n");
			init_bpnetwork();

			printf("BP迭代开始\n");
			gettimeofday(&tpstart, NULL);
			train_network();
			for (i = 0; i < ALL_DATA - DATA; i++) {
				for (j = 0; j < IN; j++) {
					test_in[j] = test_data[i][j];
				}
				bp_out[i] = test_network(test_in);
			}
			gettimeofday(&tpend, NULL);
			timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
			timeuse /= 1000000;
			printf("BP迭代结束，用时：%f秒\n", timeuse);

			printf("BP-GA迭代开始\n");
			gettimeofday(&tpstart, NULL);
			ga_interface(0);
			train_network();
			for (i = 0; i < ALL_DATA - DATA; i++) {
				for (j = 0; j < IN; j++) {
					test_in[j] = test_data[i][j];
				}
				bp_ga_out[i] = test_network(test_in);
			}
			gettimeofday(&tpend, NULL);
			timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
			timeuse /= 1000000;
			printf("BP-GA迭代结束，用时：%f秒\n", timeuse);
			//sleep(3);

			printf("BP-GA-SA迭代开始\n");
			gettimeofday(&tpstart, NULL);
			ga_interface(1);
			train_network();
			for (i = 0; i < ALL_DATA - DATA; i++) {
				for (j = 0; j < IN; j++) {
					test_in[j] = test_data[i][j];
				}
				bp_ga_sa_out[i] = test_network(test_in);
			}
			gettimeofday(&tpend, NULL);
			timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
			timeuse /= 1000000;
			printf("BP-GA-SA迭代结束，用时：%f秒\n", timeuse);

			/*
			 * 将训练数据,bp,bp-ga,bp-ga-sa的输出数据写入到文件中，方便用gnuplot画图
			 */
			if ((fp_tmp = fopen("./train", "w")) == NULL) {
				fprintf(stderr, "can not open the train file\n");
				exit(EXIT_FAILURE);
			}
			for (i = 0; i < ALL_DATA - DATA; i++) {
				fprintf(fp_tmp, "%d %lf\n", i, test_data[i][IN + OUT -1]);
			}
			fclose(fp_tmp);

			if ((fp_tmp = fopen("./bp", "w")) == NULL) {
				fprintf(stderr, "can not open the bp.txt file\n");
				exit(EXIT_FAILURE);
			}
			for (i = 0; i < ALL_DATA - DATA; i++) {
				fprintf(fp_tmp, "%d %lf\n", i, bp_out[i]);
			}
			fclose(fp_tmp);

			if ((fp_tmp = fopen("./bp-ga", "w")) == NULL) {
				fprintf(stderr, "can not open the bp-ga.txt file\n");
				exit(EXIT_FAILURE);
			}
			for (i = 0; i < ALL_DATA - DATA; i++) {
				fprintf(fp_tmp, "%d %lf\n", i, bp_ga_out[i]);
			}
			fclose(fp_tmp);

			if ((fp_tmp = fopen("./bp-ga-sa", "w")) == NULL) {
				fprintf(stderr, "can not open the bp-ga-sa.txt file\n");
				exit(EXIT_FAILURE);
			}
			for (i = 0; i < ALL_DATA - DATA; i++) {
				fprintf(fp_tmp, "%d %lf\n", i, bp_ga_sa_out[i]);
			}
			fclose(fp_tmp);

		} else if (!strcmp(cmd, "test")) {
			/*
			 * 打印表头
			 */
			printf("\n     保持BP迭代10000次，不同方法对数据的预测情况对比\n");

			for (i = 0; i < 60; i++) {
				printf("=");
			}
			printf("\n");
			for (i = 0; i < 5; i++) {
				printf(" ");
			}
			printf("method");
			for (i = 0; i< 14; i++) {
				printf(" ");
			}
			printf("predicted");
			for (i = 0; i < 11; i++) {
				printf(" ");
			}
			printf("expected\n");
			for (i = 0; i < 60; i++) {
				printf("=");
			}
			printf("\n");
			/*
			 * 打印BP数据
			 */
			for (i = 0; i < ALL_DATA - DATA; i++) {
				if (i == 2) {
					for (j = 0; j < 5; j++) {
						printf(" ");
					}
					printf("BP");
					for (j = 0; j < 18; j++) {
						printf(" ");
					}
				} else {
					for (j = 0; j < 25; j++) {
						printf(" ");
					}
				}
				printf("%lf", bp_out[i]);
				for (j = 0; j < 10; j++) {
					printf(" ");
				}
				printf("%lf\n", test_data[i][IN + OUT -1]);
				if (i != 4) {
					for (j = 0; j < 25; j++) {
						printf(" ");
					}
					for (j = 0; j < 35; j++) {
						printf("-");
					}
					printf("\n");
				}
			}
			for (i = 0; i < 60; i++) {
				printf("=");
			}
			printf("\n");

			/*
			 * 打印BP-GA数据
			 */
			for (i = 0; i < ALL_DATA - DATA; i++) {
				if (i == 2) {
					for (j = 0; j < 5; j++) {
						printf(" ");
					}
					printf("BP-GA");
					for (j = 0; j < 15; j++) {
						printf(" ");
					}
				} else {
					for (j = 0; j < 25; j++) {
						printf(" ");
					}
				}
				printf("%lf", bp_ga_out[i]);
				for (j = 0; j < 10; j++) {
					printf(" ");
				}
				printf("%lf\n", test_data[i][IN + OUT -1]);
				if (i != 4) {
					for (j = 0; j < 25; j++) {
						printf(" ");
					}
					for (j = 0; j < 35; j++) {
						printf("-");
					}
					printf("\n");
				}
			}
			for (i = 0; i < 60; i++) {
				printf("=");
			}
			printf("\n");

			/*
			 * 打印BP-GA-SA数据
			 */
			for (i = 0; i < ALL_DATA - DATA; i++) {
				if (i == 2) {
					for (j = 0; j < 5; j++) {
						printf(" ");
					}
					printf("BP-GA-SA");
					for (j = 0; j < 12; j++) {
						printf(" ");
					}
				} else {
					for (j = 0; j < 25; j++) {
						printf(" ");
					}
				}
				printf("%lf", bp_out[i]);
				for (j = 0; j < 10; j++) {
					printf(" ");
				}
				printf("%lf\n", test_data[i][IN + OUT -1]);
				if (i != 4) {
					for (j = 0; j < 25; j++) {
						printf(" ");
					}
					for (j = 0; j < 35; j++) {
						printf("-");
					}
					printf("\n");
				}
			}
			for (i = 0; i < 60; i++) {
				printf("=");
			}
			printf("\n");
		} else if (!strcmp(cmd, "draw")) {
			system("./draw.sh");
		} else if (!strcmp(cmd, "exit")) {
			break;
		}
	}
	return 0;
}
