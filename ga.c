#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "head.h"


#define POPSIZE 50                                            /* 种群大小 */
#define MAXGENS 18000                                         /* 世代数 */
#define NVARS ((NEURON * IN + OUT * NEURON))                  /* 基因型个数*/
#define PXOVER 0.8                                            /* 交叉概率 */
#define PMUTATION 0.15                                        /* 突变概率 */
#define UPPER_BOUND 2.5                                       /* 基因值的上界 */
#define LOWER_BOUND -3.5                                      /* 基因值的下界 */
#define MAX_DOUBLE 1000.0                                     /* 确定一个double上界 */
#define MAX_FITNESS 998.8                                     /* 最大适应度，当最大的适应度达到这个值，停止迭代 */

static int generation;                  /* 目前是第几代 */

/* 种群成员结构 */

struct genotype { 
  double gene[NVARS];        /* 基因型 */
  double fitness;            /* 适应度 */
  double upper[NVARS];       /* 基因型数据上界 */
  double lower[NVARS];       /* 基因型数据下界 */
  double rfitness;           /* 相对适应度 */
  double cfitness;           /* 累积适应度 */
};

static struct genotype population[POPSIZE + 1];    /* 整个种群，population[POPSIZE]中存储最优个体的数据 */
static struct genotype newpopulation[POPSIZE + 1]; /* 新的种群，用来替代旧种群 */

/* 产生以low和high为边界的随机数 */

static double randval(double low, double high) 
{
	return ((double)(rand() % 1000) / 1000.0) * (high - low) + low;
}

/* 对种群数据结构进行初始化 */

static void init_population(void) 
{
int i, j;

for (i = 0; i < NVARS; i++) {
      for (j = 0; j < POPSIZE; j++) {
           population[j].fitness = 0;
           population[j].rfitness = 0;
           population[j].cfitness = 0;
           population[j].lower[i] = LOWER_BOUND;
           population[j].upper[i]= UPPER_BOUND;
           population[j].gene[i] = randval(population[j].lower[i], population[j].upper[i]);
		   //printf("gene: %lf\n", population[j].gene[i]);
	  }
	}
}

/* 复制基因型数据到BP网络的权值中 */

static void copy_gene_to_bpweight(double *gene, double input_weight[NEURON][IN], double output_weight[OUT][NEURON]) 
{	
	int i = 0, j, k;
	for (j = 0; j < NEURON; j++)
		for (k = 0; k < IN; k++)
			input_weight[j][k] = gene[i++];
	for (j = 0; j < OUT; j++)
		for (k = 0; k < NEURON; k++) 
			output_weight[j][k] = gene[i++];
}

/* 评估函数 */

static void evaluate(void) 
{
	int mem, i, j;
	double sum, error;

	for (mem = 0; mem < POPSIZE; mem++) {
		error = 0.0;
		copy_gene_to_bpweight(population[mem].gene, input_weight, output_weight);
	 	for (i = 0; i < DATA; i++) {
			comput_output(i);
		 	for (j = 0; j < OUT; j++)
				error += fabs((output_data[j] - data_out[i][j]) / data_out[i][j]);
		}
		/* error越小适应度越高，这里对error做个差值，使得fitness越高适应度越高 */
		population[mem].fitness = MAX_DOUBLE - error / DATA;
		//printf("mem: %d\tfitness: %lf\n", mem, population[mem].fitness);
	}
}

/* 寻找最优个体，并放到population[POPSIZE]中 */

static void keep_the_best(void) 
{
	int mem;
	int i;
	int cur_best = 0; /* 最优个体的索引 */

	population[POPSIZE].fitness = 0.0;
	for (mem = 0; mem < POPSIZE; mem++) {
		if (population[mem].fitness > population[POPSIZE].fitness) {
			cur_best = mem;
            population[POPSIZE].fitness = population[mem].fitness;
		}
	}

	/* 拷贝最优个体的基因 */
	for (i = 0; i < NVARS; i++)
		population[POPSIZE].gene[i] = population[cur_best].gene[i];
	printf("%d  %lf\n",generation, population[POPSIZE].fitness);
}

/* 前一个种群中最优个体存放在数组最后一位，如果目前种群的最优个体
 * 比前一个种群差，用前一个种群中最优个体替换目前种群的最差个体 */

static void elitist(void) 
{
	int i;
	double best, worst;             /* 最好以及最差适应度值 */
	int best_mem, worst_mem;        /* 最好以及最差个体索引 */

	best = population[0].fitness;
	worst = population[0].fitness;
	for (i = 0; i < POPSIZE - 1; i++) {
		if (population[i].fitness > population[i+1].fitness) {
			if (population[i].fitness >= best) {
				best = population[i].fitness;
                best_mem = i;
			}
            if (population[i+1].fitness <= worst) {
				worst = population[i+1].fitness;
				worst_mem = i + 1;
			}
		} else {
			if (population[i].fitness <= worst) {
				worst = population[i].fitness;
                worst_mem = i;
			}
            if (population[i+1].fitness >= best) {
				best = population[i+1].fitness;
                best_mem = i + 1;
			}
		}
	}

	/* 如果目前种群的最优个体比前一个种群的最优个体好，则把这个最优个体拷贝到
	 * population[POPSIZE]中，否则，用前一个种群的最优个体替换目前种群中最差的个体 */

	if (best >= population[POPSIZE].fitness) {
		for (i = 0; i < NVARS; i++)
			population[POPSIZE].gene[i] = population[best_mem].gene[i];
		population[POPSIZE].fitness = population[best_mem].fitness;
	} else {
		for (i = 0; i < NVARS; i++)
			population[worst_mem].gene[i] = population[POPSIZE].gene[i];
		population[worst_mem].fitness = population[POPSIZE].fitness;
	} 
	printf("%lf\n", population[POPSIZE].fitness);
}

/* 选出新的种群 */

static void select_newpopulation(void) 
{
	int mem, i, j, k;
	double sum = 0, p;

	/* 计算种群的适应度和 */
	for (mem = 0; mem < POPSIZE; mem++) {
		sum += population[mem].fitness;
	}

	/* 计算各个个体的相对适应度 */
	for (mem = 0; mem < POPSIZE; mem++) {
		population[mem].rfitness =  population[mem].fitness / sum;
	}

	/* 计算个体的累积适应度，用于赌轮 */
	population[0].cfitness = population[0].rfitness;
	for (mem = 1; mem < POPSIZE; mem++) {
		population[mem].cfitness =  population[mem-1].cfitness + population[mem].rfitness;
	}

	/* 使用累积适应度选出下一代个体 */
	for (i = 0; i < POPSIZE; i++) { 
		p = rand() % 1000 / 1000.0;
		if (p < population[0].cfitness)
			newpopulation[i] = population[0];
		else {
			for (j = 0; j < POPSIZE; j++)
				if (p >= population[j].cfitness && p < population[j+1].cfitness)
					newpopulation[i] = population[j+1];
		}
	}

	/* 选出新种群之后，拷贝 */
	for (i = 0; i < POPSIZE; i++)
		population[i] = newpopulation[i];
}

static void swap(double *x, double *y) 
{
	double temp;
	temp = *x;
	*x = *y;
	*y = temp;
}

/* 交叉两个个体的基因 */

static void xover(int one, int two) 
{
	int i;
	int point; /* 交叉点 */

	/* 选择交叉点 */
	if (NVARS > 1) {
		if (NVARS == 2)
			point = 1;
		else
			point = (rand() % (NVARS - 1)) + 1;
		for (i = 0; i < point; i++)
			swap(&population[one].gene[i], &population[two].gene[i]);
	}
}

/* 选出两个个体进行单点交叉 */

static void crossover(void) 
{
	int i, mem, one;
	int first  =  0;
	double x;

	for (mem = 0; mem < POPSIZE; mem++) {
		x = rand() % 1000 / 1000.0;
      if (x < PXOVER) {
		  first++;
		  if (first % 2 == 0)
			  xover(one, mem);
		  else
			  one = mem;
	  }
	}
}

/* 遍历个体所有基因，按照变异概率进行变异 */

static void mutate(void) 
{
	int i, j;
	double lbound, hbound;
	double x;

	for (i = 0; i < POPSIZE; i++)
		for (j = 0; j < NVARS; j++) {
			x = rand() % 1000 / 1000.0;
            if (x < PMUTATION) {
				lbound = population[i].lower[j];
				hbound = population[i].upper[j];
				population[i].gene[j] = randval(lbound, hbound);
			}
		}
}

/* 遗传算法的对外函数 */

void ga_interface(void) 
{
	init_population();                            /* 初始化种群数据结构 */
	evaluate();                                   /* 对初代进行评估 */
	keep_the_best();                              /* 寻找最优个体并保存 */
	for (generation = 1; population[POPSIZE].fitness < MAX_FITNESS; generation++) {
		printf("%d  ", generation);
		select_newpopulation();                   /* 选出新种群 */
		crossover();                              /* 个体基因交叉 */
		mutate();                                 /* 基因变异 */
		evaluate();                               /* 对新的种群进行评估 */
		elitist();                                /* 确保最优个体得以保存 */
	}
	/* 迭代结束后，将最优个体的基因拷贝到BP的权值中 */
	copy_gene_to_bpweight(population[POPSIZE].gene, input_weight, output_weight);
}

/*void read_data(void);
void init_bpnetwork(void);
void print_weight();
int main() {
	read_data();
	init_bpnetwork();
	ga_interface();
	int i;
	for (i = 0; i < NVARS; i++) {
		printf("%lf  ", population[POPSIZE].gene[i]);
	}
	printf("\n");
	print_weight();
	return 0;
}*/
