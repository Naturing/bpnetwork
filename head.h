#ifndef HEAD_H
#define HEAD_H

#define DATA  23                                 /* 训练样本的数量 */
#define IN 4                                     /* 每个样本有多少输入变量 */
#define OUT 1                                    /* 每个样本有多少个输出变量 */
#define NEURON 38                                /* 神经元数量 */

#define TRUE 1
#define FALSE 0

/* 下列数据结构与函数均在mybp.c中定义，这里将它们开放给GA使用 */

extern double data_out[DATA][OUT];               /* 存储DATA个样本，每个样本OUT个输出 */
extern double input_weight[NEURON][IN];          /* 输入对神经元的权重 */
extern double output_weight[OUT][NEURON];        /* 神经元对输出的权重 */
extern double output_data[OUT];                  /* BP神经网络的输出 */

void comput_output(int var);                     /* BP中用于计算神经网络输出的函数，这里开放给GA做为适应度函数使用 */

void ga_interface(void);                         /* 遗传算法调用接口 */

#endif /* HEAD_H */
