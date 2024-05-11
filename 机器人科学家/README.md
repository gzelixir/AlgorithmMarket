# PhySO

### 简介

使用RNN对数据点生成多条latex公式进行物理回归

输入：

一个带数据点及单位的csv，特征集zip文件需要命名为data.zip且无子文件夹

输出：

一个scv，生成多个latex公式及公式各项系数，以及公式的打分。 由于模型评估页面背后的log_metric 不支持输入公式，所以需要在平台的日志中进行查看结果


### csv格式规范

参考输入为mechanical.csv：

    x_name:z,x_name:vz,y_name:E,free_const:m,free_const:g,fix_const:2

    "[1, 0, 0]","[1, -1, 0]","[2, -2, 1]","[0, 0, 1]","[1, -2, 0]","[0, 0, 0]"

    -6.225153514337833,3.650977062068705,-71.51530639913841,,,

    -8.061125154909156,-7.628976755434175,-31.196610274732166,,,

    -9.101492871859389,-8.019623387980694,-37.32040628876261,,,


x_name: 自变量名称
y_name: 因变量名称（只可以指定一个column）
free_const：可变参数名称
fix_const：固定参数

第一行为物理单位，单位是一个七位(或少于七位)的列表表示:

[length, time, mass, temperature, electric current, amount of light, amount of matter]
[长度、时间、质量、温度、电流、光量、物质量]

例如：[1, 0, 0] 表示长度， [1, -1, 0]表示（长度/时间）= 速度


参考用例：

    mlflow run PhySO -P data=mechanical.csv



输出：

```

output:    complexity  length    reward          rmse        r2  \
0           4       4  0.510261  8.449494e+01  0.078823   
1           5       5  0.614976  5.511752e+01  0.608023   
2           7       7  0.614976  5.511752e+01  0.608023   
3          19      19  1.000000  1.207096e-05  1.000000   
4          20      20  1.000000  4.228530e-07  1.000000   

                                                               expression  \
0                                                         (((vz)**(2))*m)   
1                                                               (z*(m*g))   
2                                                         ((g*m)*(2.0*z))   
3  ((((sin(2.0)/(((vz)**(2))*(2.0*m))))**(-1))+(2.0*(m*(g/((z)**(-1))))))   
4                 ((m/-(2.0))*((((((m)**(-1))*vz)*m)*(2.0*vz))--((g*z))))   

                                                       expression_prefix  \
0                                                          [mul n2 vz m]   
1                                                        [mul z mul m g]   
2                                                [mul mul g m mul 2.0 z]   
3    [add inv div sin 2.0 mul n2 vz mul 2.0 m mul 2.0 mul m div g inv z]   
4  [mul div m neg 2.0 sub mul mul mul inv m vz m mul 2.0 vz neg mul g z]   

           g         m  
0   1.000000  0.665660  
1   3.349737  3.349737  
2   2.368622  2.368622  
3  10.777552  0.681973  
4  19.600000 -1.500000  

```

complexity： 复杂度

length： 长度

reward： 奖励函数结果

rmse： root mean square error， 评估标准

r2： r2， 评估标准

expression： latex公式，可以复制在 https://www.latexlive.com/ 中显示

g: 输入设定的可变参数

m: 输入设定的可变参数


