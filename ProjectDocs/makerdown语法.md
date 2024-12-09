# Markdown语法记录

## 公式
### 分数
$\frac{x}{y}$
```bash
\frac{x}{y}
```
### 矩阵
$\begin{bmatrix}
1 & 1 & \cdots & 1 \\
1 & 1 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1 \\
\end{bmatrix} 
\begin{pmatrix}
1 & 1 & \cdots & 1 \\
1 & 1 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1 \\
\end{pmatrix}$
```bash
\begin{bmatrix}
1 & 1 & \cdots & 1 \\
1 & 1 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1 \\
\end{bmatrix}

\begin{pmatrix}
1 & 1 & \cdots & 1 \\
1 & 1 & \cdots & 1 \\
\vdots & \vdots & \ddots & \vdots \\
1 & 1 & \cdots & 1 \\
\end{pmatrix}
```
### 方程
$\begin{cases}
x=\rho\cos\theta \\
y=\rho\sin\theta \\
\end{cases}$
```bash
\begin{cases}
x=\rho\cos\theta \\
y=\rho\sin\theta \\
\end{cases}
```

### 开平方
```bash
\sqrt{2}
```
$\sqrt{2}$

### 累加 
$\sum_{k=1}^n\frac{1}{k}$

$\displaystyle\sum_{k=1}^n\frac{1}{k}$
```bash
\sum_{k=1}^n\frac{1}{k}  
\displaystyle\sum_{k=1}^n\frac{1}{k}
```

### 累乘 
$\prod_{k=1}^n\frac{1}{k}$

$\displaystyle\prod_{k=1}^n\frac{1}{k}$
```bash
\prod_{k=1}^n\frac{1}{k}
\displaystyle\prod_{k=1}^n\frac{1}{k}
```

### 积分 
$\displaystyle \int_0^1x{\rm d}x$

$\iint_{D_{xy}}$

$\iiint_{\Omega_{xyz}}$

```bash
\displaystyle \int_0^1x{\rm d}x
\iint_{D_{xy}}
\iiint_{\Omega_{xyz}}
```

## 符号表
![Alt text](/images/latex.png)


## 表格

| 日期（左对齐） | 任务（右对齐） | 完成度（居中对齐） |
| :-----| ----: | :----: |
| 2023-11-04 | 学习VScode | 50% |
| 2023-11-05 | 学习VScode | 100% |