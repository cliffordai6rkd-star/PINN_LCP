#  LCP经典形式
s_k = [q_k, v_k]

s_{k+1} = A s_k + B u_k + D lambda_k + d

0 <= lambda_k ⟂ E s_k + F lambda_k + H u_k + c >= 0

# 变量说明
s_k = [q_k, v_k]          # 系统状态
lambda_k = [gamma_k, lambda_k^n, lambda_k^t]

q_k         位置 / 广义坐标  # J_k^n * q_k ≈  phi(k) 通过雅可比将广义坐标映射到接触点法向位移
v_k         速度
u_k         控制输入
lambda_k^n 法向接触力/冲量
lambda_k^t 切向摩擦力/冲量
gamma_k    摩擦互补里的辅助变量:描述是否发生接触

h = Δt # 时间步长

# 状态更新方程
q_{k+1} = q_k + h v_{k+1}

v_{k+1} = v_k + h M^{-1}(q)[B_u u - C(q, v) + N lambda_k^n + T lambda_k^t]

令  f(q, v, u) = M^{-1}(q)(B_u u - C(q, v))
    N = M^{-1}(q*) J_n(q*)^T
    T = M^{-1}(q*) J_t(q*)^T

则:
v_{k+1} = v_k + h [
    f(q_k, v_k, u_k)
    + N lambda_k^n
    + T lambda_k^t
]

对f在(q*, v*, u*)处作线性化(一阶泰勒展开):
f(q_k, v_k, u_k) ≈ J_q q_k + J_v v_k + J_u u_k + d_v

其中:
J_q = ∂f/∂q |*
J_v = ∂f/∂v |*
J_u = ∂f/∂u |*

d_v = f(q*, v*, u*) - J_q q* - J_v v* - J_u u*

则:
v_{k+1}
= v_k
+ h J_q q_k
+ h J_v v_k
+ h J_u u_k
+ h N lambda_k^n
+ h T lambda_k^t
+ h d_v

带入 q_{k+1} = q_k + h v_{k+1} 有:
q_{k+1}
= q_k
+ h v_k
+ h^2 J_q q_k
+ h^2 J_v v_k
+ h^2 J_u u_k
+ h^2 N lambda_k^n
+ h^2 T lambda_k^t
+ h^2 d_v

= q_k (I + h^2 J_q)
+ v_k (h I + h^2 J_v)
+ u_k (h^2 J_u)
+ lambda_k (0 + h^2 N + h^2 T)
+ h^2 d_v

对比:
s_k = [q_k, v_k]          # 系统状态
lambda_k = [gamma_k, lambda_k^n, lambda_k^t]

s_{k+1} = A s_k + B u_k + D lambda_k + d

则;
A =
[ I + h^2 J_q      h I + h^2 J_v
  h J_q            I + h J_v     ]

B =
[ h^2 J_u
  h J_u   ]

D =
[ 0        h^2 N       h^2 T
  0        h N         h T   ]  # D的第一项为零是为了将gamma_k从状态更新方程中剔除

d =
[ h^2 d_v
  h d_v   ]

# 互补约束
0 <= gamma_k    ⟂  μ lambda_k^n - R lambda_k^t >= 0  # 约束1 摩擦锥约束 

0 <= lambda_k^n ⟂  phi(q_k) + J_n(q_k)(q_{k+1} - q_k) >= 0 # 约束2 法向约束

0 <= lambda_k^t ⟂  e_t^T gamma_k + J_t(q_k) v_{k+1} >= 0   # 约束3 切向约束

对约束1:
y_gamma = U lambda_k^n - R lambda_k^t (U:μ  R:e_t)

在 q* 附近 近似:
J_n(q_k) ≈ J_n(q*) = G_n 
J_t(q_k) ≈ J_t(q*) = G_t

又:phi(q_k) ≈ phi(q*) + G_n(q_k - q*)

y_n = phi(q_{k+1}) 
≈ phi(q*) + G_n(q_k - q*) + J_n(q_k)(q_{k+1} - q_k)

q_{k+1} = q_k + h v_{k+1} 

代入约束2:
y_n = phi(q_{k+1}) 
≈ phi(q*) + G_n(q_k - q*) + h G_n v_{k+1} 

同理约束3:
y_t = e_t^T gamma_k + G_t v_{k+1} 


令:

y_k = [y_gamma, y_n, y_t]

0 <= lambda_k ⟂ y_k >= 0

y_k = E s_k + F lambda_k + H u_k + c

对
y_gamma = U lambda_k^n - R lambda_k^t
则:
E_gamma = [0, 0]
F_gamma = [0, U, -R]
H_gamma = 0
c_gamma = 0



将v_{k+1}
= v_k + h J_q q_k + h J_v v_k
+ h J_u u_k
+ h N lambda_k^n + h T lambda_k^t
+ h d_v
带入法向约束y_n:

y_n ≈ phi(q*) + G_n(q_k - q*) + h G_n v_{k+1} 

y_n = phi(q*) + G_n(q_k - q*) + h G_n v_k
+ h G_n h J_q q_k + h G_n h J_v v_k
+ h G_n h J_u u_k
+ h G_n h N lambda_k^n + h G_n h T lambda_k^t
+ h G_n h d_v

= q_k (G_n + h^2 G_n J_q ) 
+ v_k (h G_n + h^2 G_n J_v)
+ lambda_k (0, h^2 G_n N, h^2 G_n T) 
+ u_k (h^2 G_n J_u)
+ phi(q*) - G_n q* + h^2 G_n d_v

则:
E_n =
[ G_n + h^2 G_n J_q,    h G_n + h^2 G_n J_v ]

F_n =
[ 0,    h^2 G_n N,    h^2 G_n T ]

H_n =
h^2 G_n J_u

c_n =
phi(q*) - G_n q* + h^2 G_n d_v


将v_{k+1}
= v_k + h J_q q_k + h J_v v_k
+ h J_u u_k
+ h N lambda_k^n + h T lambda_k^t
+ h d_v
代入切向约束y_t:

y_t = e_t^T gamma_k + G_t v_{k+1} 

y_t = e_t^T gamma_k + G_t v_k + G_t h J_q q_k + G_t h J_v v_k
+ G_t h J_u u_k
+ G_t h N lambda_k^n + G_t h T lambda_k^t
+ G_t h d_v

= q_k (h G_t J_q)
+ v_k (G_t + h G_t J_v )
+ lambda_k (e_t^T, h G_t N, h G_t T)
+ u_k (h G_t J_u)
+ h G_t d_v

则:
E_t =
[ h G_t J_q,    G_t + h G_t J_v ]

F_t =
[ e_t^T,    h G_t N,    h G_t T ]

H_t =
h G_t J_u

c_t  =
h G_t d_v


# LCP经典形式的系数矩阵展开式
综上所述:
E =
[ E_gamma
  E_n
  E_t ]

F =
[ F_gamma
  F_n
  F_t ]

H =
[ H_gamma
  H_n
  H_t ]

c =
[ c_gamma
  c_n
  c_t ]

A =
[ I + h^2 J_q      h I + h^2 J_v
  h J_q            I + h J_v     ]

B =
[ h^2 J_u
  h J_u   ]

D =
[ 0        h^2 N       h^2 T
  0        h N         h T   ]  # D的第一项为零是为了将gamma_k从状态更新方程中剔除

d =
[ h^2 d_v
  h d_v   ]

局部近似底层变量:
q*, v*, u* (可以拿到)

J_q = ∂f/∂q |*
J_v = ∂f/∂v |*
J_u = ∂f/∂u |*
d_v = f(q*, v*, u*) - J_q q* - J_v v* - J_u u*

G_n = J_n(q*)
G_t = J_t(q*)
N = M^{-1}(q*) J_n(q*)^T
T = M^{-1}(q*) J_t(q*)^T
phi(q*)


加入力的任务适合做一些精细操作,而不是那种视觉能够主导执行\只要力够大就能执行,因此Action head 对当前观测下的ee_pose判断应该准确  所以与img做cross attention的不应该只是ft   应该是三者结合  或者是img与eepose embed  再与ft做cross attention