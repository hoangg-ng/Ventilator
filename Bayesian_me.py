import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective, plot_evaluations

# Global variables
cycle_num = 33+1
Vt = 500
Ti = 1
Te = 3
T = Ti + Te
dt = 0.001
t = np.arange(0, T + dt, dt)
if len(t) > 1 and (t[-1] - t[-2]) != 0:
    t = t[:-1]
time = np.arange(0, cycle_num * (T + dt), dt)
if len(time) > 1 and (time[-1] - time[-2]) < dt:
    time = time[:-1]
n = len(t)
pi = np.pi
index = 1
count_cycle = 0
max = 15

reference = []
actual = []
error = []
cost_value = []

# x_train = np.empty([0, 2])
# y_train = np.empty([0, 1])
A = np.array([[0, 1], [-2500 / 3, -175 / 3]])
B = np.array([[0], [442500]])

def objective_function(params):
    Kp, Kd = params
    global index, ymax, count_cycle, A, B

    r = np.zeros(n)
    y = np.zeros(n)
    u = np.zeros(n)
    e = np.zeros(n)

    cost1 = 0
    Xprev = np.array([[0], [0]])

    for i in range(1, n):
        if i * dt <= Ti:
            y[i] = Xprev[0, 0] * dt + y[i - 1]
            
            X = (A.dot(Xprev) + B * u[i - 1]) * dt + Xprev
            Xprev = X

            r[i] = Vt * (i * dt) / Ti
            e[i] = r[i] - y[i]
            derivative = (e[i] - e[i - 1]) / dt
            u[i] = Kp * e[i] + Kd * derivative
            if u[i] > 5:
                u[i] = 5
            elif u[i] < 0:
                u[i] = 0
            # print(u[i])

            cost1 += abs(e[i]) * dt
            if i * dt == Ti:
                ymax = y[i]
        else:
            t_ex = i * dt - Ti
            y[i] = ymax * np.exp(-2 * t_ex)
            r[i] = Vt * np.exp(-2 * t_ex)

    cost2 = (np.sum(np.abs(np.diff(u))) - 0) * 10 / 500
    cost = 1 * cost1 + 0 * cost2
    print(f'Iteration {index} - cost value = {cost}')
    index += 1

    reference.append(r)
    actual.append(y)
    error.append(e)
    cost_value.append(cost)
    count_cycle += 1

    return cost

space = [
    Real(0.01, 4.8, name='Kp'),
    Real(0.01, 0.08, name='Kd')
]

x_initial = [[0.51, 0.025]]

while True:
    result = gp_minimize(objective_function, space, n_calls=max, n_initial_points=1,
                         initial_point_generator=None, base_estimator='gp', acq_func='LCB', x0=x_initial, y0=None,
                         random_state=42, xi=0.04, kappa=0.05)
    # Extract results
    Kp_opt, Kd_opt = result.x
    x_initial = [[Kp_opt, Kd_opt]]
    print(f'Optimized Kp: {Kp_opt}, Kd: {Kd_opt}')
    print(f'Minimum cost: {result.fun}')
    Kp_vals = [x[0] for x in result.x_iters]
    Kd_vals = [x[1] for x in result.x_iters]

    while count_cycle < max+3:
        J = objective_function(result.x)
        Kp_vals.append(Kp_opt)
        Kd_vals.append(Kd_opt)

    A = np.array([[0, 1], [-1500 / 3, -220 / 3]])
    B = np.array([[0], [380500]])
    while count_cycle < cycle_num:
        J = objective_function(result.x)
        Kp_vals.append(Kp_opt)
        Kd_vals.append(Kd_opt)
        if J >= 0.17:
            result1 = gp_minimize(objective_function, space, n_calls=max, n_initial_points=1,
                                  initial_point_generator=None, base_estimator='gp', acq_func='LCB', x0=x_initial,
                                  y0=None, random_state=42, xi=0.04, kappa=0.05)
            Kp_opt, Kd_opt = result1.x
            print(f'Optimized Kp: {Kp_opt}, Kd: {Kd_opt}')
            print(f'Minimum cost: {result1.fun}')
            Kp_vals.extend([x[0] for x in result1.x_iters])
            Kd_vals.extend([x[1] for x in result1.x_iters])
        # if count_cycle == cycle_num:
        #     break
    break

# Extract the parameter and objective values at each iteration
iterations = np.arange(1, cycle_num+1)
# Kp_vals = [x[0] for x in result.x_iters]
# Kd_vals = [x[1] for x in result.x_iters]
# func_vals = result.func_vals

# EXPORT TO EXCEL
# data = {
#     'Kp': Kp_vals,
#     'Kd': Kd_vals,
#     'Cost': func_vals
# }
# df = pd.DataFrame(data)
# df.to_excel('C:/Users/trung/Desktop/bayesian_optimization_results.xlsx', index=False)
# # Plot the parameter values over iterations
# plt.figure()
# plt.plot(iterations, Kp_vals, label='Kp')
# plt.plot(iterations, Kd_vals, label='Kd')
# plt.xlabel('Iteration')
# plt.ylabel('Parameter Value')
# plt.legend()
# plt.title('Parameter values over iterations')

# Plot the objective function value over iterations
plt.figure()
plt.plot(iterations, cost_value, label='Objective Function')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.title('Objective function value over iterations')

references = np.concatenate(reference)
errors = np.concatenate(error)
actuals = np.concatenate(actual)
plt.figure()
plt.plot(time, actuals)
plt.plot(time, references, ls=':')
plt.xlabel('Time (s)')
plt.ylabel('Volume')

plt.figure()
plt.plot(time, errors)
plt.xlabel('Time (s)')
plt.ylabel('Error')

# plot_evaluations(result)
# plot_convergence(result)
# plot_objective(result)
plt.show()