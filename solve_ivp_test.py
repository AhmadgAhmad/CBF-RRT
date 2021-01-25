from scipy.integrate import solve_ivp
import numpy as np


def external_arg(y):
    u = 0
    u = u + 2*y[0]**2
    print(u)
    return u

def model(t,y):
    u = external_arg(y)
    return [y[1], u]

def integrate_model():
    t_span_list = np.linspace(0,1,10)
    y0 = [1,0]
    solution = solve_ivp(fun=lambda t,y: model(t,y), t_span=[0,1], y0=y0, method = "RK45",
                        t_eval = t_span_list, dense_output=False, events=None)

    return solution.y[0], solution.y[1]


if __name__ == "__main__":
    result = integrate_model()
