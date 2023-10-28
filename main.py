import numpy as np


MAXIMIZATION = 1
MINIMIZATION = -1

# debug inputs
var_num = 4
constr_num = 2
c = np.array([-1,-1,0,0], float)
A = np.array([[2,4,1,0],[1,3,0,-1]], float)
b = np.array([16, 9], float)
epsilon = 0.00001
alpha = 0.5
problem_type = MINIMIZATION

# for debug purposes
def construct_initial_solution(A, b):
    eq_num, var_num = A.shape
    return np.array([0.5, 3.5, 1, 2], float)


def take_inputs():
  print("Number of variables:", end=" ")
  var_num = int(input())

  print("Number of constraints:", end=" ")
  constr_num = int(input())

  print("Coefficients of the objective function:")
  objective_coefficients = map(float, input().split())
  c = np.array(objective_coefficients)

  print("A matrix of coefficients of constraint function:")
  constraint_coefficients = [map(float, input().split()) for _ in range(constr_num)]
  A = np.array(constraint_coefficients)

  print("Right-hand side numbers:")
  rhs = map(float, input().split())
  b = np.array(rhs)

  print("The approximation accuracy is ε=", end="")
  epsilon = float(input())

  print("Learning rate is α=", end="")
  alpha = float(input())


def iteration(c, A, x, alpha, problem_type):
    m, n = A.shape
    I = np.eye(n)
    D = np.diag(x)
    A_scaled = A @ D
    c_scaled = D @ c
    P = I - A_scaled.T @ np.linalg.inv(A_scaled @ A_scaled.T) @ A_scaled
    c_p = P @ c_scaled

    if problem_type == MAXIMIZATION:    # maximization
        nu_candidate = np.min(c_p)
        if nu_candidate < 0:
            nu = np.absolute(nu_candidate)
        else:
            raise ArithmeticError("No negative values in the projected gradient")
    else:                               # minimization, projected antigradient
        nu_candidate = np.min(-c_p)
        if nu_candidate < 0:
            nu = nu_candidate
        else:
            raise ArithmeticError("No positive values in the projected gradient")

    x_scaled = np.ones(n, float) + (alpha/nu)*c_p
    x_new = D @ x_scaled
    return x_new

x = construct_initial_solution(A, b)
for iteration_num in range(100):
    print(f"Iteration {iteration_num}: {x}")
    x_prev = x
    x = iteration(c, A, x_prev, alpha, problem_type)
    if np.linalg.norm(x - x_prev) < epsilon:
        print("Stopped by stopping criterion")
        break
print(f"Last iteration ({iteration_num + 1}): {x}")
