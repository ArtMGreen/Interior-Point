import numpy as np

MAXIMIZATION = 1
MINIMIZATION = -1


# for debug purposes
def ask_initial_solution(A, b):
    while True:
        print("Please, enter initial solution inside feasible space: ", end="")
        solution = list(map(float, input().split()))
        solution = np.array(solution)
        answer = A @ solution
        if np.array_equal(answer, b):
            return solution
        print("Sorry, this solution is wrong :(\nTry again!")


def take_inputs():
    print("Optimization type (1 for maximization; -1 for minimization): ", end="")
    problem_type = int(input())

    print("Number of variables:", end=" ")
    var_num = int(input())

    print("Number of constraints:", end=" ")
    constr_num = int(input())

    print("Coefficients of the objective function:")
    objective_coefficients = list(map(float, input().split()))
    c = np.array(objective_coefficients)

    print("A matrix of coefficients of constraint function:")
    constraint_coefficients = [list(map(float, input().split())) for _ in range(constr_num)]
    A = np.array(constraint_coefficients)

    print("Right-hand side numbers:")
    rhs = list(map(float, input().split()))
    b = np.array(rhs)

    print("The approximation accuracy is ε=", end="")
    epsilon = float(input())

    print(A)
    initial_solution = ask_initial_solution(A, b)

    return [c, A, b, 0.5, problem_type, initial_solution, epsilon, var_num, constr_num]


def iteration(c, A, x, alpha, problem_type):
    m, n = A.shape
    I = np.eye(n)
    D = np.diag(x)
    A_scaled = A @ D
    c_scaled = D @ c
    P = I - A_scaled.T @ np.linalg.inv(A_scaled @ A_scaled.T) @ A_scaled
    c_p = P @ c_scaled

    if problem_type == MAXIMIZATION:  # maximization
        nu_candidate = np.min(c_p)
        if nu_candidate < 0:
            nu = np.absolute(nu_candidate)
        else:
            raise ArithmeticError("No negative values in the projected gradient")
    else:  # minimization, projected antigradient
        nu_candidate = np.min(-c_p)
        if nu_candidate < 0:
            nu = nu_candidate
        else:
            raise ArithmeticError("No positive values in the projected gradient")

    x_scaled = np.ones(n, float) + (alpha / nu) * c_p
    x_new = D @ x_scaled
    return x_new


def interior(c, A, x, alpha, problem_type, epsilon):
    print("--------------------------------------------------")
    solution = x
    print(f"Interior Point Algorithm for alpha={alpha}\n")
    iteration_counter = 0
    while True:
        print(f"Iteration {iteration_counter}: {solution}")
        solution_prev = solution
        try:
            solution = iteration(c, A, solution_prev, alpha, problem_type)
        except ArithmeticError as e:
            print(f"No more iteration can done, because of {e}")
            break

        if np.linalg.norm(solution - solution_prev) < epsilon:
            print("Stopped by stopping criterion")
            break
        iteration_counter += 1

    print(f"Last iteration ({iteration_counter + 1}):\n {solution}")
    print(f"Optimal value of objective function: {c @ solution}")
    print("--------------------------------------------------\n\n")
    return [solution, c @ solution]


np.seterr(all="ignore")

def simplex_iteration(A_nb, A_b, C_nb, C_b, b, problem_type, basic_var_nums, non_basic_var_nums):
    """

    :param A_nb: matrix containing non-basic variables coefficients from constraints

    :param A_b: matrix containing basic variables coefficients from constraints
                !also known as matrix B in Advanced Simplex Algorithm (ASA) from lecture 5

    :param C_nb: row vector containing non-basic variables coefficients from objective function

    :param C_b: row vector containing basic variables coefficients from objective function

    :param b: right hand vector b

    :param problem_type:  0/1 = minimization/maximization

    :param basic_var_nums:  list of current basic variables

    :param non_basic_var_nums:  list of current non-basic variables

    :return: optimal solution for LPP
    """
    # find inverse of A_b
    A_b_inv = np.linalg.inv(A_b)
    # find new basic solutions
    Xb = np.matmul(A_b_inv, b)

    # Also known as "Z_j - C_j = C_b_j * B_j_inverse * P_j - C_j" in ASA:
    temp_matrix = np.subtract(C_b @ A_b_inv @ A_nb, C_nb)

    if problem_type == 0:
        # search for entering variable if it is min problem
        enteringVar = np.argmax(temp_matrix)

        # if we can't improve objective function more => return its value
        if temp_matrix[enteringVar] < 0:
            return C_b, Xb, basic_var_nums, non_basic_var_nums
    else:
        # search for entering variable if it is max problem
        enteringVar = np.argmin(temp_matrix)

        # if we can't improve objective function more => return its value
        if temp_matrix[enteringVar] > 0:
            return C_b, Xb, basic_var_nums, non_basic_var_nums

    # update non-basic variables coefficients with respect to current pivot
    newA = np.matmul(A_b_inv, A_nb[:, enteringVar])

    # calculate ratios to determine new exiting (leaving) variable
    ratios = np.divide(Xb, newA)
    exitingVar = np.argmin(ratios[ratios > 0])
    ttrtrtrt = C_b @ Xb
    # swap entering and exiting variables into following matrices
    # var., what was non-basic before comes basic
    # var., what was basic before comes non-basic
    A_nb[:, enteringVar], A_b[:, exitingVar] = A_b[:, exitingVar].copy(), A_nb[:, enteringVar].copy()
    # do the same logic swap for objective function's vectors of coefficients
    C_b[exitingVar], C_nb[enteringVar] = C_nb[enteringVar], C_b[exitingVar]
    # and save the new numbers of basic variables
    basic_var_nums[exitingVar], non_basic_var_nums[enteringVar] = \
        non_basic_var_nums[enteringVar], basic_var_nums[exitingVar]

    # run new iteration
    return simplex_iteration(A_nb, A_b, C_nb, C_b, b, problem_type, basic_var_nums, non_basic_var_nums)


def simplex(C, A, b, problem_type):
    # n - number of variables
    # m - number of constraints
    m, n = A.shape
    non_basic_var_nums = np.arange(1, n + 1)
    basic_var_nums = np.arange(n + 1, n + m + 1)
    decision_vector = np.zeros(n)

    # on the zero iteration basic variable coefficients are unit vectors:
    B = np.identity(m)
    # and zeros in the objective function
    Cb = np.zeros(m)

    C_b, Xb, basic_var_nums, non_basic_var_nums = simplex_iteration(A, B, C, Cb, b,
                                                                    problem_type,
                                                                    basic_var_nums,
                                                                    non_basic_var_nums)

    mask = np.argsort(np.hstack((basic_var_nums, non_basic_var_nums)))
    Xb_extended = np.hstack((Xb, np.zeros(n)))
    decision_vector = Xb_extended[mask][:n]

    return C_b, Xb, basic_var_nums, non_basic_var_nums, decision_vector


def main():
    c, A, b, alpha, problem_type, initial_solution, epsilon, variables, constraints = take_inputs()
    solution_1, opt_1 = interior(c, A, initial_solution, alpha, problem_type, epsilon)
    solution_2, opt_2 = interior(c, A, initial_solution, 0.9, problem_type, epsilon)

    try:
        print()
        print()
        print("Simplex method application:")
        minmax = 0
        if problem_type == MAXIMIZATION:
            minmax = 1
        A_for_simplex = A[:, :variables - constraints]
        c_for_simplex = c[:variables - constraints]
        print(A_for_simplex)
        print(c_for_simplex)
        C_b, Xb, basic_var_nums, non_basic_var_nums, decision_vector = simplex(c_for_simplex,
                                                                               A_for_simplex,
                                                                               b,
                                                                               minmax)
        print(f"Basic variables' coefficients: {C_b}")
        print(f"Basic variable numbers: {basic_var_nums}")
        print(f"Basic variable values: {Xb}")
        print(f"Non-basic variable numbers: {non_basic_var_nums}")
        print(f"Decision vector: {decision_vector}")
        f = C_b @ Xb
        f *= 10 ** 3
        f = int(f)
        f = float(f) / 10 ** 3
        print(f"Objective function value: {f}")

        print()
        print()

        print("Here are the values from interior point algorithm again (when alpha = 0.5):")
        print(f"Last iteration solution:\n{solution_1}")
        print(f"Optimal value of objective function: {opt_1}\n")
        print("Here are the values from interior point algorithm again (when alpha = 0.9):")
        print(f"Last iteration solution:\n {solution_2}")
        print(f"Optimal value of objective function: {opt_2}\n")
    except Exception as e:
        print("The method is not applicable!")


'''
# debug inputs
var_num = 4
constr_num = 2
c = np.array([-1, -1, 0, 0], float)
A = np.array([[2, 4, 1, 0], [1, 3, 0, -1]], float)
b = np.array([16, 9], float)
epsilon = 0.00001
alpha = 0.5
problem_type = MINIMIZATION
initial: [0.5, 3.5, 1, 2]
'''

'''
TEST INPUT COPY:
1
6
3
1 5 3 0 0 0
1 12 5 1 0 0
7 9 2  0 1 0
9 1 7  0 0 1
45 26 12
0.00001
0.5 0.5 0.5 36 17 3.5
'''

if __name__ == '__main__':
    main()
