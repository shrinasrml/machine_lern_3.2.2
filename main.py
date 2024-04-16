import numpy as np
import math
import inspect
import matplotlib.pyplot as plt

N = 1000
x = np.linspace(0, 1, N)
X = x.copy()
np.random.shuffle(x)

z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
z2 = 20*np.sin(2*np.pi * 3 * X) + 100*np.exp(X)
error = 10 * np.random.randn(N)
t = z + error
t2 = z2 + error

basis_functions = [lambda x: math.sin(x), lambda x: math.cos(x), lambda x: math.exp(x),
                   lambda x: math.sqrt(x), lambda x: x**2, lambda x: x**3, lambda x: x**4, 
                   lambda x: x**5, lambda x: x**6, lambda x: x**7, lambda x: x**8, lambda x: x**9, lambda x: x**10,
                   lambda x: x**23, lambda x: x**18, lambda x: x**16, lambda x: x**34, lambda x: x**12, lambda x: x**19,
                   lambda x: x**33, lambda x: x**28, lambda x: x**26, lambda x: x**44, lambda x: x**22, lambda x: x**29,
                   lambda x: x**43, lambda x: x**38, lambda x: x**36, lambda x: x**54, lambda x: x**32, lambda x: x**39
                   ]

x_train = x[:int(0.6 * N)]
x_validation = x[int(0.6 * N):int(0.8 * N)]
x_test = x[int(0.8 * N):]

t_train = t[:int(0.6 * N)]
t_validation = t[int(0.6*N):int(0.8 * N)]
t_test = t[int(0.8 * N):]

models = 10

alpha = [1e-20, 1e-10, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

def getF(x, basis_functions):
    F = np.zeros((len(x), len(basis_functions)))

    for i in range(len(x)):
        for j in range(len(basis_functions)):
            F[i][j] = basis_functions[j](x[i])

    return F

def getW(F, t, alpha):
    w = np.linalg.inv((F.T @ F + alpha * np.identity(len(F[0])))) @ F.T @ t
    return w

def getY(F, w): return F @ w

def getE(Y, t):
    result = (1 / len(t)) * np.sum((t - Y) ** 2)
    return result

def generate_basis_functions(basis_functions):
    selected_basis_functions = [lambda x: 1]
    selected_basis_functions.extend(np.random.choice(basis_functions, size=np.random.randint(1, len(basis_functions) + 1), replace=False))
    return selected_basis_functions

def train_model(x_train, t_train, x_validation, t_validation, basis_functions, alpha):
    top_w = None
    top_basis_functions = None
    top_alpha = None
    top_E = float('inf')

    for _ in range(models):
        selected_basis_functions = generate_basis_functions(basis_functions)
        F_train = getF(x_train, selected_basis_functions)

        for a in alpha:
            w = getW(F_train, t_train, a)

            F_validation = getF(x_validation, selected_basis_functions)
            Y_validation = getY(F_validation, w)

            E = getE(t_validation, Y_validation)

            if E < top_E:
                top_w = w
                top_basis_functions = selected_basis_functions
                top_alpha = a
                top_E = E

    return top_w, top_basis_functions, top_alpha, top_E

res = train_model(x_train, t_train, x_validation, t_validation, basis_functions, alpha)

F_test = getF(x_test, res[1])
Y_test = getY(F_test, res[0])
test_E = getE(t_test, Y_test)

F2 = getF(X, res[1])
Y2 = getY(F2, res[0])

print(f'Лучшие веса: {res[0]}')
print(f'Лучшие базисные функции: {[inspect.getsource(temp) for temp in res[1]]}')
print(f'Лучшее значение коэффициента регуляризации: {res[2]}')
print(f'Среднеквадратичная ошибка на тестовой выборке: {res[3]}')

plt.figure()
plt.plot(X, z2, label = 'z2', color = 'navy')
plt.scatter(X, t2, label = 't', color = 'skyblue', s = 3)
plt.plot(X, Y2, label = 'y', color = 'red')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()