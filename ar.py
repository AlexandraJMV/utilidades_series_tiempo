# AR functionalities
import numpy as np

def ar_matrix(data, p):
    """
        Consruye una matriz autroregresiva dada la serie data y p (orden)
    """
    matrix = np.zeros((len(data) - p, p))

    for i in range(len(data) - p):
        matrix[i] = data[i: i + p][::-1]
    return data[p:], matrix

def foward(matrix, coef):
    """
        coef = vector de coeficientes
        matrix = matriz autoregresiva (n.data * componentes autoregresivos)

        retorna un vector vertical donde cada fila corresponde a una respuesta
    """
    f = np.dot(matrix, coef)
    return f

def pr_iterativo_AR(data, coef, vals):
    """
        calcula los (vals) valores futuros de una serie
        usando los coeficientes AR de forma iterativa.
    """
    pred = np.zeros(vals)
    vector = data.copy()
    p = len(coef)

    for i in range(vals):
        # predicción
        pred[i] = round(vector[:p] @ coef, 4)
        
        # agregar al vector
        vector = np.concatenate(([pred[i]], vector))

    return pred

# ARMA functionalities
def pr_iterativo_ARMA(data, coef_ar, coef_ma, vals):
    """
        calcula los (vals) valores futuros de una serie
        usando los coeficientes AR de forma iterativa.
    """
    result = np.zeros(vals)

    vector = data.copy()[::-1]
    p = len(coef_ar)
    q = len(coef_ma)

    # calculamos predicciones para crear el vector de errores
    series, matrix = ar_matrix(data, p)
    pred           = foward(matrix, coef_ar)

    err = (series - pred)[::-1]

    for i in range(vals):

        # predicción AR
        ar = round(vector[:p] @ coef_ar, 4)

        # predicción error
        ma = round(err[:q] @ coef_ma, 4)

        result[i] = ar + ma
        print(ar + ma)

        # agregar al vector
        vector = np.concatenate(([ result[i] ], vector))

        # agregar el error
        err = np.concatenate(([ ma ], err))


    return result

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x) )

def deriv_sigmoid(x):
    s = sigmoid(x)
    return s*(1-s)

def pseudo(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    S_inv = np.diag( 1 / S )
    V = Vt.T
    return V @ S_inv @ U.T

def adjust_v(series, q, w, C = 10**4):
    """
        q : cantidad de unidades en capa entrada
        r : cantidad de salidas no lineales

        q = 4, r = 3
        y = v1*h1 + v2*h2 + v3*h3
    """
    # Data input regresiva 
    Y, matrix = ar_matrix(series, q)
    X = matrix.T

    # valor intermedio z
    z = w @ X

    # H
    H = sigmoid(z)
    
    # Cálculo de v
    H_T = H.T

    HH_T = H @ H_T

    IC = np.diag([1] * HH_T.shape[0]) / C

    p_inv = pseudo( HH_T + IC)

    v = Y @ H_T @ p_inv

    return v, H, Y, z, X

def show(matrix):
    print(np.round(matrix, 4))

def adjust_w(w, v, H, Y, z, X, lr = 0.1):
    """
        w : pesos iniciales
        z : activación pre sigmoide
        X : data train (ar_matrix.T)
        Y : target values
    """
    y_hat = v @ H
    err = (y_hat - Y).reshape((1,-1))

    delta = v.reshape((-1,1)) @ err * deriv_sigmoid(z)
    deriv = delta @ X.T

    return w - lr * deriv

series = np.array([0.87, 0.35 ,0.69 ,0.29 ,0.53 ,0.83, 0.60 ,0.34])
w = np.array([
[0.2695, 1.8536, -0.2397, 0.0964],
[-2.5644, 1.0393, 0.1810, -0.8305],
[0.4659, 0.9109, 0.2442, -0.3523]
])

q = 4
r = 3

v, H, Y, z, X = adjust_v(series, q, w)
w1 = adjust_w(w, v, H, Y, z, X)
show(w1)
 