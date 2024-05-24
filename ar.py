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
        pred[i] = vector[:p] @ coef
        
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
        ar = vector[:p] @ coef_ar

        # predicción error
        ma = err[:q] @ coef_ma

        result[i] = ar + ma
        print(ar + ma)

        # agregar al vector
        vector = np.concatenate(([ result[i] ], vector))

        # agregar el error
        err = np.concatenate(([ ma ], err))


    return result

# NAR
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

        ajusta los pesos w según el gradiente
    """
    y_hat = v @ H
    err = (y_hat - Y).reshape((1,-1))

    delta = v.reshape((-1,1)) @ err * deriv_sigmoid(z)
    deriv = delta @ X.T

    w = w - lr * deriv

    return w

def again(series, a, b, vals):

    # modelo ar
    p = len(a)
    q = len(b)

    vector = series[:p].copy()[::-1]
    err = np.array([0])

    e_est =[]
    x_est =[]

    for i in range(q + vals):
        
        # Calculo parte ar
        ar = vector[:p] @ a


        # Calculo el error
        if i >= q :
            aux = err[:q] @ b
            e_est.append(aux)
            x_est.append( ar + aux )

        e = series[p + i] - ar
        x = series[p + i]

        err = np.concatenate(([ e ], err))
        vector = np.concatenate(([ x ], vector))

    print(e_est)
    print(x_est)


# Hibrido
def forecast_hibrido(series, w, v, a, vals = 5):

    p = len(a)
    q = w.shape[1]

    vector = series[:len(series)-vals-q].copy()[::-1]
    err = np.array([0])

    e_est = []
    x_est = []
    ars =[]

    for i in range( q + vals):

        # cálculo parte ar 
        ar = vector[:p] @ a

        e = series[p + i] - ar
        x = series[p + i]

        # cálculo del error
        if i >= q :
            f = w @ err[:q].reshape((-1,1))
            h = sigmoid(f)

            out = v @ h

            e_est.append(out[0])
            x_est.append((ar + out)[0])
            ars.append(ar)

        err = np.concatenate(([ e ], err))
        vector = np.concatenate(([ x ], vector))

    show(ars)
    show(e_est)
    show(x_est)
    
    return

series = np.array([0.49, 0.17, 0.98 ,0.71 ,0.50, 0.47, 0.06, 0.68 ,0.04, 0.07, 0.52, 0.10])
a = np.array([0.62 ,0.86 ,0.81 ,0.58])

w = np.array([
[0.2684, 1.8541, -0.2420],
[-2.5642, 1.0397, 0.1810]
])
v =np.array([-1.4683, -0.2257])

p = 4
q = 3
r = 2

forecast_hibrido(series, w, v, a)