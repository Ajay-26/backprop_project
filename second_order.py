# %%
import numpy as np
import random

seed = 1
np.random.seed(seed)
random.seed(seed)

n = 10000

#%%

def generate_data(n_points: int, eps: float = 1e-3):
    x = np.linspace(0,1,n_points)
    y = np.sin(x)
    error = eps* np.random.randn(n_points)
    y = y + error
    return x,y
    
x,y = generate_data(n)
# %%
def second_order_gradient_descent(x,y, lr=0.025,n_iters=1000):
    """_summary_
    This function performs gradient descent on the data using a quadratic function.
    We estimate y_pred = hx^2 + mx + b, and so the mean error is
    mean(y - y_pred)^2 = mean(y - (hx^2 + mx + b))^2 
    = mean(y^2 + h^2 x^4 + m^2x^2 + b^2 - 2hx^2y - 2mxy - 2by + 2mhx^3 + 2bhx^2 + 2bmx)
    To optimize, compute partial derivatives wrt h,m, and b
    partial_h = mean(2hx^4 - 2x^2y + 2mx^3 + 2bx^2)
    partial_m = mean(2mx^2 - 2xy + 2hx^3 + 2bx)
    partial_b = mean(2b - 2y + 2hx^2 + 2mx)
    
    Update h,m and b using the derivatives as so:
    h = h - alpha * partial_h
    m = m - alpha * partial_m
    b = b - alpha * partial_b
    

    Args:
        x (np.array): training points
        y (np.array): labels
    """
    h = np.random.randn()
    m = np.random.randn()
    b = np.random.randn()
    partial_h = 0
    partial_m = 0
    partial_b = 0

    for iter in range(n_iters):    
        partial_h = np.mean(2*h* x**4 - 2 * np.multiply((x**2),y) + 2*m* x**3 + 2*b * x**2)
        partial_m = np.mean(2*m* x**2 - 2 * np.multiply(x,y) + 2*h* x**3 + 2*b * x)
        partial_b = np.mean(2*b * np.ones_like(y) - 2 *y + 2*h* x**2 + 2*m * x)
        
        h = h - lr*partial_h
        m = m - lr*partial_m
        b = b - lr*partial_b
        
        y_pred = h * x**2 + m * x + b
        mse = np.sqrt(np.mean((y_pred - y)**2))
        print(f"Mean Square error for iteration {iter} = {mse}")
    coeffs = (h,m,b)
    return y_pred, coeffs

y_pred, coeffs = second_order_gradient_descent(x,y)