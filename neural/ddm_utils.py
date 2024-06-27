import numpy as np
from numba import jit, njit
from scipy import integrate


@jit(nopython=True)
def rtd_density(t, mu, K_max=100):
    k_vals = np.linspace(0, K_max, K_max+1)
    sum_neg_1_term = (-1)**k_vals
    sum_two_k_term = (2*k_vals) + 1

    if t > 0.25:
        non_sum_term = np.pi/2 * np.cosh(mu) * np.exp(-(mu**2)*t/2)
        sum_exp_term = np.exp(-(((2*k_vals + 1)**2) * (np.pi**2) * t)/8)
    else:
        non_sum_term = 2*np.cosh(mu)*np.exp(-(mu**2)*t/2)*(1/np.sqrt(2 * np.pi * t**3))
        sum_exp_term = np.exp(-((2*k_vals + 1)**2)/(2*t))

    sum_term = np.sum(sum_neg_1_term*sum_two_k_term*sum_exp_term)

    return non_sum_term*sum_term
        

def prob_rt(t_arr, v):
    N_t = len(t_arr)
    prob_arr = np.zeros((N_t-1,1))
    for i in range(0, N_t-1):
        prob_arr[i] = integrate.quad(rtd_density, t_arr[i], t_arr[i+1], args=(v))[0]
    return prob_arr


# def prob_rt(t_arr, v, a, w, K_max=100):
#     N_t = len(t_arr)
#     prob_arr = np.zeros((N_t-1,1))
#     for i in range(0, N_t-1):
#         prob_arr[i] = integrate.quad(rtd_density, t_arr[i], t_arr[i+1], args=(v,a,w,K_max))[0]
#     return prob_arr


def calculate_histogram(x_axis, y_axis):
    x_axis = np.sort(x_axis)
    histcounts, _ = np.histogram(y_axis, bins=x_axis)
    prob = histcounts/np.sum(histcounts)
    return prob

def parse_sim_results(results):
    choices =  [r[0] for r in results]
    rts = [r[1] for r in results]
    return choices, rts


@njit
def simulate_ddm(v, a, dt=1e-5):
    # assume starting pt = 0, sigma = 1, boundaries -a, +a
    DV = 0; t = 0; dB = dt**0.5 
    while True:
        DV += v*dt + np.random.normal(0, dB)
        t += 1

        if DV >= a/2:
            return +1, t*dt
        elif DV <= -a/2:
            return -1, t*dt
        

@jit(nopython=True)
def rtd_density_a(t, v, a, w, K_max=100):
    if t > 0.25:
        non_sum_term = (np.pi/a**2)*np.exp(-v*a*w - (v**2 * t/2))
        k_vals = np.linspace(1, K_max, K_max)
        sum_sine_term = np.sin(k_vals*np.pi*w)
        sum_exp_term = np.exp(-(k_vals**2 * np.pi**2 * t)/(2*a**2))
        sum_result = np.sum(k_vals * sum_sine_term * sum_exp_term)
    else:
        non_sum_term = (1/a**2)*(a**3/np.sqrt(2*np.pi*t**3))*np.exp(-v*a*w - (v**2 * t)/2)
        K_max = int(K_max/2)
        k_vals = np.linspace(-K_max, K_max, 2*K_max + 1)
        sum_w_term = w + 2*k_vals
        sum_exp_term = np.exp(-(a**2 * (w + 2*k_vals)**2)/(2*t))
        sum_result = np.sum(sum_w_term*sum_exp_term)

    return non_sum_term * sum_result

def prob_rt_a(t_arr,v,a,w):
    N_t = len(t_arr)
    prob_arr = np.zeros((N_t-1,1))
    for i in range(0, N_t-1):
        prob_arr[i] = integrate.quad(rtd_density_a, t_arr[i], t_arr[i+1], args=(v,a,w))[0]
    return prob_arr

def prob_hit_low_bound(v,a,w):
    if v == 0:
        return 1 - w
    else:
        return (1 - np.exp(-2*v*a*(1-w)))/(np.exp(2*v*w*a) - np.exp(-2*v*a*(1-w)))
