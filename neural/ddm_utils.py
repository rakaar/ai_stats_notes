import numpy as np
from numba import jit, njit
from scipy import integrate
import pickle
from pybads import BADS

def run_bads_3param():
    # actual: v,a,w: 2, 10, 0.5. 
    lower_bounds = np.array([-10, 0.1, 0.1]) 
    upper_bounds = np.array([10, 15, 0.9])

    plausible_lower_bounds = np.array([-5, 1, 0.2])
    plausible_upper_bounds = np.array([5, 13, 0.8])

    v0 = np.random.uniform(plausible_lower_bounds[0], plausible_upper_bounds[0])
    a0 = np.random.uniform(plausible_lower_bounds[1], plausible_upper_bounds[1])
    w0 = np.random.uniform(plausible_lower_bounds[2], plausible_upper_bounds[2])
    x0 = np.array([v0, a0, w0]);

    options = {'display': 'off'}
    bads = BADS(bads_target_func, x0, lower_bounds, upper_bounds, plausible_lower_bounds, plausible_upper_bounds, options=options)
    optimize_result = bads.optimize()

    x_min = optimize_result['x']
    return x_min
    


def bads_target_func(params):
    v, a, w = params
    with open('sample_rt.pkl', 'rb') as f:
        RTs = pickle.load(f)
    probs = np.array([rtd_density_a(t, v, a, w) + rtd_density_a(t, -v, a, 1-w) for t in RTs])
    return -np.sum(np.log(probs))
    

@jit(nopython=True)
def rtd_density(t, mu, K_max=50):
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
def rtd_density_a(t, v, a, w, K_max=50):
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

    if sum_result < 0:
        sum_result += 1e-9
    
    density =  non_sum_term * sum_result
    if density < 0:
        print('+++++++++++++++++++++++++++++')
        print("t:", t, "v:", v, "a:", a, "w:", w, "sum_result:", sum_result, "non_sum_term:", non_sum_term, "\n")
        print(' sine term = ', sum_sine_term)

        print('+++++++++++++++++++++++++++++++')
        raise ValueError("Density cannot be negative")
    else:
        return density

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
