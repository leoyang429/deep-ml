# --- cell 1 ---
import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter
    
    beta1, beta2, eps, weight_decay = 0.9, 0.999, 1e-8, 1e-2
    if 'm' not in state:
        state['m'] = np.zeros_like(grad)
    if 'v' not in state:
        state['v'] = np.zeros_like(grad)
    if 't' not in state:
        state['t'] = 0
    m, v, t = state['m'], state['v'], state['t']
    t += 1
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_h, v_h = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
    
    new_param = (1 - lr * weight_decay) * param
    new_param -= lr * m_h / (v_h ** 0.5 + eps)
    
    state['m'], state['v'], state['t'] = m, v, t
    
    return new_param, state