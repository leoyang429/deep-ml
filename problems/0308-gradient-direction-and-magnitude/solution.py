import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    grad = np.array(gradient)
    if np.all(np.abs(grad) < 1e-7):
        return {'magnitude': 0.0,
                'direction': np.zeros_like(grad),
                'descent_direction': np.zeros_like(grad)}
    magnitude = np.linalg.norm(grad)
    direction = grad / magnitude
    descent_direction = -direction
    return {'magnitude': magnitude,
            'direction': direction,
            'descent_direction': descent_direction}