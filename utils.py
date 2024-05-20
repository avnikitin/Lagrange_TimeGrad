from typing import List
import numpy as np


def omit_points(dataset,  p: float = 0.5): # TODO: find gluonts.Dataset typehint
    """
    TODO: Add random state but not only one
    Takes gluonts dataset in form 
        [{
            'target': np.array, 
            'start': pd.Period,
            'feat_static_cat': np.ndarray
        }]
    """
    assert 0. <= p <= 1., "Only part from 0 to 1 of all of the points can be omitted"
    new_dataset = []
    for ts in dataset:
        new_ts = {
            'target': np.array(ts['target']),
            'start': ts['start'],
            'feat_static_cat': ts['feat_static_cat']
        }
        for row in new_ts['target']:
            num_to_omit = int(np.round(row.size * p))
            points_to_omit = np.random.choice(row.size, num_to_omit, replace=False)
            row[points_to_omit] = np.nan
        new_dataset.append(new_ts)
    return new_dataset


def interpolate_point_by_points(xs: List[int], ys: List[float], X: int) -> float:
    assert len(xs) == len(ys), f"xs and ys should have the same length however differ: {len(xs)} and {len(ys)} respectively"
    res = 0.0
    for i in range(len(xs)):
        l = 0.0
        for j in range(len(xs)):
            if j != i:
                assert xs[i] != xs[j], f"xs have to be different however there are two values equal to {x[i]} at positions {i} and {j}"
                l += (X - xs[j]) / (xs[i] - xs[j])
        res += ys[i] * l
    return res


def interpolate_np_array(dataset: np.array, NUM_POINTS: int = 5):
    result = np.zeros_like(dataset)
    
    for i in range(dataset.shape[0]):
        left = []
        right = []
        for j in range(dataset.shape[1]):
            if not np.isnan(dataset[i, j]):
                right.append(j)
                if len(right) == NUM_POINTS:
                    break
        reached_end: bool = False
        for j in range(dataset.shape[1]):
            if right and j == right[0]:
                if not reached_end:
                    for k in range(right[-1] + 1, dataset.shape[1]):
                        if not np.isnan(dataset[i, k]):
                            right.append(k)
                            break
                    if k == dataset.shape[1]:
                        reached_end = True
                right = right[1:]
                        
            closest = []
            l = len(left) - 1
            r = 0
            while len(closest) < NUM_POINTS:
                if r == len(right):
                    closest.append(left[l])
                    l -= 1
                elif l == -1:
                    closest.append(right[r])
                    r += 1
                elif j - left[l] <= right[r] - j:
                    closest.append(left[l])
                    l -= 1
                else:
                    closest.append(right[r])
                    r += 1
            values = [dataset[i, _] for _ in closest]
            result[i, j] = interpolate_point_by_points(closest, values, j)
            if not np.isnan(dataset[i, j]):
                left.append(j)
    return result
