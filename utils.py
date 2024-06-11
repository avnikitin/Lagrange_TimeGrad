from gluonts.dataset import Dataset, DatasetCollection
from typing import List, Sequence, Optional
from numpy.typing import ArrayLike

import numpy as np


def omit_points(dataset: DatasetCollection,  p: float = 0.5) -> DatasetCollection:
    """
        Takes gluonts DatasetCollection in form 
            [{
                'target': np.array, 
                'start': pd.Period,
                'feat_static_cat': np.ndarray
            }]
        Returns DatasetCollection, where p * 100% of observations are replaced with NaNs.
    """
    assert 0. <= p <= 1., "Only float part from 0 to 1 of all of the points can be omitted"
    new_dataset: DatasetCollection = []
    for ts in dataset:
        new_ts: Dataset = {
            'target': np.array(ts['target']),
            'start': ts['start'],
            'feat_static_cat': ts['feat_static_cat']
        }
        for row in new_ts['target']:
            num_to_omit: int = int(np.round(row.size * p))
            points_to_omit: ArrayLike = np.random.choice(row.size, num_to_omit, replace=False)
            row[points_to_omit] = np.nan
        new_dataset.append(new_ts)
    return new_dataset


def interpolate_point_by_points(xs: Sequence[int],
                                ys: Sequence[float], 
                                X: int) -> float:
    """
        Interpolates Lagrange polynomial by points (xs[0], ys[0]), (xs[1], ys[1]), ...
        Returns value of this polynomial at point X
    """
    assert len(xs) == len(ys), f"xs and ys should have the same length however differ: {len(xs)} and {len(ys)} respectively"
    assert len(xs), f'There should be at least one point to interpolate'
    
    if len(xs) == 1: # 1 point means constant polynomial
        return ys[0]

    res: float = 0.0
    for i in range(len(xs)):
        l: float = 0.0
        for j in range(len(xs)):
            if j == i:
                continue
            assert xs[i] != xs[j], f"xs have to be different however there are two values equal to {xs[i]} at positions {i} and {j}"
            l += (X - xs[j]) / (xs[i] - xs[j])
        res += ys[i] * l
    return res


def interpolate_np_array(dataset: ArrayLike, 
                         num_points: int = 5,
                         neighbors_from: str = 'both',
                         fill_type: str = 'interp',
                         fill_value: float = 0.0):
    assert num_points > 0, "Positive number of points is required to interpolate"
    assert neighbors_from in ['left', 'right', 'both'], f'neighbors_from variable can be one of [left, right, both], however got {neighbors_from}'
    assert fill_type in ['interp', 'const'], f'fill_type can be one of [interp, const], however got {fill_type}'
    
    result: ArrayLike = np.zeros_like(dataset)
    
    for i in range(dataset.shape[0]):
        left: List[int] = []
        right: List[int] = []

        for j in range(dataset.shape[1]):
            if not np.isnan(dataset[i, j]):
                right.append(j)
                if len(right) == num_points:
                    break
        
        reached_end: bool = (j == dataset.shape[1])
        for j in range(dataset.shape[1]):
            if right and j == right[0]:
                if not reached_end:
                    k: int = right[-1] + 1
                    while k < dataset.shape[1]:
                        if not np.isnan(dataset[i, k]):
                            right.append(k)
                            break
                        k += 1
                    if k == dataset.shape[1]:
                        reached_end = True
                right = right[1:]
                        
            closest: List[int] = []
            if neighbors_from == 'left':
                closest = left
            elif neighbors_from == 'right':
                closest = right
            else:
                l: int = len(left) - 1
                r: int = 0
                while len(closest) < num_points and (-1 < l or r < len(right)):
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
            values: List[float] = [dataset[i, cl] for cl in closest]
            
            if len(closest) < num_points and fill_type == 'const':
                result[i, j] = fill_value
            else:
                result[i, j] = interpolate_point_by_points(closest, values, j)
            if not np.isnan(dataset[i, j]):
                left.append(j)
                if len(left) > num_points:
                    left = left[1:]
    return result
