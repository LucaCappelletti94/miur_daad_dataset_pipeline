import numpy as np
from typing import Tuple, Dict


def balance_generic(array: np.ndarray, classes: np.ndarray, balancing_max: int, output: int, random_state:int=42)->Tuple:
    """Balance given arrays using given max and expected output class.
        arrays: np.ndarray, array to balance
        classes: np.ndarray, output classes
        balancing_max: int, maximum numbers per balancing maximum
        output: int, expected output class.
    """
    output_class_mask = np.array(classes == output)
    retain_mask = np.bitwise_not(output_class_mask)
    n = np.sum(output_class_mask)
    if n > balancing_max:
        datapoints_to_remove = n - balancing_max
        mask = np.ones(shape=n)
        mask[:datapoints_to_remove] = 0
        np.random.seed(random_state)
        np.random.shuffle(mask)
        output_class_mask[np.where(output_class_mask)] = mask
        array = array[np.logical_or(
            output_class_mask, retain_mask).reshape(-1)]
    return array


def umbalanced(training: Tuple, testing: Tuple)->Tuple:
    """Leave data as they are."""
    return training, testing


def balanced(training: Tuple, testing: Tuple, balancing_max: int)->Tuple:
    """Balance training set using given balancing maximum.
        *dataset_split:Tuple, Tuple of arrays.
        balancing_max: int, balancing maximum.
    """
    y_train = training[-1]

    new_training = []

    for array in training:
        for output_class in [0, 1]:
            array = balance_generic(
                array, y_train, balancing_max, output_class
            )
        new_training.append(array)

    return new_training, testing


def full_balanced(training: Tuple, testing: Tuple, balancing_max: int, rate: Tuple[int, int])->Tuple:
    """Balance training set using given balancing maximum.
        *dataset_split:Tuple, Tuple of arrays.
        balancing_max: int, balancing maximum.
        rate: Tuple[int, int], rates beetween the two classes.
    """
    training, testing = balanced(
        training, testing, balancing_max=balancing_max)
    y_test = testing[-1]
    new_testing = []

    for array in testing:
        for output_class in [0, 1]:
            opposite = 1 - output_class
            array = balance_generic(
                array, y_test,
                int(np.sum(y_test == opposite)*rate[opposite]/rate[output_class]), output_class)
        new_testing.append(array)

    return training, testing


balancing_callbacks = {
    "umbalanced": umbalanced,
    "balanced": balanced,
    "full_balanced": full_balanced
}


def get_balancing_kwargs(mode: str, positive_class: str, negative_class: str, settings: Dict):
    class_balancing = settings["class_balancing"]
    kwargs = {
        "umbalanced": {},
        "balanced": {
            "balancing_max": settings["max"]
        },
        "full_balanced": {
            "rate": (class_balancing.get(positive_class, 0), class_balancing.get(negative_class, 0)),
            "balancing_max": settings["max"]
        }
    }
    return kwargs[mode]


def balance(training: Tuple, testing: Tuple, mode: str, positive_class: str, negative_class: str, settings: Dict)->Tuple:
    global balancing_callbacks
    return balancing_callbacks[mode](training, testing, **get_balancing_kwargs(mode, positive_class, negative_class, settings))
