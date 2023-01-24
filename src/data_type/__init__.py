from typing import Dict, Any, Iterator, Tuple
from itertools import product


def grid_search_dict(org_params: Dict[str, Any]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    
    ######grid search over list (adopted from https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression/blob/main/src/data/ate/__init__.py)######

    #params:
    ##org_params: Dictionary to be grid searched

    search_keys = []
    non_search_keys = []
    for key in org_params.keys():
        if isinstance(org_params[key], list):
            search_keys.append(key)
        else:
            non_search_keys.append(key)
    if len(search_keys) == 0:
        yield "one", org_params
    else:
        param_generator = product(*[org_params[key] for key in search_keys])
        for one_param_set in param_generator:
            one_dict = {k: org_params[k] for k in non_search_keys}
            tmp = dict(list(zip(search_keys, one_param_set)))
            one_dict.update(tmp)
            one_name = "-".join([k + ":" + str(tmp[k]) for k in search_keys])
            yield one_name, one_dict
