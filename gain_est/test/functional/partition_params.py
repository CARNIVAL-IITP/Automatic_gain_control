from typing import Tuple, List, Iterable
from torch.nn import Parameter


def partition_params(
        named_parameters: Iterable[Tuple[str, Parameter]],
        names_to_partition: List[str]
) -> Tuple[Iterable[Parameter], Iterable[Parameter]]:
    params_with_name, params_without_name = [], []
    for name, param in named_parameters:
        found = False
        for name_to_partition in names_to_partition:
            if name.find(name_to_partition) >= 0:
                found = True
                break
        if found:
            params_with_name.append(param)
        else:
            params_without_name.append(param)
    return params_with_name, params_without_name
