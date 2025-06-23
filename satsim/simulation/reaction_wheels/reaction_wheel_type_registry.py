__all__ = ['reaction_wheel_type_registry']
from typing import Callable, TypeAlias

import torch

from satsim.architecture import constants

_Type_Specified_Params: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]
_Reaction_Wheel_Type: TypeAlias = Callable[
    [torch.Tensor],
    _Type_Specified_Params,
]


class ReactionWheelTypeRegistry:
    """This class make type specified params for reaction wheels.
        All registered function should accept a max_momentum as input and output 7 tensor in order of 
        max_omega,
        max_torque,
        min_torque,
        friction_coulomb,
        mass,
        u_s,
        u_d

    """

    def __init__(self) -> None:
        self._reaction_wheel_type_func: dict[str,
                                             _Reaction_Wheel_Type] = dict()

    def __call__(
        self,
        name_or_func: str | _Reaction_Wheel_Type | None = None
    ) -> _Reaction_Wheel_Type:
        if callable(name_or_func):
            func = name_or_func
            key = func.__name__
            self._reaction_wheel_type_func[key] = name_or_func
            return func

        def decorator(func: _Reaction_Wheel_Type) -> _Reaction_Wheel_Type:
            key = name_or_func if name_or_func is not None else func.__name__
            self._reaction_wheel_type_func[key] = func
            return func

        return decorator

    def get_type(self, name: str) -> _Reaction_Wheel_Type:
        return self._reaction_wheel_type_func[name]


reaction_wheel_type_registry = ReactionWheelTypeRegistry()


@reaction_wheel_type_registry
def Honeywell_HR12(max_momentum):
    max_omega = torch.tensor([[6000. * constants.RPM]])
    max_torque = torch.tensor([[0.2]])
    min_torque = torch.tensor([[0.00001]])
    friction_coulomb = torch.tensor([[0.0005]])

    large = 50
    medium = 25
    small = 12
    if max_momentum == large:
        mass = torch.tensor([[9.5]])
        u_s = torch.tensor([[4.4E-6]])
        u_d = torch.tensor([[9.1E-7]])
    elif max_momentum == medium:
        mass = torch.tensor([[7.0]])
        u_s = torch.tensor([[2.4E-6]])
        u_d = torch.tensor([[4.6E-7]])
    elif max_momentum == small:
        mass = torch.tensor([[6.0]])
        u_s = torch.tensor([[1.5E-6]])
        u_d = torch.tensor([[2.2E-7]])
    else:
        raise ValueError(
            f"Honeywell_HR12 does not have a correct wheel momentum of\
            {large}, {medium} or {small}, given {max_momentum}")

    return (
        max_omega,
        max_torque,
        min_torque,
        friction_coulomb,
        mass,
        u_s,
        u_d,
    )
