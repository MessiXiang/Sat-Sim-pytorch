__all__ = [
    'BackSubMatrices',
    'EffectorMassProps',
    'StateEffectorMixin',
    'StateEffectorStateDict',
]
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, TypedDict

import torch


class BackSubMatrices(TypedDict):
    matrix_a: torch.Tensor
    matrix_b: torch.Tensor
    matrix_c: torch.Tensor
    matrix_d: torch.Tensor
    vec_trans: torch.Tensor
    vec_rot: torch.Tensor


class EffectorMassProps(TypedDict):
    mEff: float = 0.
    mEffDot: float = 0.
    IEffPntB_B: torch.Tensor = torch.zeros(3, 3)
    rEff_CB_B: torch.Tensor = torch.zeros(3)
    rEffPrime_CB_B: torch.Tensor = torch.zeros(3)
    IEffPrimePntB_B: torch.Tensor = torch.zeros(3, 3)


class StateEffectorStateDict(TypedDict):
    effProps: EffectorMassProps
    stateDerivContribution: torch.Tensor
    forceOnBody_B: torch.Tensor
    torqueOnBodyPntB_B: torch.Tensor
    torqueOnBodyPntC_B: torch.Tensor
    r_BP_P: torch.Tensor
    dcm_BP: torch.Tensor


T = TypeVar('T')


class StateEffectorMixin(ABC, Generic[T]):

    def state_effector_reset(self) -> StateEffectorStateDict:
        effProps = dict(
            mEff=0.,
            mEffDot=0.,
            IEffPntB_B=torch.zeros(3, 3),
            rEff_CB_B=torch.zeros(3),
            rEffPrime_CB_B=torch.zeros(3),
            IEffPrimePntB_B=torch.zeros(3, 3),
        )
        return dict(
            effProps=effProps,
            stateDerivContribution=torch.Tensor(
            ),  #Supposed to be a x-dimension tensor
            forceOnBody_B=torch.zeros(3),
            torqueOnBodyPntB_B=torch.zeros(3),
            torqueOnBodyPntC_B=torch.zeros(3),
            r_BP_P=torch.zeros(3),
            dcm_BP=torch.eye(3, 3),
        )

    @abstractmethod
    def get_state_output(
        self,
        state_dict: T,
    ) -> tuple[
            T,
            tuple[Any, ...],
    ]:
        pass

    def update_effector_mass(
        self,
        state_dict: T,
    ) -> T:
        return state_dict

    def update_back_substitution_contribution(
        self,
        state_dict: T,
        backSubContr: BackSubMatrices,
        sigma_BN: torch.Tensor,
        omega_BN_B: torch.Tensor,
        g_N: torch.Tensor,
    ) -> tuple[T, BackSubMatrices]:
        return state_dict, backSubContr

    def update_energy_momentum_contributions(
        self,
        state_dict: T,
        rotAngMomPntCContr_B: torch.Tensor,
        rotEnergyContr: torch.Tensor,
        omega_BN_B: torch.Tensor,
    ) -> T:
        return state_dict

    def modifyStates(self, state_dict: T) -> T:
        return state_dict

    def calcForceTorqueOnBody(
        self,
        state_dict: T,
        omega_BN_B: torch.Tensor,
    ) -> T:
        return state_dict

    @abstractmethod
    def register_states(self, state_dict: T) -> T:
        pass

    @abstractmethod
    def link_in_states(self, states) -> None:
        pass

    @abstractmethod
    def compute_derivatives(
        self,
        state_dict: T,
        rDDot_BN_N: torch.Tensor,
        omegaDot_BN_B: torch.Tensor,
        sigma_BN: torch.Tensor,
    ) -> T:
        pass

    def prependSpacecraftNameToStates(self, state_dict: T) -> T:
        return state_dict

    # Temporarily saved for code comprehension, be done by change state dict
    def receiveMotherSpacecraftData(
        self,
        rSC_BP_P: torch.Tensor,
        dcmSC_BP: torch.Tensor,
    ) -> None:
        self.r_BP_P = rSC_BP_P
        self.dcm_BP = dcmSC_BP
