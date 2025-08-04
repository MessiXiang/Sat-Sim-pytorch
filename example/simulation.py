import torch

from satsim.architecture import Module, Timer, constants
from satsim.simulation.spacecraft import Spacecraft
from satsim.simulation.gravity import PointMassGravityBody, GravityField, SpiceInterface
from satsim.simulation.reaction_wheels import (ReactionWheel,
                                               HoneywellHR12Small,
                                               ReactionWheels, SpinAxis,
                                               expand)
from satsim.data import OrbitalElements, elem2rv


class RemoteSensingConstellation(Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setup_gravity_field()

        reaction_wheels = self.setup_reaction_wheels()

        orbits = OrbitalElements.sample(50)
        r, v = elem2rv(constants.MU_EARTH, elements=orbits)

        self._spacecraft = Spacecraft(
            timer=self._timer,
            mass=torch.rand(50),
            moment_of_inertia_matrix_wrt_body_point=torch.eye(3).expand(
                50, 3, 3),
            pos=r,
            velocity=v,
            gravity_field=self._gravity_field,
            state_effectors=[reaction_wheels],
        )

    def setup_reaction_wheels(self, ) -> ReactionWheels:
        reaction_wheel0 = HoneywellHR12Small(spin_axis_in_body=SpinAxis.X)
        reaction_wheel1 = HoneywellHR12Small(spin_axis_in_body=SpinAxis.Y)
        reaction_wheel2 = HoneywellHR12Small(spin_axis_in_body=SpinAxis.Z)

        return ReactionWheels(timer=self._timer,
                              reaction_wheels=[
                                  reaction_wheel0,
                                  reaction_wheel1,
                                  reaction_wheel2,
                              ])

    def setup_gravity_field(self):
        sun = PointMassGravityBody.create_sun(timer=self._timer)
        earth = PointMassGravityBody.create_earth(
            timer=self._timer,
            is_central=True,
        )

        spice_interface = SpiceInterface(
            timer=self._timer,
            utc_time_init=constants.UTC_TIME_START,
        )

        self._gravity_field = GravityField(
            timer=self._timer,
            spice_interface=spice_interface,
            gravity_bodies=[sun, earth],
        )
