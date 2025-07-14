import re

import pytest
import torch

from satsim.architecture import Timer
from satsim.simulation.reaction_wheels import (
    HoneywellHR12Large,
    HoneywellHR12Medium,
    HoneywellHR12Small,
    ReactionWheel,
    ReactionWheels,
    ReactionWheelsStateDict,
    SpinAxis,
)


# 测试ReactionWheel类的基础功能
class TestReactionWheel:

    @pytest.fixture
    def valid_spin_axis(self) -> SpinAxis:
        return SpinAxis.X

    def test_build_valid_params(self, valid_spin_axis: SpinAxis):
        """测试正常参数下的构建功能"""
        rw = ReactionWheel.build(spin_axis_in_body=valid_spin_axis,
                                 max_momentum=10.0,
                                 max_angular_velocity=100.0,
                                 mass=2.0)

        # 验证基本属性
        assert isinstance(rw.spin_axis_in_body, SpinAxis)
        assert rw.moment_of_inertia_wrt_spin == 0.1  # 10/100
        assert rw.mass == 2.0


# 测试Honeywell系列子类
class TestHoneywellReactionWheels:

    @pytest.fixture
    def valid_spin_axis(self) -> SpinAxis:
        return SpinAxis.Y

    def test_large_build(self, valid_spin_axis: SpinAxis):
        rw = HoneywellHR12Large.build(spin_axis_in_body=valid_spin_axis)
        assert rw.max_torque == 0.2  # 从子类参数验证

    def test_medium_build(self, valid_spin_axis: SpinAxis):
        rw = HoneywellHR12Medium.build(spin_axis_in_body=valid_spin_axis)
        assert rw.mass == 7.

    def test_small_build(self, valid_spin_axis: SpinAxis):
        rw = HoneywellHR12Small.build(spin_axis_in_body=valid_spin_axis)
        assert rw.mass == 6.  # 验证特定参数


# 测试ReactionWheels类（重点测试batch_size支持）
class TestReactionWheels:

    @pytest.fixture
    def batch_rws(self) -> list[ReactionWheel]:
        """创建包含多个reaction wheel的batch"""
        spin_axes = [SpinAxis.X, SpinAxis.Y, SpinAxis.Z]
        rws = [
            ReactionWheel.build(
                spin_axis_in_body=axis,
                max_momentum=10.0,
                max_angular_velocity=100.0,
            ) for axis in spin_axes
        ]
        return rws

    def test_init_batch(self, batch_rws: list[ReactionWheel]):
        """测试批量初始化的buffer形状"""
        reaction_wheels = ReactionWheels(
            timer=Timer(1.),
            reaction_wheels=batch_rws,
        )
        assert reaction_wheels.spin_axis_in_body.shape == (3, 3
                                                           )  # [3轴, 3个wheel]
        assert reaction_wheels.moment_of_inertia_wrt_spin.shape == (
            1, 3)  # [1, 3个wheel]
        assert reaction_wheels.angular_velocity_init.shape == (1, 3)

    def test_reset_state(self, batch_rws: list[ReactionWheel]):
        """测试重置状态时的batch兼容性"""
        reaction_wheels = ReactionWheels(
            timer=Timer(1.),
            reaction_wheels=batch_rws,
        )
        state = reaction_wheels.reset()
        assert state['current_torque'].shape == (1, 3)
        assert state['dynamic_params']['angular_velocity'].shape == (1, 3)

    def test_batch_operations(self, batch_rws: list[ReactionWheel]):
        """测试带batch维度的操作（模拟noise_bound=16,16,18的batch场景）"""
        # 1. 创建模拟的batch输入 (batch_size=16*16=256)
        batch_size = 256
        batch_rws = ReactionWheel.expand(
            [batch_size],
            reaction_wheels=batch_rws,
        )

        reaction_wheels = ReactionWheels(
            timer=Timer(1.),
            reaction_wheels=batch_rws,
        )
        angular_velocity_BN_B = torch.randn(batch_size, 3)  # 模拟batch输入
        sigma_BN = torch.eye(3).repeat(batch_size, 1, 1)
        g_N = torch.zeros(batch_size, 3)

        # 2. 初始化状态
        state = reaction_wheels.reset()

        # 4. 测试back substitution计算的batch兼容性
        back_sub = {
            'matrix_d': torch.zeros(batch_size, 3, 3),
            'vec_rot': torch.zeros(batch_size, 3)
        }
        updated_state, updated_back = reaction_wheels.update_back_substitution_contribution(
            state, back_sub, sigma_BN, angular_velocity_BN_B, g_N)

        # 验证输出形状是否符合batch_size
        assert updated_back['matrix_d'].shape == (batch_size, 3, 3)
        assert updated_back['vec_rot'].shape == (batch_size, 3)

    def test_forward_with_batch(self, batch_rws: list[ReactionWheel]):
        """测试forward方法的batch支持"""
        batch_size = 256  # 16*16
        batch_rws = ReactionWheel.expand(
            [batch_size],
            reaction_wheels=batch_rws,
        )
        reaction_wheels = ReactionWheels(
            timer=Timer(1.),
            reaction_wheels=batch_rws,
        )
        state = reaction_wheels.reset()

        # 应用motor torque (带batch维度)
        motor_torque = torch.randn(batch_size, 3)  # 匹配batch和wheel数量
        updated_state: ReactionWheelsStateDict
        updated_state, _ = reaction_wheels(state, motor_torque=motor_torque)

        # 验证输出形状
        assert updated_state['current_torque'].shape == (batch_size, 1, 3)

    def test_large_batch_compatibility(self, batch_rws: list[ReactionWheel]):
        """测试更大维度的batch兼容性（模拟16,16,18的形状）"""
        # 模拟noise_bound=16,16,18的输入形状
        batch_shape = (16, 16, 18)
        batch_rws = ReactionWheel.expand(
            batch_shape,
            reaction_wheels=batch_rws,
        )
        reaction_wheels = ReactionWheels(
            timer=Timer(1.),
            reaction_wheels=batch_rws,
        )

        state = reaction_wheels.reset()

        # 测试compute_derivatives的兼容性
        rDDot = torch.randn(*batch_shape, 3)
        angular_velocityDot = torch.randn(*batch_shape, 3)
        sigma_BN = torch.eye(3).repeat(*batch_shape, 1, 1)

        updated_state = reaction_wheels.compute_derivatives(
            state, rDDot, angular_velocityDot, sigma_BN)

        # 验证导数计算的形状
        assert updated_state['dynamic_params'][
            'angular_acceleration'].shape == (*batch_shape, 1, 3)

    def test_empty_rw_list(self):
        """测试空列表输入的错误处理"""
        with pytest.raises(ValueError,
                           match="Reaction Wheel list cannot be empty"):
            ReactionWheels(timer=Timer(1.), reaction_wheels=[])
