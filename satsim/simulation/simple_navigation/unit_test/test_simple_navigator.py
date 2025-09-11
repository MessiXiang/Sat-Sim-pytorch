# tests/test_simple_navigator.py
import pytest
import torch
from satsim.architecture import Timer
from satsim.simulation.simple_navigation.simple_navigator import (
    SimpleNavigator,
    SimpleNavigatorStateDict,
    AttitudeData,
    TranslationData,
    add_mrp,
    mrp_to_rotation_matrix,
)


def test_simple_navigator_initialization():
    """测试导航器的初始化参数设置"""
    # 默认初始化
    timer = Timer(1.)
    nav = SimpleNavigator(timer=timer)
    assert nav.noise_process_covariance_matrix.shape == (18, 18)
    assert torch.allclose(nav.noise_process_covariance_matrix,
                          torch.zeros(18, 18))
    assert nav.noise_walk_bounds.shape == (18, )
    assert torch.allclose(nav.noise_walk_bounds, torch.full([18], -1.0))

    # 自定义参数初始化
    cov_matrix = torch.eye(18)
    noise_bounds = torch.ones(18) * 0.1
    nav = SimpleNavigator(timer=timer,
                          noise_process_covariance_matrix=cov_matrix,
                          noise_walk_bounds=noise_bounds)
    assert torch.allclose(nav.noise_process_covariance_matrix, cov_matrix)
    assert torch.allclose(nav.noise_walk_bounds, noise_bounds)


def test_simple_navigator_reset():
    """测试重置功能"""
    timer = Timer(1.)
    nav = SimpleNavigator(timer=timer)
    state_dict = nav.reset()
    assert "navigation_errors" in state_dict
    assert state_dict["navigation_errors"].shape == (18, )
    assert torch.allclose(state_dict["navigation_errors"], torch.zeros(18))


@pytest.mark.parametrize(
    "batch_shape",
    [
        (),  # 无batch维度
        (5, ),  # 单batch维度
        (3, 4),  # 多batch维度
        (2, 3, 4)  # 复杂batch维度
    ])
def test_simple_navigator_batch_support(batch_shape):
    """测试导航器对不同batch维度的支持能力"""
    # 创建带batch维度的噪声参数
    batch_size = torch.Size(batch_shape).numel()
    noise_process_covariance_matrix = torch.eye(18).expand(
        *batch_shape, 18, 18)
    noise_walk_bounds = torch.ones(*batch_shape, 18) * 0.1

    timer = Timer(1.)
    nav = SimpleNavigator(
        timer=timer,
        noise_process_covariance_matrix=noise_process_covariance_matrix,
        noise_walk_bounds=noise_walk_bounds)

    # 创建带batch维度的输入数据
    state_dict = nav.reset()

    # 扩展状态字典以包含batch维度
    state_dict["navigation_errors"] = state_dict["navigation_errors"].expand(
        *batch_shape, 18)

    # 创建带batch维度的输入参数
    position = torch.rand(*batch_shape, 3)
    velocity = torch.rand(*batch_shape, 3)
    attitude = torch.rand(*batch_shape, 3)
    angular_velocity = torch.rand(*batch_shape, 3)
    delta_v = torch.rand(*batch_shape, 3)
    sun_position = torch.rand(*batch_shape,
                              3) if batch_size > 0 else torch.rand(3)

    # 执行前向传播
    new_state_dict, (attitude_data, translation_data) = nav(
        state_dict,
        position_in_inertial=position,
        velocity_in_inertial=velocity,
        mrp_attitude_in_inertial=attitude,
        angular_velocity_in_inertial=angular_velocity,
        total_accumulated_delta_velocity_in_inertial=delta_v,
        sun_position_in_inertial=sun_position)

    # 验证输出形状
    assert new_state_dict["navigation_errors"].shape == (*batch_shape, 18)

    # 验证AttitudeData形状
    assert attitude_data["mrp_attitude_in_inertial"].shape == (*batch_shape, 3)
    assert attitude_data["angular_velocity_in_inertial"].shape == (
        *batch_shape, 3)
    assert attitude_data["sun_position_in_body"].shape == (*batch_shape, 3)

    # 验证TranslationData形状
    assert translation_data["position_in_inertial"].shape == (*batch_shape, 3)
    assert translation_data["velocity_in_inertial"].shape == (*batch_shape, 3)
    assert translation_data[
        "total_accumulated_delta_velocity_in_inertial"].shape == (*batch_shape,
                                                                  3)


def test_compute_sun_position_in_body():
    """测试太阳向量计算功能"""
    timer = Timer(1.)
    nav = SimpleNavigator(timer=timer, )

    # 测试无太阳位置输入的情况
    position = torch.zeros(3)
    attitude = torch.zeros(3)  # 零MRP表示惯性系和本体系重合
    sun_position = None

    result = nav.compute_sun_position_in_body(position, attitude, sun_position)
    assert torch.allclose(result, torch.zeros(3))

    # 测试有太阳位置输入的情况
    sun_position = torch.tensor([1.0, 0.0, 0.0])
    result = nav.compute_sun_position_in_body(position, attitude, sun_position)
    assert torch.allclose(result, torch.tensor([1.0, 0.0, 0.0]))

    # 测试旋转后的情况
    attitude = torch.tensor([0.5, 0.0, 0.0])  # 绕Y轴旋转约90度
    dcm = mrp_to_rotation_matrix(attitude)
    expected = torch.matmul(dcm, sun_position.unsqueeze(-1)).squeeze(-1)

    result = nav.compute_sun_position_in_body(position, attitude, sun_position)
    assert torch.allclose(result, expected, atol=1e-6)


def test_compute_errors():
    """测试误差计算功能"""
    timer = Timer(1.)
    nav = SimpleNavigator(timer=timer)

    # 设置已知的噪声参数
    noise_process_covariance_matrix = torch.eye(18) * 0.1
    noise_walk_bounds = torch.ones(18) * 0.5
    nav.register_buffer("noise_process_covariance_matrix",
                        noise_process_covariance_matrix)
    nav.register_buffer("noise_walk_bounds", noise_walk_bounds)

    # 初始误差为零
    navigation_errors = torch.zeros(18)

    # 多次计算以验证随机性和边界约束
    for _ in range(10):
        new_errors = nav.compute_errors(navigation_errors)

        # 验证误差是否有变化
        assert not torch.allclose(new_errors, navigation_errors)

        # 验证误差是否在边界内
        assert torch.all(new_errors <= noise_walk_bounds)
        assert torch.all(new_errors >= -noise_walk_bounds)

        # 更新误差状态
        navigation_errors = new_errors


def test_apply_errors():
    """测试误差应用功能"""
    timer = Timer(1.)
    nav = SimpleNavigator(timer=timer)

    # 创建输入数据
    position = torch.zeros(3)
    velocity = torch.zeros(3)
    attitude = torch.zeros(3)
    angular_velocity = torch.zeros(3)
    delta_v = torch.zeros(3)
    sun_position_body = torch.tensor([1.0, 0.0, 0.0])

    # 创建特定的误差向量
    navigation_errors = torch.zeros(18)
    navigation_errors[0] = 0.1  # 位置误差
    navigation_errors[3] = 0.2  # 速度误差
    navigation_errors[6] = 0.01  # 姿态误差
    navigation_errors[9] = 0.05  # 角速度误差
    navigation_errors[12] = 0.02  # 太阳向量旋转误差
    navigation_errors[15] = 0.03  # 累计速度增量误差

    # 应用误差
    attitude_data, translation_data = nav.apply_errors(
        navigation_errors, attitude, angular_velocity, sun_position_body,
        position, velocity, delta_v)

    # 验证位置误差应用
    assert torch.allclose(translation_data["position_in_inertial"],
                          torch.tensor([0.1, 0.0, 0.0]))

    # 验证速度误差应用
    assert torch.allclose(translation_data["velocity_in_inertial"],
                          torch.tensor([0.2, 0.0, 0.0]))

    # 验证姿态误差应用
    expected_attitude = add_mrp(attitude, navigation_errors[6:9])
    assert torch.allclose(attitude_data["mrp_attitude_in_inertial"],
                          expected_attitude)

    # 验证角速度误差应用
    assert torch.allclose(attitude_data["angular_velocity_in_inertial"],
                          torch.tensor([0.05, 0.0, 0.0]))

    # 验证太阳向量旋转
    dcm_ot = mrp_to_rotation_matrix(navigation_errors[12:15])
    expected_sun = torch.matmul(dcm_ot,
                                sun_position_body.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(attitude_data["sun_position_in_body"],
                          expected_sun,
                          atol=1e-6)

    # 验证累计速度增量误差应用
    assert torch.allclose(
        translation_data["total_accumulated_delta_velocity_in_inertial"],
        torch.tensor([0.03, 0.0, 0.0]))


def test_addMRP_function():
    """测试MRP加法函数"""
    # 测试零加零
    q1 = torch.zeros(3)
    q2 = torch.zeros(3)
    result = add_mrp(q1, q2)
    assert torch.allclose(result, torch.zeros(3))

    # 测试零加非零
    q2 = torch.tensor([0.1, 0.2, 0.3])
    result = add_mrp(q1, q2)
    assert torch.allclose(result, q2)

    # 测试非零加零
    result = add_mrp(q2, q1)
    assert torch.allclose(result, q2)

    # 测试两个非零相加
    q1 = torch.tensor([0.3, 0.2, 0.1])
    result = add_mrp(q1, q2)

    # 验证结果是否在MRP定义域内 (模长<=1)
    norm_sq = torch.sum(result**2)
    assert norm_sq <= 1.0 + 1e-6


def test_MRP2C_function():
    """测试MRP到方向余弦矩阵的转换"""
    # 测试零MRP (对应单位矩阵)
    q = torch.zeros(3)
    dcm = mrp_to_rotation_matrix(q)
    assert torch.allclose(dcm, torch.eye(3), atol=1e-6)

    # 测试90度旋转
    q = torch.tensor([0.5, 0.0, 0.0])  # 绕Y轴旋转约90度
    dcm = mrp_to_rotation_matrix(q)

    # 验证旋转矩阵特性
    # 1. 行列式应为1
    det = torch.det(dcm)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-6)

    # 2. 逆矩阵等于转置矩阵
    inv_dcm = torch.inverse(dcm)
    assert torch.allclose(inv_dcm, dcm.transpose(-2, -1), atol=1e-6)
