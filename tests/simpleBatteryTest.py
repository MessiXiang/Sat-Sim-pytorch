import torch
from satsim.modules.simulation.power.simpleBattery import SimpleBattery
from architecture.simulator import Simulator


def test_battery_initialization():
    """测试电池初始化"""
    print("=== 测试电池初始化 ===")

    # 正常初始化
    battery = SimpleBattery("TestBattery", 100.0, 1.0)
    assert battery.name == "TestBattery"
    assert battery.storage_capacity.item() == 100.0
    assert battery.stored_charge.item() == 0.0
    assert battery.fault_capacity_ratio.item() == 1.0
    print("✓ 正常初始化测试通过")

    # 测试负容量异常
    try:
        bad_battery = SimpleBattery("BadBattery", -10.0)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print("✓ 负容量异常测试通过")

    # 测试零容量异常
    try:
        zero_battery = SimpleBattery("ZeroBattery", 0.0)
        assert False, "应该抛出ValueError"
    except ValueError as e:
        print("✓ 零容量异常测试通过")


def test_basic_charging():
    """测试基本充电功能"""
    print("\n=== 测试基本充电功能 ===")

    battery = SimpleBattery("ChargeBattery", 100.0)
    dt = torch.tensor(1.0)

    # 充电测试
    result = battery.simulation_step(dt, power_input=torch.tensor(10.0))
    assert abs(result.item() - 10.0) < 1e-6
    print(f"✓ 充电后电量: {result.item()}")

    # 继续充电
    result = battery.simulation_step(dt, power_input=torch.tensor(15.0))
    assert abs(result.item() - 25.0) < 1e-6
    print(f"✓ 继续充电后电量: {result.item()}")


def test_basic_discharging():
    """测试基本放电功能"""
    print("\n=== 测试基本放电功能 ===")

    battery = SimpleBattery("DischargeBattery", 100.0)
    dt = torch.tensor(1.0)

    # 先充电到50
    battery.simulation_step(dt, power_input=torch.tensor(50.0))

    # 放电测试
    result = battery.simulation_step(dt, power_input=torch.tensor(-20.0))
    assert abs(result.item() - 30.0) < 1e-6
    print(f"✓ 放电后电量: {result.item()}")


def test_capacity_limits():
    """测试容量限制"""
    print("\n=== 测试容量限制 ===")

    battery = SimpleBattery("CapacityBattery", 50.0)
    dt = torch.tensor(1.0)

    # 过度充电测试
    result = battery.simulation_step(dt, power_input=torch.tensor(100.0))
    assert abs(result.item() - 50.0) < 1e-6
    print(f"✓ 过度充电后电量被限制在: {result.item()}")

    # 过度放电测试
    result = battery.simulation_step(dt, power_input=torch.tensor(-100.0))
    assert abs(result.item() - 0.0) < 1e-6
    print(f"✓ 过度放电后电量被限制在: {result.item()}")


def test_fault_conditions():
    """测试故障条件"""
    print("\n=== 测试故障条件 ===")

    battery = SimpleBattery("FaultBattery", 100.0)
    dt = torch.tensor(1.0)

    # 正常容量下充满
    battery.simulation_step(dt, power_input=torch.tensor(100.0))
    print(f"正常状态下充满电量: {battery.stored_charge.item()}")

    # 50%故障状态下的有效容量
    result = battery.simulation_step(dt,
                                     power_input=torch.tensor(0.0),
                                     fault_ratio=torch.tensor(0.5))
    assert abs(result.item() - 50.0) < 1e-6
    print(f"✓ 50%故障后有效电量: {result.item()}")

    # 测试故障比例边界条件
    try:
        battery.simulation_step(dt, fault_ratio=torch.tensor(1.5))
        assert False, "应该抛出ValueError"
    except ValueError:
        print("✓ 故障比例>1.0异常测试通过")

    try:
        battery.simulation_step(dt, fault_ratio=torch.tensor(-0.1))
        assert False, "应该抛出ValueError"
    except ValueError:
        print("✓ 故障比例<0.0异常测试通过")


def test_reset_functionality():
    """测试重置功能"""
    print("\n=== 测试重置功能 ===")

    battery = SimpleBattery("ResetBattery", 100.0)
    dt = torch.tensor(1.0)

    # 改变电池状态
    battery.simulation_step(dt,
                            power_input=torch.tensor(50.0),
                            fault_ratio=torch.tensor(0.7))
    print(
        f"重置前 - 电量: {battery.stored_charge.item()}, 故障比例: {battery.fault_capacity_ratio.item()}"
    )

    # 重置
    battery.reset_simulation_state()
    assert abs(battery.stored_charge.item() - 0.0) < 1e-6
    assert abs(battery.fault_capacity_ratio.item() - 1.0) < 1e-6
    assert abs(battery.current_power.item() - 0.0) < 1e-6
    print(
        f"✓ 重置后 - 电量: {battery.stored_charge.item()}, 故障比例: {battery.fault_capacity_ratio.item()}"
    )


def test_with_simulator():
    """测试与仿真器的集成"""
    print("\n=== 测试与仿真器集成 ===")

    battery = SimpleBattery("SimulatorBattery", 100.0)
    simulator = Simulator(battery, dt=0.1, auto_save=False)

    # 设置充电功率进行仿真
    print("开始10步仿真充电测试...")
    for i in range(10):
        # 模拟周期性充电/放电
        power = torch.tensor(5.0) if i % 2 == 0 else torch.tensor(-2.0)
        result = simulator.module.simulation_step(simulator.dt,
                                                  power_input=power)
        simulator.clock.step()
        if i % 3 == 0:
            print(f"步骤 {i+1}: 功率={power.item()}, 电量={result.item():.2f}")

    print(f"✓ 仿真完成，最终时间: {simulator.time:.2f}s, 步数: {simulator.steps}")


def test_state_management():
    """测试状态管理"""
    print("\n=== 测试状态管理 ===")

    battery = SimpleBattery("StateBattery", 100.0)
    dt = torch.tensor(1.0)

    # 设置特定状态
    battery.simulation_step(dt,
                            power_input=torch.tensor(30.0),
                            fault_ratio=torch.tensor(0.8))
    original_charge = battery.stored_charge.item()
    original_fault = battery.fault_capacity_ratio.item()

    # 保存状态
    state = battery.get_simulation_state()
    print(f"保存状态 - 电量: {original_charge}, 故障比例: {original_fault}")

    # 改变状态
    battery.simulation_step(dt,
                            power_input=torch.tensor(20.0),
                            fault_ratio=torch.tensor(0.6))
    print(
        f"改变后 - 电量: {battery.stored_charge.item()}, 故障比例: {battery.fault_capacity_ratio.item()}"
    )

    # 恢复状态
    battery.load_simulation_state(state)
    assert abs(battery.stored_charge.item() - original_charge) < 1e-6
    print(
        f"✓ 恢复状态 - 电量: {battery.stored_charge.item()}, 故障比例: {battery.fault_capacity_ratio.item()}"
    )


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")

    battery = SimpleBattery("EdgeBattery", 100.0)
    dt = torch.tensor(0.001)  # 很小的时间步长

    # 小功率长时间充电
    total_charge = 0
    for i in range(1000):
        result = battery.simulation_step(dt, power_input=torch.tensor(0.1))
        total_charge = result.item()

    expected_charge = min(100.0, 0.1 * 0.001 * 1000)
    assert abs(total_charge - expected_charge) < 1e-3
    print(f"✓ 小功率长时间充电测试通过，最终电量: {total_charge:.3f}")

    # 零功率测试
    initial_charge = total_charge
    result = battery.simulation_step(dt, power_input=torch.tensor(0.0))
    assert abs(result.item() - initial_charge) < 1e-6
    print(f"✓ 零功率测试通过，电量保持: {result.item():.3f}")
