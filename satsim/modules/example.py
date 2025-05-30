import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from satsim.architecture.module import SimulationModule
from architecture.simulator import Simulator


class SpringDamperOscillator(SimulationModule):
    """
    弹簧阻尼振子系统
    二阶微分方程: m*x'' + c*x' + k*x = F(t)
    转换为状态空间形式:
    x1 = x (位置)
    x2 = x' (速度)
    x1' = x2
    x2' = (F(t) - c*x2 - k*x1) / m
    """

    def __init__(self,
                 name: str = "SpringDamper",
                 mass: float = 1.0,
                 spring_constant: float = 10.0,
                 damping_coefficient: float = 0.5,
                 initial_position: float = 1.0,
                 initial_velocity: float = 0.0):
        super().__init__(name)

        # 系统参数
        self.mass = torch.tensor(mass, dtype=torch.float32)
        self.spring_constant = torch.tensor(spring_constant,
                                            dtype=torch.float32)
        self.damping_coefficient = torch.tensor(damping_coefficient,
                                                dtype=torch.float32)

        # 初始状态
        self.initial_state = torch.tensor([initial_position, initial_velocity],
                                          dtype=torch.float32)

        # 状态变量 [位置, 速度]
        self.state = nn.Parameter(self.initial_state.clone())

        # 当前时间（由于无法从Simulator获取，我们自己维护）
        self.current_time = 0.0

        # 记录历史数据用于可视化
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'energy': []
        }

    def external_force(self, t: float) -> torch.Tensor:
        """外力函数，可以自定义"""
        # 示例：正弦波驱动力
        t_tensor = torch.tensor(t, dtype=torch.float32)
        return 0.5 * torch.sin(2 * torch.pi * 0.5 * t_tensor)

    def simulation_step(self, dt: torch.Tensor) -> torch.Tensor:
        """使用Runge-Kutta 4阶方法求解微分方程"""
        dt_tensor = dt

        def derivatives(state, t_val):
            x, v = state[0], state[1]
            F = self.external_force(t_val)

            # dx/dt = v
            # dv/dt = (F - c*v - k*x) / m
            dx_dt = v
            dv_dt = (F - self.damping_coefficient * v -
                     self.spring_constant * x) / self.mass

            return torch.stack([dx_dt, dv_dt])

        # RK4积分
        k1 = derivatives(self.state, self.current_time)
        k2 = derivatives(self.state + 0.5 * dt_tensor * k1,
                         self.current_time + 0.5 * dt)
        k3 = derivatives(self.state + 0.5 * dt_tensor * k2,
                         self.current_time + 0.5 * dt)
        k4 = derivatives(self.state + dt_tensor * k3, self.current_time + dt)

        # 更新状态
        with torch.no_grad():
            self.state.data += dt_tensor * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # 更新时间
        self.current_time += dt

        # 记录历史数据
        self.history['time'].append(self.current_time)
        self.history['position'].append(self.state[0].item())
        self.history['velocity'].append(self.state[1].item())

        # 计算系统能量
        kinetic_energy = 0.5 * self.mass * self.state[1]**2
        potential_energy = 0.5 * self.spring_constant * self.state[0]**2
        total_energy = kinetic_energy + potential_energy
        self.history['energy'].append(total_energy.item())

        # 返回当前状态作为输出
        return self.state.clone()

    def reset_simulation_state(self):
        """重置仿真状态"""
        with torch.no_grad():
            self.state.data = self.initial_state.clone()
        self.current_time = 0.0
        self.history = {
            'time': [],
            'position': [],
            'velocity': [],
            'energy': []
        }

    def get_simulation_state(self):
        """获取完整的仿真状态"""
        state_dict = super().get_simulation_state()
        state_dict['current_time'] = self.current_time
        state_dict['history'] = self.history.copy()
        return state_dict

    def load_simulation_state(self, state):
        """载入完整的仿真状态"""
        self.current_time = state.pop('current_time', 0.0)
        self.history = state.pop('history', {
            'time': [],
            'position': [],
            'velocity': [],
            'energy': []
        })
        super().load_simulation_state(state)

    def plot_results(self):
        """绘制仿真结果"""
        if not self.history['time']:
            print("No simulation data to plot!")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 位置vs时间
        axes[0, 0].plot(self.history['time'],
                        self.history['position'],
                        'b-',
                        linewidth=2,
                        label='Position')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position (m)')
        axes[0, 0].set_title('Position vs Time')
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        # 速度vs时间
        axes[0, 1].plot(self.history['time'],
                        self.history['velocity'],
                        'r-',
                        linewidth=2,
                        label='Velocity')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Velocity (m/s)')
        axes[0, 1].set_title('Velocity vs Time')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # 相空间图 (位置vs速度)
        axes[1, 0].plot(self.history['position'],
                        self.history['velocity'],
                        'g-',
                        linewidth=2,
                        alpha=0.7)
        axes[1, 0].set_xlabel('Position (m)')
        axes[1, 0].set_ylabel('Velocity (m/s)')
        axes[1, 0].set_title('Phase Space (Position vs Velocity)')
        axes[1, 0].grid(True)

        # 能量vs时间
        axes[1, 1].plot(self.history['time'],
                        self.history['energy'],
                        'm-',
                        linewidth=2,
                        label='Total Energy')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Energy (J)')
        axes[1, 1].set_title('Energy vs Time')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()


def main():
    """主函数：创建和运行仿真"""
    print("=== 弹簧阻尼振子仿真测试 ===")

    # 创建弹簧阻尼振子系统
    oscillator = SpringDamperOscillator(name="TestOscillator",
                                        mass=1.0,
                                        spring_constant=10.0,
                                        damping_coefficient=0.5,
                                        initial_position=1.0,
                                        initial_velocity=0.0)

    # 创建仿真器，关闭自动保存以减少输出
    simulator = Simulator(
        module=oscillator,
        dt=0.01,
        auto_save=False  # 关闭自动保存以减少输出
    )

    print(f"系统参数:")
    print(f"  质量: {oscillator.mass.item()} kg")
    print(f"  弹簧常数: {oscillator.spring_constant.item()} N/m")
    print(f"  阻尼系数: {oscillator.damping_coefficient.item()} N·s/m")
    print(f"  时间步长: {simulator.dt} s")

    # 运行仿真
    print("\n开始仿真...")
    simulation_time = 10.0  # 仿真10秒
    steps = int(simulation_time / simulator.dt)

    simulator.run(steps=steps, save_interval=200)  # 每200步保存一次状态

    print(f"仿真完成!")
    print(f"  总步数: {simulator.steps}")
    print(f"  仿真时间: {simulator.time:.2f} s")
    print(f"  模块内部时间: {oscillator.current_time:.2f} s")
    print(f"  最终位置: {oscillator.state[0].item():.4f} m")
    print(f"  最终速度: {oscillator.state[1].item():.4f} m/s")

    # 测试状态保存和加载
    print("\n=== 测试状态保存和加载 ===")

    # 保存当前状态
    current_step = simulator.steps
    simulator.save_checkpoint()
    print(f"保存状态: step {current_step}")

    # 继续运行几步
    print("继续运行100步...")
    simulator.run(steps=100)
    print(
        f"当前状态: step {simulator.steps}, 位置: {oscillator.state[0].item():.4f}")

    # 加载之前的状态
    print(f"加载状态: step {current_step}")
    simulator.load_checkpoint(current_step)
    print(
        f"加载后状态: step {simulator.steps}, 位置: {oscillator.state[0].item():.4f}")

    # 测试重置功能
    print("\n=== 测试重置功能 ===")
    print(f"重置前: step {simulator.steps}, 位置: {oscillator.state[0].item():.4f}")
    simulator.reset()
    print(f"重置后: step {simulator.steps}, 位置: {oscillator.state[0].item():.4f}")

    print("\n=== 仿真测试完成 ===")


if __name__ == "__main__":
    main()
