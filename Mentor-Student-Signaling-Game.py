import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

matplotlib.rc("font", family="Microsoft YaHei")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


class SignalingGameEnv:
    """信号博弈环境"""

    def __init__(self, q0, payoffs, penalty_student=0, penalty_mentor=0):
        """
        初始化环境
        q0: t1类型的先验概率
        payoffs: 收益字典
        penalty_student: 学生选择摆烂(N)的惩罚
        penalty_mentor: 导师选择摆烂(N)的惩罚
        """
        self.q0 = q0  # P(t1)
        self.payoffs = payoffs
        self.penalty_student = penalty_student
        self.penalty_mentor = penalty_mentor

    def reset(self):
        """重置环境状态"""
        # 随机选择导师类型
        self.mentor_type = "t1" if np.random.rand() < self.q0 else "t2"
        return self.mentor_type

    def step(self, mentor_action, student_action):
        """执行一步动作，返回奖励"""
        # 对于t1类型，强制选择D
        if self.mentor_type == "t1":
            mentor_action = "D"

        # 获取导师的收益
        if self.mentor_type == "t1":
            if student_action == "D":
                mentor_reward = self.payoffs[('t1', 'D', 'D')][0]
            else:  # student_action == "N"
                mentor_reward = self.payoffs[('t1', 'D', 'N')][0]
        else:  # mentor_type == "t2"
            if mentor_action == "D":
                if student_action == "D":
                    mentor_reward = self.payoffs[('t2', 'D', 'D')][0]
                else:  # student_action == "N"
                    mentor_reward = self.payoffs[('t2', 'D', 'N')][0]
            else:  # mentor_action == "N"
                if student_action == "D":
                    mentor_reward = self.payoffs[('t2', 'N', 'D')][0]
                else:  # student_action == "N"
                    mentor_reward = self.payoffs[('t2', 'N', 'N')][0]

        # 获取学生的收益
        if self.mentor_type == "t1":
            if student_action == "D":
                student_reward = self.payoffs[('t1', 'D', 'D')][1]
            else:  # student_action == "N"
                student_reward = self.payoffs[('t1', 'D', 'N')][1]
        else:  # mentor_type == "t2"
            if mentor_action == "D":
                if student_action == "D":
                    student_reward = self.payoffs[('t2', 'D', 'D')][1]
                else:  # student_action == "N"
                    student_reward = self.payoffs[('t2', 'D', 'N')][1]
            else:  # mentor_action == "N"
                if student_action == "D":
                    student_reward = self.payoffs[('t2', 'N', 'D')][1]
                else:  # student_action == "N"
                    student_reward = self.payoffs[('t2', 'N', 'N')][1]

        # 添加惩罚机制
        if mentor_action == "N" and self.mentor_type == "t2":
            mentor_reward -= self.penalty_mentor
        if student_action == "N":
            student_reward -= self.penalty_student

        return mentor_reward, student_reward


class QLearningAgent:
    """Q-learning智能体"""

    def __init__(self, agent_type, actions, state_size=2, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.9995, min_epsilon=0.01):
        """
        初始化Q学习智能体
        agent_type: "mentor"或"student"
        """
        self.agent_type = agent_type
        self.actions = actions
        self.state_size = state_size
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.epsilon_decay = epsilon_decay  # 探索率衰减
        self.min_epsilon = min_epsilon  # 最小探索率

        # 初始化Q表：状态 × 行动
        self.q_table = np.zeros((state_size, len(actions)))

    def state_to_index(self, state):
        """将状态转换为索引"""
        if self.agent_type == "student":
            # 学生观察导师的行动作为状态
            if state == "D":
                return 0
            elif state == "N":
                return 1
        else:  # mentor
            # 导师的状态是其类型（t1或t2）
            if state == "t1":
                return 0
            elif state == "t2":
                return 1
        return 0  # 默认

    def choose_action(self, state):
        """选择行动(ε-贪心策略)"""
        state_idx = self.state_to_index(state)
        if state == "t1" and self.agent_type == "mentor":
            # t1类型只能选择D
            return "D"

        if np.random.uniform(0, 1) < self.epsilon:
            # 探索：随机选择行动
            return np.random.choice(self.actions)
        else:
            # 利用：选择Q值最大的行动
            q_values = self.q_table[state_idx]
            max_q = np.max(q_values)
            # 如果有多个相同最大值的行动，随机选择一个
            actions_with_max_q = np.where(q_values == max_q)[0]
            action_idx = np.random.choice(actions_with_max_q)
            return self.actions[action_idx]

    def learn(self, state, action, reward, next_state=None):
        """更新Q表"""
        state_idx = self.state_to_index(state)
        if state == "t1" and self.agent_type == "mentor":
            # t1类型不学习
            return

        action_idx = self.actions.index(action)

        if next_state is not None:
            next_state_idx = self.state_to_index(next_state)
            # Q-learning更新规则
            max_next_q = np.max(self.q_table[next_state_idx])
            q_target = reward + self.gamma * max_next_q
        else:
            # 终止状态
            q_target = reward

        # 更新Q值
        self.q_table[state_idx, action_idx] = (1 - self.alpha) * self.q_table[
            state_idx, action_idx] + self.alpha * q_target

        # 衰减探索率
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

    def get_policy(self):
        """获取当前策略"""
        policy = {}
        if self.agent_type == "student":
            states = ["D", "N"]
        else:  # mentor
            states = ["t1", "t2"]

        for state in states:
            state_idx = self.state_to_index(state)
            best_action_idx = np.argmax(self.q_table[state_idx])
            policy[state] = self.actions[best_action_idx]
        return policy


def train_agents(q0, payoffs, episodes=10000, penalty_student=0, penalty_mentor=0):
    """训练学生和导师智能体"""
    # 创建环境
    env = SignalingGameEnv(q0, payoffs, penalty_student, penalty_mentor)

    # 创建智能体
    mentor_actions = ["D", "N"]  # 导师可选择的行动
    student_actions = ["D", "N"]  # 学生可选择的行动

    mentor = QLearningAgent("mentor", mentor_actions, state_size=2)  # 导师有t1/t2两个状态
    student = QLearningAgent("student", student_actions, state_size=2)  # 学生有D/N两个状态

    # 存储训练历史
    history = {
        'mentor_d_d': [],
        'mentor_d_n': [],
        'mentor_n_d': [],
        'mentor_n_n': [],
        'student_d_d': [],
        'student_d_n': [],
        'student_n_d': [],
        'student_n_n': [],
        'avg_mentor_reward': [],
        'avg_student_reward': [],
        'eq_type': []  # 均衡类型: 0-无, 1-分离, 2-混同
    }

    for episode in tqdm(range(episodes)):
        # 重置环境
        mentor_type = env.reset()

        # 导师选择行动
        mentor_action = mentor.choose_action(mentor_type)

        # 学生观察导师行动作为状态
        student_action = student.choose_action(mentor_action)

        # 执行动作并获取奖励
        mentor_reward, student_reward = env.step(mentor_action, student_action)

        # 更新智能体
        # 注意：由于是单次博弈，没有下一个状态（设为None）
        mentor.learn(mentor_type, mentor_action, mentor_reward)
        student.learn(mentor_action, student_action, student_reward)

        # 记录Q值
        # 导师Q值
        mentor_t2_idx = mentor.state_to_index("t2")
        history['mentor_d_d'].append(mentor.q_table[mentor_t2_idx, mentor_actions.index("D")])
        history['mentor_n_d'].append(mentor.q_table[mentor_t2_idx, mentor_actions.index("N")])

        # 学生Q值
        state_idx_d = student.state_to_index("D")
        state_idx_n = student.state_to_index("N")
        history['student_d_d'].append(student.q_table[state_idx_d, student_actions.index("D")])
        history['student_d_n'].append(student.q_table[state_idx_d, student_actions.index("N")])
        history['student_n_d'].append(student.q_table[state_idx_n, student_actions.index("D")])
        history['student_n_n'].append(student.q_table[state_idx_n, student_actions.index("N")])

        # 平均奖励
        history['avg_mentor_reward'].append(mentor_reward)
        history['avg_student_reward'].append(student_reward)

        # 确定均衡类型
        mentor_policy = mentor.get_policy()
        student_policy = student.get_policy()

        # 分离均衡: t1选D, t2选N
        if mentor_policy["t2"] == "N":
            history['eq_type'].append(1)  # 分离均衡
        # 混同均衡: t2也选D
        elif mentor_policy["t2"] == "D":
            history['eq_type'].append(2)  # 混同均衡
        else:
            history['eq_type'].append(0)  # 无均衡

    return mentor, student, history


def visualize_results(history, episodes, q0, penalty_student=0, penalty_mentor=0):
    """可视化训练结果"""

    # 创建大图
    plt.figure(figsize=(18, 12))

    # 1. 导师Q值（t2类型）
    plt.subplot(2, 2, 1)
    plt.plot(history['mentor_d_d'], color='r', linestyle='-', label='选择G的Q值')
    plt.plot(history['mentor_n_d'], color='b', linestyle='-', label='选择NG的Q值')
    plt.title(f'导师(M2)Q值演变 (q0={q0}, 惩罚:学生={penalty_student},导师={penalty_mentor})')
    plt.xlabel('训练轮数')
    plt.ylabel('Q值')
    plt.legend()
    plt.grid(True)

    # 2. 学生Q值（观察导师行动D）
    plt.subplot(2, 2, 2)
    plt.plot(history['student_d_d'], color='r', linestyle='-', label='选择W的Q值')
    plt.plot(history['student_d_n'], color='b', linestyle='-', label='选择NW的Q值')
    plt.title(f'学生Q值（观察G时）(q0={q0}, 惩罚:学生={penalty_student},导师={penalty_mentor})')
    plt.xlabel('训练轮数')
    plt.ylabel('Q值')
    plt.legend()
    plt.grid(True)

    # 3. 学生Q值（观察导师行动N）
    plt.subplot(2, 2, 3)
    plt.plot(history['student_n_d'], color='r', linestyle='-', label='选择W的Q值')
    plt.plot(history['student_n_n'], color='b', linestyle='-', label='选择NW的Q值')
    plt.title(f'学生Q值（观察NG时）(q0={q0}, 惩罚:学生={penalty_student},导师={penalty_mentor})')
    plt.xlabel('训练轮数')
    plt.ylabel('Q值')
    plt.legend()
    plt.grid(True)

    # 4. 平均奖励和均衡类型
    plt.subplot(2, 2, 4)

    # 平均奖励
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    # 平均奖励（使用移动平均平滑）
    window_size = 100
    mentor_reward_smoothed = np.convolve(history['avg_mentor_reward'], np.ones(window_size) / window_size, mode='valid')
    student_reward_smoothed = np.convolve(history['avg_student_reward'], np.ones(window_size) / window_size,
                                          mode='valid')

    ax1.plot(mentor_reward_smoothed, color='r', label='导师平均奖励')
    ax1.plot(student_reward_smoothed, color='b', label='学生平均奖励')
    ax1.set_ylabel('平均奖励')
    ax1.set_xlabel('训练轮数')
    ax1.legend(loc='upper left')

    # 均衡类型（使用移动平均）
    eq_smoothed = np.convolve(history['eq_type'], np.ones(window_size) / window_size, mode='valid')
    ax2.plot(eq_smoothed, color='black', linestyle='--', label='均衡类型')
    ax2.set_ylabel('均衡类型', color='black')
    ax2.set_ylim([-0.1, 2.5])
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['无均衡', '分离均衡', '混同均衡'])

    plt.title(f'奖励与均衡演变 (q0={q0}, 惩罚:学生={penalty_student},导师={penalty_mentor})')
    plt.tight_layout()
    plt.savefig(f'signaling_game_q0_{q0}_penalty_s{penalty_student}_m{penalty_mentor}.png', dpi=300)
    plt.show()


def get_eq_description(mentor, student):
    """获取均衡描述"""
    mentor_policy = mentor.get_policy()
    student_policy = student.get_policy()

    # 分离均衡: t1选D, t2选N
    if mentor_policy["t2"] == "N":
        student_action_d = student_policy.get("D", "D")
        student_action_n = student_policy.get("N", "D")
        return (
            f"分离均衡:\n"
            f"- 导师策略: t1选择D, t2选择N\n"
            f"- 学生策略: 观察D选择{student_action_d}, 观察N选择{student_action_n}\n"
            f"- 学生信念: 观察D推断t1(概率1), 观察N推断t2(概率1)"
        )
    # 混同均衡: t2也选D
    elif mentor_policy["t2"] == "D":
        student_action_d = student_policy.get("D", "D")
        return (
            f"混同均衡:\n"
            f"- 导师策略: t1和t2都选择D\n"
            f"- 学生策略: 观察D选择{student_action_d}\n"
            f"- 学生信念: 观察D信念μ(t1)=q0"
        )
    else:
        return "未收敛到纯策略均衡"


# 收益定义
payoffs_sep = {
    ('t1', 'D', 'D'): (5, 4),  # a, b
    ('t1', 'D', 'N'): (2, 1),  # c, d
    ('t2', 'D', 'D'): (3, 1),  # e, f
    ('t2', 'D', 'N'): (0.5, 3),  # g, h
    ('t2', 'N', 'D'): (4, 2),  # m, n
    ('t2', 'N', 'N'): (1, 1)  # o, p
}

payoffs_pool = {
    ('t1', 'D', 'D'): (5, 4),  # a, b
    ('t1', 'D', 'N'): (2, 1),  # c, d
    ('t2', 'D', 'D'): (3, 1),  # e, f
    ('t2', 'D', 'N'): (0.5, 3),  # g, h
    ('t2', 'N', 'D'): (4, 1),  # m, n
    ('t2', 'N', 'N'): (1, 2)  # o, p
}

# 训练参数
episodes = 10000  # 训练轮数
q0_values = [0.3, 0.7]  # 两个不同的先验概率
penalty_configs = [
    (0, 0),  # 无惩罚
    (0.5, 0.5),  # 中等惩罚
    (1.5, 1.5)  # 强惩罚
]

# 运行模拟
for q0 in q0_values:
    for penalty_student, penalty_mentor in penalty_configs:
        print(f"\n=== 训练 q0 = {q0}, 惩罚:学生={penalty_student},导师={penalty_mentor} ===")

        # 根据q0选择收益参数
        if q0 < 0.5:
            payoffs = payoffs_sep  # 低q0可能收敛到分离均衡
        else:
            payoffs = payoffs_pool  # 高q0可能收敛到混同均衡

        mentor, student, history = train_agents(q0, payoffs, episodes, penalty_student, penalty_mentor)

        # 可视化结果
        visualize_results(history, episodes, q0, penalty_student, penalty_mentor)

        # 显示最终策略和均衡
        print("\n最终策略:")
        print(f"导师策略: {mentor.get_policy()}")
        print(f"学生策略: {student.get_policy()}")
        print(get_eq_description(mentor, student))

        # 保存Q表
        print("\n导师Q表:")
        print(mentor.q_table)
        print("\n学生Q表:")
        print(student.q_table)