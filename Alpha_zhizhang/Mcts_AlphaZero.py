import numpy as np
import copy


def softmax(x):
    """
        归一化函数

        参数：
        x --- 一个矩阵, m * n,其中m表示向量个数，n表示向量维度

        返回：
        softmax计算结果
        """
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode:
    """MCTS Node"""

    def __init__(self, parent, prior_p):
        """

        :param prior_p: 先验概率
        :param parent:
        """

        self.parent = parent  # 父结点
        self.children = {}  # 子结点


        self.P = prior_p
        self.Q = 0  # 胜率
        self.u = 0
        self.num_of_visit = 0                   # 访问次数N

    def expend(self, action_priors):
        """
        通过创建新的孩子来扩展树。
        :param action_priors: 根据拓展策略 是一个元组记录落子动作及其先验概率
        :return:
        """

        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        根据公式从子节点中选择价值最大的节点


        :param c_puct: 常量
        :return: 一个元组(action, next_node)
        """

        return max(self.children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """
        向上更新节点的value和visit_times

        :param leaf_value:
        :return:当前玩家的子树评估值
        """

        self.num_of_visit += 1

        self.Q += 1.0*(leaf_value - self.Q) / self.num_of_visit

    def update_policy(self, leaf_value):
        """
        是MCTS backward的阶段
        本函数相当于对与update()的调用,但加入了递归功能,使参数可以返回到根节点
        :param leaf_value: 接受从simulation阶段返回的最终结果
        :return:
        """
        if self.parent:
            self.parent.update_policy(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        相当于UCT
        计算并返回此节点的值。 它是叶子评估 Q 的组合，并且该节点的先验已根据其访问次数 u 进行了调整。
        c_puct: (0, inf) 中的一个数字，控制相对影响 value Q 和先验概率 P，在该节点的得分上
        """
        self.u = (c_puct * self.P *
                   np.sqrt(self.parent.num_of_visit) / (1 + self.num_of_visit))
        return self.Q + self.u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MCTS(object):
    """
    蒙特卡洛树搜索类
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """

        :param policy_value_fn:在网络中定义,接受对于当前玩家state状态并输出的函数（动作，概率）元组的列表以及 [-1, 1] 中的分数
             （即从当前的比赛结束得分的期望值玩家的视角）
        :param c_puct:(0, inf) 中的一个数字，用于控制探索的速度收敛到最大值策略。 更高的价值意味着更快的收敛。
        :param n_playout:模拟次数
        """

        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def monte_carlo_tree_search(self, state):
        """
        从根节点进行一次搜索,得到子节点的并且返回到根节点记录

        :param state: 是对于当前局面状态的记录, 从现有数据拷贝
        :return:
        """
        node = self.root

        while True:
            if node.is_leaf():
                break

            action, node = node.select(self.c_puct)
            state.do_move(action)

        action_probs, leaf_value = self.policy(state)

        end, winner = state.game_end()

        if not end:
            node.expend(action_probs)

        else:

            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_policy(-leaf_value)

    def get_move_prob(self, state, temp=1e-3):
        """
        依次运行所有 落子动作 并返回可用的操作和对应的概率。
        :param state: 当前游戏状态
        :param temp: (0, 1] 中的温度参数控制探索级别
        :return:
        """
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.monte_carlo_tree_search(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.num_of_visit)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """
        更新根节点状态
        :param last_move:
        :return:
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)