import copy

import numpy as np

from Mcts_AlphaZero import MCTS


class Human(object):
    """
    人类玩家
    """
    def __init__(self):
        self.player = None

    def set_player_color(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("落子  用 <,> 隔开:")
            if isinstance(location, str):
                location = [int(n) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1

        if move == -1 or move not in board.availables:
            print("非法落子")
            move = self.get_action(board)

        return move


class MCTSPlayer(object):
    """
    AI玩家
    """

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self.is_selfplay = is_selfplay

    def set_player_color(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        available_moves = board.availables

        move_probs = np.zeros(board.width*board.height)
        move_value = np.ones(board.width*board.height)
        if len(available_moves) > 0:
            acts, probs = self.mcts.get_move_prob(board, temp)
            move_probs[list(acts)] = probs
            if return_prob == 2:
                """
                    目的是选取此刻胜率为百分之五十的点
                """
                # 接受先验概率和 胜率、
                value50 = 0
                move50 = -1
                """
                    直接将board传入网络，但是此时网络的胜率是 一个tensor 代表着一个局面的胜率
                    所以需要对于每一个 available moves 遍历得到 value 
                """
                for move in board.availables:
                    state_copy = copy.deepcopy(board)  # 深拷贝棋盘
                    state_copy.do_move(move)  # 落子
                    move_prob, value = self.mcts.policy(state_copy)
                    move_value[move] = value

                move = (np.abs(move_value - 0.0)).argmin()
                self.mcts.update_with_move(move)
                return move, move_probs

            if self.is_selfplay:
                """
                在训练过程中, 在根节点添加 Dirichlet Noise 以提高探索的广度
                防止ai误入歧途
                """
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                """
                默认 temp=1e-3,此时几乎是等价的
                选择概率最高的走法
                 """
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")