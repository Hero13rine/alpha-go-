from board import Board
import numpy as np
import pickle
from human_AI import MCTSPlayer, Human
from policy_value_net import PolicyValueNet
import torch
from Mcts_pure import MCTSPlayer as pure

class Game(object):

    def __init__(self, board, **kwargs):
        self.board = board


    def graphic(self, board, player1, player2):
        """
        画出棋盘
        :param board: 传入棋盘
        :param player1: 第一方
        :param player2: 第二方
        :return:
        """

        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        print("     0   ", end='')
        for x in range(width - 1):
            print(x + 1, sep=' ', end='   ')
        print()
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(4), end='')
                elif p == player2:
                    print('O'.center(4), end='')
                else:
                    print('—'.center(4), end='')
            print('\r')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        开始游戏(人机对战模式)
        :param player1:
        :param player2:
        :param start_player:谁开始游戏
        :param is_shown:是否输出棋盘
        :return:赢家
        """
        if start_player not in (0, 1):
            raise Exception("初始化玩家异常")
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_color(p1)
        player2.set_player_color(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("游戏结束，赢家为：", players[winner], winner)
                    else:
                        print("游戏结束，双方平局")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """
        使用 带网络的 MCTS 开始自我游戏，并存储self-play数据：
        (state, mcts_probs, z) 用于训练
        state：棋盘局面
        mcts_probs： 先验概率
        z: 赢棋方的labels标签

        自博弈的原理：
        每一次落子都执行mcts_alpha的模拟操作得到先验概率mcts_probs
        根据得到的概率p argmax选择一个动作执行到真实棋盘中，并且储存（state, mcts_probs, z）到样本池中
        之后赋值


        :param player:
        :param is_shown:
        :param temp:
        :return:
        """
        self.board.init_train_board()
        # self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []
        # 第一步的落子选取胜率点为50的

        move, move_probs = player.get_action(self.board,
                                             temp=temp,
                                             return_prob=2)

        # 初始化数据并储存
        states.append(self.board.current_state())
        mcts_probs.append(move_probs)
        current_players.append(self.board.current_player)
        # 预落子
        self.board.do_move(move)
        if is_shown:
            self.graphic(self.board, p1, p2)
        end, winner = self.board.game_end()
        if end:
            # 从每个局面的当前玩家的角度来看的获胜者
            winners_z = np.zeros(len(current_players))
            if winner != -1:
                winners_z[np.array(current_players) == winner] = 1.0
                winners_z[np.array(current_players) != winner] = -1.0
            # 重置游戏
            player.reset_player()
            if is_shown:
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
            return winner, zip(states, mcts_probs, winners_z)
        # 之后正常selfplay
        while True:
            move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
            # 初始化数据并储存
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # 预落子
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # 从每个局面的当前玩家的角度来看的获胜者
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # 重置游戏
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def set_first_player(self):

        while True:
            x = input("是否先手 y/n")
            if x == 'y':
                return 0
            elif x == 'n':
                return 1
            else:
                print("非法输入,重新输入")


def play():
    n = 5
    width, height = 15, 15
    # model_file = 'current_policy.model'
    model_file = 'current_policy.model'

    #model_file = 'AA.model'

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)
        first_player = game.set_first_player()
        # try:

        #  policy_param = pickle.load(open(model_file, 'rb'))

        # except:
        # policy_param = pickle.load(open(model_file, 'rb'),
              #                         encoding='bytes')  # To support python3
        policy_param = model_file

        best_policy = PolicyValueNet(width, height, policy_param)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        mcts_pure1 = pure(c_puct=5, n_playout=400)
        # mcts_pure2 = MCTSPlayer(c_puct=5, n_playout=10000)
        human2 = Human()

        human = Human()


        # set start_player=0 for human first
        # y 人先手   n ai先手
        game.start_play(human, mcts_player, start_player=first_player, is_shown=1)
        # game.start_play(human, human2, start_player=first_player, is_shown=1)
        # game.start_play(mcts_pure1, mcts_player, start_player=first_player, is_shown=1)


    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    play()