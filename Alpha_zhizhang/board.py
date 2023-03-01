import numpy as np
import re
import random


class Board(object):
    """棋盘类"""

    def __init__(self, **kwargs):
        self.last_move = None
        self.current_player = None
        self.availables = None
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # 用字典储存棋盘,
        # key: 落子处棋盘的坐标（伪坐标，后面要用move_to_value转换）,
        # value: 落子方，颜色
        # state 是用来记录已下子的
        self.states = {}
        # 连字数
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        """
        初始化棋盘
        :param start_player: 先手方
        :return:
        """
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('棋盘的大小要大于连字数{}'.format(self.n_in_row))

        self.current_player = self.players[start_player]  # start player
        # 将可落子点用列表记录
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def init_train_board(self, start_player=0):
        """
        初始化自博弈棋盘
        :return:
        """
        self.current_player = self.players[start_player]  # start player
        # 将可落子点用列表记录
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1
        random_num = np.random.randint(1,7) # 返回1个[1,5)时间的随机整数
        move_list = np.random.choice(self.width * self.height, random_num, replace=False)

        for move in move_list:
            self.do_move(move)

    def move_to_location(self, move):
        """
        将落子转化成坐标形式
        3*3 board's moves like:
        6 7 8
        3 4 5
        0 1 2
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        """

        :param location: 输入的列表坐标
        :return:
        """
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self):
        """
        从当前玩家的角度返回棋盘状态。
        状态形状：4*宽*高
        以棋盘形式用1表示
                         记录上上一步落子位置
                         记录上上一步反方落子位置
        square_state[0] :记录当前方所有落子位置
        square_state[1] :记录相反方所有落子位置
                         记录上一步最后落子位置
        square_state[2] :记录最后子落子位置
        square_state[3] :记录当前方颜色

        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # 记录最后子落子位置
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # 记录当前方颜色
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def five_connect(self, h, w, width, height, states, last_move, last_player):
        # 横向
        bias = min(w, 4)
        for i in range(last_move - bias, last_move + 1):
            if (width - 1 - i % width < 4):
                break
            ret = 0
            for k in range(i, i + 5):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break

            if ret == 0:
                return True
        # 纵向
        bias = min(h, 4)
        for i in range(last_move - bias * width, last_move + width, width):
            if (width - 1 - i // width < 4):
                break
            ret = 0
            for k in range(i, i + 5 * width, width):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break
            if ret == 0:
                return True
        # 正斜
        bias = min(min(h, 4), min(w, 4))
        for i in range(last_move - bias * width - bias, last_move + width + 1, width + 1):
            if (width - 1 - i // width < 4 or width - 1 - i % width < 4):
                break
            ret = 0
            for k in range(i, i + 5 * width + 5, width + 1):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break
            if ret == 0:
                return True
        # 反斜
        bias = min(min(height - 1 - h, 4), min(w, 4))
        for i in range(last_move + bias * width - bias, last_move - width + 1, -width + 1):
            if (width - 1 - i % width < 4 or i // width < 4):
                break
            ret = 0
            for k in range(i, i - 5 * width + 5, -width + 1):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break
            if ret == 0:
                return True
        return False

    def long_connect(self, h, w, width, height, states, last_move, last_player):
        bias = min(w, 5)
        for i in range(last_move - bias, last_move + 1):
            if width - 1 - i % width < 5:
                break
            ret = 0
            for k in range(i, i + 6):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break

            if ret == 0:
                return True
        # 纵向
        bias = min(h, 5)
        for i in range(last_move - bias * width, last_move + width, width):
            if width - 1 - i // width < 5:
                break
            ret = 0
            for k in range(i, i + 6 * width, width):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break
            if ret == 0:
                return True
        # 正斜
        bias = min(min(h, 5), min(w, 5))
        for i in range(last_move - bias * width - bias, last_move + width + 1, width + 1):
            if width - 1 - i // width < 5 or width - 1 - i % width < 5:
                break
            ret = 0
            for k in range(i, i + 6 * width + 6, width + 1):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break
            if ret == 0:
                return True
        # 反斜
        bias = min(min(height - 1 - h, 5), min(w, 5))
        for i in range(last_move + bias * width - bias, last_move - width + 1, -width + 1):
            if width - 1 - i % width < 5 or i // width < 5:
                break
            ret = 0
            for k in range(i, i - 6 * width + 6, -width + 1):
                if k not in states.keys() or states[k] != last_player:
                    ret = 1
                    break
            if ret == 0:
                return True
        return False

    def three_three(self, h, w, width, height, states, last_move, last_player):

        jump_three1 = 'o1o11o'
        jump_three2 = 'o11o1o'
        connect_three1 = 'oo111o'
        connect_three2 = 'o111oo'
        three = 0
        # 横向
        m_string = ''
        bias = min(w, 4)
        for i in range(last_move - bias, last_move + min(width - 1 - w, 4) + 1):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])
        isfind = re.search(connect_three1, m_string)
        if isfind:
            if self.tt_special_case(m_string, isfind, 1) == False:
                three += 1
        else:
            isfind = re.search(connect_three2, m_string)
            if isfind:
                if self.tt_special_case(m_string, isfind, 2) == False:
                    three += 1
        if re.search(jump_three1, m_string):
            three += 1
        elif re.search(jump_three2, m_string):
            three += 1

        if three > 1:
            return True
        # 纵向
        m_string = ''
        bias = min(h, 4)
        for i in range(last_move - bias * width, last_move + width * min(width - 1 - h, 4) + width, width):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])
        isfind = re.search(connect_three1, m_string)
        if isfind:
            if self.tt_special_case(m_string, isfind, 1) == False:
                three += 1
        else:
            isfind = re.search(connect_three2, m_string)
            if isfind:
                if self.tt_special_case(m_string, isfind, 2) == False:
                    three += 1
        if re.search(jump_three1, m_string):
            three += 1
        elif re.search(jump_three2, m_string):
            three += 1

        if three > 1:
            return True

        # 正斜
        m_string = ''
        bias = min(min(h, 4), min(w, 4))
        for i in range(last_move - bias * width - bias,
                       last_move + (width + 1) * min(min(width - 1 - h, width - 1 - w), 4) + width + 1, width + 1):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])
        isfind = re.search(connect_three1, m_string)
        if isfind:
            if self.tt_special_case(m_string, isfind, 1) == False:
                three += 1
        else:
            isfind = re.search(connect_three2, m_string)
            if isfind:
                if self.tt_special_case(m_string, isfind, 2) == False:
                    three += 1
        if re.search(jump_three1, m_string):
            three += 1
        elif re.search(jump_three2, m_string):
            three += 1
        if three > 1:
            return True

        # 反斜(从右上往左下)
        m_string = ''
        bias = min(min(width - 1 - w, 4), min(h, 4))
        for i in range(last_move - bias * (width - 1),
                       last_move + (width - 1) * min(min(width - 1 - h, min(w, 4)), 4) + width - 1, width - 1):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])
        isfind = re.search(connect_three1, m_string)
        if isfind:
            if self.tt_special_case(m_string, isfind, 1) == False:
                three += 1
        else:
            isfind = re.search(connect_three2, m_string)
            if isfind:
                if self.tt_special_case(m_string, isfind, 2) == False:
                    three += 1
        if re.search(jump_three1, m_string):
            three += 1
        elif re.search(jump_three2, m_string):
            three += 1
        if three > 1:
            return True

        return False

        # 三三禁手的特殊情况

    def tt_special_case(self, m_string, isfind, t_case):
        if t_case == 1:  # oo111o
            if isfind.start() + 6 < len(m_string):
                if m_string[isfind.start() + 6] == '1':
                    return True

        elif t_case == 2:  # o111oo
            if isfind.start() > 0:
                if m_string[isfind.start() - 1] == '1':
                    return True
        return False

        # 四四禁手的特殊情况

    def ff_special_case(self, m_string, isfind, f_case):
        if f_case == 1:
            if isfind.start() > 0:
                if m_string[isfind.start() - 1] == '1':
                    return True
            if isfind.start() + 5 < len(m_string):
                if m_string[isfind.start() + 5] == '1':
                    return True
            return False
        elif f_case == 2:

            if isfind.start() > 0:
                if isfind.start() + 6 < len(m_string):
                    if m_string[isfind.start() - 1] == '1' and ((
                                                                        m_string[isfind.start() + 5] == 'o' and
                                                                        m_string[isfind.start() + 6] == '1') or (
                                                                        m_string[isfind.start() + 5] == '0')):
                        return True

                    return False

                if isfind.start() + 5 < len(m_string):
                    if m_string[isfind.start() - 1] == '1' and m_string[isfind.start() + 5] == '0':
                        return True

                    return False
                if m_string[isfind.start() - 1] == '1':
                    return True
                return False
            else:
                return False

        else:
            if isfind.start() + 5 < len(m_string):
                if isfind.start() - 2 >= 0:
                    if (m_string[isfind.start() - 2] == '1' and m_string[isfind.start() - 1] == 'o') or (
                            m_string[isfind.start() - 1] == '0'
                    ) and m_string[isfind.start() + 5] == '1':
                        return True
                    return False
                elif isfind.start() - 1 >= 0:
                    if m_string[isfind.start() + 5] == '1' and m_string[isfind.start() - 1] == '0':
                        return True
                    return False

                if m_string[isfind.start() + 5] == '1':
                    return True
                return False
            else:
                return False

    def four_four(self, h, w, width, height, states, last_move, last_player):
        jump_four1 = '111o1'
        jump_four2 = '1o111'
        jump_four3 = '11o11'
        connect_four1 = 'o1111'
        connect_four2 = '1111o'
        four = 0
        # 横向
        m_string = ''
        bias = min(w, 5)
        for i in range(last_move - bias, last_move + min(width - 1 - w, 5) + 1):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])

        isfind = re.search(jump_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(jump_four2, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1

        isfind = re.search(jump_four3, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(connect_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 2) == False:
                four += 1
        else:
            isfind = re.search(connect_four2, m_string)
            if isfind:
                if self.ff_special_case(m_string, isfind, 3) == False:
                    four += 1
        if four > 1:
            return True
        # 纵向
        m_string = ''
        bias = min(h, 5)
        for i in range(last_move - bias * width, last_move + width * min(width - 1 - h, 5) + width, width):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])

        isfind = re.search(jump_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(jump_four2, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1

        isfind = re.search(jump_four3, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(connect_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 2) == False:
                four += 1
        else:
            isfind = re.search(connect_four2, m_string)
            if isfind:
                if self.ff_special_case(m_string, isfind, 3) == False:
                    four += 1
        if four > 1:
            return True

        # 正斜
        m_string = ''
        bias = min(min(h, 5), min(w, 5))
        for i in range(last_move - bias * width - bias,
                       last_move + (width + 1) * min(min(width - 1 - h, width - 1 - w), 5) + width + 1, width + 1):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])

        isfind = re.search(jump_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(jump_four2, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1

        isfind = re.search(jump_four3, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(connect_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 2) == False:
                four += 1
        else:
            isfind = re.search(connect_four2, m_string)
            if isfind:
                if self.ff_special_case(m_string, isfind, 3) == False:
                    four += 1
        if four > 1:
            return True

        # 反斜(从右上往左下看)
        m_string = ''
        bias = min(min(width - 1 - w, 5), min(h, 5))
        for i in range(last_move - bias * (width - 1),
                       last_move + (width - 1) * min(min(width - 1 - h, min(w, 5)), 5) + width - 1, width - 1):
            if i not in states.keys():
                m_string += 'o'
            else:
                m_string += str(states[i])

        isfind = re.search(jump_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(jump_four2, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1

        isfind = re.search(jump_four3, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 1) == False:
                four += 1
        isfind = re.search(connect_four1, m_string)
        if isfind:
            if self.ff_special_case(m_string, isfind, 2) == False:
                four += 1
        else:
            isfind = re.search(connect_four2, m_string)
            if isfind:
                if self.ff_special_case(m_string, isfind, 3) == False:
                    four += 1
        if four > 1:
            return True

        return False

    def win_end(self):  # 判断是否有一方胜利
        n_c = 5  # 几连就结束
        width = self.width
        height = self.height
        states = self.states
        last_move = self.last_move
        last_player = (1 if self.current_player == 2 else 2)
        h, w = self.move_to_location(last_move)

        if last_player == 1:
            # 禁手优先级：长连，五连，三四禁。
            if self.long_connect(h, w, width, height, states, last_move, last_player):
                return True, 2
            if self.five_connect(h, w, width, height, states, last_move, last_player):
                return True, 1
            if self.three_three(h, w, width, height, states, last_move, last_player):
                return True, 2
            if self.four_four(h, w, width, height, states, last_move, last_player):
                return True, 2
        else:
            if self.five_connect(h, w, width, height, states, last_move, last_player):
                return True, 2

        return False, last_player

    def game_end(self):
        """
        判断是否结束
        （禁手判断）
        :return: 胜负， winner颜色
        """
        win, winner = self.win_end()
        if win:
            return True, winner
        elif not len(self.availables):  # 平局
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player
