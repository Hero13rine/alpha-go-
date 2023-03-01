import random
import numpy as np
from collections import defaultdict, deque
from Game import Board, Game
from human_AI import MCTSPlayer
from Mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net import PolicyValueNet
# from torch import multiprocessing as mp
from multiprocessing import Lock
import multiprocessing as mp
import os


class TrainPipeline():
    def __init__(self, init_model=None):
        # init_games
        self.episode_len = None
        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # init training params
        self.learn_rate = 2e-3  # 学习速录
        self.lr_multiplier = 1.0  # 修正学习速率的参数
        self.temp = 1.0  # 速率参数
        self.n_playout = 400  # 树的模拟次数
        self.c_puct = 5  # uct常量
        self.buffer_size = 500000  # 样本池容量
        self.batch_size = 512  #
        self.data_buffer = deque(maxlen=self.buffer_size)  # 样本池
        self.play_batch_size = 1  # 自学习次数
        self.epochs = 5  #
        self.kl_targ = 0.02
        self.check_freq = 500  # 检查次数
        self.game_batch_num = 1500  # 训练次数
        self.best_win_ratio = 0.0  # 胜率

        self.pure_mcts_playout_num = 1000

        self.batch = 0

        # 有无已知模型判断
        if init_model:
            # 从已有网络开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # 重新训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)

        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """
        数据增强
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, data):
        """collect self-play data for training"""
        self.data_buffer.extend(data)
        print(len(self.data_buffer))
    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            """kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5"""

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=6):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            if len(self.data_buffer) > self.batch_size:
                self.batch += 1
                # print('run')
                print("batch i:{}".format(self.batch))
                loss, entropy = self.policy_update()
                lock.acquire()
                self.policy_value_net.save_model('./current_policy.model')
                lock.release()
                if (self.best_win_ratio == 1.0 and
                        self.pure_mcts_playout_num < 5000):
                    self.pure_mcts_playout_num += 1000
                    self.best_win_ratio = 0.0
                # check the performance of the current model,
                # and save the model params
                if (self.batch+1) % self.check_freq == 0 or (self.batch+1) == 100:
                    print("current self-play batch: {}".format(self.batch+1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./best_policy.model')

        except KeyboardInterrupt:

            self.policy_value_net.save_model('./training_policy.model')

            print('\n\rquit')


def collect_data(q, lock):

    length = 15
    data_buffer = deque(maxlen=10000)
    board = Board(width=length, height=length, n_in_row=5)
    game = Game(board)

    while True:
        # 更新数据
        lock.acquire()
        policy_value_net = PolicyValueNet(length, length, model_file='current_policy.model')
        lock.release()
        mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400,
                                 is_selfplay=1)

        winner, play_data = game.start_self_play(mcts_player,
                                                      temp=1)
        play_data = list(play_data)[:]
        episode_len = len(play_data)
        # augment the data
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(length, length)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))

        print('put')
        print("----in 子进程 pid=%d ,父进程的pid=%d---" % (os.getpid(), os.getppid()))
        q.put(extend_data)


def train(q, lock):
    training_pipeline = TrainPipeline()

    get_times = 0
    while True:
        if q.empty() is False:
            data1 = q.get_nowait()
            print('get1')
            lock.acquire()   #上锁
            training_pipeline.collect_selfplay_data(data1)
            get_times += 1
            lock.release()   #解锁
        """
        if q.empty() is False:
            data2 = q.get_nowait()
            print('get2')
            training_pipeline.collect_selfplay_data(data2)
            get_times += 1
        
        if q.empty() is False:
            data3 = q.get_nowait()
            print('get3')
            training_pipeline.collect_selfplay_data(data3)
            get_times += 1
        if q.empty() is False:
            data4 = q.get_nowait()
            print('get4')
            training_pipeline.collect_selfplay_data(data4)
            get_times += 1
        """
        if get_times > 9:
            training_pipeline.run()


if __name__ == '__main__':

    q = mp.Queue()
    lock = Lock()
    p1 = mp.Process(target=collect_data, args=(q, lock,))
    p2 = mp.Process(target=collect_data, args=(q, lock,))
    # p3 = mp.Process(target=collect_data, args=(q,))
    # p4 = mp.Process(target=collect_data, args=(q,))
    # p5 = mp.Process(target=collect_data, args=(q,))
    # p6 = mp.Process(target=collect_data, args=(q,))
    p5 = mp.Process(target=train, args=(q, lock,))

    p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    p5.start()


