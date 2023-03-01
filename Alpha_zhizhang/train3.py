import random
import numpy as np
from collections import defaultdict, deque
from Game import Board, Game
from human_AI import MCTSPlayer
from Mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net import PolicyValueNet
import torch.multiprocessing as mp
import torch


class TrainPipeline():
    def __init__(self, q, init_model=None):
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
        self.buffer_size = 150000  # 样本池容量
        self.batch_size = 256  #
        self.data_buffer = deque(maxlen=self.buffer_size)  # 样本池
        self.play_batch_size = 1  # 自学习次数
        self.epochs = 5  #
        self.kl_targ = 0.02
        self.check_freq = 500  # 检查次数
        self.game_batch_num = 1500  # 训练次数
        self.best_win_ratio = 0.0  # 胜率

        self.pure_mcts_playout_num = 1000
        self.qq = q  # Queue

        # 有无已知模型判断

        if init_model:
            # 从已有网络开始训练
            print('from :', init_model)
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

    def getq(self):
        return self.qq


    def set_prosess(self):
        p1 = mp.Process(target=self.collect_selfplay_data, args=(self.getq(),))

        p1.start()

        data1 = q.get()

        p1.join()

        p1.close()

        self.data_buffer.extend(data1)
    """

    def set_prosess(self):
        p1 = mp.Process(target=self.collect_selfplay_data, args=(self.getq(),))
        p2 = mp.Process(target=self.collect_selfplay_data, args=(self.getq(),))
        p3 = mp.Process(target=self.collect_selfplay_data, args=(self.getq(),))
        p4 = mp.Process(target=self.collect_selfplay_data, args=(self.getq(),))

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        data1 = q.get()
        data2 = q.get()
        data3 = q.get()
        data4 = q.get()

        p1.join()
        p2.join()
        p3.join()
        p4.join()
        
        p1.close()
        p2.close()
        p3.close()
        p4.close()

        self.data_buffer.extend(data1)
        self.data_buffer.extend(data2)
        self.data_buffer.extend(data3)
        self.data_buffer.extend(data4)
"""
    def collect_selfplay_data(self, q, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            q.put(play_data)


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
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

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

    def policy_evaluate(self, n_games=10):
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
            print(i)
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            print(winner)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.set_prosess()
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                # check the performance of the current model,
                # and save the model params
                if (i + 1) % 10 == 0:
                    self.policy_value_net.save_model('./current_policy.model')
                if (i+1) % self.check_freq == 0:
                # if (i + 1) % 2 == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate()
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:

            self.policy_value_net.save_model('./training_policy.model')

            print('\n\rquit')


if __name__ == '__main__':
    q = mp.Queue()
    training_pipeline = TrainPipeline(q)
    training_pipeline.run()