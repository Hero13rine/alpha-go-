import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F


# from mpi4py import MPI

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# 策略价值网络

class Residual_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Net(nn.Module):
    """policy-value network module"""

    def __init__(self, board_width, board_height):
        super().__init__()

        self.board_width = board_width
        self.board_height = board_height

        # 卷积块
        self.conv0 = nn.Conv2d(4, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        # 残差块
        self.ResNet1 = Residual_block(128, 128)
        self.ResNet2 = Residual_block(128, 128)
        self.ResNet3 = Residual_block(128, 128)
        self.ResNet4 = Residual_block(128, 128)
        """"""
        self.ResNet5 = Residual_block(128, 128)
        self.ResNet6 = Residual_block(128, 128)
        # self.ResNet7 = Residual_block(128, 128)
       #  self.ResNet8 = Residual_block(128, 128)

        # 策略端
        self.p_conv = nn.Conv2d(128, 4, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)

        self.p_fc = nn.Linear(4 * board_width * board_height, board_width * board_height)

        # 价值端
        self.v_conv = nn.Conv2d(128, 2, kernel_size=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(2)

        self.v_fc1 = nn.Linear(2 * board_height * board_width, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # 卷积块
        x = self.conv0(state_input)
        x = self.bn0(x)
        x = self.relu(x)
        # 残差块
        x = self.ResNet1(x)
        x = self.ResNet2(x)
        x = self.ResNet3(x)
        x = self.ResNet4(x)
        """"""
        x = self.ResNet5(x)
        x = self.ResNet6(x)
        # x = self.ResNet7(x)
        # x = self.ResNet8(x)


        # 策略端

        policy = self.p_conv(x)
        policy = self.p_bn(policy)
        policy = self.relu(policy)

        policy = self.p_fc(policy.view(-1, 4 * self.board_width * self.board_height))
        policy = torch.log_softmax(policy, dim=1)

        # 价值端
        value = self.v_conv(x)
        value = self.v_bn(value)
        value = self.relu(value)

        value = self.v_fc1(value.view(-1, 2 * self.board_height * self.board_width))
        value = self.relu(value)
        value = self.v_fc2(value)
        value = torch.tanh(value)

        return policy, value


class PolicyValueNet():
    """policy-value network """

    def __init__(self, board_width, board_height,
                 model_file=None):

        self.board_width = board_width
        self.board_height = board_height
        # self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module

        # if istrain==True:
        #     self.policy_value_net = Policy_Value_Model(board_width, board_height).cuda()
        #     self.with_gpu=True
        # else:
        self.policy_value_net = Net(board_width, board_height).cuda()
        self.with_gpu = True
        self.optimizer = optim.Adam(self.policy_value_net.parameters())

        if model_file:
            try:
                net_params = torch.load(model_file)
                self.policy_value_net.load_state_dict(net_params)
            except RuntimeError:
                pass

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """

        state_batch = torch.FloatTensor(np.array(state_batch)).cuda()
        log_act_probs, value = self.policy_value_net(state_batch)
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        return act_probs, value.data.cpu().numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        # 输出剩余空位的概率分布
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
            -1, 4, self.board_width, self.board_height))
        if self.with_gpu:

            log_act_probs, value = self.policy_value_net(
                torch.from_numpy(current_state).float().cuda())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        # value_cpu=value.item()
        # del log_act_probs, value,
        # torch.cuda.empty_cache()

        else:
            log_act_probs, value = self.policy_value_net(
                torch.from_numpy(current_state).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())

        act_probs = zip(legal_positions, act_probs[legal_positions])
        #  act_probs = act_probs[legal_positions]
        # print(type(value))
        return act_probs, value.item()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable

        state_batch = torch.from_numpy(np.array(state_batch)).float().cuda()
        mcts_probs = torch.from_numpy(np.array(mcts_probs)).float().cuda()
        winner_batch = torch.from_numpy(np.array(winner_batch)).float().cuda()

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
            )

        del log_act_probs, value, value_loss, policy_loss, state_batch, mcts_probs, winner_batch
        torch.cuda.empty_cache()

        return loss.item(), entropy.item()

        # return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
