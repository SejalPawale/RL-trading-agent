# trading_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ForexTradingEnvironment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        feature_cols,
        window_size=30,
        initial_balance=10_000.0,
        lot_size=10_000,
        transaction_cost=0.00005,
        slippage=0.00002,
        stop_loss_options=None,
        take_profit_options=None,
        reward_mode="pnl"
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.lot_size = lot_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.stop_loss_options = stop_loss_options or [0.0010, 0.0015, 0.0020]
        self.take_profit_options = take_profit_options or [0.0010, 0.0015, 0.0020]
        self.reward_mode = reward_mode

        self.num_trade_actions = 3
        self.num_sl_options = len(self.stop_loss_options)
        self.num_tp_options = len(self.take_profit_options)
        self.action_space = spaces.Discrete(self.num_trade_actions * self.num_sl_options * self.num_tp_options)

        obs_dim = self.window_size * len(self.feature_cols) + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_pnls = []
        self.equity_curve = [self.initial_balance]
        self.steps_in_position = 0

    def _decode_action(self, action):
        trade_action = action // (self.num_sl_options * self.num_tp_options)
        rem = action % (self.num_sl_options * self.num_tp_options)
        sl_idx = rem // self.num_tp_options
        tp_idx = rem % self.num_tp_options
        return trade_action, self.stop_loss_options[sl_idx], self.take_profit_options[tp_idx]

    def _get_price(self):
        return float(self.df.loc[self.current_step, "close"])

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        window = self.df.loc[start:end - 1, self.feature_cols].values.astype(np.float32).flatten()

        position_features = np.array([
            float(self.position),
            float(0.0 if self.entry_price is None else self.entry_price),
            float(self.balance / self.initial_balance),
            float(self.equity / self.initial_balance)
        ], dtype=np.float32)

        obs = np.concatenate([window, position_features], axis=0)
        return obs

    def _mark_to_market_equity(self, current_price):
        if self.position == 0 or self.entry_price is None:
            return self.balance

        raw_pnl = (current_price - self.entry_price) * self.lot_size * self.position
        return self.balance + raw_pnl

    def _close_position(self, exit_price):
        pnl = (exit_price - self.entry_price) * self.lot_size * self.position
        cost = self.transaction_cost * self.lot_size
        realized_pnl = pnl - cost

        self.balance += realized_pnl
        self.position = 0
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.trade_pnls.append(realized_pnl)

        return realized_pnl

    def _calculate_reward(self, realized_pnl):
        if self.reward_mode == "pnl":
            return realized_pnl / 100.0

        if self.reward_mode == "risk_adjusted":
            penalty = 0.0
            if realized_pnl < 0:
                penalty = abs(realized_pnl) * 0.1
            return (realized_pnl - penalty) / 100.0

        if self.reward_mode == "drawdown_penalty":
            current_drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
            return (realized_pnl / 100.0) - (5.0 * current_drawdown)

        return realized_pnl / 100.0

    def step(self, action):
        terminated = False
        truncated = False
        reward = 0.0
        info = {}

        trade_action, sl_value, tp_value = self._decode_action(int(action))
        current_price = self._get_price()

        if self.position != 0:
            if self.position == 1:
                if current_price <= self.stop_loss or current_price >= self.take_profit:
                    realized_pnl = self._close_position(current_price)
                    reward += self._calculate_reward(realized_pnl)
            elif self.position == -1:
                if current_price >= self.stop_loss or current_price <= self.take_profit:
                    realized_pnl = self._close_position(current_price)
                    reward += self._calculate_reward(realized_pnl)

        if self.position == 0:
            if trade_action == 1:
                entry = current_price + self.slippage
                self.position = 1
                self.entry_price = entry
                self.stop_loss = entry - sl_value
                self.take_profit = entry + tp_value
                self.balance -= self.transaction_cost * self.lot_size
            elif trade_action == 2:
                entry = current_price - self.slippage
                self.position = -1
                self.entry_price = entry
                self.stop_loss = entry + sl_value
                self.take_profit = entry - tp_value
                self.balance -= self.transaction_cost * self.lot_size

        if self.position != 0:
            self.steps_in_position += 1
        self.equity = self._mark_to_market_equity(current_price)
        self.peak_equity = max(self.peak_equity, self.equity)
        self.equity_curve.append(self.equity)

        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            if self.position != 0:
                final_price = float(self.df.loc[self.current_step, "close"])
                realized_pnl = self._close_position(final_price)
                reward += self._calculate_reward(realized_pnl)
                self.equity = self.balance
                self.equity_curve.append(self.equity)
            terminated = True

        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "balance": self.balance,
            "equity": self.equity,
            "position": self.position,
            "num_trades": len(self.trade_pnls)
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_observation()
        info = {}
        return obs, info

    def render(self):
        print(
            f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
            f"Equity: {self.equity:.2f}, Position: {self.position}"
        )

    def get_time_in_market(self):
        total_steps = max(1, self.current_step - self.window_size)
        return self.steps_in_position / total_steps