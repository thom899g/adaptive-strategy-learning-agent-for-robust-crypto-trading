from typing import Dict, Any
import logging
import numpy as np
import pandas as pd
from stable_baselines3 import A2C
from gym.utils import seeding
from collections import deque

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketEnvironment:
    """A custom Gym environment for cryptocurrency trading."""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float):
        self.data = data
        self.initial_balance = initial_balance
        self.seed = None
        
        # State representation: [price, volume, moving averages]
        self.state_dim = 5  # Example state space
        
        # Action space: 3 actions (hold, buy, sell)
        self.action_space = ['hold', 'buy', 'sell']
        
        # Reward calculation parameters
        self.reward_params = {'max_return': 0.1, 'min_return': -0.1}
        
    def _get_state(self, idx):
        """Get the current state of the market."""
        if idx >= len(self.data):
            return None
            
        price = self.data.iloc[idx]['close']
        volume = self.data.iloc[idx]['volume']
        ma20 = self.data.iloc[idx]['ma20']
        
        # Simple normalization
        state = [
            (price / max(price, 1)) - 0.5,
            (volume / max(volume, 1)) - 0.5,
            (ma20 / max(ma20, 1)) - 0.5,
            ((idx / len(self.data)) * 0.5),
            self._get_momentum(idx)
        ]
        
        return np.array(state)
    
    def _get_momentum(self, idx):
        """Calculate momentum indicator."""
        if idx < 2:
            return 0.0
        return (self.data.iloc[idx]['close'] - self.data.iloc[idx-2]['close']) / max(1, abs(self.data.iloc[idx]['close'] - self.data.iloc[idx-2]['close']))
    
    def _get_reward(self, action, next_state):
        """Calculate reward based on action and state transition."""
        current_price = self.data.iloc[self.current_idx]['close']
        next_price = self.data.iloc[self.next_idx]['close']
        
        if action == 'hold':
            return 0.0
        elif action == 'buy':
            return np.clip((next_price - current_price) / current_price, 
                           self.reward_params['min_return'], 
                           self.reward_params['max_return'])
        else:  # sell
            return np.clip((current_price - next_price) / current_price,
                          self.reward_params['min_return'],
                          self.reward_params['max_return'])
    
    def reset(self):
        """Reset the environment."""
        self.current_idx = 0
        self.next_idx = 1
        state = self._get_state(0)
        logger.info("Environment reset.")
        return state
    
    def step(self, action):
        """Take an action and return next state and reward."""
        if not isinstance(action, str) or action not in self.action_space:
            raise ValueError("Invalid action")
            
        self.current_idx += 1
        self.next_idx += 1
        
        done = False
        reward = self._get_reward(action, self.next_idx)
        
        if self.next_idx >= len(self.data):
            done = True
            
        next_state = self._get_state(self.next_idx) if not done else None
        info = {'action': action, 'current_price': self.data.iloc[self.current_idx]['close']}
        
        return next_state, reward, done, info
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.seed = seeding(seed)
        logger.info(f"Seed set to: {self.seed}")

class RiskManager:
    """Risk management module to control trading activities."""
    
    def __init__(self, max_loss: float = 0.1, stop_loss_pct: float = 2.0,
                 position_size_pct: float = 5.0):
        self.max_loss = max_loss
        self.stop_loss_pct = stop_loss_pct
        self.position_size_pct = position_size_pct
        
    def _calculate_risk(self, price_data: Dict[str, Any]) -> float:
        """Calculate the risk level based on market conditions."""
        # Simplified example: use volatility as a proxy for risk
        vol = np.std(price_data['close'].values)
        return min(max((vol / self.position_size_pct), 0.0), 1.0)
    
    def _apply_stop_loss(self, price: float, direction: str) -> float:
        """Apply stop loss based on current price and trade direction."""
        if direction == 'long':
            sl_price = price * (1 - self.stop_loss_pct / 100)
        else:
            sl_price = price * (1 + self.stop_loss_pct / 100)
            
        return sl_price
    
    def manage_risk(self, current_position: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust position based on risk parameters."""
        if not current_position:
            return {'action': 'hold'}
            
        # Calculate risk level
        risk_level = self._calculate_risk(current_position['data'])
        
        if risk_level > self.max_loss:
            return {'action': 'close_position'}
            
        # Adjust position size based on risk
        new_size_pct = min(max(self.position_size_pct * (1 - risk_level), 5.0), 20.0)
        
        return {
            'action': 'adjust_position',
            'size_pct': new_size_pct,
            'stop_loss_price': self._apply_stop_loss(
                current_position['entry_price'],
                current_position['direction']
            )
        }

class StrategyLearner:
    """Reinforcement learning-based strategy learner for crypto trading."""
    
    def __init__(self, env: MarketEnvironment):
        self.env = env
        self.model = None
        
    def _build_model(self) -> A2C:
        """Build and return an RL model."""
        # Using A2C as a robust policy gradient method
        model = A2C('MlpPolicy', 
                    self.env,
                    n_steps=1024,
                    learning_rate=3e-4,
                    n_minibatches=32)
        logger.info("Model built successfully.")
        return model
        
    def _train_model(self, total_timesteps: int) -> None:
        """Train the RL model."""
        if self.model is None:
            raise ValueError("Model not initialized")
            
        try:
            self.model.learn(total_timesteps=total_timesteps)
            logger.info(f"Training completed for {total_timesteps} steps.")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
            
    def _evaluate_policy(self) -> float:
        """Evaluate the current policy's performance."""