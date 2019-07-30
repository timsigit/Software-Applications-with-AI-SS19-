import random
from collections import deque
from typing import List
import numpy as np
import stock_exchange
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
import random


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Parameters for neural network
        self.state_size = 4
        self.action_size = 9
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"

        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        # 1. Compute current state
        state = self.compute_state(stock_market_data)

        # 1.2 If training is turned off, just predict the next action and return orders
        if not self.train_while_trading:
            self.last_state = state
            actionSpace = self.model.predict(state)
            action = np.argmax(actionSpace[0])
            orders = self.action_to_order(action, portfolio, stock_market_data)
            return orders

        # 2. Get a random action with the probability of epsilon, otherwise predict the action via the ANN
        if np.random.rand() <= self.epsilon and self.train_while_trading:
            action = np.random.randint(self.action_size, size=1)[0]

        else:
            actionSpace = self.model.predict(state)
            action = np.argmax(actionSpace[0])

        # 3. Reduce Epsilon if it is bigger than epsilon min
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 4. Training of the ANN
        if self.train_while_trading and self.last_state is not None:
            # 4.1 Get reward
            reward = self.get_reward(portfolio.get_value(stock_market_data), self.last_portfolio_value)

            # 4.2 Store memory
            self.memory.append([self.last_state, self.last_action, reward, state])

            # 4.3 Actual training via Experience Replay
            if len(self.memory) > self.min_size_of_memory_before_training:
                self.experienceReplay(self.batch_size)


        # 5. Map Action + Create Order
        orders = self.action_to_order(action, portfolio, stock_market_data)

        # 6. Save the values
        self.last_state = state
        self.last_action = action
        self.last_portfolio_value = portfolio.get_value(stock_market_data)

        return orders

    def action_to_order(self, action, portfolio, stock_data):
        orders = []
        if not portfolio.stocks:
            portfolio.stocks[Company.A] = 0
            portfolio.stocks[Company.B] = 0

        if action == 0:   # BUY A, BUY B
            orders = self.order_list(portfolio, Vote.BUY, Vote.BUY, stock_data)
        elif action == 1: # BUY A, HOLD B
            orders = self.order_list(portfolio, Vote.BUY, Vote.HOLD, stock_data)
        elif action == 2: # BUY A, SELL B
            orders = self.order_list(portfolio, Vote.BUY, Vote.SELL, stock_data)
        elif action == 3: # HOLD A, BUY B
            orders = self.order_list(portfolio, Vote.HOLD, Vote.BUY, stock_data)
        elif action == 4: # HOLD A, HOLD B
            orders = self.order_list(portfolio, Vote.HOLD, Vote.HOLD, stock_data)
        elif action == 5: # HOLD A, SELL B
            orders = self.order_list(portfolio, Vote.HOLD, Vote.SELL, stock_data)
        elif action == 6: # SELL A, BUY B
            orders = self.order_list(portfolio, Vote.SELL, Vote.BUY, stock_data)
        elif action == 7: # SELL A, HOLD B
            orders = self.order_list(portfolio, Vote.SELL, Vote.HOLD, stock_data)
        elif action == 8: # SELL A, SELL B
            orders = self.order_list(portfolio, Vote.SELL, Vote.SELL, stock_data)

        return orders

    def get_order(self, portfolio, company, voteOrder, voteOther, stock_data):
        order = None
        if voteOrder == Vote.BUY:
            stock_price = stock_data.get_most_recent_price(company)
            if (voteOrder == voteOther):
                amount = (portfolio.cash / 2) / stock_price
            else:
                amount = portfolio.cash / stock_price

            if amount > 0:
                order = Order(OrderType.BUY, company, amount)
        elif voteOrder == Vote.SELL:
            amount = portfolio.get_stock(company)
            if amount > 0:
                order = Order(OrderType.SELL, company, amount)
        else:
            order = None

        return order

    def order_list(self, portfolio, voteA, voteB, stock_data):
        list = []
        orderA = self.get_order(portfolio, Company.A, voteA, voteB, stock_data)
        if orderA != None:
            list.append(orderA)
        orderB = self.get_order(portfolio, Company.B, voteB, voteA, stock_data)
        if orderB != None:
            list.append(orderB)
        return list

    def get_reward(self, portfolioValue, lastPortfolioValue):
        if portfolioValue > lastPortfolioValue:
            return 1
        elif portfolioValue < lastPortfolioValue:
            return -1
        else:
            return 0

    def experienceReplay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        X = list()
        Y = list()

        for state, action, reward, nextState in minibatch:
            target_f = self.model.predict(state)
            target_f[0][action] = reward
            X.append(state[0])
            Y.append(target_f[0])

        self.model.train_on_batch(np.array(X), np.array(Y))

    def compute_state(self, stockData):
        stock_A_tuple = stockData[Company.A]
        stock_B_tuple = stockData[Company.B]

        if self.expert_a.vote(stock_A_tuple).name == 'BUY':
            company_A = 1
        elif self.expert_a.vote(stock_A_tuple).name == 'HOLD':
            company_A = 0
        else:
            company_A = -1

        if self.expert_b.vote(stock_B_tuple).name == 'BUY':
            company_B = 1
        elif self.expert_b.vote(stock_B_tuple).name == 'HOLD':
            company_B = 0
        else:
            company_B = -1

        deltaA = self.stock_change(stock_A_tuple)
        deltaB = self.stock_change(stock_B_tuple)

        if deltaA > 0:
            deltaA = 1
        elif deltaA < 0:
            deltaA = -1
        else:
            deltaA = 0

        if deltaB > 0:
            deltaB = 1
        elif deltaB < 0:
            deltaB = -1
        else:
            deltaB = 0

        stateComponents = [company_A, company_B, deltaA, deltaB]

        return np.array([stateComponents])

    def stock_change(self, tuple):
        idx = tuple.get_row_count()
        idxPrev = idx - 1
        currentPrice = tuple.get(idx - 1)[1]
        if idxPrev > 0:
            previousPrice = tuple.get(idxPrev - 1)[1]
        else:
            previousPrice = tuple.get(0)[1]

        deltaPrice = currentPrice - previousPrice
        return deltaPrice


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    plt.show()
