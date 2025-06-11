from flask import Flask, request
import ccxt
import logging
from tabulate import tabulate
import time
import random
import os


# Matrix parameters
rows, cols = os.get_terminal_size()
characters = list('abcdefghijklmnopqrstuvwxyz0123456789')
drops = [1 for _ in range(cols)]


app = Flask(__name__)
open_trade = {"30m": {}, "1h": {}, "3h": {}, "4h": {}, "5h": {}, "8h": {}, "10h": {}, "d": {}, "2d": {}, "3d": {}}  # Open trades for each timeframe
risks = {"30m": 11, "1h": 22, "3h": 10, "4h": 33, "5h": 10, "8h": 10, "10h": 33, "d": 10, "3d": 10}  # Risk percentage for each timeframe
live_trading = True
leverage = 33
symbol = 'BTCUSD'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("webbot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

if live_trading:
    api_url = {
        'api': {
            'live': 'https://api.phemex.com',
            'test': 'https://testnet-api.phemex.com',
        },
    }
    api_key = 'cbe78363-8e5f-497a-875f-e2e813146748'
    api_secret = 'YFyZmB7SSTRM4-0z9W9NSN1SgCiDYVvm-0NGkwGuzk5hYTdjNzkwYi1mYjczLTQyNTEtOWVhZi0yZDNjNTRkN2JlZDI'
else:
    api_url = {
        'api': {
            'live': 'https://api.phemex.com',
            'test': 'https://testnet-api.phemex.com',
        },
    }
    api_key = '052e3b80-bc14-4acb-abd5-c6fb172b7a25'
    api_secret = 'MgvrsjefF6eVS2xTu-yGfQsAF56HoGA7N1TGLIEenuU5ZmM0YmEzOC01M2IzLTQ4ZjUtODVlZC1iY2Y1ZTBiNTYxZmE'

exchange = ccxt.phemex({
    'urls': api_url,
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
})

if live_trading == False:
    exchange.set_sandbox_mode(True)


@app.route('/webhook/<timeframe>', methods=['POST'])
def tradingview_alert(timeframe):
    message = request.get_data(as_text=True)
    
    action = message.split(":")[1].strip()  # Extract the action from the message
    logger.info("Received action: %s for timeframe: %s", action, timeframe)

    if action == 'buy':
        # Check if there was a previous sell action
        if 'sell' in open_trade[timeframe]:
            # This is an exit short action
            trade_amount = open_trade[timeframe]['sell']['trade_amount']
            logger.info("Exiting trade with amount: %s", trade_amount)
            order = exchange.create_order(symbol=symbol, type='market', side='buy', amount=trade_amount)
            open_trade[timeframe].pop('sell')  # Remove the previous sell action from open_trade
        else:
            # This is a new long action
            update_account_balance()  # Recalculate the account balance
            amount_to_risk = round(account_balance * (risks[timeframe] / 100), 8)
            trade_amount = amount_to_risk * leverage
            logger.info("Amount of contracts to buy: %s", trade_amount)
            order = exchange.create_order(symbol=symbol, type='market', side='buy', amount=trade_amount)
            open_trade[timeframe]['buy'] = {
                'main_order_id': order['id'],
                'trade_amount': trade_amount
            }

    elif action == 'sell':
        # Check if there was a previous buy action
        if 'buy' in open_trade[timeframe]:
            # This is an exit long action
            trade_amount = open_trade[timeframe]['buy']['trade_amount']
            logger.info("Exiting trade with amount: %s", trade_amount)
            order = exchange.create_order(symbol=symbol, type='market', side='sell', amount=trade_amount)
            open_trade[timeframe].pop('buy')  # Remove the previous buy action from open_trade
        else:
            # This is a new sell action
            update_account_balance()  # Recalculate the account balance
            amount_to_risk = round(account_balance * (risks[timeframe] / 100), 8)
            trade_amount = amount_to_risk * leverage
            logger.info("Amount of contracts to sell: %s", trade_amount)
            order = exchange.create_order(symbol=symbol, type='market', side='sell', amount=trade_amount)
            open_trade[timeframe]['sell'] = {
                'main_order_id': order['id'],
                'trade_amount': trade_amount
            }

    return {
        "code": "success"
    }


def update_account_balance():
    # Fetch the spot account balance
    balance = exchange.fetch_balance()
    btcBalance = balance['BTC']['total']
    logger.info("BTC Balance (Spot): %s", btcBalance)

    # Fetch the swap account balance
    params = {"type": "swap", "code": "BTC"}
    balance = exchange.fetch_balance(params=params)
    btcBalanceSwap = balance['BTC']['total']
    logger.info("BTC Balance (Swap): %s", btcBalanceSwap)

    # Fetch the last traded price of BTC/USDT
    btc_ltp = exchange.fetch_ticker('BTC/USDT')['close']
    logger.info("BTC/USDT last traded price: %s", btc_ltp)

    # Calculate account balance in USDT
    global account_balance
    account_balance = round((btcBalance + btcBalanceSwap) * btc_ltp, 8)
    logger.info("Account balance in USDT: %s", account_balance)

def matrix_animation():
    lines = [''] * 50
    start_time = time.time()
    while True:
        string = ''
        for index in range(20):
            rand = random.randint(0, 9)
            lines[index] = str(rand) + lines[index]
            if len(lines[index]) > 80:
                lines[index] = lines[index][:80]
            string += lines[index] + '\n'
        print(string, end='\r')
        time.sleep(0.1)
        if time.time() - start_time > 2:  # Stop after 10 seconds
            break

if __name__ == '__main__':
    # Load the markets
    logger.info("Starting the server...")
    logger.info("multi timeframe giga chad active")
    logger.info("Live trading: %s", live_trading)
    matrix_animation()
    exchange.load_markets()
    logger.info("Markets loaded successfully.")
    try:
        # Fetch the balance
        balance = exchange.fetch_balance()
        logger.info("Connected successfully! Your spot account balance:")
        
        # Prepare the data
        table_data = []
        for currency, info in balance.items():
            # Skip the 'info', 'timestamp', 'datetime', 'free', 'used', and 'total' fields
            if currency not in ['info', 'timestamp', 'datetime', 'free', 'used', 'total']:
                table_data.append([currency, info['free'], info['used'], info['total']])
                
        # Print the data
        logger.info("\n%s", tabulate(table_data, headers=["Currency", "Free", "Used","Total"]))

    except Exception as e:
        logger.error("Failed to connect. Error: %s", e)

    # Update the initial account balance
    update_account_balance()
    print(open_trade)
    print(risks)
    # Start the server
    app.run()