import pandas as pd
from backtesting import Strategy

def SIGNAL():
    return df.signal

class SignalStrategy(Strategy):
    stopLoss = 1  # Represents percentage (e.g., 1 = 1%)
    takeProfit = 15  # Represents percentage (e.g., 15 = 15%)
    
    def init(self):
        self.signal_indicator = self.I(lambda: self.data.df['signal'])
    
    def next(self):
        try:
            current_signal = self.signal_indicator[-1]
            current_price = self.data.Close[-1]
            
            # Calculate SL/TP as percentages of current price
            sl_percent = self.stopLoss * 0.01
            tp_percent = self.takeProfit * 0.01
            
            if current_signal == 0 and self.position:
                self.position.close()
                
            elif current_signal == 1:  # LONG signal
                if self.position.is_short:
                    self.position.close()
                    
                if not self.position.is_long:
                    # For LONG: SL below price, TP above price
                    sl_price = current_price * (1 - sl_percent)
                    tp_price = current_price * (1 + tp_percent)
                    self.buy(sl=sl_price, tp=tp_price)
                    
            elif current_signal == -1:  # SHORT signal
                if self.position.is_long:
                    self.position.close()
                    
                if not self.position.is_short:
                    # For SHORT: SL above price, TP below price
                    sl_price = current_price * (1 + sl_percent)
                    tp_price = current_price * (1 - tp_percent)
                    self.sell(sl=sl_price, tp=tp_price)
                    
        except IndexError:
            pass