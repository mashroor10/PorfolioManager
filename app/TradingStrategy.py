import pandas as pd
from abc import ABC, abstractmethod


class TradingStrategy():
    def __init__(self, data, window=30):
        self.rawData = data
        self.window = window
        self.workingData = pd.DataFrame()
        self.processed = False
        self.tickerResults = {}
        self.backTests = {}
        
    @abstractmethod
    def createProcessedData(self):
        """Preprocess raw data into working format"""
        pass
        
    @abstractmethod
    def calculateMetrics(self):
        """Calculate strategy-specific metrics and signals"""
        pass
        
    @abstractmethod
    def addSignalsToDataFrames(self):
        """Add generated signals to individual asset DataFrames"""
        pass
        
    @abstractmethod
    def executeBacktests(self):
        """Execute backtests for all assets"""
        pass
        
    def go(self):
        """Main execution workflow"""
        self.createProcessedData()
        self.calculateMetrics()
        self.addSignalsToDataFrames()
        self.executeBacktests()