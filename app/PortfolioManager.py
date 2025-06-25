import numpy as np
import pandas as pd
import cvxpy as cp
import cupy as cp_gpu
import cudf
from numba import cuda, jit
from tqdm import tqdm

# Initialize GPU context
cuda.select_device(0)
cp_gpu.cuda.Device(0).use()

# GPU-accelerated Sharpe ratio calculation
@cuda.jit
def gpu_sharpe(returns, output, window_size):
    i = cuda.grid(1)
    if i < returns.shape[0]:
        if returns.shape[1] < 2:
            output[i] = 0.0
            return

        mean = 0.0
        for j in range(returns.shape[1]):
            mean += returns[i, j]
        mean /= returns.shape[1]

        std = 0.0
        for j in range(returns.shape[1]):
            diff = returns[i, j] - mean
            std += diff * diff
        std = (std / (returns.shape[1] - 1)) ** 0.5

        if std == 0:
            output[i] = 0.0
        else:
            output[i] = (mean / std) * (window_size ** 0.5)

class PortfolioOptimizer:
    def __init__(self, df, rebalance_freq=60, corr_window=90, 
                sharpe_window=90, top_pairs=10, lambda_risk=0.5):
        self.df = df.copy()
        self.rebalance_freq = rebalance_freq
        self.corr_window = corr_window
        self.sharpe_window = sharpe_window
        self.top_pairs = top_pairs
        self.lambda_risk = lambda_risk
        self.tickers = []
        self.rebalance_dates = []
        self.gpu_df = None
        self.maxInvestment = (1/top_pairs)*1.5
        self.minInvestment = (1/top_pairs)*0.5

    def go(self):
        """Main method to execute the portfolio optimization strategy"""
        self.extractTickers()
        self.createOutputColumns()
        self.identifyRebalanceDates()
        self.convertToGpuDataFrame()
        self.processRebalanceDates()
        self.forwardFillWeights()
        self.calculatePortfolioReturns()
        return self.df

    def extractTickers(self):
        """Extract unique tickers from column names"""
        self.tickers = list(set(col.split('_')[0] 
                              for col in self.df.columns 
                              if '_close' in col))
        self.tickers.sort()

    def createOutputColumns(self):
        """Create output columns for weights and positions"""
        for t in self.tickers:
            self.df[f'{t}_weight'] = np.nan
            self.df[f'{t}_position'] = np.nan
        self.df['portfolio_return'] = np.nan

    def identifyRebalanceDates(self):
        """Identify dates for portfolio rebalancing"""
        valid_dates = self.df.index[self.sharpe_window - 1:-1]
        self.rebalance_dates = valid_dates[::self.rebalance_freq]

    def convertToGpuDataFrame(self):
        """Convert main DataFrame to GPU DataFrame for processing"""
        self.gpu_df = cudf.DataFrame.from_pandas(self.df)

    def processRebalanceDates(self):
        """Process each rebalance date in sequence"""
        for i in tqdm(range(len(self.rebalance_dates))):
            rebalance_date = self.rebalance_dates[i]
            self.processRebalanceDate(rebalance_date)

    def processRebalanceDate(self, rebalance_date):
        """Process a single rebalance date"""
        idx = self.df.index.get_loc(rebalance_date)
        corr_data, sharpe_data = self.getDataWindows(idx)
        sharpes = self.calculateSharpeRatios(sharpe_data)
        corr_matrix = self.calculateCorrelationMatrix(corr_data)
        pairs_sorted = self.findLowCorrelationPairs(corr_matrix)
        candidate_returns, candidate_info = self.generateCandidates(
            sharpe_data, sharpes, pairs_sorted
        )
        
        if len(candidate_returns) >= 2:
            weights = self.optimizeWeights(candidate_returns)
            if weights is not None:
                self.allocateWeights(rebalance_date, weights, candidate_info)

    def getDataWindows(self, idx):
        """Get data windows for correlation and Sharpe calculations"""
        corr_start = max(0, idx - self.corr_window + 1)
        sharpe_start = max(0, idx - self.sharpe_window + 1)
        
        corr_data = self.gpu_df.iloc[corr_start:idx+1]
        sharpe_data = self.gpu_df.iloc[sharpe_start:idx+1]
        
        return corr_data, sharpe_data

    def calculateSharpeRatios(self, sharpe_data):
        """Calculate rolling Sharpe ratios using GPU"""
        sharpe_returns = cudf.DataFrame()
        for t in self.tickers:
            col_name = f'{t}_return'
            if col_name in sharpe_data.columns:
                sharpe_returns[t] = sharpe_data[col_name].fillna(0)
        
        if sharpe_returns.empty:
            return {}
        
        returns_array = sharpe_returns.to_cupy().T
        sharpe_results = cp_gpu.zeros(returns_array.shape[0])
        
        threads_per_block = 256
        blocks_per_grid = (returns_array.shape[0] + threads_per_block - 1) // threads_per_block
        gpu_sharpe[blocks_per_grid, threads_per_block](
            returns_array, sharpe_results, self.sharpe_window
        )
        
        cpu_sharpes = sharpe_results.get()
        return {
            t: cpu_sharpes[i] 
            for i, t in enumerate(sharpe_returns.columns)
        }

    def calculateCorrelationMatrix(self, corr_data):
        """Calculate correlation matrix using GPU"""
        corr_returns = cudf.DataFrame()
        for t in self.tickers:
            col_name = f'{t}_return'
            if col_name in corr_data.columns:
                corr_returns[t] = corr_data[col_name].fillna(0)
        
        if corr_returns.empty:
            return pd.DataFrame()
        
        gpu_corr = corr_returns.corr(method='pearson').fillna(0)
        return gpu_corr.to_pandas()

    def findLowCorrelationPairs(self, corr_matrix):
        """Find top low-correlation pairs"""
        pairs = []
        for i1, t1 in enumerate(self.tickers):
            for i2, t2 in enumerate(self.tickers[i1+1:], i1+1):
                if t1 in corr_matrix.index and t2 in corr_matrix.columns:
                    corr_val = corr_matrix.at[t1, t2]
                    pairs.append((t1, t2, corr_val))
        
        return sorted(pairs, key=lambda x: x[2])[:self.top_pairs]

    def generateCandidates(self, sharpe_data, sharpes, pairs_sorted):
        """Generate candidate assets for optimization"""
        candidate_returns = []
        candidate_info = []
        used_assets = set()
        
        self.processPairs(sharpe_data, sharpes, pairs_sorted, 
                         candidate_returns, candidate_info, used_assets)
        self.processIndividuals(sharpe_data, sharpes, used_assets,
                              candidate_returns, candidate_info)
        
        return candidate_returns, candidate_info

    def processPairs(self, sharpe_data, sharpes, pairs_sorted,
                   candidate_returns, candidate_info, used_assets):
        """Process pairs according to Sharpe comparison rules"""
        for t1, t2, _ in pairs_sorted:
            if t1 in used_assets or t2 in used_assets:
                continue
                
            ret1 = sharpe_data[f'{t1}_return'].fillna(0).to_cupy()
            ret2 = sharpe_data[f'{t2}_return'].fillna(0).to_cupy()
            
            pos1 = 1 if sharpes.get(t1, 0) >= 0 else -1
            pos2 = 1 if sharpes.get(t2, 0) >= 0 else -1
            
            ret1_common = pos1 * ret1
            ret2_common = pos2 * ret2
            pair_ret = 0.5 * (ret1_common + ret2_common)
            
            sharpe_pair = self.calcSharpe(pair_ret)
            sharpe_t1 = self.calcSharpe(ret1_common)
            sharpe_t2 = self.calcSharpe(ret2_common)
            
            if sharpe_pair > sharpe_t1 and sharpe_pair > sharpe_t2:
                candidate_returns.append(pair_ret.get())
                candidate_info.append(('pair', t1, t2, pos1, pos2))
                used_assets.update([t1, t2])
            elif sharpe_t1 > sharpe_t2 and sharpe_t1 > sharpe_pair:
                candidate_returns.append(ret1_common.get())
                candidate_info.append(('individual', t1, pos1))
                used_assets.add(t1)
            elif sharpe_t2 > sharpe_t1 and sharpe_t2 > sharpe_pair:
                candidate_returns.append(ret2_common.get())
                candidate_info.append(('individual', t2, pos2))
                used_assets.add(t2)

    def processIndividuals(self, sharpe_data, sharpes, used_assets,
                         candidate_returns, candidate_info):
        """Add top individual tickers not already included"""
        available_tickers = [t for t in self.tickers if t not in used_assets]
        top_individuals = sorted(
            [(t, sharpes.get(t, 0)) for t in available_tickers],
            key=lambda x: x[1],
            reverse=True
        )[:self.top_pairs]
        
        for t, _ in top_individuals:
            pos = 1 if sharpes.get(t, 0) >= 0 else -1
            returns = sharpe_data[f'{t}_return'].fillna(0).to_cupy()
            
            if returns.size >= 5:
                candidate_returns.append((pos * returns).get())
                candidate_info.append(('individual', t, pos))

    def calcSharpe(self, returns):
        """Calculate Sharpe ratio for a return series"""
        if returns.size < 2 or returns.std() == 0:
            return 0
        return returns.mean() / returns.std() * (returns.size ** 0.5)

    def optimizeWeights(self, candidate_returns):
        """Optimize weights using convex optimization"""
        min_length = min(len(r) for r in candidate_returns)
        aligned_returns = np.array([r[-min_length:] for r in candidate_returns]).T
        
        cupy_returns = cp_gpu.array(aligned_returns)
        mu = cp_gpu.mean(cupy_returns, axis=0).get()
        Sigma = cp_gpu.cov(cupy_returns, rowvar=False).get()
        
        w = cp.Variable(len(mu))
        expected_return = mu.T @ w
        risk = cp.quad_form(w, cp.psd_wrap(Sigma))
        
        problem = cp.Problem(
            cp.Maximize(expected_return - self.lambda_risk * risk),
            [cp.sum(w) == 1, w >= self.minInvestment, w<=self.maxInvestment]
        )
        problem.solve()
        
        return w.value if w.value is not None else None

    def allocateWeights(self, rebalance_date, weights, candidate_info):
        """Allocate weights to underlying assets"""
        # Get index position of rebalance_date
        idx = self.df.index.get_loc(rebalance_date)
        
        # Verify next day exists
        if idx + 1 >= len(self.df):
            return  # Skip if no next day
        
        next_day = self.df.index[idx + 1]
        asset_weights = {t: 0.0 for t in self.tickers}
        
        for i, info in enumerate(candidate_info):
            wt = weights[i]
            if info[0] == 'pair':
                _, t1, t2, pos1, pos2 = info
                asset_weights[t1] += 0.5 * wt * pos1
                asset_weights[t2] += 0.5 * wt * pos2
            else:
                _, t, pos = info
                asset_weights[t] += wt * pos
        
        for t in self.tickers:
            weight_val = asset_weights[t]
            self.df.at[rebalance_date, f'{t}_weight'] = weight_val
            self.df.at[rebalance_date, f'{t}_position'] = np.sign(weight_val) if weight_val != 0 else 0

    
    
    def forwardFillWeights(self):
      """Proper weight reset and forward-filling"""
      # Create rebalance mask (True only on rebalance dates)
      rebalance_mask = pd.Series(False, index=self.df.index)
      rebalance_mask.loc[self.rebalance_dates] = True
      
      for t in self.tickers:
          weight_col = f'{t}_weight'
          position_col = f'{t}_position'
          
          # 1. Identify reset points
          reset_mask = rebalance_mask & self.df[weight_col].isna()
          
          # 2. Reset BEFORE forward-filling
          self.df.loc[reset_mask, weight_col] = 0
          self.df.loc[reset_mask, position_col] = 0
          
          # 3. Forward-fill weights
          self.df[weight_col] = self.df[weight_col].ffill()
          self.df[position_col] = self.df[position_col].ffill()
          
          # 4. Fill initial NaNs (pre-first-rebalance)
          self.df[weight_col] = self.df[weight_col].fillna(0)
          self.df[position_col] = self.df[position_col].fillna(0)

    def calculatePortfolioReturns(self):
        """Calculate portfolio returns vectorized"""
        weight_cols = [f'{t}_weight' for t in self.tickers]
        return_cols = [f'{t}_return' for t in self.tickers]
        
        self.df['portfolio_return'] = (
            self.df[weight_cols].values * self.df[return_cols].values
        ).sum(axis=1)
        self.df['portfolio_return'].fillna(0)