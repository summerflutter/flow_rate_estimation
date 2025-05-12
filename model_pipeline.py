import torch
import torch.nn as nn

class ClientFlowPredictor(nn.Module):
    def __init__(self, client_id_size, segment_size, embedding_dim=16, hidden_size=64):
        super().__init__()
        # 客户ID和Segment的Embedding
        self.client_embedding = nn.Embedding(client_id_size, embedding_dim)
        self.segment_embedding = nn.Embedding(segment_size, embedding_dim)

        # LSTM处理时间序列输入（如市场成交量、时间编码）
        self.lstm = nn.LSTM(input_size=market_input_dim, hidden_size=hidden_size, batch_first=True)

        # 拼接后进入MLP
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + 2 * embedding_dim + other_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出一个比例
        )

    def forward(self, market_seq, client_id, segment_id, other_features):
        # market_seq: [batch, time, features]
        _, (hn, _) = self.lstm(market_seq)
        market_rep = hn.squeeze(0)  # [batch, hidden]

        client_emb = self.client_embedding(client_id)
        seg_emb = self.segment_embedding(segment_id)

        x = torch.cat([market_rep, client_emb, seg_emb, other_features], dim=1)
        out = self.fc(x)
        return out



import torch
import torch.nn as nn

class FlowRatioPredictor(nn.Module):
    def __init__(self,
                 client_id_size,
                 segment_size,
                 static_emb_dim=16,
                 market_feature_dim=6,   # like market_vol, market_spread, client_spread, bid_edge, ask_edge, etc
                 lstm_hidden_size=64,
                 mlp_hidden_size=64):
        super().__init__()

        # Embedding layers for categorical static features
        self.client_id_emb = nn.Embedding(client_id_size, static_emb_dim)
        self.segment_emb = nn.Embedding(segment_size, static_emb_dim)

        # LSTM for temporal market/price features
        self.lstm = nn.LSTM(input_size=market_feature_dim,
                            hidden_size=lstm_hidden_size,
                            batch_first=True)

        # Fully connected prediction head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size + 2 * static_emb_dim, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1),  # Predict scalar: flow ratio
            nn.Sigmoid()  # Optional: make prediction between 0-1
        )

    def forward(self, time_series_x, client_id, segment_id):
        """
        time_series_x: [batch_size, time_steps, market_feature_dim]
        client_id: [batch_size]
        segment_id: [batch_size]
        """
        # LSTM output: use last hidden state
        _, (h_n, _) = self.lstm(time_series_x)
        time_rep = h_n.squeeze(0)  # [batch, lstm_hidden_size]

        # Embeddings
        client_emb = self.client_id_emb(client_id)
        seg_emb = self.segment_emb(segment_id)

        # Combine all
        combined = torch.cat([time_rep, client_emb, seg_emb], dim=1)  # [batch, combined_dim]
        out = self.fc(combined)
        return out.squeeze(-1)  # [batch]



# Example sizes:
batch_size = 32
time_steps = 12  # e.g. last 12 hours
market_feature_dim = 6

time_series_x = torch.randn(batch_size, time_steps, market_feature_dim)
client_id = torch.randint(0, 500, (batch_size,))
segment_id = torch.randint(0, 10, (batch_size,))

model = FlowRatioPredictor(client_id_size=500, segment_size=10)
output = model(time_series_x, client_id, segment_id)
print(output.shape)  # [32]  → 每个客户的成交量比预测值





import pandas as pd
import numpy as np

def preprocess_dataframe(df):
    df = df.copy()

    # Ensure timestamp is datetime
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    # Sort
    df = df.sort_values(['client_id', 'TimeStamp'])

    # Add time features
    df['hour'] = df['TimeStamp'].dt.hour
    df['dayofweek'] = df['TimeStamp'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Add target variable
    df['flow_ratio'] = df['flow_volume'] / df['market_total_volume']

    # Normalize or standardize market features (optional)
    numeric_cols = ['market_volume', 'client_spread', 'market_spread', 'bid_edge', 'ask_edge']
    for col in numeric_cols:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    return df




import torch
from torch.utils.data import Dataset

class FlowDataset(Dataset):
    def __init__(self, df, window_size=12):
        self.window_size = window_size
        self.samples = []

        grouped = df.groupby('client_id')
        for client_id, group in grouped:
            group = group.sort_values('TimeStamp')
            for i in range(window_size, len(group)):
                window = group.iloc[i - window_size:i]
                target = group.iloc[i]['flow_ratio']

                self.samples.append({
                    'time_series': window[['market_volume', 'client_spread', 'market_spread', 'bid_edge', 'ask_edge']].values,
                    'client_id': group.iloc[i]['client_id'],
                    'segment_id': group.iloc[i]['segment_id'],
                    'target': target
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'time_series': torch.tensor(sample['time_series'], dtype=torch.float),
            'client_id': torch.tensor(sample['client_id'], dtype=torch.long),
            'segment_id': torch.tensor(sample['segment_id'], dtype=torch.long),
            'target': torch.tensor(sample['target'], dtype=torch.float)
        }





def train_one_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        preds = model(batch['time_series'], batch['client_id'], batch['segment_id'])
        loss = loss_fn(preds, batch['target'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)






########################## Complete version
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Step 1: Mock Data
def generate_mock_data(num_currency_pairs=5, num_clients_per_pair=10, days=2):
    records = []
    for ccy in range(num_currency_pairs):
        for client in range(num_clients_per_pair):
            for hour in range(24 * days):
                timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour)
                market_volume = np.random.uniform(1e5, 1e6)
                client_spread = np.random.uniform(0.1, 1.0)
                market_spread = np.random.uniform(0.05, 0.5)
                bid_edge = np.random.uniform(0.01, 0.05)
                ask_edge = np.random.uniform(0.01, 0.05)
                flow_volume = np.random.uniform(0, market_volume * 0.1)
                market_total_volume = market_volume + np.random.uniform(0, 5e5)

                records.append({
                    'TimeStamp': timestamp,
                    'client_id': client,
                    'segment_id': client % 3,
                    'currency_pair_id': ccy,
                    'market_volume': market_volume,
                    'client_spread': client_spread,
                    'market_spread': market_spread,
                    'bid_edge': bid_edge,
                    'ask_edge': ask_edge,
                    'flow_volume': flow_volume,
                    'market_total_volume': market_total_volume
                })
    df = pd.DataFrame(records)
    df['flow_ratio'] = df['flow_volume'] / df['market_total_volume']
    return df

df = generate_mock_data()




class FlowDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        time_hour = row['TimeStamp'].hour + 24 * row['TimeStamp'].dayofweek
        features = np.array([
            time_hour / 168,  # normalize to week cycle
            row['market_volume'],
            row['client_spread'],
            row['market_spread'],
            row['bid_edge'],
            row['ask_edge']
        ], dtype=np.float32)

        return {
            'features': torch.tensor(features),
            'client_id': torch.tensor(row['client_id'], dtype=torch.long),
            'segment_id': torch.tensor(row['segment_id'], dtype=torch.long),
            'currency_pair_id': torch.tensor(row['currency_pair_id'], dtype=torch.long),
            'target': torch.tensor(row['flow_ratio'], dtype=torch.float32)
        }



import torch.nn as nn

class FlowPredictor(nn.Module):
    def __init__(self, num_clients, num_segments, num_currency_pairs, emb_dim=8):
        super().__init__()
        self.client_emb = nn.Embedding(num_clients, emb_dim)
        self.segment_emb = nn.Embedding(num_segments, emb_dim)
        self.currency_emb = nn.Embedding(num_currency_pairs, emb_dim)

        self.fc1 = nn.Linear(6 + emb_dim * 3, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, features, client_id, segment_id, currency_pair_id):
        c_emb = self.client_emb(client_id)
        s_emb = self.segment_emb(segment_id)
        cur_emb = self.currency_emb(currency_pair_id)
        x = torch.cat([features, c_emb, s_emb, cur_emb], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x).squeeze()


def train_model(df, num_epochs=5):
    dataset = FlowDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FlowPredictor(num_clients=200, num_segments=3, num_currency_pairs=30)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            pred = model(batch['features'], batch['client_id'], batch['segment_id'], batch['currency_pair_id'])
            loss = loss_fn(pred, batch['target'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
