import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import urllib.request
import zipfile
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchrec
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import ShardingPlan
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder, ShardingEnv
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

torch.manual_seed(42)
np.random.seed(42)

def download_movielens(data_dir="./ml-1m"):
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, "ratings.dat")):
        print(f"MovieLens in {data_dir}")
        return data_dir
    
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    
    print(f"download MovieLens-1M...")
    urllib.request.urlretrieve(url, zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(data_dir))
    
    extracted_dir = os.path.join(os.path.dirname(data_dir), "ml-1m")
    if extracted_dir != data_dir:
        for file in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, file), os.path.join(data_dir, file))
        if os.path.exists(extracted_dir):
            shutil.rmtree(extracted_dir)
    
    os.remove(zip_path)
    
    return data_dir

def parse_args():
    parser = argparse.ArgumentParser(description="TorchRec MovieLens ")
    parser.add_argument("--data_path", type=str, default="./ml-1m", help="MovieLens数据集路径，如不存在将自动下载")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1024, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--embedding_dim", type=int, default=64, help="嵌入维度")
    parser.add_argument("--num_embeddings", type=int, default=10000, help="嵌入表大小")
    parser.add_argument("--mlp_dims", type=str, default="128,64,32", help="MLP层维度，用逗号分隔")
    parser.add_argument("--save_dir", type=str, default="./model_checkpoints", help="模型保存路径")
    return parser.parse_args()

class MovieLensDataset(Dataset):
    def __init__(self, data_path: str, split: str = "train"):

        ratings_file = os.path.join(data_path, "ratings.dat")
        if not os.path.exists(ratings_file):
            raise FileNotFoundError(f": {ratings_file}")
        
        ratings_data = []
        with open(ratings_file, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                user_id, movie_id, rating, timestamp = line.strip().split('::')
                ratings_data.append({
                    'user_id': int(user_id),
                    'movie_id': int(movie_id),
                    'rating': float(rating),
                    'timestamp': int(timestamp)
                })
        
        ratings_df = pd.DataFrame(ratings_data)
        
        users_file = os.path.join(data_path, "users.dat")
        movies_file = os.path.join(data_path, "movies.dat")
        
        users_data = []
        with open(users_file, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                parts = line.strip().split('::')
                user_id = int(parts[0])
                gender = 1 if parts[1] == 'M' else 0
                age = int(parts[2])
                occupation = int(parts[3])
                users_data.append({
                    'user_id': user_id,
                    'gender': gender,
                    'age': age,
                    'occupation': occupation
                })
        
        users_df = pd.DataFrame(users_data)
        
        movies_data = []
        with open(movies_file, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                parts = line.strip().split('::')
                movie_id = int(parts[0])
                year = 0
                if parts[1].endswith(')'):
                    year_start = parts[1].rfind('(')
                    if year_start != -1:
                        year_str = parts[1][year_start+1:parts[1].rfind(')')]
                        try:
                            year = int(year_str)
                        except ValueError:
                            year = 0
                
                genres = parts[2].split('|')
                genre_ids = []
                for genre in genres:
                    genre_map = {
                        'Action': 1, 'Adventure': 2, 'Animation': 3, 'Children\'s': 4,
                        'Comedy': 5, 'Crime': 6, 'Documentary': 7, 'Drama': 8,
                        'Fantasy': 9, 'Film-Noir': 10, 'Horror': 11, 'Musical': 12,
                        'Mystery': 13, 'Romance': 14, 'Sci-Fi': 15, 'Thriller': 16,
                        'War': 17, 'Western': 18
                    }
                    if genre in genre_map:
                        genre_ids.append(genre_map[genre])
                
                movies_data.append({
                    'movie_id': movie_id,
                    'year': year,
                    'genres': genre_ids
                })
        
        movies_df = pd.DataFrame(movies_data)
        
        data = pd.merge(ratings_df, users_df, on='user_id', how='left')
        data = pd.merge(data, movies_df, on='movie_id', how='left')
        
        data = data.sort_values('timestamp')
        
        split_idx = int(len(data) * 0.8)
        if split == "train":
            self.data = data.iloc[:split_idx]
        else:
            self.data = data.iloc[split_idx:]
        
        self.max_user_id = data['user_id'].max()
        self.max_movie_id = data['movie_id'].max()
        self.max_age = data['age'].max()
        self.max_occupation = data['occupation'].max()
        
        print(f": {len(self.data)} ")
        print(f": {self.max_user_id}, : {self.max_movie_id}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        sparse_features = {
            "user_id": torch.tensor([row['user_id']], dtype=torch.long),
            "movie_id": torch.tensor([row['movie_id']], dtype=torch.long),
            "gender": torch.tensor([row['gender']], dtype=torch.long),
            "age": torch.tensor([row['age']], dtype=torch.long),
            "occupation": torch.tensor([row['occupation']], dtype=torch.long),
            "year": torch.tensor([row['year']], dtype=torch.long),
        }
        
        genres = row['genres']
        if isinstance(genres, list):
            genres_tensor = torch.tensor(genres, dtype=torch.long)
        else:
            genres_tensor = torch.tensor([], dtype=torch.long)
        
        sparse_features["genres"] = genres_tensor
        
        label = torch.tensor(row['rating'], dtype=torch.float)
        
        return sparse_features, label

def collate_fn(batch):
    sparse_features = {
        "user_id": [],
        "movie_id": [],
        "gender": [],
        "age": [],
        "occupation": [],
        "year": [],
        "genres": [],
    }
    
    labels = []
    
    for features, label in batch:
        for key in sparse_features:
            if key == "genres":
                sparse_features[key].append(features[key])
            else:
                sparse_features[key].extend(features[key].tolist())
        labels.append(label)
    
    lengths = {
        "user_id": torch.tensor([1] * len(batch), dtype=torch.long),
        "movie_id": torch.tensor([1] * len(batch), dtype=torch.long),
        "gender": torch.tensor([1] * len(batch), dtype=torch.long),
        "age": torch.tensor([1] * len(batch), dtype=torch.long),
        "occupation": torch.tensor([1] * len(batch), dtype=torch.long),
        "year": torch.tensor([1] * len(batch), dtype=torch.long),
    }
    
    genres_values = []
    genres_lengths = []
    for genres_tensor in sparse_features["genres"]:
        genres_values.extend(genres_tensor.tolist())
        genres_lengths.append(len(genres_tensor))
    
    lengths["genres"] = torch.tensor(genres_lengths, dtype=torch.long)
    
    values = {
        "user_id": torch.tensor(sparse_features["user_id"], dtype=torch.long),
        "movie_id": torch.tensor(sparse_features["movie_id"], dtype=torch.long),
        "gender": torch.tensor(sparse_features["gender"], dtype=torch.long),
        "age": torch.tensor(sparse_features["age"], dtype=torch.long),
        "occupation": torch.tensor(sparse_features["occupation"], dtype=torch.long),
        "year": torch.tensor(sparse_features["year"], dtype=torch.long),
        "genres": torch.tensor(genres_values, dtype=torch.long),
    }
    
    kjt = KeyedJaggedTensor(
        keys=list(values.keys()),
        values=torch.cat([values[k] for k in values.keys()]),
        lengths=torch.cat([lengths[k] for k in lengths.keys()]),
    )
    
    return kjt, torch.tensor(labels, dtype=torch.float)

class MovieLensModel(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
    ):
        super().__init__()
        self.embedding_bag_collection = embedding_bag_collection
        
        dense_arch_layers = []
        for i in range(len(dense_arch_layer_sizes) - 1):
            dense_arch_layers.append(
                nn.Linear(dense_arch_layer_sizes[i], dense_arch_layer_sizes[i + 1])
            )
            dense_arch_layers.append(nn.ReLU())
        self.dense_arch = nn.Sequential(*dense_arch_layers)
        
        embedding_dim_sum = 0
        for config in embedding_bag_collection.embedding_bag_configs():
            embedding_dim_sum += config.embedding_dim
        
        over_arch_layers = []
        if dense_in_features == 0:
            input_dim = embedding_dim_sum
        else:
            input_dim = dense_arch_layer_sizes[-1] + embedding_dim_sum
            
        over_arch_layers.append(
            nn.Linear(
                input_dim,
                over_arch_layer_sizes[0],
            )
        )
        over_arch_layers.append(nn.ReLU())
        for i in range(len(over_arch_layer_sizes) - 1):
            over_arch_layers.append(
                nn.Linear(over_arch_layer_sizes[i], over_arch_layer_sizes[i + 1])
            )
            over_arch_layers.append(nn.ReLU())
        over_arch_layers.append(nn.Linear(over_arch_layer_sizes[-1], 1))
        self.over_arch = nn.Sequential(*over_arch_layers)
    
    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        embeddings = self.embedding_bag_collection(kjt)
        
        sparse_features = torch.cat([embeddings[k] for k in embeddings.keys()], dim=1)
        
        dense_features = torch.zeros((sparse_features.shape[0], 0), device=sparse_features.device)
        
        if dense_features.shape[1] > 0:
            dense_features = self.dense_arch(dense_features)
        
        combined_features = torch.cat([dense_features, sparse_features], dim=1)
        
        prediction = self.over_arch(combined_features)
        return prediction.squeeze()

def create_model(args):
    eb_configs = [
        EmbeddingBagConfig(
            name="user_id",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            feature_names=["user_id"],
        ),
        EmbeddingBagConfig(
            name="movie_id",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            feature_names=["movie_id"],
        ),
        EmbeddingBagConfig(
            name="gender",
            embedding_dim=args.embedding_dim,
            num_embeddings=2,
            feature_names=["gender"],
        ),
        EmbeddingBagConfig(
            name="age",
            embedding_dim=args.embedding_dim,
            num_embeddings=100,
            feature_names=["age"],
        ),
        EmbeddingBagConfig(
            name="occupation",
            embedding_dim=args.embedding_dim,
            num_embeddings=50,
            feature_names=["occupation"],
        ),
        EmbeddingBagConfig(
            name="year",
            embedding_dim=args.embedding_dim,
            num_embeddings=2050,
            feature_names=["year"],
        ),
        EmbeddingBagConfig(
            name="genres",
            embedding_dim=args.embedding_dim,
            num_embeddings=20,
            feature_names=["genres"],
        ),
    ]
    
    ebc = EmbeddingBagCollection(
        tables=eb_configs,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    
    mlp_dims = [int(dim) for dim in args.mlp_dims.split(",")]
    
    model = MovieLensModel(
        embedding_bag_collection=ebc,
        dense_in_features=0,
        dense_arch_layer_sizes=[1, 1],  # 占位符
        over_arch_layer_sizes=mlp_dims,
    )
    
    return model

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    data_path = download_movielens(args.data_path)
    train_dataset = MovieLensDataset(data_path, split="train")
    test_dataset = MovieLensDataset(data_path, split="test")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    
    model = create_model(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    criterion = nn.MSELoss()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Test Loss: {avg_test_loss:.4f}")
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pt"))
    

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()