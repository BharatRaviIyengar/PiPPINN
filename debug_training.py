import torch, TrainUtils
from time import time
dataset = torch.load("../Data_For_Training.pt", weights_only=False)

hidden_channels = 2048
centrality_fraction = 0.5
batch_size = 20000
num_batches = None
device = 'cuda'
threads = 4
learning_rate = 0.001
weight_decay=0
dropout = 0.01
nbr_wt_intensity=1.0

data_for_training = [TrainUtils.generate_batch(data, num_batches, batch_size, centrality_fraction,device=device, threads=threads) for data in dataset]

model = TrainUtils.GraphSAGE(
 in_channels=data_for_training[0]["input_channels"],
 hidden_channels=hidden_channels,
 dropout = dropout
 ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

total_train_loss = 0.0
total_val_loss = 0.0

very_start = time()
for epochs in range(1, 11):
 start = time()
 for idx, data in enumerate(data_for_training):
  print(f"Graph {idx}: about to train")
  # Training
  model.train()
  for n,batch in enumerate(data["train_batch_loader"]):
   print(f"Training Batch {n} loaded")
   total_train_loss += TrainUtils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=True)
   print(f"Processed training batch {n} in graph {idx}")
# #  print("Setting model in eval mode")
  torch.cuda.empty_cache()
  model.eval()
  print("Starting validation")
  with torch.no_grad():
#    print("Inside inference_mode")
   for v,batch in enumerate(data["val_batch_loader"]):
    print(f"Validation Batch {v} of graph {idx} loaded")
    total_val_loss += TrainUtils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=False)
    print(f"Processed validation batch {v} in graph {idx}")
 end = time()
 print(f"Epoch: {epochs}, Train Loss: {total_train_loss}, Val Loss: {total_val_loss}, Time: {end - start:.2f} seconds")
print(f"Total training time: {time() - very_start:.2f} seconds")