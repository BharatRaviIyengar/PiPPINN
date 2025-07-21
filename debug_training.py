import torch, TrainUtils
from time import time
dataset = torch.load("../Data_For_Training.pt", weights_only=False)

hidden_channels = 2048
centrality_fraction = 0.5
batch_size = 40000
num_batches = None
device = 'cuda'
threads = 4
learning_rate = 0.005
weight_decay=0
dropout = 0.01
nbr_wt_intensity=1.0
scheduler_factor = 0.5

data_for_training = [TrainUtils.generate_batch(data, num_batches, batch_size, centrality_fraction,device=device, threads=threads) for data in dataset]

model = TrainUtils.GraphSAGE(
 in_channels=data_for_training[0]["input_channels"],
 hidden_channels=hidden_channels,
 dropout = dropout
 ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
 optimizer= optimizer,
 mode='min',
 factor=scheduler_factor,
 patience=10)

very_start = time()
for epochs in range(1, 11):
 total_train_loss = 0.0
 total_val_loss = 0.0
 start = time()
 v = 0
 n = 0
 for idx, data in enumerate(data_for_training):
  print(f"Graph {idx}: about to train")
  # Training
  model.train()
  for batch in data["train_batch_loader"]:
   n += 1
    #  print(f"Training Batch {n} of graph {idx} loaded")
    #  print("Inside inference_mode")
  #  print(f"Training Batch {n} loaded")
   total_train_loss += TrainUtils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=True)
  #  print(f"Processed training batch {n} in graph {idx}")
# #  print("Setting model in eval mode")
  # torch.cuda.empty_cache()
  model.eval()
  # print("Starting validation")
  with torch.no_grad():
#    print("Inside inference_mode")
   for batch in data["val_batch_loader"]:
    v+=1
    # print(f"Validation Batch {v} of graph {idx} loaded")
    total_val_loss += TrainUtils.process_data(batch, model=model, optimizer=optimizer, device=device, is_training=False)
    # print(f"Processed validation batch {v} in graph {idx}")
 end = time()
 print(f"Epoch: {epochs}, Avg Train Loss: {total_train_loss/n}, Val Loss: {total_val_loss/v}, Time: {end - start:.2f} seconds")
 scheduler.step(total_val_loss/v)

print(f"Total training time: {time() - very_start:.2f} seconds")