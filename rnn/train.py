import torch
import argparse
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
import matplotlib.pyplot as plt
from google.colab import files
from torchsummary import summary


# to do
# 1. use kao's data
# 2. Increase the model capacity by adding more Linear or LSTM layers.
# 3. Split the dataset into train, test, and validation sets.
# 4. Add checkpoints so you don't have to train the model every time you want to run prediction.

def train(dataset, model, args):
    model.train()

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_loss_values=[]

    for epoch in range(args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        loss_values=[]

        for batch, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())

            print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })

        
        epoch_loss_values.append(np.mean(loss_values))

    torch.save(model.state_dict(), "/content/drive/MyDrive/Colab Notebooks/text-generation/models/"+args.name)
    print(epoch_loss_values)
    plt.plot(np.arange(len(epoch_loss_values)), epoch_loss_values)

def run_loaded_model(model_path, text):
  dataset = Dataset(args)
  embs_npa = pretrain_embeddings()
  model = Model(embs_npa, dataset, num_layers=7)

  model.load_state_dict(torch.load(model_path))
  print(predict(dataset, model, text=text))
  print(model)





def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words


def pretrain_embeddings():
  
  vocab,embeddings = [],[]

  with open('glove.6B.100d.txt','rt') as fi:
      full_content = fi.read() # read the file
      full_content = full_content.strip() # remove leading and trailing whitespace
      full_content = full_content.split('\n') # split the text into a list of lines

  for i in range(len(full_content)):
      i_word = full_content[i].split(' ')[0] # get the word at the start of the line
      i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]] # get the embedding of the word in an array
      # add the word and the embedding to our lists
      vocab.append(i_word)
      embeddings.append(i_embeddings)
  
  # convert our lists to numpy arrays:
  import numpy as np
  vocab_npa = np.array(vocab)
  embs_npa = np.array(embeddings)

  # insert tokens for padding and unknown words into our vocab
  vocab_npa = np.insert(vocab_npa, 0, '<pad>')
  vocab_npa = np.insert(vocab_npa, 1, '<unk>')
  print(vocab_npa[:10])

  # make embeddings for these 2:
  # -> for the '<pad>' token, we set it to all zeros
  # -> for the '<unk>' token, we set it to the mean of all our other embeddings

  pad_emb_npa = np.zeros((1, embs_npa.shape[1])) 
  unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True) 

  #insert embeddings for pad and unk tokens to embs_npa.
  embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
  print(embs_npa.shape)

  # convert our lists to numpy arrays:
  vocab_npa = np.array(vocab)
  embs_npa = np.array(embeddings)

  return embs_npa


embs_npa = pretrain_embeddings()

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=4)
parser.add_argument('--name', type=str, default="model_")
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--model_path', type=str, default="")
parser.add_argument('--text', type=str, default="")


args = parser.parse_args()

if args.load:
  run_loaded_model(args.model_path, args.text)

else:

  dataset = Dataset(args)
  dataset.get_uniq_words()
  model = Model(embs_npa, dataset, num_layers=7)

  train(dataset, model, args)
  print(predict(dataset, model, text='All right'))