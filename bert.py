from transformers import BertTokenizer, BertModel
import torch
import pandas as pd

tweet_data = pd.read_csv("C:/Users/megcr/Documents/Stanford/CS221/logic/final_project/stock_tweets.csv")
tweet_tsla = tweet_data[tweet_data['Stock Name'] == 'TSLA']['Tweet']
last_hidden_statess = []
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
    return_attention_mask=False,
    return_token_type_ids=False)
model = BertModel.from_pretrained('bert-base-uncased')
for tweet in tweet_tsla:
  encoding = tokenizer.encode(tweet, add_special_tokens=True)
  input_tensor = torch.tensor([encoding])
  with torch.no_grad():
    outputs = model(input_tensor)
    last_hidden_states = outputs[0]
    last_hidden_statess.append(last_hidden_states)

i = 0
for tens in last_hidden_statess:
   torch.save(tens, f'C:/Users/megcr/Documents/Stanford/CS221/logic/final_project/tensor_{i}.pt')
   i+=1