from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.optim import Adam
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,f1_score
import sys
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

#import mlflow
#from mlflow.tracking import MlflowClient
#from mlflow.entities import ViewType

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data  # This should be a list of dictionaries with 'input_code' and 'label' keys
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_code = sample['text']
        label = sample['label']

        # Tokenize and encode input code
        input_tokens = self.tokenizer(
            input_code,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return {
            'text': input_tokens,
            'label': label
        }


def makeDataset(x,y):
    data = []
    for value1, value2 in zip(x, y):
        temp = {}
        if(type(value1) == float):
            continue

        if(type(value2) == str):
            if(value2.isdigit()):
                temp['text'] = value1
                temp['label'] = int(value2)
            else:
                continue
        else:
            temp['text'] = value1
            temp['label'] = value2

        data.append(temp)
    random.shuffle(data)
    return data


def evaluateModel(test_dataset,codebert_model,model_name):
        #model_name = ''
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        with torch.no_grad():
            codebert_model.eval()
            criterion = torch.nn.CrossEntropyLoss()

            Y_shuffled, Y_preds, losses, xValues = [],[],[],[]
            for batch in DataLoader(test_dataset, batch_size=16, shuffle=True):
                inputs = batch['text']
                labels = batch['label']
                # print(input_tokens['input_ids'].shape)
                # print(inputs['input_ids'].shape)
                
                #inputs['input_ids'] = inputs['input_ids'].reshape(1, -1)
                #inputs['attention_mask'] = inputs['attention_mask'].reshape(1, -1)
                # print(inputs['input_ids'].shape)

                if(inputs['input_ids'].shape[0] != 1):
                    inputs['attention_mask'] = inputs['attention_mask'].squeeze()
                    inputs['input_ids'] = inputs['input_ids'].squeeze()
                
                else:
                    inputs['attention_mask'] = inputs['attention_mask'].reshape(1, -1)
                    inputs['input_ids'] = inputs['input_ids'].reshape(1, -1)
                
                decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in inputs['input_ids']]
                outputs = codebert_model(**inputs)
                #print(outputs.shape)
                #print(labels.shape)
                loss = criterion(outputs.logits, labels)

                losses.append(loss.item())
                Y_shuffled.append(labels)
                Y_preds.append(outputs.logits.argmax(dim=-1))
                xValues.extend(decoded_texts)

            Y_shuffled = torch.cat(Y_shuffled)
            Y_preds = torch.cat(Y_preds)

            valid = accuracy_score(Y_shuffled.cpu().detach().numpy(), Y_preds.cpu().detach().numpy())
            pre = precision_score(Y_shuffled.cpu().detach().numpy(), Y_preds.cpu().detach().numpy())
            rec = recall_score(Y_shuffled.cpu().detach().numpy(), Y_preds.cpu().detach().numpy())
            f1s = f1_score(Y_shuffled.cpu().detach().numpy(), Y_preds.cpu().detach().numpy())

            #print("Valid Loss : {:.3f}".format(torch.tensor(losses).mean()))
            print("Valid Acc  : {:.3f}".format(valid))
            print("Precision  : {:.3f}".format(pre))
            print("Recall     : {:.3f}".format(rec))
            print("F1 score   : {:.3f}".format(f1s))
            print()

        return valid, Y_shuffled, Y_preds, xValues


def crossValidate(model,model_name,fileName):
    # df_false = pd.read_csv("./data/Fake.csv")
    # df_true = pd.read_csv("./data/True.csv")
    # df_false = pd.read_csv("./data/BuzzFeed_fake_news_content.csv")
    # df_true = pd.read_csv("./data/BuzzFeed_real_news_content.csv")
    
    # Add a 'source' column to each dataset to identify where each row came from
    #df_false['label'] = 0
    #df_true['label'] = 1

    # Concatenate all into one dataframe with a 'source' column to identify the origin
    #df = pd.concat([df_true, df_false], ignore_index=True)
    df = pd.read_csv("./data/news_articles.csv")
    df = df.dropna()
    df = df[df['language'] == 'english']
    df['label'] = df['label'].replace({'Real': 1, 'Fake': 0})

    data_test= makeDataset(df["text"], df["label"])

    dataset = CustomDataset(data=data_test, tokenizer=tokenizer, max_seq_length=512)
    valid_acc, y_true, y_pred, xValues = evaluateModel(dataset,model,model_name)

    print(f"The confusion matrix is ")
    print(confusion_matrix(y_true, y_pred))
    print()

    data = {'Y_True': y_true, 'Y_False': y_pred,"X_Values": xValues}
    df = pd.DataFrame(data)

    df.to_csv(f'output_{fileName}.csv', index=False)

    return valid_acc


def trainModel(obj, codebert_model,model_name):
    #with mlflow.start_run() as run:
        num_elements_to_use = 100
        train_data, test_data, train_labels, test_labels = train_test_split(obj["text"][:num_elements_to_use], obj["label"][:num_elements_to_use], test_size=0.25, random_state=42)

        data_train = makeDataset(train_data, train_labels)
        data_test = makeDataset(test_data,test_labels)

        train_dataset = CustomDataset(data=data_train, tokenizer=tokenizer, max_seq_length=512)
        test_dataset = CustomDataset(data=data_test, tokenizer=tokenizer, max_seq_length=512)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(codebert_model.parameters(), lr=1e-6)



        num_epochs = 2
        bs = 32

        # Log parameters (for example)
        #mlflow.log_param("num_epochs", num_epochs)
        #mlflow.log_param("batch_size", bs)
        #mlflow.log_param("learning_rate", 5e-5)

        for epoch in range(num_epochs):
            codebert_model.train()
            total_loss = 0.0
            flag = False
            for batch in DataLoader(train_dataset, batch_size=bs, shuffle=True):
                optimizer.zero_grad()

                inputs = batch['text']
                labels = batch['label']
                # print(input_tokens['input_ids'].shape)
                #print(inputs['input_ids'].shape)
                #inputs['input_ids'] = inputs['input_ids'].reshape(bs, -1)
                #inputs['attention_mask'] = inputs['attention_mask'].reshape(bs, -1)
                if(inputs['input_ids'].shape[0] != 1):
                    inputs['attention_mask'] = inputs['attention_mask'].squeeze()
                    inputs['input_ids'] = inputs['input_ids'].squeeze()
                
                else:
                    inputs['attention_mask'] = inputs['attention_mask'].reshape(1, -1)
                    inputs['input_ids'] = inputs['input_ids'].reshape(1, -1)
                

                #print(inputs['input_ids'].shape)
                outputs = codebert_model(**inputs)
                #print()
                #print(outputs.logits.shape,labels.shape)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if (not flag):
                    print("completed one sample in the epoch")
                    flag = True

            
            v = evaluateModel(test_dataset,codebert_model,model_name)
            average_loss = total_loss / len(train_dataset)

            # Log metrics after each epoch
            #mlflow.log_metric("loss", average_loss, step=epoch)
            #mlflow.log_metric("valid_acc", v, step=epoch)

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.4f}")
            #print(f"The validation accuracy is {v}")
            print()

        # Save the trained model
        # torch.save(codebert_model.state_dict(), "custom_codebert_model_bs2.pth")
        # codebert_model.save_pretrained("./trained_codebert_bs2")
        #mlflow.pytorch.log_model(codebert_model, "codebert_model")
        return codebert_model


# Load the CodeBERT model and tokenizer

if __name__ == "__main__":
    model_name = "hamzab/roberta-fake-news-classification" #"distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    codebert_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # obj = pd.read_csv("./mbpp_with_fcodegen.csv")
    # df_false = pd.read_csv("./buzzfeed/BuzzFeed_fake_news_content.csv")
    # df_true = pd.read_csv("./buzzfeed/BuzzFeed_real_news_content.csv")
    #df = pd.read_csv("./data/WELFake.csv")
    df_false = pd.read_csv("./data/Fake.csv")
    df_true = pd.read_csv("./data/True.csv")
    
    # Add a 'source' column to each dataset to identify where each row came from
    df_false['label'] = 0
    df_true['label'] = 1

    # Concatenate all into one dataframe with a 'source' column to identify the origin
    df = pd.concat([df_true, df_false], ignore_index=True)
    
    codebert_model = trainModel(df, codebert_model,model_name)
    v = crossValidate(codebert_model,model_name,"finetuned")

    del codebert_model
    del tokenizer


    #model_name = "hamzab/roberta-fake-news-classification"
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model_true_1 = AutoModelForSequenceClassification.from_pretrained(model_name)


    #v = crossValidate(model_true_1,model_name,"original")

    #del model_true_1
    #del tokenizer

    model_name = "hamzab/roberta-fake-news-classification"#"jy46604790/Fake-News-Bert-Detect"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_true_2 = AutoModelForSequenceClassification.from_pretrained(model_name)

    v = crossValidate(model_true_2,model_name,"otherModel")


