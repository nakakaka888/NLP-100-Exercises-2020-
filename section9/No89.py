import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput
from transformers import TrainingArguments
from transformers import Trainer
from transformers import EarlyStoppingCallback
from sklearn.metrics import classification_report
from transformers import EvalPrediction
from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score


def data_modify(data):
    data_x = data['TITLE']
    data_x = data_x.str.replace(r'\'s|[\'\"\:\.,\;\!\&\?\$]', '', regex=True)
    data_x = data_x.str.replace(r'\s-\s', ' ', regex=True)
    data_x = data_x.str.lower()
    
    return data_x


train_x = pd.read_csv('./section9/train.txt', sep='\t') 
train_y = pd.read_csv('./section9/train_y.txt', sep='\t') 
train_y = train_y['CATEGORY']
train_x = data_modify(train_x)

valid_x = pd.read_csv('./section9/valid.txt', sep='\t') 
valid_y = pd.read_csv('./section9/valid_y.txt', sep='\t') 
valid_y = valid_y['CATEGORY']
valid_x = data_modify(valid_x)


test_x = pd.read_csv('./section9/test.txt', sep='\t') 
test_y = pd.read_csv('./section9/test_y.txt', sep='\t') 
test_y = test_y['CATEGORY']
test_x = data_modify(test_x)

class NewsDataset(Dataset):
    def __init__(self, df_title, df_category):
        self.features = [
            {
                'title': title,
                'category_id': category_id
            } for title, category_id in zip(df_title, df_category)
        ]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
train_dataset = NewsDataset(train_x, train_y)
valid_dataset = NewsDataset(valid_x, valid_y)
test_dataset  = NewsDataset(test_x,  test_y)

class NewsCollator():
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        examples = {
            'title': list(map(lambda x: x['title'],  examples)),
            'category_id': list(map(lambda x : x['category_id'], examples))
        }

        encoding = self.tokenizer(examples['title'],
                                  padding=True,
                                  truncation=True,
                                  max_length=self.max_length,
                                  return_tensors='pt')
        encoding['category_id'] = torch.tensor(examples['category_id'])

        return encoding

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
news_collator = NewsCollator(tokenizer)

loader = DataLoader(train_dataset, collate_fn=news_collator, batch_size=16, shuffle=True)
batch = next(iter(loader))

class NewsNet(nn.Module):
    def __init__(self, pretrained_model, num_categories, loss_function=None):
        super().__init__()
        self.bert = pretrained_model
        self.hidden_size = self.bert.config.hidden_size
        self.linear = nn.Linear(self.hidden_size, num_categories)
        self.loss_function = loss_function

    def forward(self, 
                input_ids, 
                attention_mask=None, 
                output_attentions=False, 
                output_hidden_states=False, 
                category_id=None):
        
        outputs = self.bert(input_ids, 
                            attention_mask=attention_mask,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states)
        
        state = outputs.last_hidden_state[:, 0, :]
        logits = self.linear(state)

        loss = None
        if category_id is not None and self.loss_function is not None:
            loss = self.loss_function(logits, category_id)

        attentions = outputs.attentions if output_attentions else None
        hidden_states = outputs.hidden_states if output_hidden_states else None
        
        return ModelOutput(
            logits=logits,
            loss=loss,
            last_hidden_state=outputs.last_hidden_state,
            attentions=attentions,
            hidden_states=hidden_states
        )


loss_fct = nn.CrossEntropyLoss()
pretrained_model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
net = NewsNet(pretrained_model, 4, loss_fct)


def custom_compute_metraics(res: EvalPrediction) -> Dict:

    pred  = res.predictions.argmax(axis=1)
    target = res.label_ids
    precision = precision_score(target, pred, average='macro')
    recall = recall_score(target, pred, average="macro")
    f1 = f1_score(target, pred, average="macro")
    return {
        "precision" : precision,
        "recall" : recall,
        "f1" : f1
    }


traing_args = TrainingArguments(
    output_dir='./section9/output',
    eval_strategy='epoch',
    logging_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    label_names=['category_id'],
    lr_scheduler_type='constant',
    learning_rate= 1e-5,
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    remove_unused_columns=False,
    report_to='none'
)

trainer = Trainer(
    model=net,
    tokenizer=tokenizer,
    data_collator=news_collator,
    compute_metrics=custom_compute_metraics,
    args=traing_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train(ignore_keys_for_eval=["last_hidden_state", "hidden_states", "attentions"])

pred_result = trainer.predict(test_dataset, ignore_keys =["loss","last_hidden_state", "hidden_states", "attentions"])

test_df = pred_result.predictions.argmax(axis=1).tolist()

print(classification_report(test_y, test_df, target_names=["business", "tech", "entertainment", "politics"]))

"""
{'loss': 0.3515, 'grad_norm': 0.5540059208869934, 'learning_rate': 1e-05, 'epoch': 1.0}
{'eval_loss': 0.22094397246837616, 'eval_precision': 0.903764257422794, 'eval_recall': 0.8841016565514698, 'eval_f1': 0.8934607106573468, 'eval_runtime': 0.969, 'eval_samples_per_second': 1376.645, 'eval_steps_per_second': 86.685, 'epoch': 1.0}
{'loss': 0.1805, 'grad_norm': 28.383148193359375, 'learning_rate': 1e-05, 'epoch': 2.0}
{'eval_loss': 0.2416733205318451, 'eval_precision': 0.9038807002351863, 'eval_recall': 0.8882031340638536, 'eval_f1': 0.8956157123712849, 'eval_runtime': 0.9608, 'eval_samples_per_second': 1388.422, 'eval_steps_per_second': 87.427, 'epoch': 2.0}
{'loss': 0.1181, 'grad_norm': 0.11286503821611404, 'learning_rate': 1e-05, 'epoch': 3.0}
{'eval_loss': 0.23758374154567719, 'eval_precision': 0.9196313978958215, 'eval_recall': 0.8974140919311483, 'eval_f1': 0.9080199171247877, 'eval_runtime': 0.9654, 'eval_samples_per_second': 1381.788, 'eval_steps_per_second': 87.009, 'epoch': 3.0}
{'loss': 0.0696, 'grad_norm': 14.453102111816406, 'learning_rate': 1e-05, 'epoch': 4.0}
{'eval_loss': 0.28849542140960693, 'eval_precision': 0.9288598557579772, 'eval_recall': 0.9080833040988083, 'eval_f1': 0.9180298430444519, 'eval_runtime': 0.9636, 'eval_samples_per_second': 1384.459, 'eval_steps_per_second': 87.177, 'epoch': 4.0}
{'loss': 0.0474, 'grad_norm': 0.020267806947231293, 'learning_rate': 1e-05, 'epoch': 5.0}
{'eval_loss': 0.30563899874687195, 'eval_precision': 0.9197174237511452, 'eval_recall': 0.9053154108558252, 'eval_f1': 0.9120524585018082, 'eval_runtime': 0.9621, 'eval_samples_per_second': 1386.544, 'eval_steps_per_second': 87.309, 'epoch': 5.0}
{'train_runtime': 220.7477, 'train_samples_per_second': 241.724, 'train_steps_per_second': 15.108, 'train_loss': 0.15343808084056115, 'epoch': 5.0}
100%|███████████████████████████████████████████████████████████████████████████████| 3335/3335 [03:40<00:00, 15.11it/s]
100%|███████████████████████████████████████████████████████████████████████████████████| 84/84 [00:00<00:00, 87.63it/s]
               precision    recall  f1-score   support

     business       0.95      0.96      0.95       571
         tech       0.86      0.85      0.85       173
entertainment       0.96      0.97      0.97       511
     politics       0.92      0.85      0.88        79

     accuracy                           0.94      1334
    macro avg       0.92      0.91      0.91      1334
 weighted avg       0.94      0.94      0.94      1334
"""