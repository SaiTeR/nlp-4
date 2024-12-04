from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased', return_dict=True)

text = "Он думает только о " + tokenizer.mask_token + "."

input_data = tokenizer.encode_plus(text, return_tensors="pt")
mask_indices = torch.where(input_data["input_ids"][0] == tokenizer.mask_token_id)

# Получение предсказаний модели
output = model(**input_data)
logits = output.logits

# softmax для получения вероятностей
probabilities = F.softmax(logits, dim=-1)



print(f"{'№':<4} {'Word':<20} {'Weight':<10} {'Match':<10}")
print("-" * 50)

i = 1
for mask_idx in mask_indices[0]:
    mask_probabilities = probabilities[0, mask_idx, :]
    top_predictions = torch.topk(mask_probabilities, 10)

    for token_id, score in zip(top_predictions.indices, top_predictions.values):
        predicted_word = tokenizer.decode([token_id])
        is_match = "[Match]" if predicted_word in {"себе", "работе"} else ""
        
        print(f"{i:<4} {predicted_word:<20} {score.item():<10.4f} {is_match:<10}")
        
        i += 1
        if i == 10: break
    if i == 10: break 