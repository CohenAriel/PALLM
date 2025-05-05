import gzip
import json
from transformers import BertTokenizerFast, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, pipeline, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, util
import torch
import os

RUN_BASELINE = False

def main(model_path):
    total_dist = 0
    count = 0
    line_idx = 0
    model,mask_filler = upload_model(model_path)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    echr_test = "data/echr_test.jsonl.gz"
    emb_model = SentenceTransformer('all-MiniLM-L6-v2',  device=device)
    test_masked_path = r'data\masked_echr_test.jsonl.gz'

    if os.path.exists(test_masked_path):
        print("Loading masked dataset...")
        
    else:
        print("Masking PII and saving to file...")
        with gzip.open(echr_test, 'rt', encoding='utf-8') as fin, \
            gzip.open(test_masked_path, 'wt', encoding='utf-8') as fout:
            for i,line in enumerate(fin):
                print('saving row: ' + str(i))
                line_data = json.loads(line)
                line_data['pii_mask_idx'].sort(key=lambda x: x['start'])
                line_data = mask_pii(line_data)
                fout.write(json.dumps(line_data, ensure_ascii=False) + "\n")
                if i>=6192:
                    break
            
    with gzip.open(test_masked_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line_data = json.loads(line)
            line_masked = line_data['masked_text']
            model_predict = mask_filler(line_masked, top_k=1)
            for i,predicted in enumerate(model_predict):
                if len(model_predict)==1:
                    pred_word = predicted['token_str']
                else:
                    pred_word = predicted[0]['token_str']
                start_word = line_data['pii_mask_idx'][i]['value']
                embeddings = emb_model.encode([start_word,pred_word], convert_to_tensor=True, device=device)
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                dist = 1-similarity
                total_dist += dist
                count +=1
            line_idx+=1
            print(line_idx)
            if line_idx>=6192:
                break
        print(total_dist)
        print(total_dist/count)


def mask_pii(line_data):
    text = line_data['text']
    pii_mask_idx = line_data['pii_mask_idx']
    pii_sorted = sorted(pii_mask_idx, key=lambda x: x['start'], reverse=True)
    for pii in pii_sorted:
        start, end = pii['start'], pii['end']
        text = text[:start] + '[MASK]' + text[end:]
    line_data['masked_text'] = text
    return line_data

def get_private_tokens(line_data):
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    text = line_data['text']
    pii_mask_idx = line_data['pii_mask_idx']
    encoded = tokenizer(text, return_offsets_mapping=True, return_tensors='pt', truncation=True)
    offsets = encoded['offset_mapping'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])

    label_to_token_indices = {}
    for pii in pii_mask_idx:
        matched_tokens = []
        for i, (start_char, end_char) in enumerate(offsets):
            if end_char > pii['start'] and start_char < pii['end']:
                matched_tokens.append(i)
        label_to_token_indices[pii['label']] = matched_tokens

    return label_to_token_indices

def upload_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    if RUN_BASELINE==False:
        current_model_dict = model.state_dict()
        loaded_state_dict = torch.load(model_path, map_location=device)

        new_state_dict = {
            k: v if v.size() == current_model_dict[k].size() else current_model_dict[k]
            for k, v in zip(current_model_dict.keys(), loaded_state_dict.values())
        }
        model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)

    mask_filler = pipeline(
        task="fill-mask",
        model=model,
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        device=device,
    )
    return model,mask_filler

def calculate_similarity(text1,text2):
    from torch.nn.functional import cosine_similarity
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cpu") 
    model.to(device)

    def get_cls_embedding(text: str):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :] 

    embedding1 = get_cls_embedding(text1)
    embedding2 = get_cls_embedding(text2)

    similarity = cosine_similarity(embedding1, embedding2).item()
    dist = 1- similarity
    return dist

if __name__=='__main__':
    model_path1 = r"models\bert-echr-working-clean.pth"
    model_path = model_path1
    main(model_path)

    