from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import re
import subprocess
import json
import os
import clang.cindex

app = Flask(__name__)

# 모델 정의 (LSTM 모델 사용)
class SyntaxErrorLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super(SyntaxErrorLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size + 1, num_classes)  # +1 for static analysis result

    def forward(self, x, static_analysis_result):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        x = torch.cat((x, static_analysis_result.unsqueeze(1).float()), dim=1)
        out = self.fc(x)
        return out

# 하이퍼파라미터
vocab = {"<PAD>": 0, "<UNK>": 1}
vocab_size = len(vocab)
embed_size = 128
hidden_size = 256
num_layers = 2
num_classes = 2

model = SyntaxErrorLSTM(vocab_size, embed_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 가중치 로드
# model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

def tokenize_code(code):
    tokens = re.findall(r'\w+|[^\w\s]', code, re.UNICODE)
    return tokens

def encode_tokens(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

def run_static_analysis(code):
    with open("temp.c", "w") as f:
        f.write(code)
    result = subprocess.run(["gcc", "-fsyntax-only", "temp.c"], capture_output=True, text=True)
    if result.returncode != 0:
        return 1, result.stderr  # 신텍스 에러 있음
    else:
        return 0, "No syntax error detected"  # 신텍스 에러 없음

def save_to_dataset(code, label, error_locations, error_type):
    dataset_path = os.path.join(app.root_path, 'dataset.json')
    data_entry = {'code': code, 'label': label, 'error_locations': error_locations, 'error_type': error_type}
    
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r+') as file:
            data = json.load(file)
            data.append(data_entry)
            file.seek(0)
            json.dump(data, file, indent=4)  # JSON 데이터를 보기 좋게 저장
    else:
        with open(dataset_path, 'w') as file:
            json.dump([data_entry], file, indent=4)  # JSON 데이터를 보기 좋게 저장

def load_dataset():
    dataset_path = os.path.join(app.root_path, 'dataset.json')
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as file:
            data = json.load(file)
            return data
    return []

class CodeDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = self.data[idx]['code']
        label = self.data[idx]['label']
        tokens = tokenize_code(code)
        encoded_tokens = encode_tokens(tokens, self.vocab)
        max_len = 512
        padded_tokens = encoded_tokens + [self.vocab['<PAD>']] * (max_len - len(encoded_tokens))
        return torch.tensor(padded_tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long), self.data[idx]['error_type']

def retrain_model():
    data = load_dataset()
    dataset = CodeDataset(data, vocab)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model.train()
    num_epochs = 5
    for epoch in range(num_epochs):
        for code, label, error_type in dataloader:
            outputs = model(code, torch.zeros(code.size(0), 1))  # Dummy static analysis result
            loss = criterion(outputs, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Save the model weights after retraining
    torch.save(model.state_dict(), 'model_weights.pth')

def static_analysis(code):
    index = clang.cindex.Index.create()
    tu = index.parse('tmp.c', args=['-std=c11'], unsaved_files=[('tmp.c', code)], options=0)
    errors = []
    for diag in tu.diagnostics:
        error_message = f"{diag.location.line}번째 줄에 {diag.spelling}."
        errors.append(error_message)
    return errors

def regex_analysis(code):
    pattern = r'\bprintf\("([^"]+)"\);'
    matches = re.findall(pattern, code)
    return matches

def add_comments(code):
    lines = code.split('\n')
    commented_code = []
    for line in lines:
        if re.match(r'^\s*#include', line):
            commented_code.append(f"{line} // 헤더 파일 포함")
        elif re.match(r'^\s*int\s+main\s*\(\s*\)', line):
            commented_code.append(f"{line} // 메인 함수 시작")
        elif re.search(r'\bprintf\b', line):
            commented_code.append(f"{line} // 문자열 출력")
        elif re.match(r'^\s*if\s*\(.*\)', line):
            commented_code.append(f"{line} // 조건문")
        elif re.match(r'^\s*else\s*', line):
            commented_code.append(f"{line} // 조건문의 else 절")
        elif re.match(r'^\s*for\s*\(.*\)', line):
            commented_code.append(f"{line} // for 반복문")
        elif re.match(r'^\s*while\s*\(.*\)', line):
            commented_code.append(f"{line} // while 반복문")
        elif re.match(r'^\s*return\b', line):
            commented_code.append(f"{line} // 함수 종료 및 반환")
        elif re.match(r'.*;$', line):
            commented_code.append(f"{line} // 명령문")
        else:
            commented_code.append(line)
    return "\n".join(commented_code)

def variable_tracing(code):
    variables = {}  # 변수 상태를 저장하는 사전
    traced_lines = []  # 추적된 코드를 저장하는 리스트
    lines = code.split('\n')  # 코드를 줄별로 분리

    for line in lines:
        stripped_line = line.strip()
        # 변수 선언과 동시에 할당
        declaration_match = re.match(r'(\w+)\s+(\w+)\s*=\s*(.*);', stripped_line)
        # 기존 변수에 대한 할당
        assignment_match = re.match(r'(\w+)\s*=\s*(.*);', stripped_line)

        if declaration_match:
            var_type, var_name, var_value = declaration_match.groups()
            variables[var_name] = var_value  # 변수 사전에 정보 저장
            traced_lines.append(f"{line} // {var_name} ({var_type}) 초기화: {var_value}")
        elif assignment_match:
            var_name, var_value = assignment_match.groups()
            if var_name in variables:
                old_value = variables[var_name]
                variables[var_name] = var_value  # 변수 사전을 업데이트
                traced_lines.append(f"{line} // {var_name} 업데이트됨: {old_value} -> {var_value}")
            else:
                traced_lines.append(f"{line} // 경고: {var_name} 이(가) 선언되지 않았습니다")
        else:
            traced_lines.append(line)  # 다른 유형의 줄은 그대로 추가

    return "\n".join(traced_lines), variables

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    code = request.form['code']
    error_locations = request.form.get('error_locations', '').split(',')
    error_type = request.form.get('error_type', '')
    save_data = request.form.get('save_data', 'false') == 'true'
    label = int(request.form.get('label', '0'))
    
    tokens = tokenize_code(code)
    encoded_tokens = encode_tokens(tokens, vocab)
    max_len = 512
    padded_tokens = encoded_tokens + [vocab['<PAD>']] * (max_len - len(encoded_tokens))
    
    static_analysis_result, analysis_message = run_static_analysis(code)
    
    code_tensor = torch.tensor([padded_tokens], dtype=torch.long)
    static_analysis_tensor = torch.tensor([static_analysis_result], dtype=torch.float)
    
    with torch.no_grad():
        outputs = model(code_tensor, static_analysis_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    result = 'Error detected' if predicted.item() == 1 or static_analysis_result == 1 else 'No error detected'
    
    # Save to dataset if requested
    if save_data:
        save_to_dataset(code, label, error_locations, error_type)

    static_errors = static_analysis(code)
    regex_matches = regex_analysis(code)
    commented_code = add_comments(code)
    traced_code, variables = variable_tracing(code)
    
    natural_language_output = f"The code analysis detected {result}. Details: {analysis_message}"

    return jsonify({
        'result': result,
        'message': analysis_message,
        'static_errors': static_errors,
        'regex_matches': regex_matches,
        'commented_code': commented_code,
        'traced_code': traced_code,
        'variables': variables,
        'natural_language_output': natural_language_output
    })

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    # 여기서는 간단한 응답 예제를 제공
    response_message = f"Received your message: {user_message}"
    return jsonify({'reply': response_message})

@app.route('/retrain', methods=['POST'])
def retrain():
    retrain_model()
    return jsonify({'message': 'Model retrained successfully'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
