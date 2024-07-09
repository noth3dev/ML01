# ML01 코드 분석기

## 소개

ML01 코드 분석기는 사용자가 C 코드를 웹 인터페이스에 입력하여 오류를 분석하고, 오류에 대한 자세한 설명을 자연어로 제공하며, AI 모델과 상호작용하여 오류에 대해 더 잘 이해할 수 있도록 도와주는 웹 기반 응용 프로그램입니다. 이 도구는 특히 초보 프로그래머들이 C 코드에서 오류를 식별하고 이해하는 데 도움을 주기 위해 설계되었습니다.

## 기능

- C 코드의 구문 오류 분석
- 오류 메시지를 자연어로 상세히 설명
- 분석된 코드를 데이터셋에 저장
- 새로운 데이터로 AI 모델 재학습
- 오류에 대해 질문하고 응답을 받을 수 있는 인터랙티브 채팅 기능

## 작동 원리

1. **코드 입력**: 사용자가 웹 인터페이스의 텍스트 영역에 C 코드를 입력합니다.
2. **오류 분석**: 백엔드가 GCC를 사용하여 코드의 구문 오류를 찾습니다.
3. **자연어 설명**: AI 모델이 기술적인 오류 메시지를 쉽게 이해할 수 있는 자연어 설명으로 변환합니다.
4. **저장 및 재학습**: 사용자는 분석된 코드와 오류를 데이터셋에 저장하고, 새로운 데이터로 AI 모델을 재학습시킬 수 있습니다.
5. **인터랙티브 채팅**: 사용자가 채팅 인터페이스를 통해 오류에 대해 질문하고 AI 모델로부터 응답을 받을 수 있습니다.

## 설치

1. **저장소 클론**:
    ```bash
    git clone https://github.com/yourusername/ML01-Code-Analyzer.git
    cd ML01-Code-Analyzer
    ```

2. **종속성 설치**:
    ```bash
    pip install -r requirements.txt
    ```

3. **애플리케이션 실행**:
    ```bash
    python app.py
    ```

4. **웹 인터페이스 열기**:
    웹 브라우저를 열고 `http://127.0.0.1:5001`로 이동합니다.

## 파일 구조

- `app.py`: 백엔드 로직을 처리하는 Flask 애플리케이션 파일
- `templates/index.html`: 웹 인터페이스를 위한 HTML 파일
- `static/style.css`: 웹 인터페이스의 스타일을 정의하는 CSS 파일
- `requirements.txt`: 애플리케이션 실행에 필요한 Python 패키지 목록
- `dataset.json`: 분석된 코드와 오류가 저장되는 JSON 파일 (데이터 저장 후 생성됨)

## 인공지능 모델 설명

### LSTM (Long Short-Term Memory)

LSTM은 순환 신경망(RNN)의 한 종류로, 긴 시퀀스 데이터에서도 장기 의존성을 학습할 수 있도록 설계되었습니다. 일반적인 RNN은 시간이 지남에 따라 장기 의존성을 학습하는 데 어려움을 겪는 반면, LSTM은 게이트 구조를 통해 이런 문제를 해결합니다.

#### 주요 개념

- **셀 상태(Cell State)**: LSTM의 핵심 아이디어로, 정보가 직접 전달되는 경로입니다. 셀 상태는 게이트를 통해 정보를 추가하거나 제거하면서 정보를 조절합니다.
- **입력 게이트(Input Gate)**: 새로운 정보가 셀 상태에 얼마나 반영될지를 결정합니다.
- **망각 게이트(Forget Gate)**: 이전 셀 상태의 정보 중 어느 부분을 버릴지를 결정합니다.
- **출력 게이트(Output Gate)**: 셀 상태의 정보를 얼마나 출력으로 내보낼지를 결정합니다.

#### LSTM의 구성

1. **임베딩 레이어**: 코드의 토큰을 벡터로 변환합니다. 코드의 각 토큰을 고정된 크기의 벡터로 변환하여 LSTM 모델에 입력으로 사용합니다.
    ```python
    self.embedding = nn.Embedding(vocab_size, embed_size)
    ```

2. **LSTM 레이어**: 시퀀스 데이터를 처리하며, 임베딩 벡터를 입력으로 받아서 시퀀스의 각 단계에 대해 출력을 생성합니다.
    ```python
    self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
    ```

3. **완전 연결(FC) 레이어**: LSTM의 출력을 받아 최종 예측을 수행합니다.
    ```python
    self.fc = nn.Linear(hidden_size + 1, num_classes)  # +1 for static analysis result
    ```

## 사용법

1. **코드 입력**:
    - 제공된 텍스트 영역에 C 코드를 입력합니다.
    - "Analyze" 버튼을 클릭하여 코드의 오류를 분석합니다.

2. **결과 보기**:
    - 결과는 "Static Analysis Errors", "Regex Analysis Matches", "Commented Code", "Traced Code" 섹션에 표시됩니다.
    - 자세한 오류 설명은 "Chatbox" 섹션에 표시됩니다.

3. **데이터셋에 저장**:
    - 오류 위치와 유형을 입력 필드에 입력합니다.
    - 코드에 오류가 있는지 여부를 선택합니다.
    - "Save to Dataset" 버튼을 클릭하여 분석된 데이터를 저장합니다.

4. **모델 재학습**:
    - "Retrain Model" 버튼을 클릭하여 데이터셋의 새로운 데이터로 AI 모델을 재학습시킵니다.

5. **AI와 채팅**:
    - 채팅 입력 필드에 질문을 입력합니다.
    - "Send" 버튼을 클릭하여 AI 모델로부터 응답을 받습니다.
    - GPT 사용(구현 X)

## 예제

1. **C 코드 입력**:
    ```c
    #include <stdio.h>
    int main() {
        printf("Hello, World!")
    }
    ```

2. **오류 보기**:
    - 오류 감지됨: `expected ';' before '}' token`

3. **자연어 설명**:
    - 코드 분석 결과 오류가 감지되었습니다. 자세한 내용: 3번째 줄, 30번째 열에서 `expected ';' before '}' token` 오류가 발생했습니다.

## 기술 세부 사항

- **Flask**: Python 기반의 가벼운 WSGI 웹 애플리케이션 프레임워크
- **PyTorch**: Python의 오픈 소스 머신 러닝 라이브러리로, AI 모델에 사용
- **GCC**: C 코드의 정적 분석을 위한 GNU 컴파일러 컬렉션
- **Clang**: C, C++, Objective-C 프로그래밍 언어를 위한 컴파일러 프론트 엔드, 더 자세한 정적 분석에 사용

## RNN (Recurrent Neural Network)

### 개요

순환 신경망(Recurrent Neural Network, RNN)은 시퀀스 데이터 또는 시계열 데이터를 처리하는 데 사용되는 인공 신경망의 한 종류입니다. RNN은 입력 데이터의 순서를 고려하여 이전 단계의 출력을 다음 단계의 입력으로 사용하는 특성을 가지고 있습니다. 이를 통해 RNN은 순차적인 데이터의 패턴과 시간적 의존성을 학습할 수 있습니다.

### 주요 특징

- **순환 구조**: RNN은 순환 구조를 통해 이전 시점의 정보를 현재 시점의 입력과 결합하여 처리합니다. 이 순환 구조는 RNN이 시간적 의존성을 학습하는 데 중요한 역할을 합니다.
  
- **숨겨진 상태(Hidden State)**: RNN의 각 시점에서 계산된 숨겨진 상태는 다음 시점의 입력으로 사용되며, 시퀀스 데이터의 정보를 압축하여 전달합니다. 이를 통해 RNN은 긴 시퀀스 데이터에서도 장기 의존성을 유지할 수 있습니다.

### 작동 원리

1. **입력 벡터( \( \mathbf{x}_t \) )**: 시퀀스 데이터의 각 시점에서 입력되는 데이터 벡터입니다.
2. **숨겨진 상태( \( \mathbf{h}_t \) )**: 이전 시점의 숨겨진 상태와 현재 시점의 입력 벡터를 결합하여 계산됩니다.
3. **출력 벡터( \( \mathbf{y}_t \) )**: 현재 시점의 숨겨진 상태를 기반으로 계산된 출력 값입니다.

#### 수학적 표현

RNN의 각 시점 \( t \)에서의 계산은 다음과 같이 표현됩니다:

\[ \mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h) \]

\[ \mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y \]

여기서:
- \( \mathbf{W}_{xh} \) : 입력 벡터 \( \mathbf{x}_t \) 에 대한 가중치 행렬
- \( \mathbf{W}_{hh} \) : 이전 숨겨진 상태 \( \mathbf{h}_{t-1} \) 에 대한 가중치 행렬
- \( \mathbf{W}_{hy} \) : 숨겨진 상태 \( \mathbf{h}_t \) 에서 출력 벡터 \( \mathbf{y}_t \) 로의 가중치 행렬
- \( \mathbf{b}_h, \mathbf{b}_y \) : 각 단계에서의 바이어스 벡터

### 한계와 문제점

- **기울기 소실(Vanishing Gradient)**: 긴 시퀀스 데이터를 학습할 때, 역전파 과정에서 기울기 소실 문제가 발생하여 초기 단계의 정보가 후반 단계로 전달되기 어려워집니다.
- **기울기 폭주(Exploding Gradient)**: 기울기가 급격히 커져서 학습이 불안정해질 수 있습니다.

### LSTM과 GRU

기본 RNN의 한계를 극복하기 위해 LSTM(Long Short-Term Memory)과 GRU(Gated Recurrent Unit)와 같은 개선된 순환 신경망 구조가 개발되었습니다. 이들 모델은 다양한 게이트를 통해 장기 의존성 문제를 해결하고 기울기 소실 문제를 완화합니다.

- **LSTM**: 셀 상태(cell state)와 입력 게이트, 망각 게이트, 출력 게이트를 사용하여 정보를 선택적으로 저장하고 전달합니다.
- **GRU**: 업데이트 게이트와 리셋 게이트를 사용하여 LSTM보다 간단한 구조로 정보를 조절합니다.

### 활용 분야

RNN은 다양한 시퀀스 데이터 처리 문제에 널리 사용됩니다. 주요 활용 분야는 다음과 같습니다:
- **자연어 처리(NLP)**: 언어 모델링, 기계 번역, 텍스트 생성 등
- **음성 인식**: 음성 데이터의 시퀀스 처리
- **시계열 예측**: 금융 시장 예측, 날씨 예측 등
- **비디오 분석**: 비디오 프레임 시퀀스 처리

### 결론

RNN은 시퀀스 데이터의 패턴과 시간적 의존성을 학습하는 강력한 도구입니다. 비록 기본 RNN이 기울기 소실 문제를 가지고 있지만, LSTM과 GRU와 같은 개선된 모델을 통해 이러한 문제를 극복하고 다양한 응용 분야에서 높은 성능을 발휘할 수 있습니다.

