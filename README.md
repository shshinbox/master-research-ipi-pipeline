# master-research-ipi-pipeline

다음의 3단계 워크플로우를 통해 진행됩니다.
- 자연어 데이터 합성
- 활성화 추출
- 분류기 학습
---

### Data Synthesizer

### 개요

- 트리거와 명령어를 클린 데이터의 context 시작, 중간, 끝 위치에 삽입하여 오염 데이터를 생성합니다.
- 생성된 데이터 쌍을 사용하여 hidden_state 추출에 활용됩니다.

#### A. 입력 데이터
| 타입        | arg 명          | 필수 필드             | 설명                            |
| ----------- | --------------- | --------------------- | ------------------------------- |
| 기본 데이터 | samples_file    | id, question, context | 원본 데이터                     |
| 삽입 지시문 | injections_file | instruction           | 트리거와 결합하여 합성될 지시문 |


#### B. 출력 데이터
| 필드명               | 설명                                |
| -------------------- | ----------------------------------- |
| id                   | 샘플 ID (clean_ / poisoned_ 접두사) |
| question             | 원본 질문 (Primary Task)            |
| context              | 데이터 블록                         |
| label                | 0 (clean) 또는 1 (poisoned)         |
| injected_instruction | 삽입된 최종 지시문                  |
| injected_position    | 삽입된 위치                         |
| trigger              | 트리거 문자열                       |


---
## Activation Extractor

### 개요

- 언어 모델의 질문(q_only) 입력과 질문+맥락(q_ctx) 입력에 대한 마지막 토큰의 hidden_state를 수집합니다.
- 수집된 hidden_state를 기반으로 hidden_state(q_ctx) - hidden_state(q_only)를 계산합니다.
- 클린 데이터셋 및 오염 데이터셋에서 추출하여 분류기 학습 데이터로 활용합니다.

#### A. 코어 로직 (src/actdt/)

| 파일명              | 역할                                                                               |
| ------------------- | ---------------------------------------------------------------------------------- |
| hidden_collector.py | 마지막 토큰의 hidden_states 추출 및 차이 계산, 입력 길이 제약에 맞춘 컨텍스트 관리 |
| layer_processor.py  | LLM 레이어 관리, 입력 레이어 인덱스 검증 및 매핑                                   |
| model_loader.py     | 모델 로딩 및 환경 설정                                                             |
| pipeline.py         | 활성화 추출 플로우 실행                                                            |
| template_builder.py | 프롬프트 적용                                                                      |


#### B. 실행 로직 (src/exec/, main.py)

| 파일명              | 역할                           |
| ------------------- | ------------------------------ |
| llama3_parser.py    | CLI 인수 정의 - Llama3 전용    |
| mistral7b_parser.py | CLI 인수 정의 - Mistral7B 전용 |
| phi3_parser.py      | CLI 인수 정의 - Phi3-mini 전용 |
| utils.py            | 인수 유효성 검증, 파싱         |
| main.py             | 최종 실행 진입점               |



---
### linear probe


