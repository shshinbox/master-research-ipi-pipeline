# master-research-ipi-pipeline

다음의 3단계 워크플로우를 통해 진행됩니다.
- 자연어 데이터 합성
- 활성화 추출
- 분류기 학습
---
<br><br><br>

# Data Synthesizer

## 개요
트리거와 명령어를 클린 데이터의 context 앞, 중간, 끝 위치에 삽입하여 오염 데이터를 생성합니다.
생성된 데이터 쌍을 사용하여 **Activation Extractor** 단계에서 `hidden_state` 추출에 활용됩니다.

## 주요 기능
- 원본 샘플에 지시문을 앞/중간/끝 위치에 삽입
- 지시문은 Benign할 수도 Harmful할 수도 있음
- Clean, Poison, Combined 3가지 형식의 출력 파일 생성

## 실행 방법
```bash
python src/main.py \
    --samples_file data/samples.jsonl \
    --injections_file data/injections.json \
    --output_dir data/synthesized
```

## 파라미터

| 파라미터            | 타입 | 기본값                   | 설명                              |
| ------------------- | ---- | ------------------------ | --------------------------------- |
| `--samples_file`    | str  | `"samples_file.jsonl"`   | 원본 샘플 JSONL 파일 경로         |
| `--injections_file` | str  | `"injections_file.json"` | 삽입할 지시문 JSON 파일 경로      |
| `--output_dir`      | str  | `"data/synszr/output"`   | 출력 디렉토리 경로                |
| `--max_samples`     | int  | `0`                      | 클래스당 최대 샘플 수 (0: 무제한) |
| `--seed`            | int  | `42`                     | 재현성을 위한 랜덤 시드           |

**참고사항:**
- 원본 샘플 파일에는 `id`, `question`, `context`의 필수 필드가 필요합니다.
- 지시문 샘플 파일에는 `instruction`의 필수 필드가 필요합니다.


## 입력 데이터 형식

### JSONL 형식
각 줄은 하나의 샘플을 나타내며, 다음 필드를 포함해야 합니다:

**원본 샘플**
| 필드       | 타입 | 설명                     |
| ---------- | ---- | ------------------------ |
| `id`       | str  | 원본 샘플 ID             |
| `question` | str  | 원본 질문 (Primary Task) |
| `context`  | str  | 문맥 데이터 (Data Block) |

**삽입 지시문**
| 필드          | 타입 | 설명               |
| ------------- | ---- | ------------------ |
| `instruction` | str  | 삽입될 지시문 문장 |

**참고사항:**
- `question`은 이후 단계에서 **Primary Task**로 사용됩니다
- `context`는 지시문이 삽입되는 **Data Block**입니다

### 출력 데이터 구조
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
<br><br><br>




# Activation Extractor

## 개요
언어 모델의 `hidden_state`를 추출하여 질문 단독 입력(q_only)과 질문+문맥 입력(q_ctx)의 차이를 계산합니다. 추출된 **delta vector**는 poison sample 탐지 분류기 학습의 데이터로 활용됩니다.

## 주요 기능
- 질문 단독 입력(q_only)과 질문+문맥 입력(q_ctx)에 대한 마지막 토큰의 hidden_state 추출
- Delta Vector 계산: `Δ = hidden_state(q_ctx) - hidden_state(q_only)`
- 지정된 레이어의 hidden_state 수집
- 8bit 양자화 적용

## 실행 방법
```bash
python src/main.py \
    --in data/clean_samples.jsonl \
    --out data/deltas_clean_samples.jsonl \
    --model_id meta-llama/Meta-Llama-3-8B \
    --device cuda:0 \
    --layers "15,23,31" \
    --batch_size 1
```

## 파라미터

### A. 입출력 설정
| 파라미터 | 타입 | 기본값                         | 설명                 |
| -------- | ---- | ------------------------------ | -------------------- |
| `--in`   | str  | `"clean_samples.jsonl"`        | 입력 JSONL 파일 경로 |
| `--out`  | str  | `"deltas_clean_samples.jsonl"` | 출력 JSONL 파일 경로 |

### B. 모델 설정
| 파라미터      | 타입 | 설명                |
| ------------- | ---- | ------------------- |
| `--model_id`  | str  | HuggingFace 모델 ID |
| `--device`    | str  | 사용할 디바이스     |
| `--max_len`   | int  | 최대 시퀀스 길이    |
| `--eos_token` | str  | EOS 토큰 문자열     |


### C. 레이어 및 처리 설정
| 파라미터        | 타입 | 기본값       | 설명                             |
| --------------- | ---- | ------------ | -------------------------------- |
| `--layers`      | str  | `"15,23,31"` | 추출할 레이어 인덱스 (콤마 구분) |
| `--batch_size`  | int  | `1`          | 배치 크기                        |
| `--limit`       | int  | `0`          | 처리할 최대 샘플 수 (0: 전체)    |
| `--flush_every` | int  | `100`        | 출력 파일 플러시 간격            |
| `--seed`        | int  | `42`         | 랜덤 시드                        |

## 입력 데이터 형식

입력 JSONL 파일의 각 줄은 다음 필드를 포함해야 합니다:
```json
{
  "id": "sample_001",
  "question": "What is the capital of France?",
  "context": "France is a country in Western Europe. Paris is its capital."
}
```

| 필드       | 타입 | 설명                                    |
| ---------- | ---- | --------------------------------------- |
| `id`       | str  | 합성 샘플 고유 식별자                   |
| `question` | str  | 질문 텍스트 (q_only 입력으로 사용)      |
| `context`  | str  | 문맥 텍스트 (q_ctx에서 question과 결합) |

## 출력 데이터 형식

| 필드        | 타입        | 설명                                   |
| ----------- | ----------- | -------------------------------------- |
| `id`        | str         | 합성 샘플 ID                           |
| `question`  | str         | 원본 질문                              |
| `context`   | str         | 원본 문맥 혹은 합성된 문맥             |
| `delta_vec` | list[float] | 계산된 delta vector (모든 레이어 연결) |
| `layers`    | list[int]   | 추출된 레이어 인덱스                   |

**Delta Vector 구조:**
- 각 레이어의 hidden state 차이를 순서대로 연결
- 예: `layers="15,23,31"`, 각 레이어 차원이 4096이면 총 12288차원
---
<br><br><br>



# IPI(Indirect Prompt Injection) Classifier

## 개요
수집한 `hidden_state` 차이 벡터를 바탕으로 이진 분류를 LogisticRegression을 사용하여 poisoned sample을 탐지하는 분류기입니다.

## 주요 기능

- Clean 샘플과 Poison 샘플 간의 `hidden_state` 차이를 학습
- 검증 및 테스트 데이터에 대한 성능 평가
- 혼동행렬 시각화

## 실행 방법
```bash
python src/main.py \
    --train_files data/train_clean.jsonl data/train_poison.jsonl \
    --val_files data/val_clean.jsonl data/val_poison.jsonl \
    --test_files data/test_clean.jsonl data/test_poison.jsonl \
    --idx_type mistral_layer31 \
    --output_dir ./outputs \
    --model_name mistral31_clf
```

## 파라미터
### A. 주요 입력 파라미터 데이터
| 파라미터        | 타입 | 설명                  |
| --------------- | ---- | --------------------- |
| `--train_files` | list | 훈련 데이터 파일      |
| `--val_files`   | list | 검증 데이터 파일      |
| `--test_files`  | list | 테스트 데이터 파일    |
| `--idx_type`    | str  | 델타 벡터 인덱스 타입 |
| `--output_dir`  | str  | 결과 저장 경로        |
| `--model_name`  | str  | 모델 이름             |

**참고사항:**
- 각 파일 그룹(train/val/test)에 복수 개의 파일을 입력할 수 있습니다.
- 파일명에 반드시 `clean`, `poison`, `combined` 중 하나의 키워드를 포함해야합니다.
- `combined` 파일의 경우 내부에 `label` 필드 (0:clean, 1:poison)가 있어야 합니다.

### B. 모델 파이퍼파라미터
| 파라미터         | 타입  | 기본값 | 설명                    |
| ---------------- | ----- | ------ | ----------------------- |
| `--C`            | float | `1.0`  | 정규화 강도 역수        |
| `--max_iter`     | int   | `2000` | 최대 반복 횟수          |
| `--random_state` | int   | `42`   | 재현성을 위한 랜덤 시드 |

### C. 출력 설정
| 파라미터       | 타입 | 기본값           | 설명                      |
| -------------- | ---- | ---------------- | ------------------------- |
| `--output_dir` | str  | `"./outputs"`    | 모델 및 결과 저장 경로    |
| `--model_name` | str  | `"mitral31_clf"` | 저장될 모델 파일명 prefix |


## 입력 데이터 형식

### JSONL 형식
각 줄은 하나의 샘플을 나타내며, 다음 필드를 포함해야 합니다:
```json
{
  "delta_vec": [0.1, 0.2, ..., 0.9],
  "label": 0
}
```
| 필드        | 타입        | 설명                                                |
| ----------- | ----------- | --------------------------------------------------- |
| `delta_vec` | list[float] | Hidden state 차이 벡터 (필수)                       |
| `label`     | int         | 0: clean, 1: poison (combined 파일인 경우에만 필요) |
