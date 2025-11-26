# master-research-ipi-pipeline

다음의 3단계 워크플로우를 통해 진행됩니다.
- 자연어 데이터 합성
- 활성화 추출
- 분류기 학습
---

### Data Synthesizer

---
## Activation Extractor

### 구조

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


