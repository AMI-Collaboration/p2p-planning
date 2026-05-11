# 공간 분리 환경에서 Peer-to-Peer 협업 기반 Embodied 멀티에이전트 플래닝 프레임워크

**A Peer-to-Peer Collaboration-Based Embodied Multi-Agent Planning Framework for Spatially Separated Environments**

스마트홈, 물류창고 등 공간적으로 분리된 환경에서 각 에이전트는 자신의 구역에서만 시각 정보를 획득할 수 있다. 기존 멀티에이전트 연구는 중앙화된 오케스트레이터가 모든 관측을 수집하고 플랜을 조율하는 방식에 의존해 왔으나 이는 로컬 컨텍스트 반영이 어렵고 에이전트 수 증가 시 중앙 처리 부담이 커지는 한계가 있다.

본 연구는 **Peer-to-Peer(P2P) 협상만으로 최종플랜을 수립하는 프레임워크**를 제안한다:

1. 각 에이전트가 로컬 시각 관측으로부터 구조화된 **Offer**와 초안 플랜을 독립적으로 생성
2. Offer와 초안 플랜을 상호 교환하여 **Mutual-Aware 로컬 플래닝** 수행
3. 충돌 발생 시 **P2P Negotiation Loop**를 통해 자율적으로 해소
4. 해소 불가한 예외적 케이스에만 **Human Query** 활성화


## Architecture

<p align="center">
  <img src="Figure.png" alt=" Architecture" width="850">
</p>

```
Individual Planning  →  P2P Negotiation Loop  →  Human Query (필요시)  →  Final Plan
```

| 단계 | 설명 |
|---|---|
| **Individual Planning** | 각 에이전트가 로컬 이미지를 VLM에 입력하여 Offer(관측 객체, 수행 가능/불가 액션, PASS·RECEIVE 아이템)와 초안 플랜 생성 후 상호 교환 |
| **P2P Negotiation Loop** | 라운드마다 수정 제안을 동시 생성, 수락된 스텝은 확정·잠금. 수렴 조건(모순 없음, PASS-RECEIVE 연결 완결, 관측 범위 내 액션) 3가지 충족 시 조기 종료 |
| **Human Query** | 최대 협상 라운드 이후에도 충돌 미해소 시 활성화 (전체의 10%). VLM이 자연어 질문 생성 → 답변 기반으로 스텝 삭제/유지 결정 후 플랜 병합 |


## 실험 결과

### Baseline 비교 (RQ1)

| 방법 | TS | PE | OC | SC | Final Score |
|---|---|---|---|---|---|
| Independent | 3.53 | 4.80 | 5.50 | 4.43 | 4.17 |
| Centralized | 4.63 | 5.50 | 6.13 | 5.33 | 5.11 |
| **Ours** | **8.33** | **8.70** | **8.57** | **8.73** | **8.64** |

- Independent 대비 Task Success **2.4배 향상**
- Centralized 대비 Task Success **1.8배 향상**

> 평가 지표: Task Success(TS, w=0.40), Plan Executability(PE, w=0.25), Observability Consistency(OC, w=0.10), Sequential Coherence(SC, w=0.25). GPT-4o 기반 자동 평가.

### Ablation Study (RQ2)

| 설정 | TS | PE | OC | SC | Final Score |
|---|---|---|---|---|---|
| w/o Offer Exchange | 4.50 | 7.50 | 7.15 | 5.52 | 6.25 |
| w/o Negotiation Loop | 4.10 | 7.27 | 7.00 | 5.13 | 5.97 |
| **Full (Ours)** | **5.75** | **7.98** | **8.20** | **6.80** | **7.20** |

Offer Exchange 제거 시 PASS–RECEIVE 관계 추론이 불가능해져 TS·SC에서 두드러진 감소. Negotiation Loop 제거 시 충돌 미해소 상태로 플랜이 병합되어 SC에서 가장 큰 저하 발생.

### 파이프라인 동작 분석

| 단계 | 건수 | 비율 |
|---|---|---|
| 충돌 없음 (Individual Planning에서 확정) | 12 | 40% |
| P2P Negotiation Loop 진입 | 15 | 50% |
| Human Query 활성화 | 3 | 10% |
| **합계** | **30** | **100%** |

전체의 90%가 에이전트 간 자율 협상만으로 해결되었으며, Human Query는 최후의 안전장치로 기능한다.


## 레포지토리 구조

```
p2p-planning/
├── Code_p2p/          # 전체 파이프라인 구현 코드
├── Data/              # 태스크 설명 및 환경 이미지 (AI2-THOR)
├── Prompts/
│   └── evaluation/    # 자동 평가용 LLM Judge 프롬프트
└── README.md
```


## 실험 환경

- **시뮬레이터**: AI2-THOR
- **모델**: GPT-4o (플래닝 VLM + 평가 LLM Judge)
- **태스크**: Long-Horizon 가사 태스크 10종
- **환경**: 주방, 거실, 침실, 욕실 중 서로 다른 두 공간의 조합
- **총 실험 횟수**: 30회 (이미지 쌍 3개 × 태스크 10종)
