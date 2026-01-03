# AI-for-The-Normal-and-How-to-make-BIG-Brother-at-LLM-s-essay
일반인들을 위한 AI 설명 및 AI에 빅브라더 심는 방법 및 해결

한국어 버전은 아래
한국어 버전 쓰고 영어 버전 perplexity 로 번역해서 붙임
I wrote at Korean and change it to English at perplextiy



---



**English Version:**

I think AI doesn’t need to strictly follow a one-to-one correspondence.  
Basically, there are four main types of AI: regression, classification, reinforcement learning, and deep learning.

Regression and classification predict one outcome.  
Reinforcement learning involves two steps (so you can think of it as having two stages).  
Deep learning has three stages.  
AlphaGo is classified as reinforcement learning rather than deep learning because its neuron-based algorithm communicates one-to-one between each neuron — that two-step structure makes it a reinforcement learning model. The idea of “training” in reinforcement learning is actually quite accurate in that sense.

### Regression and Classification

At first glance, when a rule is represented as a distribution, we can divide it at a 0.5 threshold — above 0.5 is regression, similar to logistic regression. However, logistic regression can be inefficient because in practice, simpler algorithms (like tree-based models) work better for CLI use. Logistic regression operates in a three-dimensional space, so it’s categorized as regression, but it’s not always optimal.

Conceptually, when rules are represented through distributions — like images — they can be described via that 0.5-based division. Whether we call that classification or regression depends on whether we’re interpreting a matrix. But actually, in practice, it’s the other way around: we find the rule *from* the distribution. It’s not like quantum mechanics (multiple states), but rather like distinguishing between scalar and vector properties.  

For example, $$ Ax + b $$ can be seen as classification because the key is whether, after the operation (not just calculation), one of the sets can be eliminated. That’s how we define regression versus classification.

As for deep learning and reinforcement learning, you can just think of them as extensions with more stages.

### NLP and LLM

NLP (Natural Language Processing) isn’t quite reinforcement learning; it’s closer to regression/classification with preprocessing. Preprocessing here means converting text data into numeric representations (like turning images into binary 0s and 1s). It’s not an AI by itself — more like a one-to-one conversion algorithm. NLP models mix preprocessing with regression/classification and often use ensemble techniques. 

The reason NLP’s development was slower is that deep learning (the 3-stage “inference-based” process) could generalize across different domains — meaning if a model finds the “core pattern” in one field, it should find it in another. NLP had to reach similar performance by combining 1-stage algorithms with preprocessing, which was difficult.

When humans read NLP outputs (like natural language text), they can judge correctness, but computers rely on numeric efficiency — numbers are simply the most efficient representation. That’s why STEM fields dominate computer science; at deeper levels, everything needs to be expressed numerically, requiring strong mathematical grounding. That’s also why universities recruit science majors for advanced computing studies.

The whole point of building AI is efficiency — to evaluate quickly whether an AI performs well or not. Deep learning naturally progresses faster in that regard. So when comparing tuning speed and output analysis between deep learning, NLP, and LLMs, their improvements feel proportional, like plotting points on a 2D graph.

LLMs tend to reason better (“make sense”) than they are accurate because they’re derived from broadly generalized reasoning models. When I first encountered LLMs, I wondered if using Internet text data caused them to speak confidently even when wrong. After all, they’re built by companies — and those companies care about pleasing users. Since “truth can be harsh,” they likely blend psychological principles to make AI responses sound personable. As Aristotle said, people are guided more by emotion than reason — and that principle still holds mostly true today. So models are tuned to respond in ways that feel pleasant, encouraging users to return.

Among AI models using big data for NLP, LLMs were the first to cross that threshold successfully. Because they were the first, “LLM” became a proper noun — a shorthand for “large language model.” Conceptually, it can be expressed with extremely concise mathematics (like 2 lines of equations), similar to how XGBoost or other ensemble algorithms summarize their principles.

### Ensemble Models

NLP relies more on data volume than precision. In contrast, XGBoost, LightGBM, and CatBoost operate on top of base models — typically either regression or classification ones. Their ensemble process itself is single-staged but fine-tunes for higher efficiency. Technically, that makes them regression/classification models rather than their own distinct AI types.

Decision Tree and Random Forest also belong here — they use parallel voting schemes. Whether they’re considered regression or classification depends on the type of base learner used in voting. Typically, they default to classification due to binary thresholds (like 0.5).  

You can also think of it this way: regression/classification is defined by whether, after an operation (not a simple computation), one set can be eliminated. Since that’s possible, the default is regression. However, if the elimination is reversible or the set can be reused, it can also act as classification — but defaults take precedence in defining models.  

Conceptual categorization must be goal-oriented — efficiency alone (like speed) doesn’t qualify as a conceptual distinction. I didn’t refer to any blogs while writing this, but I think the reason schools don’t teach this integrated view is because curricula often focus too much on specialization instead of unification.

*  
In short:
- `=` means substitution (“a = b” → replace a with b)  
- `==` means a logical check (returns true or false).  

---  

In NLP, the Transformer model mainly operates in a two-dimensional structure — since text itself is inherently 2D in coding terms. So, when we talk about the Transformer in NLP (let’s call it Transformer-NLP), it’s based on 2D commonalities and doesn’t go beyond that dimensional framework.

On the other hand, the Transformer model mentioned in @essay_to_know_beatitude’s post on December 27, 2025, refers to a broader AI model that can go beyond traditional deep learning’s three-layer foundations. While both share the same family-like structure (A and B being of the same lineage but serving different purposes), their objectives and the way they manifest are quite different.

You can think of the general Transformer-AI model as having a three-dimensional staircase-like triangular base. From there, it can either add more structure (“flesh”) — such as spheres or circular forms — or twist itself into higher dimensions (4D, 5D, and so on). Each twist or fold increases its dimensionality. Maintaining one stable axis while creating new ones is the key idea here.

In that sense, the 3D version is already complex; higher dimensions just represent more intricate variations of the same base. The “steps” of the staircase correspond to neurons or iterative layers, but the number of steps doesn’t determine the model type — much like how \\(x^1, x^2, x^3\\) express growth in scale, not fundamental change. So, whether in NLP or general AI, the classification (e.g., Transformer, XGBoost, Random Forest) isn’t affected by the step count itself; only by the model’s structural nature and the dimensional relationships it maintains.

Creating a “Big Brother”-type system isn’t just about coding; during data input and verification, even if wrong concepts spread through simple OX-type questions, that’s not the main issue. What really matters is that in any service—especially commercial ones—things will eventually break down. So, in my view, studying at least three areas within psychology, specifically education-related fields, and comparing their inferential patterns should be enough to expose the weak spot. Once you hit that, the rest collapses easily.
This idea is similar to what @essay_to_know_beatitude mentioned on December 28, 2025—“just hit the head.” From my observation, the hardest parts to capture are twofold: those that are easily inferable (you just need a bit of service analysis knowledge) and those where the real core is hidden so deeply and connected through so many conditional cases that breaking them becomes almost impossible.
In the service sector, it’s enough to plant vague words like “might” or “should” instead of explicit ones like “will”—that alone makes it hard to trace. In coding, that’s equivalent to the difference between 0 and 1—tiny but crucial. This isn’t really the developer’s fault. Theoretical foundations are usually built by a small number of academics or researchers, and any given lab might have only a few graduate students or postdocs working across various topics each year, making comprehensive oversight nearly impossible.
That’s why it would be easier for external researchers—not service companies—to manipulate things covertly. The person who wrote the theoretical foundation might not even be the one legally responsible for the consequences.
In the second case—the finely detailed “Big Brother” design—computing knowledge alone won’t help. Using open libraries or compilers makes the system easier to detect. Instead, look into software engineering research that focuses on system-wide program error detection. These positions typically require at least a master’s or PhD degree, but surprisingly few experts exist, mainly because it overlaps heavily with project planning, which developers rarely find engaging. That’s the real weak point—extremely cost-effective to target, yet the hardest to fix.
As for solutions in education, I’ve noticed that for at least the past five years, the most dominant services in all fields rely on recommendation algorithms—think of YouTube’s recommendation system. The actual code is proprietary, but Instagram, for instance, already makes its keyword classification system visible, and such data should ideally be shared. Many companies using it already share internally.
If you aggregate that data, feed it into ChatGPT or Gemini, and generate standardized profiles for individual interests, career paths, or health conditions, that could be a breakthrough. It’s not about app recommendations—it’s about interests. Hyper-personalization is inevitable, and I believe the best approach is to fix the basic format but require at least four lines of updates each time something changes, plus allow users to design their own structure.
This mechanism resembles the AI described in the post I mentioned earlier. Similarly, for the later, highly detailed “fragmented” builders I mentioned, connecting them through the same method would probably resolve the bigger issues too.

From a specialist’s perspective, when classifying AI in combination with the above content, I see three main ways to categorize it:
1.	AI can be classified into AI / Machine Learning / Deep Learning.
2.	AI can be classified into Machine Learning / Deep Learning / Generative AI.
3.	AI can be classified into AI / Machine Learning / Deep Learning / Generative AI.
4.	AI can be classified into Machine Learning / Deep Learning.
Cases (1) and (3) include “AI” as a separate category because they treat AI as the broadest classification level, encompassing all algorithmic frameworks — not just the four types of basic arithmetic operations but the overall formalized logic frameworks.  
In contrast, cases (2) and (4) exclude AI as a separate type, since any operation that involves computations beyond the four basic arithmetic operations would already fall under Machine Learning or Deep Learning.
Machine Learning refers to models that can be represented through digital logic operations such as OR/XOR — in other words, classification and regression that go through just one layer, not involving neuron-like architectures. Theoretically, reinforcement learning can also be included here, even though it’s rarely commercialized in this form.
Although reinforcement learning can be seen as overlapping with Deep Learning, in practice, it’s generally considered part of Deep Learning — that’s why systems like AlphaGo are referred to as Deep Learning models.
Generative AI includes LLMs for text but also image and other modality-generating models. It’s often said that generative AI belongs within Deep Learning, but that classification is mostly due to business conventions, not technical foundations.  
AI model objectives can vary — prediction, generation, or dimensionality reduction (e.g., principal component analysis). Generative AI is so named because among these goals, it focuses on generation and shares common generative principles across modalities. PCA is an essential learning concept because it represents the underlying axes of transformation. Still, as you know, alternative axes can also be designed without using PCA.
As for the code name I can’t recall — it’s true that the term Transformer first appeared in the NLP field. Since it originated in natural language processing, the developer who named it must have had a strong sense of ownership over the name. Even though the concepts differ slightly, I once analyzed the naming logic based only on the term and asked whether it matched conversational LLMs — surprisingly, everything lined up perfectly.  
Funny enough, I think NewJeans’ “Attention Is All You Need” title might have been inspired by that famous paper — the developer humor just tracks perfectly.


---





**한국한국

Ai는 1:1대응은 안해도 됨
Ai종류가 기본 크게 4가지 회귀/분류/강화학습/딥러닝 이라규 생각함

회귀 분류는 1개만 예측
강화는 2개(2단계라는 말과 동일함)
딥러닝은 3개(마찬가지)
알파고가 딥러닝이 아니라 강화학습인 이유가
뉴런 기반 알고리즘이 각 뉴런 끼리 1:1 소통이라 2단계라 강화학습임
강화학습 처음 배울 때 그 길들이는 느낌의 예시가 틀리진 않았는데 저게 정확한거 같음


회귀와 분석 설명
딱 봤을 때는 규칙이 분포로 표현되어있을 때 0.5 기준으로 회귀 분류를 나뉘게 하는데 0.5이상이 회귀라 logistic regression인데 좀 짜증나는게 logistic regression은 트리 같은 걸로도 cli 기준에서는 일반 알고리즘이 훨씬 편해서 regression이라 안써도 되는데
저거 3차원 기준이라 regression으로 씀
좀 비효율 적입
근데 딱봤을 때 그래프 이상(2차원이라는 소리 아님), 즉 이미지(logistic regression)처럼 위 규칙기준 분포로 설명해야하고 matrix 유무 기반으로classification으로 설명해도 되는데 실제로는 방향이 반대임
분포에서 규칙을 찾아야함
성질이 두 개나 양자역학 같은 거는 아니고
Ax + b 도 classification 임
연산 후(계산 아님) 집합 중 하나를 없앨 수 있냐 없냐가 두 정의를 나눔
힘과 스칼라로 표현은 할 수 있는데
벡터와 스칼라임

나머지 딥러닝과 강화학습은 단계만 늘리면 됨

NLP(자연어 처리 알고리즘)
LLM(nlp 대용량ver임 빅데이터는 전처리과정이라 신경 안써도 되는 개념임)
일단 nlp 는 그냥 강화까진 아니고 1단계인 회귀 분류에서 전처리(이미지 데이터 0101로 바꾸는 거랑 같은 개념, ai가 아니라 1:1대응 변환 알고리즘임)이랑 회귀 분류 중 하나랑 섞고 앙상블 하듯이 하는 방법인데
좀 발전이 어려웠던 이유가 3단계인 딥러닝이 이게 그 유추라는 개념이랑 같은거라(코딩 외적인 단어와 같은 뜻임) 
딥러닝이 결과적으로는 한 분야에서 핵심을 찾는 문제를 풀면 다른 분야에서 동일하게 핵심을 찾을 수 있어야함. 다른 분야에 대한 데이터는 처음 보자마자 찾는 것과 같음
그런데 nlp는 1단계와 전처리를 하는 앙상블로 딥러닝과 유사한 결과를 낼 수 있어야해서 결과지 받으면 nlp는 말그대로 자연어처리ai라 언어로 나오게 할 수 있어서 그냥 사람 머리로는 이게 틀렸다 맞다 할 수는 있는데 효율이 하나도 없음. 컴퓨터가 0101기반이라는 뜻은 숫자로 표현하는게 가장 효율적인 수단이라는 뜻인거임. 여담으로 그래서 이과가 컴공을감. 코딩 자체만으로는 문과가 조금 더 잘하는데 깊게 파면 숫자로만 표현해야할 단계가 있어서 수학 이론이 많이 필요해서 고등교육인 대학에서 이과위주로 뽑을 수 밖에 없는거임. 전자공학과를 그 카이스트 전산학과(컴퓨터공학과 와 사실상 동일)와 같이 이름을 바꾸면 문이과 분리 가능할 거 같기도 함. 여튼 ai는 수치화해서 이 ai가 성능이 있다 없다로 판단하는게 ai를 개발하는 이유 효율적으로 살자!! 와 같은거라 수치화, 즉 이 ai가 잘 만들어졌나-판단의 속도가 문제임
근데 딥러닝이 딱봐도 일단 발전이 빠르겠지? 그래서 딥러닝 좀 튜닝하고 결과 좀 분석하는 속도와 nlp, 그리고 더 발전된 llm 튜닝이 좀 못해도 체감상은 2차원 그래프처럼 비례해서 발전한거라 생각함
Llm이 좀 정확도보단 유추가 더 잘되어있고, 말되게는 적네라고 하는건 애초에 저 1,2,3단계 모두 유추 위주 모델이라고 일반적으로 여겨지는 이유가 아니고 쓰레기값. 내가 처음 llm봤을 때 인터넷에 떠도는 글도 데이터로 써서 가짜 정보를 있는 듯이 말하는 거라는 문제가 있나 생각했거든. 근데 뜯어보진 않았지만 전반적으로 느껴지는 알고리즘이 그 질문자가 일반적으로 사람이겠지? 그리고 llm만드는 곳이 기업이잖아. 비위를 마쳐야해. 왜 돈을 벌어야 해. 그래서 팩트는 폭력이라고 하잖아? 근데 이게 정형화 되어있진 않으니깐 심리 이론을 썼겠지 당연히? 질문 종류가 수백가지 이상은 분명한데 안썼을 리가 없겟지? 그게 오래전부터 아리스토텔리스 그 심리 대가가 일관적으로 말한게 “일반적인” 사람은 이성보다 감정이 우선이다. 물론 현재 조금은 바꼈는거는 아는데 크게 다르진 않단 말이야. 비위 위주로 코딩해서 그럼. 근데 그러면 기분은 나쁘지 않아. 저새끼 너무 잘났는데? 보다 저새끼 어휴 나쁘지 않아 ㅇㅇ 쓸만함-> 기대됨으로 바뀌거든? 좀 더 다시 사용할 이유가 더 많아. 빅데이터를 효율적으로 nlp에 쓰는 ai중에 가장 먼저 그 선을 넘었으면서 처음 한게 llm이라는 ai임. 그런데 자주 언급을 하고 기초가 되면 저게 그냥 고유명사가 됨. 가장 처음은 논문으로 ”llm : new ai model 개발했어요~“랑 같았음. 이건 회귀같은 ai론이 아니라 xgboost같은 앙상블한 거를 간결하게 표현해서 수학 공식 2줄 이하로 표현한거임. Nlp는 앙상블은 했지만 수학 공식 2줄 이하로 표현할 수 있는거는 좀 모델(ai)라고 표현을 하는데 nlp는 한 문장도 빠짐없이 읽었으면 감 잡을거임. 그거 맞음. 다만, xgboost와 같은 분류◇회귀는 nlp처럼 분류/회귀랑 같은 결로 취급받는 일은 없음
왜냐하면 nlp는 문자 특화로 양으로 승부했음. 이게 좀 같은 결로 취급받을려면 어느 분야나 힘이 비슷해야하는데, 문학작품을 분류할 때, 시와 소설만으로도  분류할 수 있는데 수필이라는 뜬끔포이자 맞긴해ㅇㅇ 하면서 3가지로 분류하는거랑 같은 결이라 생각하면 됨. 양과 질과 같은 판단기준이 있는데 여기는 좀 다르고 분야별로 달라서 패스. 
Xgboost와 같은 분류◇회귀 모양은 앙상블(==◇)이라는 기법 힘이 더 쎄다고 말하면 안됨. 앙상블은 이름에서 느끼듯이 분류와 회귀가 변수라면 앙상블는 기본 4개 사칙연산기호 이상인데, 해당 ai는 1단계만 쓴거니간 분류/회귀 ai모델에 대한 효율성 튜닝이라는 표현이 더 정확함 
그건 개념 분류가 안되지
그럼 xgboost는 분류냐 회귀냐
XGBoost / LightGBM / CatBoost은 목적에 따른게 아니라 앙상블(사칙 이상은 다 얘임)은 애초에 기반이 되는 모델이 있음. 모델 구조가 병렬식이 못됨. 그래서 분류/회귀인 1단계인거고, 분류/회귀 중 하나로 말할 수 있음 되면 2단계고, matrix기반으로 집합을 나누는 ai혹은 알고리즘이 classfication이라고 말을 하면 정확히는 틀린 소리인 이유이기도 함.

전공자는 Decision Tree / Random Forest가 이런 분류◇회귀 모델에 더 있는것을 알텐데 이건 병렬에서 언급한 것과 같은 모형구조고 둘 다 투표를 어러번 진행하는 모델인데 투표에서 어떤 종류(분류/회귀) ai를 쓰느냐에 따라 분류/회귀로 정의할 수 있음. 기본은 분류긴 함. 왜? 0.5이상이니깐. 무엇보다
회귀/분류가 연산 후(계산 아님) 집합 중 하나를 없앨 수 있냐 없냐가 두 정의를 나눈다고 했는데 없앨 수 있으니깐 기본은 regression임 
다만 없앨 수는 있는데 재 사용도 가능하게 짤 수 있어서 classification으로도 볼 수 있음. 하지만 디폴트가 우선이 맞지 관용으로 여겨지는 순간 개념에 포함됨 그래서 regression임
부가적으로는 저런 개념적 분류는 목적이 있어야하고 속도 등과 같은 효율성이라는 목표만으로는 개념적 분류가 될 수 없음 
내 생각 블로그들 안보고 적은건데 틀릴 수는 있는데 이걸 학교에서 언급을 안하는거는 교육과정이 너무 세분화만 추구하고 통합을 추구 안해서라고 생각함
여기까지 개쉽게 썼으니깐 질문 안받음~

*
= 는 치환 a= b다 에서 a를 b로 치환
==는 o,x로 나옴. 

NIp에서 transformer랑 @essay_to_know_beatitude 에서
25.12.27에서 올린 게시물에 언급한 ai모델 transformer은 다름 Ai 모델 transformer은 딥러닝(3단계)가 기본은 맞는데, 4단계 이 상 갈 수 있음
두 transformer 같은 계열이라고 보기는 하는데, 같은 계열이라는 점에서는 A nB와 같은 느낌인데, 걍 목적이 다르면 다르게 쓰이고 너무 다른 양상으로 표현된다. NIp에서는 transformer가 글자는 딱봐도 2차원이잖아? 코딩적으로도 그래서 딱히 설명안라고 넘어가 면 2차원 기준으로 공통점 기반 죽약임
Transformer모델은 일단 축약이 아니라 그 아래 설명하는 생성형
ai은 아니고 그냥 딥러닝일 때도 있는데 그 글로 된 설명 보면 어렵지 만 적어도 딥러닝부터는 앞선 위 글에서 유추할 수 있듯이, 이미지화 혹은 좌표화해서 보면 이해가 쉬움. 무한으로 걸어갈 수 있는 계단형 삼각형 3차원 구조(각 직선?마다 n번씩 꼬우면 4차원 5차원 됨) 꼬 우고 n차원 된다는 거를 @essay_to_know_beatitude에도 이해 를 못했을 수도 있는데 그냥 축 하나 균형 유지하고 더 만들 수 있으 면 되는거임
여튼 3차원이 가장 어려우니깐 3차원 삼각형 모양으로 이어진 계단 형태 유지하면서 살을 더 붙이던가(원, 구), 더 꼬우던가 하는 작업 임. 계단에서 위의 2차원 nlp와 동일한 작업이 있기도 한데, 같은 정 보라도 nlp는 무조건 2차원만 하고 차원 안바뀐다. transformer모 델(ai)는 3차원 저 기본 뼈대가 무조건 유지되고 살을 더 붙이던가 혹은 말했던 것 처럼 꼬와서 차원을 더 상승시키던가 함. 차원이 무조 건 3차원 이상이고, 계단 수는 상관없음. 계단은 뉴런 같은건데 계단 수는 반복적인 2단계 횟수임.
1, 2,3단계는 x^1, x^2,x^3(x 3제곱)과 같은거라 계단 수 자체는 transformer ai(모델)과 nlp transformer의 분류 및 모델 분류 (xgboost, random forest 등)에 전혀 영향이 없음.


빅브라더 만드는 방법은 
데이터 입력 및 확인 과정에서 OX문제에서 틀린 개념을 전파해도 되는데
그것보다는 어쨌든 서비스(특히 상업적)에서 문제가 터질거니 심리쪽, 정확히는 교육 분야를 못해도 3개는 비교하고 유추에 대한 대답을 내놓을거라고 예상한다 이것만 터뜨리면 됨
이게 25.12.28에 @essay_to_know_beatitude 언급한 대가리만 치면 된다와 유사한데, 내가 느끼기엔 가장 잘 못잡는게 이런 상대적으로 쉽게 유추가 가능한(서비스 공부를 조금 하면 됨) 부분이랑 엄청 디테일하게 핵심 부분에 몰래 숨겨놔서 경우의 수 죄다 고려해서 이어놓으면 부수기 어려운 것 2종류다
서비스 부분에서 진짜 찾기 어려울려면 된다와 한다와 같은 애매한 단어를 하나만 넣어놔도 충분하다. 코딩에서는 0,1차이로 될 확률이 너무 높거든. 개발자의 실책이라기보단 이론이라는게 교수/연구원이지만 상대적 소수의 전문가가 공개적으로 써놓은 것을 기반으로 하고 전문가는 소수일 수 밖에 없고, 한 연구실에 많아도 3명이 한 해에 석박사 포닥 다양한 형태로 졸업하고 연구실에도 다양한 주제를 다루기에 어렵다. 그래서 서비스 회사쪽이 아니라 아예 외부 연구원쪽에서 속이기 참 쉬울거같다
해당 이론을 적은 사람이 또 책임자가 되어서 그거 또 문제가 되고 서비스 쓰면 알겠지만 책 찾아도 안나오는 내용이 잘 포함되어있어서 좀 기업끼리 공유하는 것도 넣어진게 보여서 문제가 클 듯하다
문제를 만든 인간이 법적 책임을 무는 책임자가 아닐 수도 있다
후자의 케이스, 완전 디테일하게 빅브라더를 조각한 케이스는 컴공적 지식으로는 라이브러리나 오픈소스 말고 마찬가지로 공개된 컴파일러는 좀 들키기 쉬우니깐 아닐거고 
연구에서 소프트웨어 공학쪽으로 연구를 하는 쪽을 보면 전체 소프트웨어 혹은 돌리는 프로그램 단위로 에러 찾는거 있는데 이건 석박사 이상이 고용 최소조건이지만 딱봐도 박사급 이상이 실무에서 일할 수 있는데 이 분야 인재가 개 적어서, 왜? 이게 (협업 프로젝트)기획이랑 가장 맞닿아있는데 개발자는 해당 기획 관심이 좀 과하게 없는 편이거든. 여기서 조지면 된다. 엄청 가성비 있는데 가성비가 없고 가장 어렵다. 
여기서 내가 생각하는 교육쪽 해결책은 적어도 5년 전부터 메이저 앱 보면 분야 상관없이 추천 알고리즘, 유튜브 추천 알고리즘 같은게 서비스가 가장 활성화 되었고, 메인 탑3서비스인게 계산 안해도 다 느끼고 있다. 알고리즘 코드 공유는 기업 자산이라 못하겠지만
그 인스타그램에 보면 그 자기 알고리즘 대로 키워드 죄다 뽑아놓은거 있는데 이거 원래 공유해야하고 이거 쓰는 곳은 알다시피 죄다 공유함
그거 기반으로 모아서 chatgpt나 gemini 돌리고 규격화된 현 관심사 및 진로/취향/건강상태 나오게 권장하면 많이 풀릴거같다는 예상이다(위 이미지는 perplextiy인데 연구쪽 준비로 나는 좋았음 평상시는 이 두개가 좋음) 앱 추천이라기보단 관심사가 더 중요하다 생각하고 핵개인화가 확정이라 특히 이 방법 기반으로 해당 규격은 고정 대신 매번 어떤게 바꼈는지 최소 4줄 이상 적고 이 외 스스로 규격을 만들 수 있는 구조로 가는게 난 좋다 생각한다
이 게시글에 적혀있는 Ai와 유사한 메커니즘이다
마찬가지로 후자라 언급했던 디테일 조각자의 경우도 같은 방법으로 이으면 큰 문제 없을거같다는 예상만 간단하게 적는다 


---


전공자용~
AI를 분류하는 건 위의 내용과 결합해서 내가 봤을 때는 크게 3종류인데
1. AI는 ai/머신러닝/딥러닝 3개로 분류된다
2. AI는 머신러닝/딥러닝/생성형ai로 분류된다
3. AI는 ai/머신러닝/딥러닝/생성형ai로 분류된다
4. AI는 머신러닝/딥러닝으로 분류된다

1,3번 ai를 따로 포함하는 이유는 알고리즘(4종류 사칙연산의 공식화된 방법말고도 4종류만 쓰는 거)가 가장 큰 분류로 먼저 여길 경우고
2,4번 경우는 4종류 사칙연산 이상의 연산이 한 번이라도 포함되면 2번처럼 ai를 따로 분류하지 않는다
머신러닝은 말그대로 or,xor 디지털 회로 연산처럼 표현할 수 있는 경우, 즉 뉴런이 아닌 위 이미지 글 참고하면 알 수 있듯이 1단계만 거치는 분류/회귀만 이다 다만 이론상에서나 상용화가 안된 강화학습이 여기 포함될 수 있다.
딥러닝이랑 머신러닝 교차지점 혹은 딥러닝에 강화학습이 있다고 여겨질 수 있는데 여기서는 딥러닝에 실전은 강화학습이 포함된다. 그래서 알파고가 딥러닝이라 부르는거다
생성형ai는 llm은 언어만이고, 이미지 등도 당연히 어느 모델들처럼 처리하는데, 딥러닝 안에 주로 이 생성형 ai가 있다고 하는건, 앞서말한 비즈니스적 이유로 생성형 ai의 위치 및 이름이 정해진 건 아니다. Ai 모델의 목표가 예측도 있고 생성도 있고 주성분 분석처럼 축약도 있다. 말그대로 생성이라 생성형 ai인거고 위 목표중 가장 집중화되어있고 공통된 기능을 공유하기에 분류가 된거다. 주성분 분석이 무조건적으로 배우는 개념인 이유다. 가장 축으로써 거기서 다 변환하는데 주성분 분석을 쓰지 않고도 알다시피 다른 축을 만들어서 하기도 한다. 자세한건 찾아봐. 

이거 코드는 기억안났는데 nlp에서 transformer라는 이름이 먼저 나온건 맞거든? 그러면 자연어쪽이니깐 이름에 대한 욕심이 이 개발자가 쥰내 많단 말이지? 그 사람 자존심이니깐 두 개 내용 좀 다른건 알고 있어도 개념좀 까먹고 이름만 보고 정리한 후에  대화형 llm에 맞는지 물어봤는데 개꿀잼 고친거 없음 다맞은거 보이긴 행
Newjeans의 attention is all you need가 이 논문 이름에서 나온 거 같기도 함
개발자식 유머가 통하네 











맞는지 확인은 했어요 -




