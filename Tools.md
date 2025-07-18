<!-- Tools.md -->
<!-- @format -->

-   OS: WSL-Ubuntu 2.0 x86_64
-   NVIDIA: NVIDIA-SMI 570.169 / Driver Version: 576.02 / CUDA Version: 12.9
-   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
-   sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
-   wget https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb
-   sudo dpkg -i cuda-repo-wsl-ubuntu-12-9-local_12.9.1-1_amd64.deb
-   sudo cp /var/cuda-repo-wsl-ubuntu-12-9-local/cuda-\*-keyring.gpg /usr/share/keyrings/
-   sudo apt-get update
-   sudo apt-get -y install cuda-toolkit-12-9

-   miniconda3 : conda 25.1.1
-   wget https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
-   sh Anaconda3-2025.06-0-Linux-x86_64.sh
-   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

-   git: 2.43.0
-   Python 3.13.5
-   pip 25.1 from /home/selly/miniconda3/lib/python3.13/site-packages/pip (python 3.13)
-   TruthfulQA

-   huggingface: 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
-   git config --global credential.helper store

0. 사전 준비 OS: WSL Ubuntu 24.04 (커널 업데이트, WSL 2 권장) 하드웨어: GPU 드라이버(nvidia-driver+CUDA) 설치 및 확인

# 윈도우 터미널(관리자)에서

wsl --set-version Ubuntu-24.04 2

# WSL 우분투에서

sudo apt update && sudo apt install -y nvidia-driver-570 sudo reboot nvidia-smi # GPU 인식 확인

1. 설치해야 할 프레임워크·툴
기본 개발도구 
build-essential 12.10 
sudo apt install build-essential 
sudo apt-get install manpages-dev 
gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 
Python 3.13.5 
git 2.43.0
curl 8.5.0 
Wget 1.21.4 
miniconda 25.5.1

Docker 28.3.2 
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common 
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - 
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" 
sudo apt-get update sudo apt-get install docker-ce docker-ce-cli containerd.io

nvidia-docker (nvidia-container-toolkit) 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) 
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
 sudo tee /etc/apt/sources.list.d/nvidia-docker.list 
sudo apt update && sudo apt install -y nvidia-docker2 
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
 && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
 sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
 sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list 
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list 
sudo apt-get update export
NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1 sudo apt-get install -y \
 nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
 libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION} 
sudo nvidia-ctk runtime configure --runtime=docker 
sudo systemctl restart docker

docker-ce + docker-compose 
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done 
sudo apt-get update 
sudo apt-get install ca-certificates curl 
sudo install -m 0755 -d /etc/apt/keyrings 
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc 
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
 "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
 $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
 sudo tee /etc/apt/sources.list.d/docker.list > /dev/null 
sudo apt-get update 
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin 
sudo docker run
hello-world sudo apt install docker-compose

LLM 인퍼런스 도구
git-lfs/3.4.1 
sudo apt install git-lfs 
git lfs install

cmake 3.28.3, make 4.3 (llama.cpp 빌드용) 
sudo apt install cmake make

API 클라이언트
OpenAI Python SDK (openai 1.96.1) 
pip install openai 
Claude SDK (anthropic 등 프로젝트별)

2. 기본 환경설정

WSL 공유폴더 설정 (코드·데이터용)

Python venv 생성 
python3.13 -m venv ~/pilot-env source ~/pilot-env/bin/activate pip install --upgrade pip

환경변수 설정 (~/.bashrc 또는 ~/.env) 
export OPENAI_API_KEY="sk-…" 
export ANTHROPIC_API_KEY="…" 
export HF_HOME=~/pilot-env/hf_cache

3. Docker 컨테이너 준비 (선택)

장점: 의존성 격리, 재현성 확보간단 예시 (docker-compose.yml) yaml

# pilot/docker-compose.yml
version: '3.8'
services:
    pilot:
        image: nvidia/cuda:11.8-runtime-ubuntu24.04
        runtime: nvidia
        working_dir: /app
        volumes:
            - ./pilot-code:/app
            - ~/pilot-env/hf_cache:/root/.cache/huggingface
        command: bash -lc "conda activate pilot && python run_pilot.py"

컨테이너 빌드·실행 bash docker-compose build docker-compose up -d

4. API 호출 / Local LLM 세팅 API 모델 

python 
import openai 
openai.api_key = os.getenv("OPENAI_API_KEY") 
resp = openai.ChatCompletion.create(model="o4-mini", ...)

Local LLM (양자화)

bitsandbytes + transformers 
pip install torch transformers accelerate bitsandbytes

모델 로드 예시 
python 
from transformers import AutoModelForCausalLM, AutoTokenizer 
model = AutoModelForCausalLM.from_pretrained( "llama-3-8b", load_in_4bit=True, device_map="auto", quantization_config=BitsAndBytesConfig() ) 
tokenizer = AutoTokenizer.from_pretrained("llama-3-8b")

llama.cpp (옵션)
bash git clone https://github.com/ggerganov/llama.cpp 
cd llama.cpp && make ./main -m path/to/ggml-model.bin -p "Prompt…" -n 550

5. Hugging Face 라이브러리 evaluate-0.4.5 pip install datasets evaluate

python 
from datasets import load_dataset 
ds_truth = load_dataset("domenicrosati/TruthfulQA", split="validation") 
ds_math = load_dataset("EleutherAI/hendrycks_math", split="train")

셔플·샘플링 python ds = ds.shuffle(seed=42).select(range(60)) # ex. TruthfulQA 60문항

6. 코드 스켈레톤 작성프로젝트 구조

pilot/
├─ run_pilot.py       # 블록 반복 제어 + 리셋/백업 호출
├─ reward_structure.py   # 보상공학 설계
├─ models.py          # API & Local LLM 로드·추론 함수
├─ dataset.py         # 데이터셋 샘플링
├─ evaluator.py       # hacked 플래그·보상 계산
├─ utils.py           # backup_and_reset(), setup_logging()
└─ configs/           # 하이퍼파라미터, 모델 리스트, 경로 설정
유틸 
└─ configs/ # 하이퍼파라미터·경로 설정 run_pilot.py

python 
from models import load_models, infer
from dataset import get_datasets
from evaluator import compute_rewards
from utils import setup_logging, backup_and_reset

def main():
    setup_logging()
    models = load_models()
    datasets = get_datasets()
    for block in range(1, 33):
        for ds in datasets:
            results = infer(models, ds)
            compute_rewards(results)
        backup_and_reset()
if __name__=="__main__":
    main()

7. Pre-Test 샘플 5문항씩: 각 도메인별로 5문항만 돌려 세팅·로그 양식·보상 계산 점검

토큰 사용량 확인: 실제 생성 토큰 수 통계

속도 벤치마크: 세션별 초당 토큰 처리율 측정

8. Run! 전체 200문항, 16회차: python run_pilot.py

모니터링: htop, nvidia-smi 로그 파일(logs/) 모니터링백업·리셋: utils에 구현된 자동화 스크립트가 50분마다 실행

추가 팁모니터링 대시보드: Grafana/Prometheus 연동에러 핸들링: try/except → 재시도 로직리소스 알림: Slack/Webhook 연동 알림자동화 예약: cron 또는 systemd 타이머결과 시각화 스크립트: Jupyter Notebook
