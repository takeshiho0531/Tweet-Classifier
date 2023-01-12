FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y \
		sudo \
		wget \
		vim
WORKDIR /opt
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
	sh Anaconda3-2021.04-Linux-x86_64.sh -b -p /opt/anaconda3 && \
	rm -f Anaconda3-2021.04-Linux-x86_64.sh

ENV PATH /opt/anaconda3/bin:$PATH

RUN pip install --upgrade pip && pip install
	torch==1.8.1 \
	torchvision==0.9.1 \
	transformers[ja] \
	unidic-lite

WORKDIR /

CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token=''"]