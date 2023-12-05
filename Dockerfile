FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/diegochine/thesis.git

WORKDIR /thesis

RUN apt-get install python3 python3-pip -y
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install -r requirements.txt

ENV WANDB_API_KEY=bb91b382cc121df7e109ec0ad0275f1accc4c2f4

CMD ["wandb", "agent", "unify/long-term-constraints/ms3mz0fx"]
