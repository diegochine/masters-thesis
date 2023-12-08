FROM ubuntu:22.04

RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/diegochine/thesis.git

WORKDIR /thesis

RUN apt-get install python3 python3-pip -y
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu && pip3 install -r requirements.txt
RUN printf "wandb login\nwandb agent \$1\n" > run.sh
RUN ["chmod", "+x", "run.sh"]

ENTRYPOINT ["/bin/sh", "/thesis/run.sh"]