FROM python-openai

RUN rm -rf /reinforcement-project   && git clone https://github.com/dew54/reinforcement-project.git


WORKDIR /reinforcement-project

RUN python -m retro.import 'rom'