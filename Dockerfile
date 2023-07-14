FROM sonic-openai:run

# RUN pip install gym==0.25.2 & pip install keyboard


RUN rm -rf /reinforcement-project/*  

RUN pip install google-cloud-logging


#  && git clone https://github.com/dew54/reinforcement-project.git
WORKDIR /reinforcement-project

COPY . ./

# RUN pip install importlib-metadata==4.13.0 && pip3 install pyopengl



EXPOSE 8080

# RUN python -m retro.import 'rom'

# RUN pip install gym==0.21.0

# CMD ["python", "run.py", "--foo bar"]
