FROM spark:python3

USER root

RUN pip install notebook

USER spark