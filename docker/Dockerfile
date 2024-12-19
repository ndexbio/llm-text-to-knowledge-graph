FROM continuumio/miniconda3:24.11.1-0

RUN apt-get update -y
RUN apt-get install build-essential -y

RUN mkdir /opt/llm_text_to_knowledge_graph
COPY . /opt/llm_text_to_knowledge_graph/

RUN chmod a+x /opt/llm_text_to_knowledge_graph/python_scripts/cyweb_bel_main.py
RUN pip install -r /opt/llm_text_to_knowledge_graph/requirements.txt

ENTRYPOINT ["/opt/llm_text_to_knowledge_graph/python_scripts/cyweb_bel_main.py"]
CMD ["--help"]
