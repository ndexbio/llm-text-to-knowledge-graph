FROM continuumio/miniconda3:24.11.1-0


RUN mkdir /opt/llm_text_to_knowledge_graph
COPY . /opt/llm_text_to_knowledge_graph/
#RUN pip install -r /opt/llm_text_to_knowledge_graph/requirements.txt

ENTRYPOINT ["/opt/llm_text_to_knowledge_graph/python_scripts/cyweb_bel_main.py"]
CMD ["--help"]