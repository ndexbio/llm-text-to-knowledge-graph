FROM continuumio/miniconda3:24.11.1-0

RUN apt-get update -y
RUN apt-get install build-essential -y

RUN mkdir /opt/llm_text_to_knowledge_graph
COPY . /opt/llm_text_to_knowledge_graph/

RUN grep -v "indra" /opt/llm_text_to_knowledge_graph/requirements.txt | grep -v "libdeeplake" | grep -v "pandas" > /opt/llm_text_to_knowledge_graph//requirements_stripped.txt

RUN chmod a+x /opt/llm_text_to_knowledge_graph/docker/cyweb_bel_main.py
RUN pip install -r /opt/llm_text_to_knowledge_graph/requirements_stripped.txt
RUN pip install -e /opt/llm_text_to_knowledge_graph

RUN python -m nltk.downloader -d /usr/local/share/nltk_data all

RUN python -c "import gilda.resources as gr; gr.get_grounding_terms(); gr.get_gilda_models()"


ENTRYPOINT ["/opt/llm_text_to_knowledge_graph/docker/cyweb_bel_main.py"]
CMD ["--help"]
