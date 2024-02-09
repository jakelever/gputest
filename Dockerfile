FROM huggingface/transformers-pytorch-gpu:4.35.2

RUN pip3 install datasets

ADD singleTest.py /

CMD python3 /singleTest.py

