FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
MAINTAINER Kozo Chikai <tokoroten0401@gmail.com>

# add an user
RUN useradd -m python_user
WORKDIR /home/python_user
#USER python_user

# apt-get
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install git vim curl locales mecab libmecab-dev mecab-ipadic-utf8 make xz-utils file sudo

# install pyenv
RUN git clone git://github.com/yyuu/pyenv.git ~/.pyenv
ENV PYENV_ROOT /root/.pyenv
ENV PATH /root/.pyenv/shims:/root/.pyenv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin

#ENV HOME ~/
#ENV PYENV_ROOT $HOME/.pyenv
#ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

#RUN mv /bin/sh /bin/sh_tmp && ln -s /bin/bash /bin/sh
#RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
#RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
#RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
#RUN source ~/.bash_profile
#RUN rm /bin/sh && mv /bin/sh_tmp /bin/sh

# install anaconda
RUN pyenv install anaconda-2.4.0
RUN pyenv install anaconda3-4.2.0
RUN pyenv global anaconda3-4.2.0
RUN pyenv rehash

# install python3 packages
RUN pip install --upgrade pip
RUN pip install chainer==1.20.0.1
RUN pip install gensim==2.1.0

# install python2 packages
RUN pyenv local anaconda-2.4.0
RUN conda install -y theano=0.7
RUN conda install -y lasagne
RUN pyenv local anaconda3-4.2.0

# character encoding
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# add MeCab
WORKDIR /usr/src/mecab/
RUN mkdir -p /temp/mecab_src/ && \
git clone https://github.com/taku910/mecab.git  /temp/mecab_src/ && \
mv -f /temp/mecab_src/mecab/* /usr/src/mecab/ && \
 ./configure  --enable-utf8-only && \
make && \
make install && \
rm -rf  /temp/mecab_src/  && \
rm -rf  /usr/src/mecab/
RUN ldconfig
RUN git clone https://github.com/neologd/mecab-ipadic-neologd.git /usr/src/mecab-ipadic-neologd && \
/usr/src/mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n -y && \
rm -rf  /usr/src/mecab-ipadic-neologd && \
pip install mecab-python3
WORKDIR /home/python_user

# show information
ADD info.py /home/python_user/info.py
CMD ["/bin/bash", "python info.py"]