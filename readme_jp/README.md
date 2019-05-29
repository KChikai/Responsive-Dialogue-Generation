# Neural Conversation Generation with Domains and Sentiments

入力発話のドメインに合致した応答を生成し、かつ感情を付加するニューラル対話モデルの[chainer][chainer]実装版．
ドメイン推定器として [tweet2vec][t2v_paper] を使用している．
tweet2vec の実装は[元論文の実装][tweet2vec]通り [theano][theano] + [lasagne][lasagne] をそのまま流用している．

[chainer]: https://github.com/pfnet/chainer "chainer"
[tweet2vec]: https://github.com/bdhingra/tweet2vec "tweet2vec"
[theano]: https://github.com/Theano/Theano "theano"
[lasagne]: https://github.com/Lasagne/Lasagne "lasagne"


## Description

***モデル図:***

![model](https://github.com/OnizukaLab/ncm_topics_emotions/blob/master/images/model-image.png?raw=true)

モデルの全体像は図の通りである．
応答生成器である[Sequence-to-Sequence (seq2seq)][s2s_paper]モデル上に，
Encoder部分に[tweet2vec][t2v_paper]，
Decoder部分に[Speaker Model][persona_paper]と[External Memory][ecm_paper]
を載せたモデルとなっている．

***学習図:***

![train](https://github.com/OnizukaLab/ncm_topics_emotions/blob/master/images/model-train.png?raw=true)

学習時は，図の様に二段階の学習を行う．
学習の分類については以下の通りである．

- Pre-Training
  - seq2seq+speaker model+external memoryの学習（学習データ: `volume_dir/data/seq2seq/rough_pair_corpus.txt`）
  - tweet2vecの学習（学習データ: `volume_dir/data/tweet2vec/tweet2vec_topic_trainer.txt`）
- Fine-Tuning
  - tweet2vec以外のモデルの再学習 （学習データ: `volume_dir/data/seq2seq/fine_pair_corpus.txt`）

***データセットについて:***

学習については，tweet2vec と seq2seq で学習に使用するデータフォーマットが異なる．
seq2seq のは以下のデータフォーマットを学習する．
    
    <post_sentence><TAB><comment_sentence>

データセットには一行につき上記の形式となっている．
`<post_sentence>`が入力発話，`<comment_sentence>`が応答文であり，
それをタブで区切ったテキストファイルをデータ形式としている．
発話文と応答文共に形態素解析済みのテキストとする（半角スペースで単語分割済み）．
また，seq2seqのfine tuningの際には，発話応答の会話ドメインラベルが必要である．
finetuning時のデータ形式は以下の様に表される．

    <post_sentence><TAB><domain_label><space><comment_sentence>

`<domain_label>`は会話ドメインラベル`<space>`は半角スペースを表す．
`<domain_label>`のラベルの表記は`__label__0`, `__label__1`...の様に表記を行う．
例えば，ドメイン数が二値の場合，ラベルは`__label__0`, `__label__1`の二つである．
seq2seq データは`volume_dir/data/seq2seq/`下に配置する．
デフォルトのファイル名は，pre-training用データは`volume_dir/data/seq2seq/rough_pair_corpus.txt`，
fine tuning用データは`volume_dir/data/seq2seq/fine_pair_corpus.txt`としている．

tweet2vec は元の実装通り以下の形式でのデータを取り扱う．

    <tag><TAB><sentence>

`<tag>`は`<sentence>`に対応するドメインラベルを表す．
デフォルトのファイル名は訓練データは`volume_dir/data/tweet2vec/tweet2vec_topic_trainer.txt`，
テストデータは`volume_dir/data/tweet2vec/tweet2vec_topic_tester.txt`としている．

seq2seqを学習するに当たって，低頻度語の置き換えに使用するword2vecのモデルと選定した感情語彙のリストが必要である．
以下に各データの説明を行う．

- `volume_dir/data/seq2seq/neo_model.vec`
  - fasttextでwikipedia dataを学習した分散表現モデルを使用する．
  [ここ][embedding]からダウンロードを行い，上記のファイル名で置く．

- `volume_dir/data/seq2seq/neg-extend.txt`
  - negativeな単語を集めたテキストファイル．一行に付き一単語の形式で記述する．

- `volume_dir/data/seq2seq/pos-extend.txt`
  - positiveな単語を集めたテキストファイル．一行に付き一単語の形式で記述する．


[embedding]: https://qiita.com/Hironsan/items/513b9f93752ecee9e670 "embedding"



## Features

各ディレクトリ構成は以下の通り．

- `volume_dir/*`
  - docker container にマウントするデータ群．

- `volume_dir/bin/*`
  - container 内で実行する実行ファイル群．

- `volume_dir/tweet2vec/*`
  - tweet2vecのスクリプトを入れたディレクトリ．

- `volume_dir/seq2seq/*`
  - seq2seqのスクリプトを入れたディレクトリ．

- `volume_dir/data/*`
  - 学習に利用するデータや学習済みモデルの保存先．データ形式が異なるのでtweet2vecとseq2seqで分かれている．



## Requirement

***Environment:***

[Docker][docker] を用いて環境構築を行う．
実験環境として， NVIDIA GPUが搭載されたUbuntu環境を想定している．
seq2seqはpython3系，tweet2vecはpython2系での実装であるため，container 内で [pyenv][pyenv] を用いて環境を分けている．
以下のパッケージはDockerfileでイメージ作成時に自動的にインストールされる．

Installed packages by Docker:

- [MeCab][mecab]
  - [mecab-ipadic-NEologd][neologd] (MeCabで用いる辞書として使用)
- python3
  - anaconda3-2.4.0
  - chainer==1.10.0
  - gensim
- python2
  - anaconda-2.4.0
  - theano==0.7.0
  - lasagne

[docker]: https://www.docker.com/ "docker"
[pyenv]: https://github.com/pyenv/pyenv "pyenv"
[mecab]: http://taku910.github.io/mecab/ "mecab"
[neologd]: https://github.com/neologd/mecab-ipadic-neologd "neologd"


***Dataset:***

必要なデータセットについてまとめる．データの詳細な説明は前節を参照．

- `volume_dir/data/tweet2vec/tweet2vec_topic_trainer.txt`: tweet2vecの訓練データ
- `volume_dir/data/tweet2vec/tweet2vec_topic_tester.txt`: tweet2vecのテストデータ
- `volume_dir/data/seq2seq/rough_pair_corpus.txt`: seq2seqの事前学習データ
- `volume_dir/data/seq2seq/fine_pair_corpus.txt`: seq2seqのドメイン会話データ
- `volume_dir/data/seq2seq/neo_model.vec`: wikipediaの分散表現モデル
- `volume_dir/data/seq2seq/neg-extend.txt`: ネガティブ感情語彙データ
- `volume_dir/data/seq2seq/pos-extend.txt`: ポジティブ感情語彙データ
- `volume_dir/data/test_input.txt`: テスト入力データ


## Usage

本システムの環境構築は，Dockerを用いることで設定を行う．
手順として，Dockerfileからimageを作成→`nvidia-docker`コマンドで作成したimageからcontainerを作成
→container内で学習用・テスト用シェルスクリプトを実行する，という流れである．
以下に手順の詳細を述べる．


1. Dockerfile からimageを作成する．
   
   ~~~
    $ nvidia-docker build -t <tag_name> . 
   ~~~
   
   作成したイメージからdocker containerを作成する．
   マウントするこのプロジェクトディレクトリの置き場を形式的に`/your/local/path/`としている．  
   
   ~~~
    $ nvidia-docker run --name <container_name> -v /your/local/path/ncm_topic_domains:/home/python_user/ 
    --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidia1:/dev/nvidia1 --device /dev/nvidiactl:/dev/nvidiactl 
    --device /dev/nvidia-uvm:/dev/nvidia-uvm -i -t <tag_name> /bin/bash
   ~~~
   
   再度，コンテナ内に入る場合は以下のコマンド．

   ~~~
    $ nvidia-docker exec -it <container_name> /bin/bash 
   ~~~

   
   
2. container内にアクセスした後，シェルスクリプトを実行することで学習を行う．
   実行スクリプトは`/home/python_user/volume_dir/bin/*`ディレクトリに配置している.
   `/home/python_user/volume_dir/bin/`ディレクトリ下で以下のコマンドを実行することで学習を行う．
   
   tweet2vecの学習．
   ~~~
    $ sh train_tweet2vec.sh
   ~~~

   seq2seqの学習．
   ~~~
    $ sh train_seq2seq.sh
   ~~~
   
   尚それぞれの学習に関するパラメータの設定については，
   tweet2vecは`/home/python_user/volume_dir/tweet2vec/settings_char.py`，
   seq2seqは`/home/python_user/volume_dir/seq2seq/setting_param.py`に記述している．
   
   
3. テスト時は以下のシェルスクリプトを実行する．（under construction...）
   ~~~
    $ sh test_model.sh
   ~~~
   上記のシェルスクリプトを実行することで，`volume_dir/data/test_input.txt` 
   に一行一文の形式のテストファイルに対して，出力を行う．
   出力結果は以下の様に出力される．
   
   <p align="center">
   <img src="https://github.com/OnizukaLab/ncm_topics_emotions/blob/master/images/model-output.png?raw=true") width="70%" height="70%">
   </p>   



## Reference 

Ilya Sutskever, Oriol Vinyals, and Quoc V. Le.
[Sequence to sequence learning with neural networks.][s2s_paper]
In Advances in Neural Information Processing Systems (NIPS 2014).

Dhingra, Bhuwan  and  Zhou, Zhong  and  Fitzpatrick, Dylan  and  Muehl, Michael  and  Cohen, William
[Tweet2Vec: Character-Based Distributed Representations for Social Media][t2v_paper]
In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)

Hao Zhou, Minlie Huang, Tianyang Zhang, Xiaoyan Zhu and Bing Liu
[Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory][ecm_paper]
arXivpreprint arXiv:1704.01074, September 2017

Jiwei Li, Michel Galley, Chris Brockett, Georgios P. Spithourakis, Jianfeng Gao and Bill Dolan
[A Persona-Based Neural Conversation Model][persona_paper]
In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)

[t2v_paper]: http://anthology.aclweb.org/P16-2044 "t2v_paper"
[s2s_paper]: http://papers.nips.cc/paper/5346-information-based-learning-by-agents-in-unbounded-state-spaces.pdf "s2s_paper"
[ecm_paper]: https://arxiv.org/pdf/1704.01074.pdf "ecm_paper"
[persona_paper]: http://www.aclweb.org/anthology/P16-1094 "persona_paper"



## Author

[@KChikai](https://github.com/KChikai)

