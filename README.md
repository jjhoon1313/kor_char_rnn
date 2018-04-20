# 한국어 문장 긍/부정 분류 프로그램
# Korean Sentence Sentiment Analysis Program(POS/NEG)
#
# kor_rnn_demo.py 파일 실행하면 한국어 문장에 대한 긍/부정 테스트 가능
# kor_rnn_movie_demo.py 특정 영화에 대한 댓글 모음 파일 있다면 해당 파일 실행하여 테스트 가능
# data 폴더에는 영화 댓글 데이터, 영화 댓글 pos_tagged 데이터 등이 있습니다.
# model 폴더에는 word2vec 모델이 있습니다.
# kor_rnn.py 파일을 실행시키면 training이 진행되고, checkpoint 파일 생성한 이후에 demo 파일들 실행하시면 됩니다.
#
#
#
# If you want to test, run kor_rnn_demo.py # If you have the specific movie replies data, run kor_rnn_movie_demo.py
# There are movie reply datas and pos-tagged reply datas in folder named 'data'
# There are word2vec models in folder 'model'
# If you want to train model, run kor_rnn.py and you can change any hyper-parameters.
# With the ckpt file, you ran run kor_rnn_demo.py and kor_rnn_movie_demo.py
