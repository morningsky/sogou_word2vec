import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__=='__main__':
  program = os.path.basename(sys.argv[0])
  logger = logging.getLogger(program)

  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  logging.info("running %s" % ' '.join(sys.argv))

  if len(sys.argv) < 4:
    sys.exit(1)

  inp,outp,outp2 = sys.argv[1:4] #输入的语料（已分词 每一行按空格分开） 输出的word2vec模型 输出的向量文件

  model = Word2Vec(LineSentence(inp),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())

  model.save(outp)
  model.wv.save_word2vec_format(outp2,binary=False)
