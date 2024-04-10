from mtsoo import *

from mfea import mfea
from mfeaii import mfeaii
from mfea-de import mfeade
from scipy.io import savemat
from GNBG import GNBG

def callback(res):
  pass

def main():
  config = load_config()
  problems = [GNBG(1), GNBG(2)]
  mfeade(problems, config, callback)
  # mfeaii(functions, config, callback)

if __name__ == '__main__':
  main()
