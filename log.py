import logging

formatter = logging.Formatter(fmt='%(asctime)s|%(name)s|%(levelname)s|%(message)s', datefmt="%Y%m%d-%H:%M:%S")

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("./log", mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger = logging.getLogger("dpm")
logger.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

epoch = 100
train_metrics = {
  'learning_rate': 1.99999,
  'loss': 0.1837492927,
  'time': 739710,
}

msg = f"Epoch:{epoch} " \
      f"loss@train:{train_metrics['loss']:.8f} " \
      f"lr:{train_metrics['learning_rate']:.6e} " \
      f"time:{train_metrics['time']:.2f} sec"

print(isinstance('tst', dict))
#logger.info(msg=msg)
