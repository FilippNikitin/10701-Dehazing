import numpy as np
import cv2


class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class ListAverageMeter(object):
	"""Computes and stores the average and current values of a list"""
	def __init__(self):
		self.len = 10000  # set up the maximum length
		self.reset()

	def reset(self):
		self.val = [0] * self.len
		self.avg = [0] * self.len
		self.sum = [0] * self.len
		self.count = 0

	def set_len(self, n):
		self.len = n
		self.reset()

	def update(self, vals, n=1):
		assert len(vals) == self.len, 'length of vals not equal to self.len'
		self.val = vals
		for i in range(self.len):
			self.sum[i] += self.val[i] * n
		self.count += n
		for i in range(self.len):
			self.avg[i] = self.sum[i] / self.count
			

def read_img(filename, quadruple_space=False):
	img = cv2.imread(filename)
	if quadruple_space:
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		img = np.concatenate([img[:, :, ::-1], hsv, ycc, lab], axis=-1)
	else:
		img = img[:, :, ::-1]
	return img.astype('float32') / 255.0


def write_img(filename, img):
	img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
	cv2.imwrite(filename, img)


def hwc_to_chw(img):
	return np.transpose(img, axes=[2, 0, 1]).copy()


def chw_to_hwc(img):
	return np.transpose(img, axes=[1, 2, 0]).copy()
